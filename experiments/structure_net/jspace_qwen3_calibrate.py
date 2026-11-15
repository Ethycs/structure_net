"""Train-only choice-margin calibration for one native Qwen3 block.

The caller owns source authentication, reservations, the frozen native prefix
cache, and the accepted suffix. Teacher execution is BF16; captured activations
and derivatives are exact FP32 conversions of those BF16 values. Geometry and
polynomial fitting use FP32. Development rows are diagnostics, never fit data.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_bench_data import sha, write
from .jspace_quadratic_basis import block_krylov_basis
from .jspace_allblocks_rankfit import fit_benchmark_groebner


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 20861
    families: tuple = ("arc", "boolq", "hellaswag", "winogrande")
    cases_per_family: int = 32
    activation_positions: int = 16
    position_capacity: int = 416
    rank: int = 2048
    krylov_block_width: int = 2048
    krylov_steps: int = 1
    head_dim: int = 64
    anchors: int = 24
    remove_count: int = 1024
    svd_rank: int = 64
    oversample: int = 16
    power_iterations: int = 1
    ridge: float = .01
    eligibility_nrmse: float = .5
    chunk_rows: int = 256

    def __post_init__(self):
        positive = (self.cases_per_family, self.activation_positions, self.position_capacity,
                    self.rank, self.krylov_block_width, self.head_dim, self.anchors,
                    self.remove_count, self.svd_rank, self.chunk_rows)
        if any(type(v) is not int or v < 1 for v in positive):
            raise ValueError("Positive integer calibration dimensions required")
        if (type(self.seed) is not int or any(type(v) is not int or v < 0 for v in
                (self.krylov_steps, self.oversample, self.power_iterations)) or
                self.rank > self.krylov_block_width or self.anchors > self.svd_rank or
                self.head_dim % 2 or not math.isfinite(self.ridge) or self.ridge < 0 or
                not math.isfinite(self.eligibility_nrmse) or self.eligibility_nrmse <= 0 or
                not self.families or len(set(self.families)) != len(self.families)):
            raise ValueError("Invalid fixed calibration settings")


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA calibration requires a reservation guard")
    if guard is not None:
        guard()
    return device


def _config(metadata):
    value = dict(metadata["config"])
    value["families"] = tuple(value["families"])
    return CalibrationConfig(**value)


def _length(case):
    return len(case["input_ids"])


def select_calibration(cases, config):
    """Deterministic balanced training sample; reject evaluation even as input."""
    identities = [c["identity"] for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Train/development identities must be unique and disjoint")
    if any(c["split"] not in ("train", "development") for c in cases):
        raise ValueError("Evaluation cases are forbidden in calibration")
    if any(c["family"] not in config.families or not 0 < _length(c) <= config.position_capacity or
           not 2 <= len(c["choices"]) <= 4 or not 0 <= c["answer"] < len(c["choices"]) or
           c["loss_positions"][-1] != _length(c)-1 for c in cases):
        raise ValueError("Invalid family, context, choices or final answer position")
    chosen = []
    for family in config.families:
        available = sorted((i for i,c in enumerate(cases) if c["split"] == "train" and c["family"] == family),
                           key=lambda i: cases[i]["identity"])
        if len(available) < config.cases_per_family or not any(c["split"] == "development" and c["family"] == family for c in cases):
            raise ValueError("Each family needs the fixed training sample and development diagnostics")
        random.Random(f"{config.seed}:{family}").shuffle(available)
        chosen.extend(available[:config.cases_per_family])
    return sorted(chosen)


def pair_mask_derivative(q, k, gq, gk, *, query_heads, kv_heads, head_dim):
    """Derivative of a shared post-QK-norm RoPE-pair mask, before squaring.

    Inputs are normalized Q/K *before* RoPE. A pair mask commutes with RoPE.
    It does not commute with RMSNorm; raw q_proj/k_proj hooks are incorrect.
    """
    if query_heads % kv_heads or head_dim % 2:
        raise ValueError("Invalid GQA/RoPE geometry")
    length = q.shape[1]
    if q.numel() != length*query_heads*head_dim or k.numel() != length*kv_heads*head_dim or q.shape != gq.shape or k.shape != gk.shape:
        raise ValueError("Aligned one-case normalized Q/K and gradients required")
    query = (q.detach().float()*gq.float()).reshape(length, kv_heads, query_heads//kv_heads, head_dim).sum((0,2))
    key = (k.detach().float()*gk.float()).reshape(length, kv_heads, head_dim).sum(0)
    value = query + key
    return value[:, :head_dim//2] + value[:, head_dim//2:]


def _frozen(modules, choice_weight):
    tensors = []
    for module in modules:
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError("Every supplied teacher module must be frozen and eval")
        tensors.extend(module.parameters()); tensors.extend(module.buffers())
    tensors.append(choice_weight)
    if choice_weight.requires_grad or any(t.is_inference() for t in tensors):
        raise ValueError("Teacher tensors must be frozen normal tensors, not inference tensors")
    return [(id(t), t._version) for t in tensors]


def _float_rows(value):
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite calibration activation or derivative")
    return value.cpu().numpy()


def calibrate(cases, read_boundary, native_block, forward_teacher, choice_weight, output_dir, *,
              teacher_modules=(), teacher_fingerprint, config=CalibrationConfig(), guard=None,
              input_fingerprint=None, progress=lambda **_kw: None):
    """Capture one block against the caller's immutable accepted teacher.

    read_boundary(index, device) returns BF16 [L,d]; forward_teacher(x, case)
    executes this native block plus the frozen accepted suffix and final norm,
    returning normalized BF16 [1,L,d]. choice_weight contains the four original
    untied LM-head rows in A/B/C/D order. teacher_modules must include the norm
    and any suffix modules used by the callback, for mutation/freeze checks.
    No teacher logits or intermediate student states are fabricated here.
    """
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    cases = list(cases); selected = select_calibration(cases, config)
    if not isinstance(teacher_fingerprint,str) or not teacher_fingerprint:
        raise ValueError("An authenticated teacher fingerprint is required")
    device = _guard(native_block.self_attn.q_proj.weight.device, guard)
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian capture cannot run inside inference_mode")
    modules = (native_block, *teacher_modules)
    snapshot = _frozen(modules, choice_weight)
    attention = native_block.self_attn
    width, intermediate = attention.q_proj.in_features, native_block.mlp.down_proj.in_features
    nq, nkv, hd = attention.config.num_attention_heads, attention.config.num_key_value_heads, attention.head_dim
    if (choice_weight.shape != (4,width) or choice_weight.dtype != torch.bfloat16 or
            choice_weight.device != device or attention.q_proj.weight.dtype != torch.bfloat16 or
            not hasattr(attention,"q_norm") or not hasattr(attention,"k_norm")):
        raise ValueError("Native BF16 Qwen3 and four BF16 untied readout rows required")
    development = [i for i,c in enumerate(cases) if c["split"] == "development"]
    cal_rows = {i:r for r,i in enumerate(selected)}; dev_rows = {i:r for r,i in enumerate(development)}
    offsets = {}; cursor = 0
    for i in selected:
        offsets[i] = cursor; cursor += _length(cases[i])
    root.mkdir(parents=True)
    arrays = {}
    shapes = {"native_train": (len(selected)*config.activation_positions, intermediate),
              "native_development": (len(development)*config.activation_positions, intermediate),
              "mlp_jacobian": (cursor,intermediate)}
    for name,shape in shapes.items():
        arrays[name] = np.lib.format.open_memmap(root/(name+".npy"),mode="w+",dtype=np.float32,shape=shape)
    zero = lambda *shape: torch.zeros(*shape,device=device,dtype=torch.float32)
    gram, pair_scores, mask_gram = zero(width,width), zero(nkv,hd//2), zero(nkv*hd//2,nkv*hd//2)
    value_moments, value_means = zero(nkv,hd,hd), zero(nkv,hd)
    mean_sums, counts = zero(config.position_capacity,width), zero(config.position_capacity)
    case_ranges = []; records = []
    try:
        for index,case in enumerate(cases):
            _guard(device,guard)
            if _frozen(modules,choice_weight) != snapshot:
                raise ValueError("Accepted teacher changed during capture")
            length = _length(case); source = read_boundary(index,device)
            if source.shape != (length,width) or source.dtype != torch.bfloat16 or source.device != device:
                raise ValueError("Boundary callback must return exact BF16 [valid_tokens,width] on teacher device")
            if not torch.isfinite(source).all():
                raise FloatingPointError("Nonfinite native prefix boundary")
            if case["split"] == "train":
                mean_sums[:length].add_(source.detach().float()); counts[:length].add_(1)
            if index not in cal_rows and index not in dev_rows:
                continue
            x = source.detach().clone()[None].requires_grad_(index in cal_rows)
            captured = {}; handles = []
            handles.append(native_block.register_forward_pre_hook(lambda _m,a: captured.__setitem__("input",a[0])))
            for name,module in (("q",attention.q_norm),("k",attention.k_norm),("v",attention.v_proj)):
                handles.append(module.register_forward_hook(lambda _m,_a,out,key=name: captured.__setitem__(key,out)))
            handles.append(native_block.mlp.down_proj.register_forward_pre_hook(lambda _m,a: captured.__setitem__("mlp",a[0])))
            try:
                with torch.enable_grad():
                    output = forward_teacher(x,case)
                    if (output.shape != (1,length,width) or output.dtype != torch.bfloat16 or
                            set(captured) != {"input","q","k","v","mlp"} or
                            not torch.equal(captured["input"],x)):
                        raise ValueError("Teacher callback must execute the requested native block on the supplied boundary")
                    logits = F.linear(output[0,-1],choice_weight[:len(case["choices"])]).float()
                    other = [j for j in range(len(case["choices"])) if j != case["answer"]]
                    margin = logits[case["answer"]] - logits[other].mean()
                    sample = torch.linspace(0,length-1,config.activation_positions,device=device).long()
                    name,row = ("native_train",cal_rows[index]) if index in cal_rows else ("native_development",dev_rows[index])
                    arrays[name][row*config.activation_positions:(row+1)*config.activation_positions] = _float_rows(captured["mlp"][0,sample])
                    if index in cal_rows:
                        gx,gq,gk,gm = torch.autograd.grad(margin,[captured[k] for k in ("input","q","k","mlp")])
                        gx = gx[0].detach().float()
                        if not torch.isfinite(gx).all():
                            raise FloatingPointError("Nonfinite task-margin input Jacobian")
                        gram.addmm_(gx.T,gx)
                        mask = pair_mask_derivative(captured["q"],captured["k"],gq,gk,query_heads=nq,kv_heads=nkv,head_dim=hd)
                        pair_scores.add_(mask.square()); flat = mask.flatten(); mask_gram.add_(flat[:,None]*flat[None,:])
                        values = captured["v"][0].detach().float().reshape(length,nkv,hd).transpose(0,1)
                        value_moments.add_(values.transpose(1,2)@values/length); value_means.add_(values.mean(1))
                        start = offsets[index]; arrays["mlp_jacobian"][start:start+length] = _float_rows(gm[0])
                        case_ranges.append(dict(identity=case["identity"],family=case["family"],start=start,stop=start+length))
                records.append(dict(identity=case["identity"],split=case["split"],family=case["family"],
                                    answer=case["answer"],prediction=int(logits.argmax()),logits=logits.detach().cpu().tolist(),
                                    margin=float(margin.detach())))
            finally:
                for handle in handles: handle.remove()
            if _frozen(modules,choice_weight) != snapshot:
                raise ValueError("Accepted teacher changed during capture")
            progress(phase="choice_margin_capture",completed=index+1,total=len(cases))
        means = mean_sums/counts[:,None].clamp_min(1)
        global_mean = mean_sums.sum(0)/counts.sum()
        means[counts == 0] = global_mean
        number = len(selected)
        stats = dict(input_gram=gram/number,pair_scores=pair_scores/number,pair_mask_gram=mask_gram/number,
                     value_moments=value_moments/number,value_means=value_means/number,
                     means=means,counts=counts,global_mean=global_mean)
        if any(not torch.isfinite(t).all() for t in stats.values()):
            raise FloatingPointError("Nonfinite completed statistics")
        _save(root/"statistics.pt",{k:t.cpu() for k,t in stats.items()})
        write(root/"items.json",dict(items=cases))
        write(root/"choice_records.json",dict(records=records))
        for array in arrays.values(): array.flush()
        if _frozen(modules,choice_weight) != snapshot:
            raise ValueError("Accepted teacher changed during capture")
    finally:
        for array in arrays.values(): array._mmap.close()
    metadata = dict(status="completed",schema="jspace.qwen3-choice-calibration.v1",teacher_fingerprint=teacher_fingerprint,
        input_fingerprint=input_fingerprint,config=asdict(config),shape=dict(width=width,intermediate=intermediate,
        query_heads=nq,kv_heads=nkv,head_dim=hd),calibration_indices=selected,development_indices=development,
        calibration_families=[cases[i]["family"] for i in selected for _ in range(config.activation_positions)],
        development_families=[cases[i]["family"] for i in development for _ in range(config.activation_positions)],
        jacobian_case_ranges=case_ranges,normalization="mean over cases of full-valid-token input J.T@J; no per-row normalization",
        objective="correct final answer-choice logit minus arithmetic mean of other available choice logits; supervised task-margin, not full-vocabulary",
        head_intervention=f"shared KV-group RoPE-pair mask after native {hd}-dimensional Q/K RMSNorm and before RoPE; sum then square",
        means_provenance="all training valid-token boundaries, per absolute position; global training mean for unobserved positions",
        storage="FP32 exact conversions of BF16 teacher activations/derivatives; FP32 accumulated geometry",
        development_scope="native/compiled approximation diagnostics only; no basis, anchor, coefficient or removal fitting",
        artifact_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()})
    write(root/"capture.json",metadata)
    return metadata


def verify_capture(root):
    root = Path(root); metadata = _read(root/"capture.json")
    if metadata.get("status") != "completed" or metadata.get("schema") != "jspace.qwen3-choice-calibration.v1":
        raise ValueError("Completed Qwen3 calibration required")
    for name,digest in metadata["artifact_sha256"].items():
        if sha(root/name) != digest: raise ValueError("Changed calibration artifact: "+name)
    return metadata


@torch.no_grad()
def fit_basis(capture_dir, output_dir, *, device="cpu",guard=None):
    device = _guard(device,guard); root = Path(capture_dir); target = Path(output_dir)
    if target.exists(): raise FileExistsError(target)
    metadata = verify_capture(root); config = _config(metadata)
    stats = torch.load(root/"statistics.pt",map_location="cpu",weights_only=True)
    gram = stats["input_gram"].to(device)
    basis,info = block_krylov_basis(gram,config.rank,block_width=config.krylov_block_width,
        steps=config.krylov_steps,seed=config.seed,guard=guard)
    scores = stats["pair_scores"].to(device); moment = stats["value_moments"].to(device)
    if config.head_dim > metadata["shape"]["head_dim"] or (scores.sum(1)<=0).any():
        raise ValueError("Invalid head compression or degenerate task head sensitivity")
    pairs = torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:config.head_dim//2].sort(-1).values
    eigenvalues,vectors = torch.linalg.eigh((moment+moment.transpose(1,2))*.5)
    value_basis = vectors[:,:,-config.head_dim:].flip(-1).contiguous()
    if (eigenvalues.sum(1)<=0).any(): raise ValueError("Degenerate value second moment")
    removed = torch.ones_like(scores).scatter_(1,pairs,0).flatten()
    info.update(teacher_fingerprint=metadata["teacher_fingerprint"],objective=metadata["objective"],
        calibration_count=len(metadata["calibration_indices"]),head_pair_fraction=(scores.gather(1,pairs).sum(1)/scores.sum(1)).cpu().tolist(),
        joint_removed_mask_energy=float(removed@stats["pair_mask_gram"].to(device)@removed),
        value_second_moment_fraction=(eigenvalues[:,-config.head_dim:].sum(1)/eigenvalues.sum(1)).cpu().tolist(),
        unobserved_positions=(stats["counts"]==0).nonzero().flatten().tolist(),
        capture_sha256=sha(root/"capture.json"),head_intervention=metadata["head_intervention"])
    fit = dict(basis=basis.cpu(),means=stats["means"],rank=config.rank,head_dim=config.head_dim,
               rope_pairs=pairs.cpu(),value_basis=value_basis.cpu(),metadata=info)
    target.mkdir(parents=True); _save(target/"fit.pt",fit); write(target/"fit.json",info)
    return fit


@torch.no_grad()
def capture_compiled_development(capture_dir, fit_dir, read_boundary, block, forward_block, output_dir, *,guard=None):
    """Diagnostic h before polynomial substitution at the *actual* compact interface."""
    root,target = Path(capture_dir),Path(output_dir)
    if target.exists(): raise FileExistsError(target)
    device = _guard(block.mlp_output.device,guard); metadata = verify_capture(root); config = _config(metadata)
    if block.training or any(p.requires_grad for p in block.parameters()) or block.has_polynomial:
        raise ValueError("Frozen non-polynomial compact initializer required")
    fit = torch.load(Path(fit_dir)/"fit.pt",map_location="cpu",weights_only=True)
    basis,mean = fit["basis"].to(device),fit["means"].to(device)
    expected_anchor = mean-(mean@basis)@basis.T if basis.shape[1]<basis.shape[0] else mean.new_zeros(1,basis.shape[0])
    if not torch.equal(block.basis.detach(),basis) or not torch.equal(block.anchor.detach(),expected_anchor):
        raise ValueError("Compiled diagnostic geometry differs from the fitted basis/means")
    items = _read(root/"items.json")["items"]; positions = config.activation_positions
    target.mkdir(parents=True); path = target/"compiled_development.npy"
    array = np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,
        shape=(len(metadata["development_indices"])*positions,metadata["shape"]["intermediate"]))
    versions = [(id(t),t._version) for t in [*block.parameters(),*block.buffers()]]
    block.set_capture_mlp(True)
    try:
        for row,index in enumerate(metadata["development_indices"]):
            _guard(device,guard);case = items[index];x = read_boundary(index,device)
            if x.dtype != torch.bfloat16 or x.shape != (_length(case),metadata["shape"]["width"]):
                raise ValueError("Wrong compact development boundary")
            block.last_mlp_hidden = None
            forward_block(block,x[None],case)
            if block.last_mlp_hidden is None: raise ValueError("Callback did not execute the compact block")
            sample = torch.linspace(0,_length(case)-1,positions,device=device).long()
            array[row*positions:(row+1)*positions] = _float_rows(block.last_mlp_hidden[0,sample])
        array.flush()
    finally:
        block.set_capture_mlp(False);array._mmap.close()
    if versions != [(id(t),t._version) for t in [*block.parameters(),*block.buffers()]]:
        raise ValueError("Compiled diagnostic changed weights or geometry")
    _save(target/"output_weight.pt",block.mlp_output.detach().float().cpu())
    result = dict(status="completed",capture_sha256=sha(root/"capture.json"),fit_sha256=sha(Path(fit_dir)/"fit.pt"),
        output_weight_precision="FP32 exact conversion of the actual BF16 compiled non-polynomial output map",
        artifact_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()})
    write(target/"interface.json",result);return result


class StreamedFloat32Rows:
    """Owned bounded FP32 slices; compatible with the unchanged rank fitter."""
    def __init__(self,path,*,device="cpu",max_rows=256):
        self.path,self.device,self.max_rows = Path(path),torch.device(device),max_rows
        self._matrix = np.load(path,mmap_mode="r",allow_pickle=False)
        if self._matrix.dtype != np.dtype("float32") or self._matrix.ndim != 2 or min(self._matrix.shape)<1 or max_rows<1:
            self.close();raise ValueError("Nonempty FP32 row bank and positive slice bound required")
        self.shape,self.ndim,self.dtype = tuple(self._matrix.shape),2,torch.float32
        self.rows_read = 0
    def __len__(self):return self.shape[0]
    def is_floating_point(self):return True
    def __getitem__(self,index):
        if self._matrix is None:raise ValueError("Row bank is closed")
        if not isinstance(index,slice) or index.step not in (None,1):raise ValueError("Contiguous row slices required")
        start,stop,_ = index.indices(len(self))
        if stop-start>self.max_rows:raise ValueError("Slice exceeds bounded staging size")
        self.rows_read += stop-start
        return torch.from_numpy(np.array(self._matrix[start:stop],copy=True)).to(self.device)
    def close(self):
        if self._matrix is not None:self._matrix._mmap.close();self._matrix=None
    def __enter__(self):return self
    def __exit__(self,*_args):self.close()


@torch.no_grad()
def fit_relations(capture_dir,fit_dir,interface_dir,*,device="cpu",guard=None):
    """A single fixed anchor width: development cannot select fit or removals."""
    device = _guard(device,guard);root,fit_dir,interface_dir = map(Path,(capture_dir,fit_dir,interface_dir))
    if (fit_dir/"relation.pt").exists() or (fit_dir/"relations.json").exists():raise FileExistsError("Relations already fitted")
    metadata = verify_capture(root);config = _config(metadata);interface = _read(interface_dir/"interface.json")
    if (interface["status"] != "completed" or interface["capture_sha256"] != sha(root/"capture.json") or
            interface["fit_sha256"] != sha(fit_dir/"fit.pt")):
        raise ValueError("Compiled interface does not match calibration geometry")
    for name,digest in interface["artifact_sha256"].items():
        if sha(interface_dir/name)!=digest:raise ValueError("Changed compiled interface")
    def rows(path):
        bank = np.load(path,mmap_mode="r",allow_pickle=False)
        try:
            if bank.dtype != np.dtype("float32"):raise ValueError("Expected exact FP32 activation cache")
            return torch.from_numpy(np.array(bank,copy=True)).to(device)
        finally:bank._mmap.close()
    train = rows(root/"native_train.npy");dev = rows(root/"native_development.npy")
    compiled = rows(interface_dir/"compiled_development.npy")
    weight = torch.load(interface_dir/"output_weight.pt",map_location="cpu",weights_only=True).to(device)
    with StreamedFloat32Rows(root/"mlp_jacobian.npy",device=device,max_rows=config.chunk_rows) as jacobian:
        result = fit_benchmark_groebner(train,metadata["calibration_families"],jacobian,weight,
            dev,metadata["development_families"],compressed_dev_activations=compiled,
            anchor_widths=(config.anchors,),remove_count=config.remove_count,seed=config.seed,
            svd_rank=config.svd_rank,oversample=config.oversample,power_iterations=config.power_iterations,
            ridge=config.ridge,eligibility_nrmse=config.eligibility_nrmse,chunk_rows=config.chunk_rows,
            selector_kwargs={"guard":guard})
        read_count = jacobian.rows_read
    if result["selected_anchor_width"] != config.anchors:raise ValueError("Fixed anchor width changed")
    fit = result.pop("fits")[str(config.anchors)];_save(fit_dir/"relation.pt",fit)
    logical = metadata["shape"]["intermediate"]-config.remove_count+1+config.anchors+config.anchors*(config.anchors+1)//2
    result.update(status="completed",teacher_fingerprint=metadata["teacher_fingerprint"],
        capture_sha256=sha(root/"capture.json"),fit_sha256=sha(fit_dir/"fit.pt"),interface_sha256=sha(interface_dir/"interface.json"),
        relation_sha256=sha(fit_dir/"relation.pt"),jacobian_rows_read=read_count,
        fixed_anchor_width=config.anchors,removed_channels=config.remove_count,
        runtime_padding=dict(logical_width=logical,padded_width=((logical+127)//128)*128,multiple=128),
        fitting_scope="Training-only normalization, task-Jacobian anchor selection, ridge coefficients, and removal ranking; single fixed width. Development only measures native/compiled errors and per-family redundancy.",
        coefficient_rounding="Existing graph fitter's 2^-20 rational coefficient grid; no BF16 roundtrip in fitting",
        algebraic_scope="Exact certificate of the fitted graph ideal; neural approximation quality is measured separately")
    write(fit_dir/"relations.json",result);_guard(device,guard);return result
