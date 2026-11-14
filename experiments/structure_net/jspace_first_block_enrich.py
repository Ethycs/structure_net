"""Lock observed J directions and fit reachable first-block input/update states.

One predeclared basis candidate: preserve all resolved input-Jacobian columns,
then the global input mean and separate position-zero input/update means.
Use the remaining rank for equal trace-normalized input and native-update
second moments at nonzero training positions. No development fitting occurs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from . import jspace_quadratic_recipe as recipe


def guard_device(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA fitting requires a reservation guard")
    if guard is not None: guard()
    return device


def projection_metrics(basis, mean, h0, h1, g0=None, g1=None):
    x, update = h0-mean, h1-h0
    ex, ed = x-(x@basis)@basis.T, update-(update@basis)@basis.T
    def energy(t): return t.double().square().sum()
    value = dict(input_centered_energy_lost=float(energy(ex)/energy(x).clamp_min(1e-30)),
        native_update_energy_lost=float(energy(ed)/energy(update).clamp_min(1e-30)))
    for name, error, reference, gram in (("input_G0_residual_ratio",ex,x,g0), ("update_G1_residual_ratio",ed,update,g1)):
        if gram is not None:
            numerator = (error*(error@gram)).double().sum()
            denominator = (reference*(reference@gram)).double().sum().clamp_min(1e-30)
            value[name] = float(numerator/denominator)
    return value


@torch.no_grad()
def enrich_basis(original, mean, h0, h1, position0, *, functional_columns, guard=None):
    """Keep strong columns bit-identical; optimize only their orthogonal complement."""
    guard_device(original.device, guard)
    width, rank = original.shape
    if (original.dtype not in (torch.float32,torch.float64) or not 0 < functional_columns < rank <= width or
            h0.shape != h1.shape or h0.ndim != 2 or h0.shape[1] != width or
            mean.shape != (1,width) or position0.shape != (len(h0),) or position0.dtype != torch.bool or
            not position0.any() or position0.all()):
        raise ValueError("Invalid locked-J geometry or training sample shapes")
    if any(t.device != original.device for t in (mean,h0,h1,position0)):
        raise ValueError("All fitting tensors must share the selected device")
    if not all(torch.isfinite(t).all() for t in (original,mean,h0,h1)):
        raise ValueError("Nonfinite training geometry")
    strong = original[:,:functional_columns].clone()
    q = torch.linalg.qr(strong,mode="reduced").Q
    update = h1-h0
    locked_means = [mean[0],h0[position0].mean(0),update[position0].mean(0)]
    extras = []
    for vector in locked_means:
        residual = vector.clone()
        for _ in range(2): residual -= q@(q.T@residual)
        tolerance = width*torch.finfo(original.dtype).eps*float(vector.norm())
        if float(residual.norm()) > tolerance:
            if q.shape[1] == rank:
                raise ValueError("Rank budget cannot retain the declared mean directions")
            direction = residual/residual.norm();extras.append(direction)
            q = torch.cat((q,direction[:,None]),dim=1)
    locked = q.shape[1]
    x, delta = (h0-mean)[~position0], update[~position0]
    rows = []
    for values in (x,delta):
        total = values.square().sum()
        if float(total) > 0: rows.append(values/total.sqrt())
    if not rows:
        raise ValueError("No nonzero non-position0 training second moment")
    design = torch.cat(rows)/len(rows)**.5
    for _ in range(2): design -= (design@q)@q.T
    if guard is not None:guard()
    covariance = design.T@design
    eigenvalues,eigenvectors = torch.linalg.eigh((covariance+covariance.T)*.5)
    values,vectors = eigenvalues.flip(0),eigenvectors.flip(1)
    threshold = width*torch.finfo(original.dtype).eps*max(0.,float(values[0]))
    count = min(rank-locked,int((values>threshold).sum()))
    selected = vectors[:,:count]
    for _ in range(2):selected -= q@(q.T@selected)
    if count:
        selected = torch.linalg.qr(selected,mode="reduced").Q
        q = torch.cat((q,selected),dim=1)
    completion = rank-q.shape[1]
    if completion:
        residual = original.clone()
        for _ in range(2):residual -= q@(q.T@residual)
        left,singular,_ = torch.linalg.svd(residual,full_matrices=False)
        if int((singular>width*torch.finfo(original.dtype).eps*float(singular[0])).sum()) < completion:
            raise ValueError("Original weak completion cannot fill the unchanged storage rank")
        fill=left[:,:completion]
        for _ in range(2):fill -= q@(q.T@fill)
        fill=torch.linalg.qr(fill,mode="reduced").Q
    else:fill=original[:,:0]
    pieces=[strong]
    if extras:pieces.append(torch.stack(extras,dim=1))
    pieces.extend((selected,fill))
    result=torch.cat(pieces,dim=1).contiguous()
    orthogonality=float((result.T@result-torch.eye(rank,device=result.device,dtype=result.dtype)).abs().max())
    if result.shape != original.shape or orthogonality > 3e-5 or not torch.equal(result[:,:functional_columns],strong):
        raise ValueError("Enriched geometry lost rank, strong columns, or orthogonality")
    return result,dict(functional_columns_locked=functional_columns,strong_columns_bit_identical=True,
        additional_locked_mean_directions=len(extras),moment_directions=count,old_weak_completion=completion,
        moment_numerical_threshold=threshold,moment_energy_selected=float(values[:count].clamp_min(0).sum()),
        moment_energy_total=float(values.clamp_min(0).sum()),orthogonality_max=orthogonality,
        objective="equal trace-normalized centered-input and uncentered-native-update second moments at nonzero positions",
        locked_means="global native input mean; separate position0 input mean and native-update mean",
        position0_input_variance=float((h0[position0]-h0[position0].mean(0)).square().mean()),
        position0_update_variance=float((update[position0]-update[position0].mean(0)).square().mean()))


def sample_training(items, cache, *, start=0, per_family=32):
    cases=[case for family in ("arc","boolq","hellaswag","winogrande") for case in
        [c for c in items if c["split"]=="train" and c["family"]==family][start:start+per_family]]
    if len(cases)!=4*per_family:raise ValueError("Insufficient balanced training-only cases")
    indices=[];zero=[]
    for case in cases:
        positions=torch.linspace(0,len(case["input_ids"])-1,16).long().tolist()
        indices.extend(case["offset"]+position for position in positions);zero.extend(position==0 for position in positions)
    states=[]
    for layer in (0,1):
        bank=np.load(Path(cache)/f"boundary_{layer:02d}.npy",mmap_mode="r")
        try:states.append(torch.from_numpy(np.array(bank[indices],copy=True)).view(torch.bfloat16).float())
        finally:bank._mmap.close()
    return *states,torch.tensor(zero,dtype=torch.bool),[c["identity"] for c in cases]


def fit(root,source,device="cpu"):
    from .jspace_full_lease import wait_ready
    root,source=Path(root).resolve(),Path(source).resolve()
    if root.exists():raise FileExistsError(root)
    lease=wait_ready([0]) if torch.device(device).type=="cuda" else None
    guard=(lambda:wait_ready([0])) if lease else None
    device=guard_device(device,guard);torch.set_num_threads(2)
    if device.type=="cuda":torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    parent=recipe.verify(source);cache=Path(parent["teacher_root"])/"capture"
    original_path=recipe.fit_path(source,0);manifest=recipe.checked(source,source/"basis",hashes=False)
    if recipe.sha(original_path)!=manifest["artifacts"][str(original_path.relative_to(source))]:raise ValueError("Changed original fit")
    inputs={str(p):recipe.sha(p) for p in (source/"plan.json",source/"items.json",source/"basis/manifest.json",original_path)}
    for path in (cache/"boundary_00.npy",cache/"boundary_01.npy",cache/"statistics.pt"):
        expected=parent["input_sha256"][str(path)]
        if recipe.sha(path)!=expected:raise ValueError("Changed native training bank")
        inputs[str(path)]=expected
    items=recipe.read(source/"items.json")["items"]
    h0,h1,pos0,identities=sample_training(items,cache)
    ch0,ch1,cp0,check_ids=sample_training(items,cache,start=32)
    if set(identities)&set(check_ids):raise ValueError("Internal training projection-check identities overlap fit")
    original=recipe.load(original_path);u,mean=original["basis"].to(device),original["means"].to(device)
    h0,h1,pos0,ch0,ch1,cp0=[v.to(device) for v in (h0,h1,pos0,ch0,ch1,cp0)]
    statistics=recipe.load(cache/"statistics.pt")
    g0=statistics["grams"][0].to(device);g0=g0/g0.trace()
    g1=statistics["grams"][1].to(device);g1=g1/g1.trace()
    producers=[Path(__file__),recipe.ROOT/"tests/structure_net/test_jspace_first_block_enrich.py",
        Path(__file__).with_name("jspace_first_block_ablation.py"),Path(__file__).with_name("jspace_residual_carry.py")]
    plan=dict(schema="jspace.first-block-enriched-basis.v1",source_root=str(source),source_fingerprint=parent["fingerprint"],
        layers=[0],rank=1024,probe_head_dim=128,fit_training_identities=identities,internal_training_check_identities=check_ids,
        positions_per_case=16,original_functional_columns=original["metadata"]["retained_functional_directions"],
        objective="lock resolved input-J columns plus input-global/position0-input/position0-update means; equal trace-normalized input/update moments on nonzero positions",
        input_sha256=inputs,source_sha256={str(p):recipe.sha(p) for p in producers},
        scope="One training-only basis repair; internal training check is not fresh confirmation; no development choice of objective")
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path in producers:
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan)
    try:
        enriched,metadata=enrich_basis(u,mean,h0,h1,pos0,functional_columns=plan["original_functional_columns"],guard=guard)
        metadata["diagnostics"]={}
        for name,basis in (("original",u),("enriched",enriched)):
            if guard is not None:guard()
            metadata["diagnostics"][name]=dict(input_gram_trace_retained=float((basis*(g0@basis)).double().sum()),
                output_gram_trace_retained=float((basis*(g1@basis)).double().sum()),
                fitting_train=projection_metrics(basis,mean,h0,h1,g0,g1),
                separate_training_check=projection_metrics(basis,mean,ch0,ch1,g0,g1),
                position0_training_check=projection_metrics(basis,mean,ch0[cp0],ch1[cp0],g0,g1))
        directory=root/"basis/block_00";directory.mkdir(parents=True)
        value={**original,"basis":enriched.cpu(),"metadata":{**original["metadata"],"enrichment":metadata}}
        recipe.save(directory/"fit.pt",value);recipe.write(directory/"fit.json",metadata)
        files=[directory/"fit.pt",directory/"fit.json"]
        peak=torch.cuda.max_memory_reserved() if device.type=="cuda" else None
        if lease and peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Basis fit exceeded reservation")
        recipe.write(root/"basis/manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            source_fingerprint=parent["fingerprint"],artifacts={str(p.relative_to(root)):recipe.sha(p) for p in files},
            lease_id=None if lease is None else lease["id"],peak_reserved_bytes=peak))
        return dict(status="completed",root=str(root),fingerprint=plan["fingerprint"],diagnostics=metadata["diagnostics"])
    except BaseException as error:
        recipe.write(root/"failed.json",dict(error=repr(error),fingerprint=plan["fingerprint"]));raise


def probe(root,source,fit_root):
    # The existing runner's mean_rank_full_head variant loads the supplied
    # candidate fit verbatim. The candidate fingerprint identifies this richer
    # training-only repair; no claim is made that it is merely a mean change.
    from .jspace_first_block_ablation import run
    result=run(root,source,"mean_rank_full_head",mean_root=fit_root)
    return {**result,"basis_intervention":"locked_J_input_and_native_update_enrichment"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","probe"));parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=recipe.DEFAULT_ROOT)
    parser.add_argument("--device",default="cpu");parser.add_argument("--fit-root",type=Path)
    args=parser.parse_args()
    print(fit(args.root,args.source,args.device) if args.stage=="fit" else probe(args.root,args.source,args.fit_root))
