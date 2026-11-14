"""Matched FP32 original-Qwen teacher, partitioned across two reserved GPUs.

Only midpoint/answer full-vocabulary logits are persisted. The original
512/64 population, native weights and all previous caches remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import jspace_quadratic_recipe as recipe
from .jspace_single_block_verify import validate_records

SCHEMA = "jspace.recovered-fp32-teacher.v1"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-recovered-joint-distillation.md"
SLOTS = [1,3]
ROLES = {"front":("GPU-0370fdb0",11264),"tail":("GPU-7b1db3cb",7168)}
ATOL, RTOL = 5e-4, 1e-5


def validate_items(items, vocabulary, label_ids, *, full=False):
    if (not items or len({c["identity"] for c in items}) != len(items)
            or {c["split"] for c in items} != {"train","development"}
            or not label_ids or len(set(label_ids)) != len(label_ids)
            or any(type(i) is not int or not 0 <= i < vocabulary for i in label_ids)):
        raise ValueError("Unique disjoint train/development identities and valid choice labels required")
    for c in items:
        ids, positions = c["input_ids"], c["loss_positions"]
        if (not 1 <= len(ids) <= 384 or any(type(i) is not int or not 0 <= i < vocabulary for i in ids)
                or len(positions) != 4 or any(type(p) is not int or not 0 <= p < len(ids) for p in positions)
                or positions[-1] != len(ids)-1 or not 0 <= c["answer"] < len(c["choices"]) <= len(label_ids)):
            raise ValueError("Invalid native tokens, loss positions or answer choices")
    if full:
        wanted = {(split,family):count for split,count in (("train",128),("development",16))
                  for family in ("arc","boolq","hellaswag","winogrande")}
        if Counter((c["split"],c["family"]) for c in items) != wanted:
            raise ValueError("Original512 training and64 balanced development cases required")


def token_batch(items, indices, device):
    if not indices or any(type(i) is not int or not 0 <= i < len(items) for i in indices):
        raise ValueError("Nonempty valid case indices required")
    cases = [items[i] for i in indices]; lengths = [len(c["input_ids"]) for c in cases]
    ids = torch.zeros((len(cases),max(lengths)),device=device,dtype=torch.long)
    for row,c in enumerate(cases): ids[row,:lengths[row]] = torch.tensor(c["input_ids"],device=device)
    positions = torch.arange(max(lengths),device=device).expand(len(cases),-1)
    valid = positions < torch.tensor(lengths,device=device)[:,None]
    allowed = (positions[:,:,None] >= positions[:,None,:]) & valid[:,:,None] & valid[:,None,:]
    return dict(indices=indices,input_ids=ids,position_ids=positions,attention_mask=allowed[:,None],
        loss_positions=torch.tensor([c["loss_positions"] for c in cases],device=device,dtype=torch.long))


class FP32JointDataset:
    """Root-authenticated dataset; explicit items/path/labels also support CPU tests."""
    def __init__(self, root_or_items, teacher_logits_path=None, label_ids=None):
        if teacher_logits_path is None:
            root = Path(root_or_items); plan = verify(root,completed=True)
            items = recipe.read(root/"items.json")["items"]
            teacher_logits_path, label_ids = root/"cache/teacher_logits.npy", plan["label_ids"]
        else:
            items = root_or_items
        self.items, self.label_ids = list(items), list(label_ids)
        self.teacher_logits = np.load(teacher_logits_path,mmap_mode="r",allow_pickle=False)
        try:
            if (self.teacher_logits.dtype != np.float32 or self.teacher_logits.ndim != 3
                    or self.teacher_logits.shape[:2] != (len(self.items),2)):
                raise ValueError("Teacher logits must be FP32 [cases,2,full vocabulary]")
            self.vocabulary = self.teacher_logits.shape[-1]
            validate_items(self.items,self.vocabulary,self.label_ids)
            self.training = [i for i,c in enumerate(self.items) if c["split"] == "train"]
            self.development = [i for i,c in enumerate(self.items) if c["split"] == "development"]
        except Exception:
            self.close();raise

    def close(self):
        bank = getattr(self,"teacher_logits",None)
        if bank is not None and not bank._mmap.closed: bank._mmap.close()

    def batch(self, indices, device):
        batch = token_batch(self.items,indices,device)
        teacher = torch.from_numpy(np.array(self.teacher_logits[indices],copy=True)).to(device)
        if not torch.isfinite(teacher).all(): raise FloatingPointError("Nonfinite FP32 teacher logits")
        return dict(**batch,teacher=teacher,target_tokens=torch.tensor(
            [self.label_ids[self.items[i]["answer"]] for i in indices],device=device,dtype=torch.long))


class PartitionedNativeTeacher(nn.Module):
    """Native causal blocks; full hidden transfer once, only2 rows return for E."""
    def __init__(self, embedding, blocks, norm, rotary_front, rotary_tail, *, cut):
        super().__init__()
        if not 0 < cut < len(blocks) or embedding.ndim != 2 or embedding.dtype != torch.float32:
            raise ValueError("Two nonempty native partitions and FP32 tied embedding required")
        self.register_buffer("embedding",embedding.detach());self.blocks = nn.ModuleList(blocks)
        self.norm,self.rotary_front,self.rotary_tail = norm,rotary_front,rotary_tail
        self.cut = cut;self.front = embedding.device;self.tail = next(blocks[cut].parameters()).device
        for index,block in enumerate(blocks):
            device = self.front if index < cut else self.tail
            if any(t.device != device or t.is_floating_point() and t.dtype != torch.float32
                   for t in (*block.parameters(),*block.buffers())):
                raise ValueError("Every native partition must remain FP32 on its assigned device")
        if any(p.device != self.front or p.dtype != torch.float32 for p in norm.parameters()):
            raise ValueError("Final native norm and tied E must share front FP32 device")
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, input_ids, position_ids, attention_mask, loss_positions):
        if input_ids.device != self.front or loss_positions.shape != (input_ids.shape[0],4):
            raise ValueError("Front-device input and original four loss positions required")
        hidden = F.embedding(input_ids,self.embedding)
        mask = torch.zeros(attention_mask.shape,dtype=torch.float32,device=self.front).masked_fill_(
            ~attention_mask,torch.finfo(torch.float32).min)
        rope = self.rotary_front(hidden,position_ids)
        for index,block in enumerate(self.blocks):
            if index == self.cut:
                hidden = hidden.to(self.tail);position_ids = position_ids.to(self.tail);mask = mask.to(self.tail)
                rope = self.rotary_tail(hidden,position_ids)
            value = block(hidden,attention_mask=mask,position_ids=position_ids,
                cache_position=position_ids[0],position_embeddings=rope,use_cache=False)
            hidden = value[0] if isinstance(value,tuple) else value
        rows = torch.arange(hidden.shape[0],device=self.tail)[:,None]
        selected = hidden[rows,loss_positions[:,SLOTS].to(self.tail)].to(self.front)
        return F.linear(self.norm(selected),self.embedding).float()


@torch.inference_mode()
def capture(model, items, label_ids, directory, *, microbatch=4, guard=None, progress=lambda **_k:None):
    directory = Path(directory)
    if directory.exists(): raise FileExistsError(directory)
    if model.front.type == "cuda" and guard is None: raise ValueError("Live reservation guard required for CUDA capture")
    guard = guard or (lambda:None)
    if type(microbatch) is not int or microbatch < 1: raise ValueError("Positive microbatch required")
    validate_items(items,model.embedding.shape[0],label_ids);guard();directory.mkdir(parents=True)
    shape = (len(items),2,model.embedding.shape[0])
    bank = np.lib.format.open_memmap(directory/"teacher_logits.npy",mode="w+",dtype=np.float32,shape=shape)
    records = []
    try:
        for start in range(0,len(items),microbatch):
            guard();indices = list(range(start,min(start+microbatch,len(items))))
            batch = token_batch(items,indices,model.front)
            logits = model(**{k:v for k,v in batch.items() if k != "indices"})
            if logits.dtype != torch.float32 or logits.shape != (len(indices),*shape[1:]) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite or malformed native FP32 output")
            host = logits.cpu().numpy();bank[indices] = host
            for row,index in enumerate(indices):
                case = items[index]
                if case["split"] == "development":
                    choices = host[row,-1,label_ids[:len(case["choices"])]]
                    records.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                        prediction=int(choices.argmax()),logits=choices.tolist()))
            del logits,batch,host
            progress(cases_written=min(start+microbatch,len(items)),total_cases=len(items))
        bank.flush();guard()
    finally:
        bank._mmap.close()
    return dict(records=records,shape=list(shape),dtype="float32",slots=SLOTS,microbatch=microbatch,
                tensor_bytes=math.prod(shape)*4,full_boundary_banks_written=0)


def check_control(records, reference, cases, *, expected_correct=50, atol=ATOL, rtol=RTOL):
    validate_records(records,cases);validate_records(reference,cases)
    exact = all(a["prediction"] == b["prediction"] for a,b in zip(records,reference))
    logit_pass = all(abs(x-y) <= atol+rtol*abs(y) for a,b in zip(records,reference) for x,y in zip(a["logits"],b["logits"]))
    correct = sum(r["prediction"] == r["answer"] for r in records)
    reference_correct = sum(r["prediction"] == r["answer"] for r in reference)
    passed = exact and logit_pass and correct == reference_correct == expected_correct
    return dict(passed=passed,predictions_exact=exact,logits_pass=logit_pass,correct=correct,count=len(records),
        reference_correct=reference_correct,logit_max_abs_delta=max(abs(x-y) for a,b in zip(records,reference)
            for x,y in zip(a["logits"],b["logits"])),logit_atol=atol,logit_rtol=rtol)


def allocator_settings(lease, *, cuda=None, visible=None):
    """Called only AFTER live lease verification; resolve roles from physical UUIDs."""
    cuda = torch.cuda if cuda is None else cuda
    ids = (os.environ.get("CUDA_VISIBLE_DEVICES","") if visible is None else visible).split(",")
    allocation = lease["allocation"]
    if len(ids) != 2 or len(set(ids)) != 2 or set(ids) != set(allocation):
        raise ValueError("Exactly two visible leased physical GPU UUIDs required")
    roles = {}
    for role,(prefix,mib) in ROLES.items():
        matches = [i for i,name in enumerate(ids) if name.startswith(prefix)]
        if len(matches) != 1 or allocation[ids[matches[0]]] != mib:
            raise ValueError("Expected3060=11264MiB and2060SUPER=7168MiB")
        roles[role] = matches[0]
    if cuda.is_initialized(): raise RuntimeError("Fresh CUDA process required for allocator caps")
    settings = {}
    for role,index in roles.items():
        total = int(cuda.get_device_properties(index).total_memory)
        allowed = (allocation[ids[index]]-256)*1024**2
        if not 0 < allowed <= total: raise ValueError("Allocator budget exceeds physical memory")
        if cuda.memory.get_allocator_backend() != "native": raise RuntimeError("Native allocator required")
        cuda.set_per_process_memory_fraction(allowed/total,device=index)
        settings[role] = dict(device=index,uuid=ids[index],lease_bytes=allocation[ids[index]]*1024**2,
            allocator_limit_bytes=allowed,headroom_bytes=256*1024**2,device_total_memory=total)
    return settings


def prepare(root, source, native_cache_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Hide CUDA for CPU preparation")
    from . import jspace_native_dev_cache as native_cache
    root,source,native_cache_root = map(lambda p:Path(p).resolve(),(root,source,native_cache_root))
    if root.exists(): raise FileExistsError(root)
    original = recipe.verify(source);native_plan = native_cache.verify(native_cache_root,completed=False)
    items = recipe.read(source/"items.json");config = recipe.native_config(original)
    validate_items(items["items"],config.vocab_size,original["label_ids"],full=True)
    if (config.num_hidden_layers,config.hidden_size,config.vocab_size) != (36,2048,151936) or not config.tie_word_embeddings:
        raise ValueError("Pinned tied Qwen2.5-3B architecture required")
    reference_path = native_cache_root/"cache/result.json";reference = recipe.read(reference_path)
    manifest = recipe.read(native_cache_root/"cache/manifest.json")
    cases = [c for c in items["items"] if c["split"] == "development"]
    native_cases = recipe.read(native_cache_root/"items.json")["items"]
    fields = ("identity","input_ids","loss_positions","family","answer","choices")
    if (len(native_cases) != len(cases) or any(any(a[k] != b[k] for k in fields) for a,b in zip(cases,native_cases))
            or native_plan["source_fingerprint"] != original["fingerprint"]
            or manifest["status"] != "completed" or manifest["fingerprint"] != native_plan["fingerprint"]
            or manifest["artifact_sha256"]["cache/result.json"] != recipe.sha(reference_path)
            or reference["status"] != "completed" or reference["fingerprint"] != native_plan["fingerprint"]
            or reference["native_baseline_predictions_exact"] is not True
            or not check_control(reference["records"],reference["records"],cases)["passed"]):
        raise ValueError("Authenticated matching native FP32 reference50/64 required")
    inputs = [source/"plan.json",source/"items.json",native_cache_root/"plan.json",native_cache_root/"items.json",
              reference_path,native_cache_root/"cache/manifest.json"]
    sources = {**original["source_sha256"],**native_plan["source_sha256"]}
    for path in (Path(__file__),recipe.ROOT/"tests/structure_net/test_jspace_recovered_teacher.py",PROTOCOL,
                 Path(__file__).with_name("jspace_qwen_parallel_model.py"),Path(__file__).with_name("jspace_full_lease.py")):
        sources[str(path)] = recipe.sha(path)
    plan = dict(schema=SCHEMA,source=str(source),source_fingerprint=original["fingerprint"],
        native_cache_root=str(native_cache_root),native_cache_fingerprint=native_plan["fingerprint"],
        native_reference=str(reference_path),native_reference_sha256=recipe.sha(reference_path),native_correct=50,
        model=original["model"],label_ids=original["label_ids"],items_digest=recipe.digest(items),
        shape=[576,2,151936],dtype="float32",slots=SLOTS,cut=24,microbatch=4,
        precision="Original BF16 stored weights promoted exactly to FP32; FP32 forward and full-vocabulary readout; TF32 disabled",
        roles=ROLES,headroom_mib=256,logit_atol=ATOL,logit_rtol=RTOL,confirmation_used=False,
        input_sha256={str(p):recipe.sha(p) for p in inputs},input_identity={str(p):recipe.file_identity(p) for p in inputs},
        source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        if recipe.sha(path) != digest: raise ValueError("Changed inherited source")
        target = root/"producer"/Path(path).relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan);recipe.write(root/"items.json",items)
    return dict(status="prepared",fingerprint=plan["fingerprint"],shape=plan["shape"],bytes=576*2*151936*4)


def verify(root, completed=False):
    root = Path(root);plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed FP32 teacher plan")
    if recipe.verify(plan["source"])["fingerprint"] != plan["source_fingerprint"]:
        raise ValueError("Changed original teacher provenance")
    if recipe.digest(recipe.read(root/"items.json")) != plan["items_digest"]: raise ValueError("Changed teacher identities/tokens")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != digest:
            raise ValueError("Changed FP32 teacher producer")
    for path,digest in plan["input_sha256"].items():
        if recipe.sha(path) != digest: raise ValueError("Changed teacher metadata/reference")
    if completed:
        manifest = recipe.read(root/"cache/manifest.json")
        names = {"cache/teacher_logits.npy","cache/result.json","items.json"}
        if (manifest.get("status") != "completed" or manifest.get("passed") is not True
                or manifest.get("fingerprint") != plan["fingerprint"] or set(manifest["artifact_sha256"]) != names):
            raise ValueError("Completed passing FP32 teacher cache required")
        for name,digest in manifest["artifact_sha256"].items():
            if recipe.sha(root/name) != digest: raise ValueError("Changed teacher cache artifact")
        result = recipe.read(root/"cache/result.json")
        if (result.get("status") != "completed" or result.get("passed") is not True
                or result.get("fingerprint") != plan["fingerprint"] or result["shape"] != plan["shape"]
                or result["control"]["passed"] is not True or result["correct"] != 50 or result["count"] != 64):
            raise ValueError("Native reference control failed")
    return plan


def reservation():
    from .jspace_full_lease import wait_ready
    return wait_ready([0,1])


def run(root):
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    root = Path(root).resolve();plan = verify(root)
    if any((root/name).exists() for name in ("cache","run_started.json","failure.json")):
        raise FileExistsError("Teacher capture already attempted")
    lease = None;started = time.monotonic()
    try:
        lease = reservation()  # Must precede every CUDA query/allocation.
        settings = allocator_settings(lease)
        recipe.write(root/"run_started.json",dict(fingerprint=plan["fingerprint"],lease_id=lease["id"],allocator=settings))
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = False
        for value in settings.values(): torch.cuda.reset_peak_memory_stats(value["device"])
        def guard():
            current = reservation()
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Teacher reservation changed")
            for value in settings.values():
                if torch.cuda.max_memory_reserved(value["device"]) > value["allocator_limit_bytes"]:
                    raise MemoryError("Teacher allocator exceeded reserved device cap")
        front,tail = [torch.device("cuda",settings[name]["device"]) for name in ("front","tail")]
        guard();embedding,norm,rotary_front = load_readout_norm(plan["model"]["path"],device=front,dtype=torch.float32)
        original = recipe.read(Path(plan["source"])/"plan.json")
        rotary_tail = recipe.rotary_module(original,tail);blocks = []
        for layer in range(36):
            guard();blocks.append(load_native_block(plan["model"]["path"],layer,
                device=front if layer < plan["cut"] else tail,dtype=torch.float32))
        model = PartitionedNativeTeacher(embedding,blocks,norm,rotary_front,rotary_tail,cut=plan["cut"])
        items = recipe.read(root/"items.json")["items"]
        value = capture(model,items,plan["label_ids"],root/"cache",microbatch=plan["microbatch"],guard=guard,
            progress=lambda **v:recipe.event(root,"teacher",**v))
        control = check_control(value["records"],recipe.read(plan["native_reference"])["records"],
            [c for c in items if c["split"] == "development"],atol=plan["logit_atol"],rtol=plan["logit_rtol"])
        result = dict(**value,status="completed" if control["passed"] else "failed",passed=control["passed"],
            control=control,correct=control["correct"],count=control["count"],fingerprint=plan["fingerprint"],confirmed=False)
        recipe.write(root/"cache/result.json",result)
        if not control["passed"]: raise RuntimeError("Matched FP32 teacher did not reproduce native control")
        guard();names = ["cache/teacher_logits.npy","cache/result.json","items.json"]
        manifest = dict(status="completed",passed=True,fingerprint=plan["fingerprint"],
            artifact_sha256={name:recipe.sha(root/name) for name in names},lease_id=lease["id"],allocation=lease["allocation"],
            allocator=settings,peak_reserved_bytes={name:torch.cuda.max_memory_reserved(v["device"]) for name,v in settings.items()},
            elapsed_seconds=time.monotonic()-started)
        recipe.write(root/"cache/manifest.json",manifest)
        return dict(status="completed",passed=True,fingerprint=plan["fingerprint"],correct=result["correct"],count=64)
    except Exception as error:
        recipe.write(root/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],
            lease_id=None if lease is None else lease["id"],error_type=type(error).__name__,error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--source",type=Path)
    parser.add_argument("--native-cache-root",type=Path);args = parser.parse_args()
    if args.stage == "prepare" and (args.source is None or args.native_cache_root is None):
        parser.error("prepare requires --source and --native-cache-root")
    result = prepare(args.root,args.source,args.native_cache_root) if args.stage == "prepare" else run(args.root)
    print(json.dumps(result,allow_nan=False),flush=True)
