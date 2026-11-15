"""Cache the packed accepted teacher's inference outputs, separately from VJPs.

The differentiable calibration teacher is intentionally unchanged. This helper
uses the deployed packed parent's inference path and the exact development
batch grouping used by the original scorer. It stores existing BF16 values
losslessly as int16 bits; it performs no quantization or training.
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_bench_data import sha
from .jspace_qwen_progressive_calibration import write_bfloat16_bits


def _guard(device,guard):
    if torch.device(device).type=="cuda" and guard is None:
        raise ValueError("Packed teacher CUDA capture requires a reservation guard")
    if guard is not None:guard()


def _fixed(stack,norm,readout,rotary):
    modules=(stack,norm,rotary)
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in modules) or readout.requires_grad:
        raise ValueError("Packed accepted parent, norm, readout and rotary must already be frozen and eval")
    tensors=[readout,*(v for m in modules for v in [*m.parameters(),*m.buffers()])]
    if any(v.device!=readout.device or v.is_inference() for v in tensors):
        raise ValueError("Frozen teacher state must use normal tensors on one device")
    return [(id(v),v._version) for v in tensors]


@torch.inference_mode()
def capture_packed_teacher(path,stack,norm,E,rotary,dataset,*,guard=None,progress=None,
                           microbatch=4,label_ids=None):
    """Write [cases,4,vocabulary] BF16 bits using split-specific batches of4.

    ``dataset`` is the existing BoundaryDataset at the accepted parent's cut,
    i.e. native boundary ``new_cut+1``. Its original case indices define file
    rows even when train and development items are interleaved. All training
    batches run first; development starts a fresh grouping, matching old.score.
    The caller authenticates the boundary and selected parent and owns the
    stage manifest. On failure the partial file is retained and never reused.
    """
    _guard(E.device,guard)
    if type(microbatch) is not int or microbatch!=4:
        raise ValueError("Packed targets must use the fixed scoring microbatch4")
    if E.dtype!=torch.bfloat16 or E.ndim!=2 or E.shape!=(dataset.vocabulary,dataset.width):
        raise ValueError("Frozen full-vocabulary BF16 readout must match the boundary dataset")
    if getattr(stack,"packed",False) is not True:
        raise ValueError("Accepted parent must already be packed")
    before=_fixed(stack,norm,E,rotary)
    items=dataset.items;count=len(items)
    train=list(dataset.training);development=list(dataset.development)
    if (not train or not development or len({c["identity"] for c in items})!=count or
            sorted(train+development)!=list(range(count)) or
            train!=[i for i,c in enumerate(items) if c["split"]=="train"] or
            development!=[i for i,c in enumerate(items) if c["split"]=="development"]):
        raise ValueError("Disjoint complete train/development groups in original case order required")
    if label_ids is not None and (len(label_ids)!=4 or len(set(label_ids))!=4 or
            any(type(i) is not int or not 0<=i<len(E) for i in label_ids)):
        raise ValueError("Optional four answer-label IDs must be distinct vocabulary rows")
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb"):pass
    array=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(count,4,len(E)))
    batches=[];records=[];seen=set();start_reads=dict(getattr(dataset,"read_bytes",{}))
    try:
        for split,indices in (("train",train),("development",development)):
            for start in range(0,len(indices),microbatch):
                _guard(E.device,guard)
                if _fixed(stack,norm,E,rotary)!=before:raise ValueError("Accepted teacher changed during target capture")
                selected=indices[start:start+microbatch]
                batch=dataset.batch(selected,device=E.device)
                if batch["indices"]!=selected or batch["hidden"].dtype!=torch.bfloat16 or batch["loss_positions"].shape!=(len(selected),4):
                    raise ValueError("Batch must retain exact BF16 boundaries and four existing output positions")
                hidden=batch["hidden"]
                rope=rotary(hidden,batch["position_ids"])
                output=stack(hidden,attention_mask=batch["attention_mask"],position_ids=batch["position_ids"],
                    cache_position=batch["position_ids"][0],position_embeddings=rope,use_cache=False)
                hidden=output[0] if isinstance(output,tuple) else output
                hidden=hidden[torch.arange(len(selected),device=E.device)[:,None],batch["loss_positions"]]
                logits=F.linear(norm(hidden),E)
                if logits.shape!=(len(selected),4,len(E)) or logits.dtype!=torch.bfloat16:
                    raise ValueError("Packed inference must produce actual BF16 full-vocabulary logits")
                write_bfloat16_bits(array,selected,logits)
                if label_ids is not None and split=="development":
                    for j,index in enumerate(selected):
                        case=items[index];values=logits[j,-1,label_ids[:len(case["choices"])]].float()
                        records.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                            prediction=int(values.argmax()),logits=values.cpu().tolist()))
                if _fixed(stack,norm,E,rotary)!=before:raise ValueError("Accepted teacher changed during target capture")
                seen.update(selected);batches.append(dict(split=split,indices=selected))
                if progress is not None:progress(dict(phase="packed_teacher_targets",split=split,completed=len(seen),
                    total=count,indices=selected))
                del batch,hidden,output,rope,logits
        if seen!=set(range(count)):raise ValueError("Incomplete packed teacher rows")
        array.flush()
    finally:array._mmap.close()
    _guard(E.device,guard)
    if _fixed(stack,norm,E,rotary)!=before:raise ValueError("Accepted teacher changed during target capture")
    return dict(status="completed",path=str(path.resolve()),sha256=sha(path),shape=[count,4,len(E)],
        storage="Lossless existing BF16 logit bits in native-endian int16; no dtype conversion before storage",
        teacher_execution="Packed accepted parent in inference_mode; original native prefix boundary at parent cut",
        row_order="Original dataset indices; training batches then independently grouped development batches",
        microbatch=4,batches=batches,training_count=len(train),development_count=len(development),
        fixed_state_unchanged=True,records=records if label_ids is not None else None,
        label_ids=label_ids,read_bytes={k:v-start_reads.get(k,0) for k,v in getattr(dataset,"read_bytes",{}).items()},
        calibration_jacobians_changed=False,scope="Packed parent output-only recovery targets; differentiable VJP teacher remains separate")
