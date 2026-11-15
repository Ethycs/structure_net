"""Retokenized Qwen3 tasks and one-resident-block native capture.

Execution precision is explicit; disk arrays are always float32. BF16 native
values are exactly representable in those arrays, and BF16 dataset batches
recover the same values. This module does not allocate a complete model, acquire
a GPU lease, choose outcomes, or import previous experiment producers.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

SYSTEM_PROMPT = "Answer the multiple-choice question. Reply with only the letter of the correct option."
FAMILIES = ("arc", "boolq", "hellaswag", "winogrande")
TOKEN_FIELDS = {"input_ids", "loss_positions", "targets", "offset", "stop", "source_offset", "source_stop"}


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def _validate_cases(cases, *, systems_only):
    if not cases or len({c["identity"] for c in cases}) != len(cases):
        raise ValueError("Nonempty distinct task identities required")
    for c in cases:
        if (c["split"] not in ("train", "development") or c["family"] not in FAMILIES
                or not isinstance(c["prompt"], str) or not c["prompt"]
                or len(c["choices"]) not in (2, 4) or type(c["answer"]) is not int
                or not 0 <= c["answer"] < len(c["choices"])):
            raise ValueError("Only valid original training/development multiple-choice tasks are allowed")
    if not systems_only:
        wanted = {(s, f): n for s, n in (("train", 128), ("development", 16)) for f in FAMILIES}
        if Counter((c["split"], c["family"]) for c in cases) != wanted:
            raise ValueError("Scientific capture requires all512 train/64 development cases; subsets are systems-only")


def load_raw_cases(path, *, systems_only=False):
    """Load raw prompts/provenance, deliberately removing every old token field."""
    payload = json.loads(Path(path).read_text())
    if "items" in payload:
        cases = payload["items"]
    elif "splits" in payload:
        cases = [dict(c, split=split) for split in ("train", "development") for c in payload["splits"][split]]
    else:
        raise ValueError("Expected tasks.json splits or capture/items.json")
    result = [{k: deepcopy(v) for k, v in c.items() if k not in TOKEN_FIELDS} for c in cases]
    _validate_cases(result, systems_only=systems_only)
    return result


class TokenizationLimitError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Qwen3 tokenization has {report['over_limit_count']} prompts above "
                         f"{report['max_tokens']} tokens; maximum {report['max_observed_length']}. No cases truncated or dropped.")


def prepare_items(cases, tokenizer, *, max_tokens=416, systems_only=False):
    """Apply Qwen3's non-thinking template and rebuild all four causal targets."""
    _validate_cases(cases, systems_only=systems_only)
    if type(max_tokens) is not int or max_tokens < 4:
        raise ValueError("Explicit token capacity of at least four is required")
    labels = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(4)]
    if any(len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0 for ids in labels):
        raise ValueError("Each answer letter must be one Qwen3 vocabulary token")
    label_ids = [ids[0] for ids in labels]
    if len(set(label_ids)) != 4:
        raise ValueError("Answer labels must have distinct token IDs")
    items, offset, too_long = [], 0, []
    for case in cases:
        messages = [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=case["prompt"])]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if (not isinstance(ids, list) or len(ids) < 4
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError("Tokenizer must return a plain valid token list")
        length = len(ids)
        if length > max_tokens:
            too_long.append(dict(identity=case["identity"], length=length))
        positions = [0, (length-2)//2, length-2, length-1]
        targets = [ids[p+1] for p in positions[:-1]]+[label_ids[case["answer"]]]
        raw = {k: deepcopy(v) for k,v in case.items() if k not in TOKEN_FIELDS}
        items.append(dict(raw, input_ids=ids, loss_positions=positions, targets=targets,
                          offset=offset, stop=offset+length))
        offset += length
    report = dict(count=len(items), split_counts=dict(Counter(c["split"] for c in items)),
        total_tokens=offset, max_tokens=max_tokens, max_observed_length=max(len(c["input_ids"]) for c in items),
        over_limit_count=len(too_long), over_limit=too_long, truncated_count=0, dropped_count=0,
        enable_thinking=False, retokenized=True, systems_only=systems_only)
    if too_long:
        raise TokenizationLimitError(report)
    return dict(items=items, label_ids=label_ids, tokenization=report)


def systems_subset(items, indices):
    """Explicit systems-only subset with fresh contiguous disk offsets."""
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(items) for i in indices):
        raise ValueError("Distinct nonempty systems indices required")
    result, offset = [], 0
    for index in indices:
        c = deepcopy(items[index]); c.update(offset=offset, stop=offset+len(c["input_ids"]))
        result.append(c); offset = c["stop"]
    return result


def _validate_items(items, label_ids, vocabulary, *, systems_only=True, max_tokens=None):
    _validate_cases(items, systems_only=systems_only)
    if len(label_ids) != 4 or len(set(label_ids)) != 4 or any(type(i) is not int or not 0 <= i < vocabulary for i in label_ids):
        raise ValueError("Four distinct valid answer-token IDs required")
    offset, lengths = 0, []
    for c in items:
        ids, positions, targets = c["input_ids"], c["loss_positions"], c["targets"]
        length = len(ids)
        if (length < 4 or max_tokens is not None and length > max_tokens
                or any(type(t) is not int or not 0 <= t < vocabulary for t in ids)
                or c["offset"] != offset or c["stop"] != offset+length
                or positions != [0, (length-2)//2, length-2, length-1]
                or targets != [ids[p+1] for p in positions[:-1]]+[label_ids[c["answer"]]]):
            raise ValueError("Changed token layout, causal positions, or final gold-letter target")
        lengths.append(length); offset += length
    return lengths


def _batch(items, bank, indices, *, device, dtype):
    lengths = [len(items[i]["input_ids"]) for i in indices]; longest = max(lengths)
    host = np.zeros((len(indices), longest, bank.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        c = items[index]; host[row, :lengths[row]] = bank[c["offset"]:c["stop"]]
    hidden = torch.from_numpy(host).to(device=device, dtype=dtype)
    positions = torch.arange(longest, device=device)
    position_ids = positions[None].expand(len(indices), -1)
    valid = positions[None] < torch.tensor(lengths, device=device)[:, None]
    allowed = valid[:, None, None, :] & (positions[None, :] <= positions[:, None])[None, None]
    mask = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
    return dict(indices=list(indices), hidden=hidden, lengths=lengths, position_ids=position_ids,
        attention_mask=mask, loss_positions=torch.tensor([items[i]["loss_positions"] for i in indices],device=device),
        targets=torch.tensor([items[i]["targets"] for i in indices],device=device))


class FP32BoundaryDataset:
    """Read-only float32 disk banks; output batches may be FP32 or BF16."""
    def __init__(self, items, boundary_path, accepted_logits_path, *, label_ids,
                 original_logits_path=None, original_indices=None):
        self.items = deepcopy(list(items)); self.label_ids = list(label_ids)
        self.boundary = self._open(boundary_path, 2); self.accepted = self._open(accepted_logits_path, 3)
        self.original = self._open(original_logits_path, 3) if original_logits_path is not None else None
        self.width, self.vocabulary = self.boundary.shape[1], self.accepted.shape[-1]
        self.lengths = _validate_items(self.items, self.label_ids, self.vocabulary)
        if self.boundary.shape[0] != sum(self.lengths) or self.accepted.shape[:2] != (len(items), 4):
            raise ValueError("Boundary and four-position teacher bank layouts disagree")
        self.original_indices = None if original_indices is None else list(original_indices)
        if self.original is not None and (self.original_indices is None or len(self.original_indices) != len(items)
                or self.original.shape[1:] != (4, self.vocabulary)
                or any(type(i) is not int or not 0 <= i < len(self.original) for i in self.original_indices)):
            raise ValueError("Original teacher needs an explicit valid identity-row mapping")
        self.training = [i for i,c in enumerate(items) if c["split"] == "train"]
        self.development = [i for i,c in enumerate(items) if c["split"] == "development"]
        self.read_bytes = dict(boundary=0, accepted=0, original=0)

    @staticmethod
    def _open(path, ndim):
        bank = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(bank,np.memmap) or bank.dtype != np.dtype("float32") or not bank.dtype.isnative or bank.ndim != ndim or min(bank.shape) < 1:
            raise ValueError("Expected native-endian, nonempty float32 memmap; BF16-bit arrays are unsupported")
        return bank

    def batch(self, indices, *, device="cpu", dtype=torch.float32, include_original=False):
        indices = list(indices)
        if dtype not in (torch.float32,torch.bfloat16) or not indices or any(type(i) is not int or not 0 <= i < len(self.items) for i in indices):
            raise ValueError("Explicit FP32/BF16 batches and valid nonempty indices required")
        result = _batch(self.items,self.boundary,indices,device=device,dtype=dtype)
        result["teacher"] = torch.from_numpy(np.array(self.accepted[indices],copy=True)).to(device=device)
        self.read_bytes["boundary"] += sum(self.lengths[i] for i in indices)*self.width*4
        self.read_bytes["accepted"] += len(indices)*4*self.vocabulary*4
        if include_original and self.original is not None:
            result["original"] = torch.from_numpy(np.array(self.original[[self.original_indices[i] for i in indices]],copy=True)).to(device=device)
            self.read_bytes["original"] += len(indices)*4*self.vocabulary*4
        if any(not torch.isfinite(result[k]).all() for k in ("hidden","teacher","original") if k in result):
            raise FloatingPointError("Nonfinite captured values or activation cast overflow")
        return result

    def close(self):
        for bank in (self.boundary,self.accepted,self.original):
            if bank is not None and not bank._mmap.closed:bank._mmap.close()


@contextmanager
def _execution(device):
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode(), torch.autocast(device_type=device.type,enabled=False):yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=cudnn


def _check_module(module,device,dtype):
    if any(p.device != device or p.dtype != dtype for p in module.parameters()):
        raise ValueError("Loaded native parameters must match the explicit device/execution dtype")
    if any(b.device != device or b.is_floating_point() and b.dtype not in (dtype,torch.float32) for b in module.buffers()):
        raise ValueError("Native buffers are on the wrong device or precision")


def _summary(records):
    correct=sum(r["prediction"] == r["answer"] for r in records)
    return dict(count=len(records),correct=correct,accuracy=correct/len(records) if records else None,
        by_family={f:dict(count=sum(r["family"] == f for r in records),
                         correct=sum(r["family"] == f and r["prediction"] == r["answer"] for r in records))
                   for f in FAMILIES})


def stream_native_capture(items, *, input_embedding, block_loader, readout_loader, rotary, label_ids,
                          directory, capture_cuts=(34,35), layer_count=36, device="cpu", dtype=torch.bfloat16,
                          microbatch=4, max_tokens=416, systems_only=False, guard=None, progress=None):
    """Write requested boundary cuts and native full-vocabulary teacher logits.

    ``block_loader(i)`` returns one native block already on device in ``dtype``.
    ``readout_loader()`` lazily returns ``(untied_readout_weight, final_norm)``.
    Only this function's scratch files are deleted; requested boundary files and
    failure receipts are preserved. Caller must hold a reservation for CUDA.
    """
    directory,device=Path(directory).resolve(),torch.device(device)
    if device.type == "cuda" and device.index is None:device=torch.device("cuda",0)
    if device.type == "cuda" and guard is None:raise ValueError("CUDA capture requires a reservation guard")
    guard = guard or (lambda:None);progress = progress or (lambda **_kw:None)
    if dtype not in (torch.float32,torch.bfloat16) or type(layer_count) is not int or layer_count < 1 or type(microbatch) is not int or microbatch < 1:
        raise ValueError("Explicit FP32/BF16 execution, layer count and positive microbatch required")
    cuts=list(capture_cuts)
    if len(set(cuts)) != len(cuts) or any(type(i) is not int or not 0 <= i <= layer_count for i in cuts):
        raise ValueError("Distinct boundary cuts must be in0..layer_count")
    cuts=sorted(cuts);items=deepcopy(list(items))
    if input_embedding.device.type != "cpu" or input_embedding.ndim != 2 or input_embedding.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError("Pass only the native CPU embedding matrix, not a complete model")
    lengths=_validate_items(items,label_ids,input_embedding.shape[0],systems_only=systems_only,max_tokens=max_tokens)
    shape=(sum(lengths),input_embedding.shape[1]);vocabulary=input_embedding.shape[0]
    guard()  # Before a CUDA context, model loader, output directory or tensor transfer.
    directory.mkdir(parents=True,exist_ok=False);scratch=directory/"_scratch";scratch.mkdir()
    source=destination=teacher=None;source_path=None;temporary=set();artifacts={};started=time.perf_counter()
    io=dict(hidden_read_bytes=0,hidden_written_bytes=0,teacher_written_bytes=0)
    def path_for(cut):
        path=(directory if cut in cuts else scratch)/f"boundary_{cut:02d}.npy"
        if cut not in cuts:temporary.add(path)
        return path
    def bank_info(path,shape):
        return dict(path=str(path),shape=list(shape),dtype="float32",tensor_bytes=int(np.prod(shape))*4,file_bytes=path.stat().st_size)
    try:
        _write(directory/"items.json",dict(items=items,label_ids=list(label_ids),tokens=shape[0]))
        with _execution(device):
            source_path=path_for(0)
            source=np.lib.format.open_memmap(source_path,mode="w+",dtype=np.float32,shape=shape)
            for case in items:
                guard();hidden=input_embedding.index_select(0,torch.tensor(case["input_ids"])).to(dtype=dtype).float()
                if not torch.isfinite(hidden).all():raise FloatingPointError("Nonfinite native embedding")
                source[case["offset"]:case["stop"]]=hidden.numpy()
                io["hidden_written_bytes"]+=hidden.numel()*4;del hidden
            source.flush()
            if 0 in cuts:artifacts[0]=bank_info(source_path,shape)
            for layer in range(layer_count):
                guard();block=block_loader(layer).eval().requires_grad_(False);_check_module(block,device,dtype)
                destination_path=path_for(layer+1)
                destination=np.lib.format.open_memmap(destination_path,mode="w+",dtype=np.float32,shape=shape)
                for start in range(0,len(items),microbatch):
                    guard();indices=list(range(start,min(start+microbatch,len(items))))
                    batch=_batch(items,source,indices,device=device,dtype=dtype)
                    hidden=batch["hidden"];rope=rotary(hidden,batch["position_ids"])
                    value=block(hidden,attention_mask=batch["attention_mask"],position_ids=batch["position_ids"],
                        cache_position=batch["position_ids"][0],position_embeddings=rope,use_cache=False)
                    value=value[0] if isinstance(value,tuple) else value
                    if value.shape != hidden.shape or value.dtype != dtype or not torch.isfinite(value).all():
                        raise FloatingPointError("Invalid native block output shape/precision/values")
                    host=value.float().cpu().numpy()
                    for row,index in enumerate(indices):
                        c=items[index];destination[c["offset"]:c["stop"]]=host[row,:lengths[index]]
                    valid=sum(lengths[i] for i in indices)*shape[1]*4
                    io["hidden_read_bytes"]+=valid;io["hidden_written_bytes"]+=valid
                    del batch,hidden,rope,value,host
                destination.flush();source._mmap.close()
                if source_path in temporary:source_path.unlink();temporary.remove(source_path)
                source,destination=destination,None;source_path=destination_path
                if layer+1 in cuts:artifacts[layer+1]=bank_info(source_path,shape)
                del block;gc.collect()
                if device.type == "cuda":guard();torch.cuda.empty_cache()
                progress(event="boundary",layer=layer,boundary=layer+1,retained=layer+1 in cuts)
            # No native decoder block remains when the untied vocabulary head loads.
            guard();readout,norm=readout_loader();norm.eval().requires_grad_(False);_check_module(norm,device,dtype)
            if readout.device != device or readout.dtype != dtype or readout.shape != (vocabulary,shape[1]):
                raise ValueError("Untied native readout must match vocabulary, width, device and execution dtype")
            teacher_path=directory/"teacher_logits.npy"
            teacher=np.lib.format.open_memmap(teacher_path,mode="w+",dtype=np.float32,shape=(len(items),4,vocabulary))
            records=[]
            for start in range(0,len(items),microbatch):
                guard();indices=list(range(start,min(start+microbatch,len(items))))
                selected=np.stack([np.array(source[np.array(items[i]["loss_positions"])+items[i]["offset"]],copy=True) for i in indices])
                hidden=torch.from_numpy(selected).to(device=device,dtype=dtype)
                logits=F.linear(norm(hidden),readout).float()
                if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite native teacher logits")
                host=logits.cpu().numpy();teacher[indices]=host
                io["hidden_read_bytes"]+=selected.nbytes;io["teacher_written_bytes"]+=host.nbytes
                for row,index in enumerate(indices):
                    c=items[index];values=host[row,-1,list(label_ids[:len(c["choices"])])]
                    records.append(dict(identity=c["identity"],split=c["split"],family=c["family"],answer=c["answer"],
                                        prediction=int(values.argmax()),logits=values.tolist()))
                del selected,hidden,logits,host
            teacher.flush();teacher_info=bank_info(teacher_path,teacher.shape)
            del readout,norm
        result=dict(status="completed",systems_only=systems_only,execution_dtype=str(dtype).removeprefix("torch."),
            cache_dtype="float32",cache_precision="Exact float32 values; BF16 execution values are losslessly promoted",
            autocast=False,tf32=False,layer_count=layer_count,microbatch=microbatch,capture_cuts=cuts,
            shape=list(shape),vocabulary=vocabulary,boundary_paths={str(k):v["path"] for k,v in artifacts.items()},
            boundary_inventory={str(k):v for k,v in artifacts.items()},teacher_logits_path=str(teacher_path),teacher_inventory=teacher_info,
            items_path=str(directory/"items.json"),records=records,
            by_split={split:_summary([r for r in records if r["split"] == split]) for split in ("train","development")},
            owned_native_blocks_max=1,readout_loaded_after_blocks=True,io_bytes=io,elapsed_seconds=time.perf_counter()-started)
        _write(directory/"result.json",result)
        return result
    except BaseException as error:
        _write(directory/"failure.json",dict(status="failed",error=repr(error),completed_cuts=sorted(artifacts),
                                           elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        for bank in (source,destination,teacher):
            if bank is not None and not bank._mmap.closed:bank._mmap.close()
        for path in temporary:
            if path.exists():path.unlink()
        scratch.rmdir()
