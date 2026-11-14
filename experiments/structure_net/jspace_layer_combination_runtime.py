"""Frozen native/recovered Qwen banks with call-local shared-prefix scoring.

This module evaluates caller-supplied masks; it has no search/selection policy.
Each True mask entry selects the independently recovered compact block at that
original layer index. All masks retain the original tied embedding and norm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os

import torch
from torch import nn
from torch.nn import functional as F

from . import jspace_recovered_assembly as assembly
from .jspace_single_block_verify import validate_records

CARDS = {"3060":("GPU-0370fdb0",11264), "2060_SUPER":("GPU-7b1db3cb",7168), "2060":("GPU-8cf440df",5120)}
PARTITION = {"3060":tuple(range(18)), "2060_SUPER":tuple(range(18,29)), "2060":tuple(range(29,36))}


class MixedBlockBank(nn.Module):
    """Read-only modules; execution contexts and branch states live in each call."""
    def __init__(self, embedding, native_blocks, compact_blocks, norm, rotary_by_device):
        super().__init__()
        if embedding.ndim != 2 or embedding.dtype != torch.float32 or not native_blocks or len(native_blocks) != len(compact_blocks):
            raise ValueError("FP32 embedding and equally sized nonempty block banks required")
        self.register_buffer("embedding", embedding.detach())
        self.native, self.compact = nn.ModuleList(native_blocks), nn.ModuleList(compact_blocks)
        self.norm = norm
        self._rotary_keys = {str(torch.device(k)):"device_"+str(torch.device(k)).replace(":","_") for k in rotary_by_device}
        self.rotary = nn.ModuleDict({self._rotary_keys[str(torch.device(k))]:v for k,v in rotary_by_device.items()})
        self.layer_devices = tuple(next(b.parameters()).device for b in self.native)
        for layer, (native, compact) in enumerate(zip(self.native, self.compact)):
            device = self.layer_devices[layer]
            if str(device) not in self._rotary_keys: raise ValueError("Every block device needs original RoPE")
            if any(t.device != device for b in (native,compact) for t in (*b.parameters(),*b.buffers())):
                raise ValueError("Each native/compact layer pair must share one device")
        if self.layer_devices[0] != embedding.device or norm.weight.device != embedding.device:
            raise ValueError("Embedding, first layer and original final norm must share the front device")
        if norm.weight.shape != (embedding.shape[1],) or not hasattr(norm,"variance_epsilon"):
            raise ValueError("Native-width Qwen RMSNorm required")
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in (*self.parameters(),*self.buffers())):
            raise ValueError("All resident model tensors must remain FP32")
        self.eval().requires_grad_(False)

    @property
    def layers(self): return len(self.native)

    def rotary_for(self, device): return self.rotary[self._rotary_keys[str(torch.device(device))]]

    def storage_inventory(self):
        per_device = {}
        for tensor in (*self.parameters(),*self.buffers()):
            device = str(tensor.device); storage = tensor.untyped_storage()
            per_device.setdefault(device,{})[storage.data_ptr()] = storage.nbytes()
        sizes = {device:sum(storages.values()) for device,storages in per_device.items()}
        return dict(storage_bytes=sum(sizes.values()), by_device=sizes,
            scope="Both resident decoder banks plus original tiedE/norm/RoPE; excludes temporary states, label-row copies and allocator reserve")


def validate_masks(masks, layers):
    if not isinstance(masks,dict) or not masks: raise ValueError("Named nonempty mask dictionary required")
    seen = set()
    for name,mask in masks.items():
        if not isinstance(name,str) or not name or not isinstance(mask,tuple) or len(mask) != layers or any(type(x) is not bool for x in mask):
            raise ValueError("Each named mask must be a tuple of strictly boolean entries, one per layer")
        if mask in seen: raise ValueError("Caller must deduplicate identical masks")
        seen.add(mask)


def _trie(masks):
    root = {"children":{}}
    for name,mask in masks.items():
        node = root
        for choice in mask: node = node["children"].setdefault(choice,{"children":{}})
        node["name"] = name
    return root


def _validate_cases(cases, bank, label_ids, teacher_records):
    labels = list(label_ids)
    if (not cases or len({c["identity"] for c in cases}) != len(cases) or not labels
            or len(set(labels)) != len(labels) or any(type(i) is not int or not 0 <= i < len(bank.embedding) for i in labels)):
        raise ValueError("Distinct cases and valid original choice token IDs required")
    for c in cases:
        if (c.get("split","development") != "development" or not 1 <= len(c["input_ids"]) <= 384
                or any(type(i) is not int or not 0 <= i < len(bank.embedding) for i in c["input_ids"])
                or type(c["answer"]) is not int or not 0 <= c["answer"] < len(c["choices"]) <= len(labels)
                or len(c["choices"]) < 2):
            raise ValueError("Valid development-only raw tokens and answer choices required")
    validate_records(teacher_records, cases)
    return labels


@torch.inference_mode()
def score_masks(bank, cases, masks, label_ids, *, teacher_records, microbatch=4, share_prefixes=True,
                guard=None, progress=lambda **_kw:None):
    """Return per-mask records/metrics; shared-prefix state never survives a call.

    Prefix reuse is exact within each case batch. A depth-first traversal keeps
    only live branch ancestors, and unbranched runs replace their prior state.
    `share_prefixes=False` supplies an independent full-forward parity control.
    """
    if not isinstance(bank,MixedBlockBank) or bank.training or any(p.requires_grad for p in bank.parameters()):
        raise ValueError("Frozen evaluation-only resident bank required")
    if any(d.type == "cuda" for d in bank.layer_devices) and guard is None:
        raise ValueError("A live reservation guard is required before GPU scoring")
    if type(microbatch) is not int or microbatch < 1 or type(share_prefixes) is not bool:
        raise ValueError("Positive microbatch and explicit shared-prefix boolean required")
    validate_masks(masks, bank.layers); labels = _validate_cases(cases,bank,label_ids,teacher_records)
    guard = guard or (lambda:None); guard()
    front = bank.embedding.device; tree = _trie(masks)
    label_weight = bank.embedding.index_select(0,torch.tensor(labels,device=front))
    records = {name:[] for name in masks}
    telemetry = dict(decoder_calls=0,native_calls=0,compact_calls=0,max_live_branch_states=0,
        case_batches=0,shared_prefixes=share_prefixes)
    for start in range(0,len(cases),microbatch):
        guard(); batch = cases[start:start+microbatch]; lengths = [len(c["input_ids"]) for c in batch]
        width = max(lengths); positions = torch.arange(width,device=front)
        ids = torch.zeros(len(batch),width,device=front,dtype=torch.long)
        for row,c in enumerate(batch): ids[row,:lengths[row]] = torch.tensor(c["input_ids"],device=front)
        initial = F.embedding(ids,bank.embedding)
        position_ids = positions[None].expand(len(batch),-1)
        valid = positions[None] < torch.tensor(lengths,device=front)[:,None]
        allowed = (positions[:,None] >= positions[None,:])[None,None] & valid[:,None,None,:]
        mask = torch.zeros(allowed.shape,device=front,dtype=torch.float32).masked_fill_(~allowed,torch.finfo(torch.float32).min)
        contexts = {}
        for device in dict.fromkeys(bank.layer_devices):
            guard(); local_ids = position_ids.to(device)
            dummy = torch.empty(1,1,bank.embedding.shape[1],device=device,dtype=torch.float32)
            rope = bank.rotary_for(device)(dummy,local_ids)
            contexts[device] = dict(attention_mask=mask.to(device),position_ids=local_ids,
                cache_position=positions.to(device),position_embeddings=rope,use_cache=False,output_attentions=False)
        def step(hidden,layer,choice):
            guard(); device = bank.layer_devices[layer]
            block = bank.compact[layer] if choice else bank.native[layer]
            value = block(hidden.to(device),**contexts[device]); value = value[0] if isinstance(value,tuple) else value
            if value.shape != hidden.shape or value.dtype != torch.float32:
                raise ValueError("Native-width FP32 block output required")
            telemetry["decoder_calls"] += 1; telemetry["compact_calls" if choice else "native_calls"] += 1
            return value
        def finish(hidden,name):
            guard(); row = torch.arange(len(batch),device=hidden.device)
            final = hidden[row,torch.tensor(lengths,device=hidden.device)-1].to(front)
            final = final*torch.rsqrt(final.square().mean(-1,keepdim=True)+bank.norm.variance_epsilon)
            logits = F.linear(final*bank.norm.weight,label_weight).float().cpu()
            if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite mixed-model answer logits")
            added = []
            for row,c in enumerate(batch):
                scores = logits[row,:len(c["choices"])]; answer = c["answer"]
                wrong = scores.clone(); wrong[answer] = -torch.inf
                values = dict(identity=c["identity"],family=c["family"],answer=answer,prediction=int(scores.argmax()),
                    logits=scores.tolist(),teacher_prediction=teacher_records[start+row]["prediction"],
                    choice_ce=float(-F.log_softmax(scores,dim=-1)[answer]),choice_margin=float(scores[answer]-wrong.max()))
                added.append(values)
            records[name].extend(added)
            progress(kind="mask_batch",mask_id=name,completed=len(records[name]),total=len(cases),records=added)
        def walk(hidden,node,layer,branches):
            telemetry["max_live_branch_states"] = max(telemetry["max_live_branch_states"],branches)
            while layer < bank.layers:
                children = node["children"]
                if len(children) == 1:
                    choice,node = next(iter(children.items())); hidden = step(hidden,layer,choice); layer += 1
                else:
                    for choice in (False,True):
                        child = step(hidden,layer,choice)
                        walk(child,children[choice],layer+1,branches+1)
                        del child
                    return
            finish(hidden,node["name"])
        if share_prefixes: walk(initial,tree,0,0)
        else:
            for name,choices in masks.items():
                hidden = initial
                for layer,choice in enumerate(choices): hidden = step(hidden,layer,choice)
                finish(hidden,name)
        telemetry["case_batches"] += 1
        progress(kind="batch",completed=min(start+microbatch,len(cases)),total=len(cases))
    results = {}
    for name,rows in records.items():
        validate_records(rows,cases)
        results[name] = dict(records=rows,**assembly.single.ranking_summary(rows),
            choice_ce=sum(r["choice_ce"] for r in rows)/len(rows),
            choice_margin=sum(r["choice_margin"] for r in rows)/len(rows),
            teacher_agreement=sum(r["prediction"] == r["teacher_prediction"] for r in rows)/len(rows))
        progress(kind="mask",mask_id=name,result=results[name])
    telemetry["independent_decoder_calls"] = len(masks)*bank.layers*telemetry["case_batches"]
    telemetry["shared_prefix_saved_calls"] = telemetry["independent_decoder_calls"]-telemetry["decoder_calls"]
    return dict(results=results,execution=telemetry)


def allocator_settings(lease, *, cuda=None, visible=None):
    cuda = torch.cuda if cuda is None else cuda
    visible = os.environ.get("CUDA_VISIBLE_DEVICES","") if visible is None else visible
    tokens, allocation = visible.split(","),lease["allocation"]
    if len(tokens) != 3 or len(allocation) != 3 or cuda.is_initialized():
        raise ValueError("Fresh process with three visible leased physical GPU UUIDs required")
    resolved = []
    for token in tokens:
        matches = [uuid for uuid in allocation if token.startswith("GPU-") and uuid.startswith(token)]
        if len(matches) != 1: raise ValueError("Every visible UUID must identify one leased physical GPU")
        uuid = matches[0]; roles = [role for role,(prefix,mib) in CARDS.items() if uuid.startswith(prefix)]
        if len(roles) != 1 or type(allocation[uuid]) is not int or allocation[uuid] != CARDS[roles[0]][1]:
            raise ValueError("Frozen3060/SUPER/2060 allocation budgets required")
        resolved.append((uuid,roles[0]))
    if len({u for u,_ in resolved}) != 3: raise ValueError("Three distinct leased cards required")
    devices,settings = {},[]
    for logical,(uuid,role) in enumerate(resolved):
        total = int(cuda.get_device_properties(logical).total_memory); allowed = (allocation[uuid]-256)*1024**2
        if not 0 < allowed <= total or cuda.memory.get_allocator_backend() != "native":
            raise ValueError("Native allocator and sufficient physical memory required")
        cuda.set_per_process_memory_fraction(allowed/total,device=logical); devices[role] = f"cuda:{logical}"
        settings.append(dict(logical_device=logical,physical_uuid=uuid,role=role,lease_bytes=allocation[uuid]*1024**2,
            allocator_limit_bytes=allowed,device_total_memory=total,memory_fraction=allowed/total))
    return dict(devices=devices,settings=settings,lease_id=lease["id"],allocation=dict(allocation))


def device_peaks(settings, *, cuda=None):
    cuda = torch.cuda if cuda is None else cuda
    result = {}
    for row in settings["settings"]:
        reserved = cuda.max_memory_reserved(row["logical_device"])
        if reserved > row["allocator_limit_bytes"]: raise MemoryError("Resident bank exceeded a card's allocator cap")
        result[row["physical_uuid"]] = dict(peak_reserved_bytes=reserved,
            peak_allocated_bytes=cuda.max_memory_allocated(row["logical_device"]),
            allocator_limit_bytes=row["allocator_limit_bytes"],role=row["role"])
    return result


def state_digest(bank, *, chunk_bytes=32*1024**2, guard=lambda:None):
    """Hash all actual weights and buffers in bounded CPU transfer chunks."""
    if type(chunk_bytes) is not int or chunk_bytes < 4: raise ValueError("Positive bounded hash chunk size required")
    digest = hashlib.sha256(); count = total = 0
    for kind,values in (("parameter",bank.named_parameters()),("buffer",bank.named_buffers())):
        for name,value in values:
            guard(); digest.update(json.dumps([kind,name,list(value.shape),str(value.dtype)]).encode())
            flat = value.detach().reshape(-1); step = max(1,chunk_bytes//value.element_size())
            for start in range(0,flat.numel(),step): digest.update(flat[start:start+step].cpu().numpy().tobytes())
            count += 1; total += value.numel()*value.element_size()
    return dict(sha256=digest.hexdigest(),tensor_count=count,bytes=total)


def load_bank(assembly_plan, devices, *, guard):
    """Use only frozen independent selections; load separate native sources for compilation."""
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    guard(); front = devices["3060"]
    embedding,norm,rotary = load_readout_norm(assembly_plan["model"]["path"],device=front,dtype=torch.bfloat16)
    embedding,norm,rotary = embedding.float(),norm.float(),rotary.float()
    rotary_by_device = {str(torch.device(device)):deepcopy(rotary).to(device) for device in devices.values()}
    native,compact = [],[]
    for layer in range(36):
        role = next(role for role,indices in PARTITION.items() if layer in indices); device = devices[role]
        guard(); compact.append(assembly.load_block(assembly_plan,"all_compact",layer,device=device,guard=guard))
        guard(); native.append(load_native_block(assembly_plan["model"]["path"],layer,device=device,dtype=torch.float32))
    bank = MixedBlockBank(embedding,native,compact,norm,rotary_by_device); guard()
    return bank
