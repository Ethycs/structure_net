"""Broad reconstruction training of a new copy of the released learned codec."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_learned_codec_corpus as corpus
from experiments.structure_net.ncm_llama import INITIAL, VENDOR, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-broad-learned-codec-reconstruction-plan.md")
PILOT = ROOT / "20261023_broad_learned_codec_pilot_v1"
TRAIN = ROOT / "20261023_broad_learned_codec_train_v1"
RECON = ROOT / "20261023_broad_learned_codec_reconstruction_v1"


@dataclass(frozen=True)
class Config:
    updates: int = 2048
    microbatch: int = 4
    accumulation: int = 8
    max_lr: float = 2e-5
    warmup: int = 128
    seed: int = 8142
    sampler_seed: int = 9142


def rate(update, config=Config()):
    if not 1 <= update <= config.updates:
        raise ValueError("update outside fixed codec budget")
    if update <= config.warmup:
        return config.max_lr * update / config.warmup
    return config.max_lr * .5 * (1 + math.cos(math.pi * (update-config.warmup)/(config.updates-config.warmup)))


def require_gpu():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 4_000_000_000):
        raise ValueError("planned GPU2 with4GBfree required")
    torch.backends.cuda.matmul.allow_tf32 = False


def terminal(root):
    if json.loads((root/"worker_terminal_receipt.json").read_text())["exit_code"] != 0:
        raise ValueError("observed successful worker termination required")
    verify(json.loads((root/"producer.json").read_text()))


def producing(root, pilot, extra=()):
    files = [Path(__file__), PLAN, Path(corpus.__file__), corpus.PLAN]
    files += sorted(VENDOR.rglob("*.py"))
    inputs = [INITIAL/"tokenizer.pt", INITIAL/"config.yaml"]
    inputs += [corpus.DEST/n for n in ("producer.json", "result.json", "fit_windows.pt",
        "fit_origins.json", "summary_diagnostic.json", "worker_terminal_receipt.json")]
    producer = freeze(root, "learned_codec_systems_pilot" if pilot else "learned_codec_training_or_reconstruction",
        files+inputs+list(extra), {**asdict(Config()), "pilot": pilot, "physical_gpu": 2,
        "actual_updates":2 if pilot else Config().updates, "actual_accumulation":1 if pilot else Config().accumulation,
        "precision": "FP32", "allow_tf32": False, "parent_loaded": False,
        "oracle_target_codes": True, "task_accuracy": None, "torch": torch.__version__, "cuda": torch.version.cuda})
    producer["hypothesis_id"] = "ncm-broad-learned-codec-reconstruction-v1"
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/f"codec_{index:03d}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = corpus.quality.run.q.digest({k:v for k,v in producer.items() if k!="fingerprint"})
    write_json(root/"producer.json", producer)
    return producer


def load_codec(state=None, device="cuda"):
    OmegaConf, VQVAE, _ = vendor_classes()
    cfg = OmegaConf.load(INITIAL/"config.yaml")
    codec = VQVAE(**cfg.tokenizer.vqvae)
    released = torch.load(INITIAL/"tokenizer.pt", map_location="cpu", weights_only=True)
    codec.load_state_dict(released["model"], strict=True)
    if state is not None:
        codec.load_state_dict(state, strict=True)
    if codec.context_length != 48 or list(codec.vq.levels) != [4,6,10,14,20,26,32,40,48]:
        raise ValueError("released48-position codec architecture differs")
    if any(e.weight.requires_grad for e in codec.vq.embeddings):
        raise ValueError("EMA books must stay outside optimizer gradients")
    return codec.to(device).float().eval()


@torch.no_grad()
def reconstruction(codec, cases, tokenizer):
    before = codec.training; codec.eval(); result = []
    for case in cases:
        packed = torch.tensor(case["packed_ids"], dtype=torch.long, device=next(codec.parameters()).device)
        codes = codec.encode(packed)
        logits = codec.decode(codes)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite codec reconstruction")
        raw = logits.argmax(-1).flatten().cpu().tolist()
        end = next((i for i,t in enumerate(raw) if t==50256), len(raw))
        text = tokenizer.decode(raw[:end], skip_special_tokens=True)
        gold = case["target_ids"]
        numbers = lambda value: re.findall(r"\b\d+(?:[,.]\d+)*\b", value)
        result.append({"id":case["id"], "target":case["text"], "target_ids":gold,
            "raw_ids":raw, "codes":codes.cpu().tolist(), "text":text,
            "correct":sum(a==b for a,b in zip(raw,gold)), "total":len(gold),
            "exact_text":text==case["text"], "first_EOS_correct":end==len(gold)-1,
            "active_invalid_ids":sum(t>=len(tokenizer) for t in raw[:end]),
            "changed_numbers":numbers(text)!=numbers(case["text"])})
    codec.train(before)
    return result


def save(path, codec, optimizer, sampler, step, producer):
    if path.exists():
        raise ValueError("checkpoint overwrite forbidden")
    torch.save({"model":codec.state_dict(), "optimizer":optimizer.state_dict(), "step":step,
        "sampler_rng":sampler.get_state(), "torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all(), "fine_dropout_step":codec.vq._fd_step.cpu().clone(),
        "producer_fingerprint":producer["fingerprint"], "config":asdict(Config())}, path)


def reload_check(path, codec, optimizer, sampler, cases, tokenizer):
    before = reconstruction(codec,cases,tokenizer)
    saved = torch.load(path,map_location="cpu",weights_only=True)
    with torch.no_grad():
        for value in codec.state_dict().values(): value.zero_()
        codec.vq._fd_step.zero_()
    optimizer.state.clear();sampler.manual_seed(0);torch.manual_seed(0)
    codec.load_state_dict(saved["model"],strict=True);optimizer.load_state_dict(saved["optimizer"])
    codec.vq._fd_step.copy_(saved["fine_dropout_step"])
    sampler.set_state(saved["sampler_rng"]);torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if any(not torch.equal(v.cpu(),saved["model"][k]) for k,v in codec.state_dict().items()):
        raise ValueError("codec tensor reload differs")
    state=optimizer.state_dict()
    if state["param_groups"]!=saved["optimizer"]["param_groups"]:
        raise ValueError("optimizer groups differ")
    for index, values in saved["optimizer"]["state"].items():
        if any(not torch.equal(v.cpu(),state["state"][index][k].cpu()) for k,v in values.items()):
            raise ValueError("optimizer moments differ")
    if (not torch.equal(sampler.get_state(),saved["sampler_rng"])
            or not torch.equal(torch.get_rng_state(),saved["torch_rng"])
            or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved["cuda_rng"]))
            or not torch.equal(codec.vq._fd_step.cpu(),saved["fine_dropout_step"])):
        raise ValueError("codec RNG/nonpersistent dropout state differs")
    if reconstruction(codec,cases,tokenizer)!=before:
        raise ValueError("codec reconstruction reload differs")
    return before


def train(pilot=False):
    require_gpu();terminal(corpus.DEST);config=Config()
    if not pilot:
        terminal(PILOT)
        if json.loads((PILOT/"result.json").read_text())["status"]!="systems_codec_training_reload_passed":
            raise ValueError("real matching CUDA pilot required")
    root=PILOT if pilot else TRAIN
    extra=[] if pilot else [PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    p=producing(root,pilot,extra)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(corpus.GPT2,local_files_only=True)
    tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    cases=json.loads((corpus.DEST/"summary_diagnostic.json").read_text())
    longest=sorted(cases,key=lambda c:(-len(c["target_ids"]),c["id"]))[0]
    probe_cases=[cases[0],longest]
    windows=torch.load(corpus.DEST/"fit_windows.pt",map_location="cpu",weights_only=True)
    codec=load_codec();initial_hash=module_hash(codec)
    if not pilot and initial_hash!=json.loads((PILOT/"result.json").read_text())["initial_model_sha256"]:
        raise ValueError("full codec run did not restore fresh released initialization")
    initial_probe=reconstruction(codec,probe_cases,tokenizer)
    write_json(root/"initial_probe.json",initial_probe)
    optimizer=torch.optim.AdamW([v for v in codec.parameters() if v.requires_grad],lr=config.max_lr,
        betas=(.9,.95),weight_decay=.01)
    sampler=torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    updates,accumulation=(2,1) if pilot else (config.updates,config.accumulation)
    codec.train();torch.cuda.reset_peak_memory_stats();(root/"checkpoints").mkdir()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group["lr"]=rate(step+1,config)
        indices=[];losses=[]
        for micro in range(accumulation):
            chosen=(list(range(config.microbatch)) if step==0 else list(range(len(windows)-config.microbatch,len(windows)))) if pilot else torch.randint(len(windows),(config.microbatch,),generator=sampler).tolist()
            targets=windows[chosen].cuda();logits,vq,*_=codec(targets)
            ce=F.cross_entropy(logits.float().flatten(0,1),targets.flatten(),ignore_index=50257)
            loss=ce+vq
            if not torch.isfinite(loss):raise ValueError("nonfinite learned codec objective")
            (loss/accumulation).backward();indices.append(chosen);losses.append([float(ce.detach()),float(vq.detach())])
            del logits,targets,loss,ce,vq
        for part in (codec.encoder,codec.decoder):
            if not any(v.grad is not None and torch.count_nonzero(v.grad).item() for v in part.parameters() if v.requires_grad):
                raise ValueError("missing encoder/decoder gradient")
        norm=torch.nn.utils.clip_grad_norm_([v for v in codec.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated()>4_500_000_000:raise ValueError("codec allocation exceeds pilot budget")
        row={"step":step+1,"window_indices":indices,"CE_VQ":losses,"gradient_norm":float(norm),"lr":rate(step+1)}
        with (root/"trace.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        if pilot or (step+1)%32==0:print(json.dumps(row),flush=True)
        if not pilot and (step+1)%512==0 and step+1<updates:
            save(root/"checkpoints"/f"step_{step+1:06d}.pt",codec,optimizer,sampler,step+1,p)
    save(root/"model.pt",codec,optimizer,sampler,updates,p)
    reloaded=reload_check(root/"model.pt",codec,optimizer,sampler,probe_cases,tokenizer)
    if module_hash(codec)==initial_hash:raise ValueError("codec did not change")
    verify(p);write_json(root/"reload_probe.json",reloaded)
    result={"status":"systems_codec_training_reload_passed" if pilot else "trained_pending_reconstruction_diagnostic",
        "updates":updates,"accumulation":accumulation,"initial_model_sha256":initial_hash,
        "checkpoint_sha256":file_hash(root/"model.pt"),"reload_exact":True,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"parent_loaded":False,
        "task_accuracy":None,"NCM_success_established":False,"final_test_opened":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--pilot",action="store_true")
    train(parser.parse_args().pilot)
