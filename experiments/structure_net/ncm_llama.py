"""Pinned NCM architecture and contracts for sequence distillation from Llama."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import torch

from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json

HYPOTHESIS_ID = "llama32-3b-ncm-distillation-v1"
SCHEMA_VERSION = "nal.ncm-llama.v1"
BASE = Path("data/experiments/ncm_llama")
VENDOR = BASE / "vendor/ncm-20261021"
DEPENDENCIES = BASE / "dependencies"
INITIAL = BASE / "inputs/hazy-xsum-d342ce8/xsum"
GPT2 = Path("/home/rabbit/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
PLAN = Path("docs/07 - Status Reports/2026-10-21_ncm-llama-distillation-plan.md")
FIXTURES = BASE / "20261021_fixtures_v1/examples.json"
LABELS = BASE / "20261021_teacher_v1/labels.json"


@dataclass(frozen=True)
class Config:
    train_examples: int = 1024
    development_examples: int = 64
    heldout_examples: int = 100
    seed: int = 42
    prefix: int = 512
    canvas: int = 48
    codec_steps: int = 512
    generator_steps: int = 1024
    codec_batch: int = 16
    generator_batch: int = 8
    context: int = 1024

    def __post_init__(self):
        if asdict(self) != {"train_examples":1024,"development_examples":64,"heldout_examples":100,
            "seed":42,"prefix":512,"canvas":48,"codec_steps":512,"generator_steps":1024,
            "codec_batch":16,"generator_batch":8,"context":1024}:
            raise ValueError("this pilot's fixed configuration requires a new protocol for changes")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def vendor_classes():
    for p in (DEPENDENCIES, VENDOR):
        if not p.is_dir():
            raise FileNotFoundError(f"Missing pinned NCM prerequisite: {p}")
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0, str(p.resolve()))
    from omegaconf import OmegaConf
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    return OmegaConf, VQVAE, NextConceptTransformer


def tokenizers():
    from transformers import AutoTokenizer
    gpt = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    gpt.add_special_tokens({"pad_token":"<|pad|>"})
    llama = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if gpt.pad_token_id != 50257 or gpt.eos_token_id != 50256:
        raise ValueError("released text-vocabulary identity differs")
    return gpt, llama


def load_ncm(codec=None, generator=None, device="cuda", adaptation=False):
    OmegaConf, VQVAE, NextConceptTransformer = vendor_classes()
    cfg = OmegaConf.load(INITIAL / "config.yaml")
    if adaptation:
        cfg.model.dropout = .1
    vae = VQVAE(**cfg.tokenizer.vqvae)
    state = torch.load(codec or INITIAL / "tokenizer.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(state["model"], strict=True)
    vae.to(device).eval().requires_grad_(False)
    net = NextConceptTransformer(vae, cfg.model)
    state = torch.load(generator or INITIAL / "generator.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"], strict=True)
    net.to(device).eval()
    if net.prefix_len != 512 or vae.context_length != 48 or not net.query_level0:
        raise ValueError("released conditional architecture differs")
    return vae, net, cfg


def autocast(device="cuda"):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def text_check(text, example, common_tokenizer=None):
    low = re.sub(r"\s+", " ", text.lower()).strip()
    money = {int(x.replace(",", "")) for x in re.findall(r"\$\s*([0-9][0-9,]*)", text)}
    required, old = example["required"], example["superseded"]
    fields = {"project":required["project"].lower() in low,
        "owner":required["owner"].lower() in low,
        "budget":int(required["budget"]) in money,
        "deadline":bool(re.search(re.escape(required["deadline"].lower()) + r"(?!\d)", low)),
        "no_superseded_owner":old["owner"].lower() not in low,
        "no_superseded_budget":int(old["budget"]) not in money,
        "no_superseded_deadline":not bool(re.search(re.escape(old["deadline"].lower()) + r"(?!\d)", low)),
        "nonempty":bool(low), "one_paragraph":"\n\n" not in text.strip()}
    if common_tokenizer is not None:
        fields["within_128_tokens"] = len(common_tokenizer.encode(text, add_special_tokens=False)) <= 128
    return {"fields":fields,"diagnostic_pass":all(fields.values()),
        "boundary":"field diagnostics only; extra claims need source review"}


def label_ids(text, gpt, canvas=48):
    ids = gpt.encode(text, add_special_tokens=False)
    if len(ids) + 1 > canvas:
        raise ValueError("teacher label exceeds the codec canvas; no truncation permitted")
    return ids + [gpt.eos_token_id] + [gpt.pad_token_id] * (canvas-len(ids)-1)


@torch.no_grad()
def encode_codes(vae, target_ids):
    # Nearest-code assignment always uses FP32, independent of training autocast.
    with torch.autocast(target_ids.device.type, enabled=False):
        return vae.encode(target_ids)


def generator_input(vae, net, prefix, codes):
    with torch.no_grad():
        source = vae.encoder.wte(prefix)
        previous = vae.vq.idx_to_ncm_input(codes)
    # Keep the learned level0 query's gradient; no target codes enter this block.
    query = net.build_l0_query_input(prefix.shape[0], previous.dtype)
    return torch.cat((source, query, previous), dim=1)


def code_loss(vae, net, prefix, codes):
    x = generator_input(vae, net, prefix, codes)
    logits = net(x, prefix_pad_mask=prefix.eq(50257))
    if logits.shape[:2] != codes.shape:
        raise ValueError("all-level query/target alignment differs")
    return torch.nn.functional.cross_entropy(logits.float().flatten(0,1), codes.flatten())


def final_latent(vae, codes):
    with torch.no_grad():
        return vae.vq.idx_to_zhats(codes, return_zlevels=False)[0][-1]


def decode_ids(ids, gpt):
    stop = next((i for i,t in enumerate(ids) if t in (50256,50257)), len(ids))
    valid = ids[:stop]
    if any(t < 0 or t >= len(gpt) for t in valid):
        return "[INVALID TEXT TOKEN]", "invalid_token"
    return gpt.decode(valid, skip_special_tokens=True).strip(), (
        "eos" if stop < len(ids) and ids[stop] == 50256 else "pad" if stop < len(ids) else "canvas_limit")


@torch.inference_mode()
def infer_ncm(vae, net, prefix, gpt):
    with autocast():
        codes = net.generate(None, prefix, temperature=1., top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
        logits = vae.decoder(final_latent(vae, codes))
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite NCM output logits")
    raw = logits.argmax(-1).tolist()
    return [{"text":decode_ids(ids,gpt)[0],"termination":decode_ids(ids,gpt)[1],
        "native_output_ids":ids,"generator_steps":len(vae.vq.levels)} for ids in raw]


def storage_bytes(*modules):
    storages = {}
    for module in modules:
        for x in list(module.parameters()) + list(module.buffers()):
            s=x.untyped_storage()
            storages[(x.device.type,x.device.index,s.data_ptr())]=s.nbytes()
    return sum(storages.values())


def freeze(root, stage, config, inputs=()):
    if root.exists():
        raise ValueError("fresh append-only evidence root required")
    root.mkdir(parents=True)
    paths=[Path(__file__),Path("experiments/structure_net/run_ncm_llama.py"),PLAN,
        Path("experiments/structure_net/llama_summary_throughput.py"),
        Path("experiments/structure_net/llama_tensor_runtime.py"),
        Path("experiments/structure_net/llama_noise_tensor_continuation.py"),
        Path("experiments/structure_net/llama_neighbourhood_tensorization.py"),
        Path("experiments/structure_net/prepare_llama_cotenancy.py"),VENDOR/"LICENSE"]
    paths += sorted(VENDOR.rglob("*.py"))
    copies=root/"producer_source";copies.mkdir()
    for i,p in enumerate(paths):
        shutil.copyfile(p,copies/f"{i:03d}_{p.name}")
    p={"hypothesis_id":HYPOTHESIS_ID,"schema":SCHEMA_VERSION,"stage":stage,
        "config":config,"source_hashes":{str(p):sha256(p) for p in paths},
        "input_hashes":{str(p):sha256(p) for p in inputs},"command":sys.argv,
        "created_utc":datetime.now(timezone.utc).isoformat(),"client_date":"2026-10-21",
        "torch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "physical_gpu":0 if torch.cuda.is_available() else None,"slots":1}
    p["fingerprint"]=digest(p);write_json(root/"producer.json",p)
    return p


def unchanged(producer):
    for p,h in {**producer["source_hashes"],**producer["input_hashes"]}.items():
        if sha256(p) != h:
            raise RuntimeError(f"producing source/input changed: {p}")


def seal(root, producer, result):
    unchanged(producer)
    result.update(producer_fingerprint=producer["fingerprint"])
    if torch.cuda.is_available():
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(root/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("outputs","trace")},allow_nan=False),flush=True)


def training_examples(fixtures, labels):
    all_examples={e["id"]:e for e in fixtures}
    if len(all_examples)!=len(fixtures) or len({r["id"] for r in labels})!=len(labels) or {r["id"] for r in labels}!=set(all_examples):
        raise ValueError("complete unique source-label pairing required")
    if any(r["source_sha256"]!=all_examples[r["id"]]["source_sha256"] for r in labels):
        raise ValueError("teacher label source identity differs")
    ex={e["id"]:e for e in fixtures if e["split"]=="train"}
    kept=[r for r in labels if r["id"] in ex and r["usable_label"]]
    if len(kept) < .95 * len(ex):
        raise ValueError("training_label_capacity_or_control_failed")
    return [(ex[r["id"]],r) for r in kept]
