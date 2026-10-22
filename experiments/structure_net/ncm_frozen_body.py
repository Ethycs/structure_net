"""Train an NCM output module while retaining a frozen Llama context body."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import types

import torch
from torch import nn

from experiments.structure_net.ncm_llama import (
    INITIAL, VENDOR, autocast, decode_ids, encode_codes, final_latent,
    generator_input, infer_ncm, label_ids, load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH
from experiments.structure_net.train_ncm_memory import rows, save, word_unigram_f1

HYPOTHESIS = "ncm-frozen-llama-body-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-frozen-body-plan.md")
DATA = Path("data/experiments/ncm_memory/20261022_data_v1")
PRIOR = Path("data/experiments/ncm_memory/20261022_training_v1")
CODEC, GENERATOR = PRIOR / "codec.pt", PRIOR / "generator.pt"
EXPECTED = {CODEC: "905e6563394f37f539ff55a0466620589d7f5feab9a0aa2c7eb2dbd9e3b05eb6",
            GENERATOR: "9a27a3e214915d30221640b0ff0c7fb6745a3240f1f5fbfe3b83b3015cc9de66"}


@dataclass(frozen=True)
class Config:
    train_cases: int = 2048
    test_cases: int = 64
    steps: int = 512
    batch: int = 8
    seed: int = 42
    sampler_seed: int = 2042
    prefix: int = 512
    parent_width: int = 3072
    head_width: int = 768
    lr: float = 1e-4
    warmup: int = 32


def prompt_ids(tokenizer, source):
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Summarize the passage in one coherent paragraph of about 32 words. "
         "Preserve the main point and key facts. Do not invent facts. Return only the summary."},
        {"role": "user", "content": source}], add_generation_prompt=True)


def select(records, tokenizer, split, count):
    eligible, excluded = [], 0
    for record in sorted(records, key=lambda r: text_hash("frozen-body-" + split + ":" + r["id"])):
        ids = prompt_ids(tokenizer, record["source"])
        if len(ids) <= Config().prefix:
            eligible.append({**record, "parent_ids": ids})
        else:
            excluded += 1
        if len(eligible) == count:
            break
    if len(eligible) != count:
        raise ValueError("not enough eligible nontruncated examples")
    return eligible, excluded


def parent_inputs(records, tokenizer, device):
    ids = torch.full((len(records), Config().prefix), tokenizer.eos_token_id,
                     dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, record in enumerate(records):
        tokens = record["parent_ids"]
        if not 0 < len(tokens) <= Config().prefix:
            raise ValueError("source truncation prohibited")
        ids[row, -len(tokens):] = torch.tensor(tokens, device=device)
        mask[row, -len(tokens):] = 1
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}


def module_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        # uint8 view supports BF16 without NumPy conversion of that dtype.
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def supplied_context(net, context):
    """Use the vendor's external-context path, without changing its parameters.

    The override is scoped to one call and restored even after an exception.
    Existing forward and generation both call _encode_context on that path.
    """
    previous_flag = net.use_ctx_encoder
    had_override = "_encode_context" in net.__dict__
    previous_method = net.__dict__.get("_encode_context")
    net.use_ctx_encoder = True
    net._encode_context = types.MethodType(lambda self, ids: context, net)
    try:
        yield
    finally:
        net.use_ctx_encoder = previous_flag
        if had_override:
            net._encode_context = previous_method
        else:
            del net._encode_context


def fake_prefix(pad_mask):
    return torch.where(pad_mask, 50257, 0).long()


def external_logits(vae, net, context, pad_mask, codes):
    with torch.no_grad():
        previous = vae.vq.idx_to_ncm_input(codes)
    query = net.build_l0_query_input(len(codes), previous.dtype)
    target = torch.cat((query, previous), 1)
    with supplied_context(net, context):
        return net(target, prefix_ids=fake_prefix(pad_mask), prefix_pad_mask=pad_mask)


@torch.no_grad()
def external_codes(net, context, pad_mask):
    with supplied_context(net, context):
        return net.generate(None, fake_prefix(pad_mask), temperature=1., top_k=1,
                            top_p=0., cfg_scale=0., use_kv_cache=False)


def bridge():
    return nn.Sequential(nn.LayerNorm(Config().parent_width),
                         nn.Linear(Config().parent_width, Config().head_width))


def verify(producer):
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("frozen source/input changed: " + path)


def freeze(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if args.output.exists():
        raise ValueError("fresh append-only output required")
    for p, expected in EXPECTED.items():
        if file_hash(p) != expected:
            raise ValueError("initial checkpoint changed")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name in ("native_train.jsonl", "native_test.jsonl"):
        if file_hash(DATA / name) != manifest["files"][name]["sha256"]:
            raise ValueError("data partition changed")
    args.output.mkdir(parents=True)
    source = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_llama.py"),
              Path("experiments/structure_net/ncm_memory_data.py"),
              Path("experiments/structure_net/train_ncm_memory.py"),
              Path("experiments/structure_net/ncm_vocabulary.py"),
              Path("experiments/structure_net/prepare_llama_cotenancy.py")]
    source += sorted(VENDOR.rglob("*.py"))
    copies = args.output / "producer_source"
    copies.mkdir()
    for i, p in enumerate(source):
        shutil.copyfile(p, copies / f"{i:03d}_{p.name}")
    inputs = [DATA / n for n in ("manifest.json", "native_train.jsonl", "native_test.jsonl")]
    inputs += [CODEC, GENERATOR, INITIAL / "config.yaml"]
    inputs += sorted(p for p in MODEL_PATH.iterdir() if p.is_file())
    record = {"hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-frozen-body.v1",
              "stage": args.command, "config": asdict(Config()),
              "source_hashes": {str(p): file_hash(p) for p in source},
              "input_hashes": {str(p): file_hash(p) for p in inputs},
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "client_date": "2026-10-22", "device": torch.cuda.get_device_name(),
              "physical_gpu": 0, "workers": 1, "torch": torch.__version__,
              "underpowered": True}
    record["fingerprint"] = text_hash(json.dumps(record, sort_keys=True))
    write_json(args.output / "producer.json", record)
    return record


def main(args):
    cfg = Config()
    producer = freeze(args)
    if args.command == "train":
        pilot = json.loads((args.pilot / "result.json").read_text())
        pp = json.loads((args.pilot / "producer.json").read_text())
        if pilot["status"] != "passed" or any(pp[k] != producer[k]
                for k in ("source_hashes", "input_hashes", "config")):
            raise ValueError("successful matching CUDA lifecycle pilot required")
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    gpt, tokenizer = tokenizers()
    gpt = CachedVocabulary(gpt)
    training, excluded_train = select(rows(DATA / "native_train.jsonl"), tokenizer, "train",
                                      16 if args.command == "pilot" else cfg.train_cases)
    testing, excluded_test = select(rows(DATA / "native_test.jsonl"), tokenizer, "test",
                                    2 if args.command == "pilot" else cfg.test_cases)
    if ({r["source_id"] for r in training} & {r["source_id"] for r in testing}
            or {r["source_sha256"] for r in training} & {r["source_sha256"] for r in testing}):
        raise ValueError("source train/test overlap")
    write_json(args.output / "selection.json", {"train": training, "test": testing,
               "excluded_parent_length_before_selection": {"train": excluded_train, "test": excluded_test}})
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    before_parent = module_hash(parent.model)
    extracted = {}
    for split, records in (("train", training), ("test", testing)):
        destination = args.output / (split + "_features")
        destination.mkdir()
        for start in range(0, len(records), 4):
            chunk = records[start:start+4]
            inputs = parent_inputs(chunk, tokenizer, "cuda")
            with torch.inference_mode():
                hidden = parent.model(**inputs, use_cache=False).last_hidden_state
            for record, value, mask in zip(chunk, hidden, inputs["attention_mask"]):
                # A normal detached CPU tensor can be consumed by a trainable adapter.
                save(destination / (record["id"] + ".pt"),
                     {"hidden": value.detach().cpu().clone(), "pad_mask": mask.eq(0).cpu().clone()})
            if start % 128 == 0:
                print(json.dumps({"stage": "frozen_features", "split": split,
                                  "completed": start+len(chunk), "total": len(records)}), flush=True)
        extracted[split] = len(records)
    parent_outputs = []
    if args.command == "train":
        for i, record in enumerate(testing):
            inputs = parent_inputs([record], tokenizer, "cuda")
            with torch.inference_mode():
                output = parent.generate(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], max_new_tokens=96, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            ids = output[0, cfg.prefix:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            parent_outputs.append({"id": record["id"], "source_id": record["source_id"],
                "source": record["source"], "target": record["target"], "text": text,
                "native_output_ids": ids})
            if (i+1) % 8 == 0:
                print(json.dumps({"stage": "full_parent_baseline", "completed": i+1,
                                  "total": len(testing)}), flush=True)
        write_json(args.output / "test_parent.json", {"outputs": parent_outputs})
    if module_hash(parent.model) != before_parent or any(p.grad is not None for p in parent.parameters()):
        raise RuntimeError("frozen parent body changed")
    del parent, hidden, value, inputs
    gc.collect()
    torch.cuda.empty_cache()
    feature_seconds = time.perf_counter()-started
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR, adaptation=True)
    frozen_codec = module_hash(vae)
    packed = {}
    for split, records in (("train", training), ("test", testing)):
        targets = torch.tensor([label_ids(r["target"], gpt) for r in records])
        codes = []
        for start in range(0, len(records), 64):
            codes.append(encode_codes(vae, targets[start:start+64].cuda()).cpu())
        prefixes = []
        for record in records:
            ids = gpt.encode(record["source"], add_special_tokens=False)
            if len(ids) > cfg.prefix:
                raise ValueError("native source truncation prohibited")
            prefixes.append(ids + [50257]*(cfg.prefix-len(ids)))
        packed[split] = {"targets": targets, "codes": torch.cat(codes),
                         "prefixes": torch.tensor(prefixes)}
        save(args.output / (split + "_codes.pt"), packed[split])
    net.eval()
    p = packed["train"]["prefixes"][:1].cuda()
    c = packed["train"]["codes"][:1].cuda()
    with torch.no_grad():
        context = net.transformer.wte(vae.encoder.wte(p))
        native = net(generator_input(vae, net, p, c), prefix_pad_mask=p.eq(50257))
        external = external_logits(vae, net, context, p.eq(50257), c)
        if not torch.allclose(native, external, atol=1e-4, rtol=1e-5):
            raise RuntimeError("external-context teacher-forced parity failed")
        native_codes = net.generate(None, p, top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
        supplied_codes = external_codes(net, context, p.eq(50257))
        if not torch.equal(native_codes, supplied_codes):
            raise RuntimeError("external-context greedy code parity failed")
    traces, arm_results = {}, {}
    for arm in ("hybrid", "native"):
        torch.manual_seed(cfg.seed)
        net.load_state_dict(torch.load(GENERATOR, map_location="cpu", weights_only=True)["model"])
        adapter = bridge().cuda() if arm == "hybrid" else None
        # Adapter initialization must not shift the shared dropout RNG stream.
        torch.manual_seed(cfg.seed + 10)
        parameters = list(net.parameters()) + ([] if adapter is None else list(adapter.parameters()))
        opt = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=.01)
        sampler = torch.Generator().manual_seed(cfg.sampler_seed)
        trace = []
        steps = 2 if args.command == "pilot" else cfg.steps
        arm_start = time.perf_counter()
        max_adapter_gradient = 0.
        for step in range(1, steps+1):
            pick = torch.randint(len(training), (cfg.batch,), generator=sampler)
            if arm == "hybrid":
                features = [torch.load(args.output / "train_features" / (training[i]["id"] + ".pt"),
                                       weights_only=True) for i in pick.tolist()]
                hidden = torch.stack([f["hidden"] for f in features]).cuda()
                pad_mask = torch.stack([f["pad_mask"] for f in features]).cuda()
            net.train()
            opt.zero_grad(set_to_none=True)
            factor = step/cfg.warmup if step <= cfg.warmup else .1 + .45*(1+math.cos(
                math.pi*(step-cfg.warmup)/(cfg.steps-cfg.warmup)))
            for group in opt.param_groups:
                group["lr"] = cfg.lr*factor
            codes = packed["train"]["codes"][pick].cuda()
            with autocast():
                if arm == "hybrid":
                    logits = external_logits(vae, net, adapter(hidden.float()), pad_mask, codes)
                else:
                    prefix = packed["train"]["prefixes"][pick].cuda()
                    logits = net(generator_input(vae, net, prefix, codes), prefix_pad_mask=prefix.eq(50257))
                loss = torch.nn.functional.cross_entropy(logits.float().flatten(0,1), codes.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            if adapter is not None:
                gradient = sum(float(p.grad.float().square().sum()) for p in adapter.parameters()
                               if p.grad is not None) ** .5
                max_adapter_gradient = max(max_adapter_gradient, gradient)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            opt.step()
            trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                          "pick_sha256": text_hash(json.dumps(pick.tolist()))})
            if step % 32 == 0 or step == steps:
                print(json.dumps({"arm": arm, **trace[-1]}), flush=True)
                write_json(args.output / (arm+"_trace.json"), trace)
        if arm == "hybrid" and max_adapter_gradient <= 0:
            raise RuntimeError("no adapter gradient")
        path = args.output / (arm + ".pt")
        save(path, {"generator": net.state_dict(), "adapter": None if adapter is None else adapter.state_dict(),
                    "optimizer": opt.state_dict(), "step": steps, "sampler_state": sampler.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "producer_fingerprint": producer["fingerprint"]})
        net.eval()
        if adapter is not None:
            adapter.eval()
        outputs = []
        with torch.no_grad():
            for i, record in enumerate(testing):
                codes = packed["test"]["codes"][i:i+1].cuda()
                if arm == "hybrid":
                    feature = torch.load(args.output / "test_features" / (record["id"]+".pt"), weights_only=True)
                    mask = feature["pad_mask"][None].cuda()
                    with autocast():
                        context = adapter(feature["hidden"][None].cuda().float())
                        logits = external_logits(vae, net, context, mask, codes)
                        generated = external_codes(net, context, mask)
                        text_logits = vae.decoder(final_latent(vae, generated))
                    ids = text_logits.argmax(-1)[0].tolist()
                    text, termination = decode_ids(ids, gpt)
                else:
                    prefix = packed["test"]["prefixes"][i:i+1].cuda()
                    with autocast():
                        logits = net(generator_input(vae, net, prefix, codes), prefix_pad_mask=prefix.eq(50257))
                    output = infer_ncm(vae, net, prefix, gpt)[0]
                    text, ids, termination = output["text"], output["native_output_ids"], output["termination"]
                loss = float(torch.nn.functional.cross_entropy(logits.float().flatten(0,1), codes.flatten()))
                outputs.append({"id": record["id"], "source_id": record["source_id"], "source": record["source"],
                    "target": record["target"], "text": text, "termination": termination,
                    "native_output_ids": ids, "code_cross_entropy": loss,
                    "word_unigram_f1": word_unigram_f1(text, record["target"])})
        write_json(args.output / ("test_"+arm+".json"), {"outputs": outputs})
        # Exact save/reload check on the final test context, without any update.
        before = logits.clone()
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state["generator"], strict=True)
        if adapter is not None:
            adapter.load_state_dict(state["adapter"], strict=True)
        with torch.no_grad(), autocast():
            if adapter is not None:
                context = adapter(feature["hidden"][None].cuda().float())
                after = external_logits(vae, net, context, mask, codes)
            else:
                after = net(generator_input(vae, net, prefix, codes), prefix_pad_mask=prefix.eq(50257))
        if not torch.equal(before, after):
            raise RuntimeError("save/reload output changed")
        arm_results[arm] = {"updates": steps, "trainable_parameters": sum(p.numel() for p in parameters),
            "max_adapter_gradient": max_adapter_gradient, "checkpoint": str(path), "sha256": file_hash(path),
            "seconds": time.perf_counter()-arm_start, "reload_equal": True,
            "first_mean_loss": sum(t["loss"] for t in trace[:32])/min(32,len(trace)),
            "last_mean_loss": sum(t["loss"] for t in trace[-32:])/min(32,len(trace))}
        traces[arm] = trace
        del opt, state, parameters
        gc.collect()
        torch.cuda.empty_cache()
    if [x["pick_sha256"] for x in traces["hybrid"]] != [x["pick_sha256"] for x in traces["native"]]:
        raise RuntimeError("matched minibatches differed")
    if module_hash(vae) != frozen_codec or any(p.grad is not None for p in vae.parameters()):
        raise RuntimeError("frozen codec changed")
    verify(producer)
    result = {"status": "passed" if args.command == "pilot" else "trained_quality_pending",
        "hypothesis_id": HYPOTHESIS, "producer_fingerprint": producer["fingerprint"],
        "label": "systems_lifecycle_only_not_quality_evidence" if args.command == "pilot" else "exploratory",
        "train_cases": len(training), "test_cases": len(testing), "body_unchanged": True,
        "body_sha256": before_parent, "codec_unchanged": True, "codec_sha256": frozen_codec,
        "native_external_context_parity": True, "minibatches_matched": True, "arms": arm_results,
        "feature_and_parent_generation_seconds": feature_seconds, "elapsed_seconds": time.perf_counter()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "speed_tested": False,
        "task_accuracy": None, "underpowered": True}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "error.json", {"type": type(error).__name__, "error": str(error)[:400]})
        raise
