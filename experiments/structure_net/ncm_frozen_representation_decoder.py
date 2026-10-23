"""Fit only a new codec decoder against unchanged full-code representations."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_broad_learned_codec as core
from experiments.structure_net.ncm_sample_scaled_codec_ema import FixedVocabularySize
from experiments.structure_net.ncm_memory_data import file_hash, write_json

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-frozen-representation-decoder-plan.md")
PILOT = core.ROOT / "20261023_frozen_representation_decoder_pilot_v1"
TRAIN = core.ROOT / "20261023_frozen_representation_decoder_train_v1"
RECON = core.ROOT / "20261023_frozen_representation_decoder_reconstruction_v1"
PREVIOUS = core.ROOT / "20261023_sample_scaled_codec_ema_results_v1"
REFERENCE = core.ROOT / "20261023_sample_scaled_codec_ema_reconstruction_v1"
HYPOTHESIS = "ncm-frozen-representation-decoder-transfer-v1"
ORIGINAL_TRAIN = core.TRAIN
ORIGINAL_PILOT = core.PILOT
_load, _producing, _reconstruction = core.load_codec, core.producing, core.reconstruction


def freeze_representation(codec):
    codec.eval().requires_grad_(False)
    codec.decoder.requires_grad_(True)
    return codec


def representation_hash(codec):
    from experiments.structure_net.ncm_native_word_mechanism import tensor_digest
    values = {name: value for name, value in codec.state_dict().items() if not name.startswith("decoder.")}
    values["nonpersistent_fine_dropout_step"] = codec.vq._fd_step
    return tensor_digest(values)


def decoder_logits(codec, targets):
    if codec.training or codec.encoder.training or codec.vq.training:
        raise ValueError("representation must stay in evaluation mode")
    with torch.no_grad():
        codes = codec.encode(targets)
        latent = codec.vq.idx_to_z_q(codes)
    return codec.decoder(latent.detach())


def load_codec(state=None, device="cuda"):
    return freeze_representation(_load(state, device))


def reconstruction(codec, cases, tokenizer):
    return _reconstruction(codec, cases, FixedVocabularySize(tokenizer))


def producing(root, pilot, extra=()):
    controls = [PREVIOUS / "campaign_results.json",
        PREVIOUS / "ncm-sample-scaled-codec-ema-v1_registry_readback.json",
        ORIGINAL_TRAIN / "trace.jsonl", ORIGINAL_PILOT / "initial_probe.json",
        REFERENCE / "original_outputs.json", Path(__file__), PLAN,
        Path("experiments/structure_net/ncm_sample_scaled_codec_ema.py")]
    p = _producing(root, pilot, list(extra) + controls)
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(decoder_only=True, full_codes=True, representation_frozen=True,
        encoder_dropout=False, quantizer_EMA_updates=False, fine_level_dropout=False,
        VQ_loss=False, objective="non_PAD_word_CE", decoder_dropout=.1,
        runtime_overrides=["PLAN", "PILOT", "TRAIN", "RECON", "load_codec", "producing", "reconstruction"])
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("decoder_only_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = core.corpus.quality.run.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def configure():
    record = json.loads((PREVIOUS / "ncm-sample-scaled-codec-ema-v1_registry_readback.json").read_text())
    if (record["status"] != "registered_and_read_back"
            or record["record"]["campaign_sha256"] != file_hash(PREVIOUS / "campaign_results.json")
            or record["record"]["confirmation_status"] != "codec_reconstruction_fidelity_failed"):
        raise ValueError("preceding failed EMA control must be registered")
    core.PLAN, core.PILOT, core.TRAIN, core.RECON = PLAN, PILOT, TRAIN, RECON
    core.load_codec, core.producing, core.reconstruction = load_codec, producing, reconstruction


def train(pilot=False):
    core.require_gpu()
    core.terminal(core.corpus.DEST)
    if not pilot:
        core.terminal(PILOT)
        if json.loads((PILOT / "result.json").read_text())["status"] != "systems_decoder_only_training_reload_passed":
            raise ValueError("real decoder-only training/reload pilot required")
    root = PILOT if pilot else TRAIN
    extra = [] if pilot else [PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    p = producing(root, pilot, extra)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(core.corpus.GPT2, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer = FixedVocabularySize(tokenizer)
    cases = json.loads((core.corpus.DEST / "summary_diagnostic.json").read_text())
    probes = [cases[0], sorted(cases, key=lambda c: (-len(c["target_ids"]), c["id"]))[0]]
    windows = torch.load(core.corpus.DEST / "fit_windows.pt", map_location="cpu", weights_only=True)
    codec = load_codec()
    initial = core.module_hash(codec)
    frozen = representation_hash(codec)
    initial_probe = reconstruction(codec, probes, tokenizer)
    if initial_probe != json.loads((ORIGINAL_PILOT / "initial_probe.json").read_text()):
        raise ValueError("initial decoder-only inference differs from released reference")
    if not pilot and initial != json.loads((PILOT / "result.json").read_text())["initial_model_sha256"]:
        raise ValueError("full training must start from fresh released initialization")
    write_json(root / "initial_probe.json", initial_probe)
    parameters = [v for v in codec.parameters() if v.requires_grad]
    names = [k for k, v in codec.named_parameters() if v.requires_grad]
    if not names or any(not name.startswith("decoder.") for name in names):
        raise ValueError("only decoder parameters may enter AdamW")
    write_json(root / "trainable_parameters.json", {"names": names, "count": sum(v.numel() for v in parameters)})
    config = core.Config()
    optimizer = torch.optim.AdamW(parameters, lr=config.max_lr, betas=(.9,.95), weight_decay=.01)
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    updates, accumulation = (2,1) if pilot else (config.updates,config.accumulation)
    codec.decoder.train()
    torch.cuda.reset_peak_memory_stats()
    (root / "checkpoints").mkdir()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = core.rate(step + 1, config)
        indices, losses = [], []
        for _ in range(accumulation):
            chosen = (list(range(4)) if step == 0 else list(range(len(windows)-4, len(windows)))) if pilot else torch.randint(len(windows),(4,),generator=sampler).tolist()
            targets = windows[chosen].cuda()
            logits = decoder_logits(codec, targets)
            loss = F.cross_entropy(logits.float().flatten(0,1), targets.flatten(), ignore_index=50257)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite frozen-representation decoding objective")
            (loss / accumulation).backward()
            indices.append(chosen)
            losses.append(float(loss.detach()))
            del logits, targets, loss
        if (not any(v.grad is not None and torch.count_nonzero(v.grad).item() for v in parameters)
                or any(v.grad is not None for k, v in codec.named_parameters() if not k.startswith("decoder."))):
            raise ValueError("decoder/frozen gradient routing differs")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated() > 4_500_000_000:
            raise ValueError("decoder allocation exceeds pilot budget")
        row = {"step":step+1,"window_indices":indices,"CE":losses,"gradient_norm":float(norm),"lr":core.rate(step+1)}
        with (root / "trace.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if pilot or (step+1)%32 == 0:
            print(json.dumps({"step":step+1,"CE":sum(losses)/len(losses),"gradient_norm":float(norm)}), flush=True)
        if (step+1)%512 == 0:
            if representation_hash(codec) != frozen:
                raise ValueError("frozen representation changed during decoder training")
            if step+1 < updates:
                core.save(root / "checkpoints" / f"step_{step+1:06d}.pt", codec, optimizer, sampler, step+1, p)
    if representation_hash(codec) != frozen or core.module_hash(codec) == initial:
        raise ValueError("decoder must change while representation remains exact")
    core.save(root / "model.pt", codec, optimizer, sampler, updates, p)
    reloaded = core.reload_check(root / "model.pt", codec, optimizer, sampler, probes, tokenizer)
    if representation_hash(codec) != frozen:
        raise ValueError("frozen representation changed on reload")
    if not pilot:
        previous = [json.loads(s) for s in (ORIGINAL_TRAIN / "trace.jsonl").read_text().splitlines()]
        actual = [json.loads(s) for s in (root / "trace.jsonl").read_text().splitlines()]
        if len(previous) != 2048 or len(actual) != 2048 or any((a["window_indices"],a["lr"]) != (b["window_indices"],b["lr"]) for a,b in zip(previous,actual)):
            raise ValueError("sample lists and learning-rate schedule changed")
    core.verify(p)
    write_json(root / "reload_probe.json", reloaded)
    result = {"status":"systems_decoder_only_training_reload_passed" if pilot else "trained_pending_reconstruction_diagnostic",
        "updates":updates,"accumulation":accumulation,"initial_model_sha256":initial,
        "checkpoint_sha256":file_hash(root / "model.pt"),"reload_exact":True,
        "initial_inference_matches_original_exactly":True,"representation_sha256_unchanged":frozen,
        "only_decoder_trained":True,"trainable_parameters":sum(v.numel() for v in parameters),
        "all_sampled_windows_and_LRs_match_previous":not pilot,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"parent_loaded":False,
        "task_accuracy":None,"NCM_success_established":False,"final_test_opened":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def measure():
    from experiments.structure_net import ncm_broad_codec_reconstruction as measurement
    measurement.main()
    original = json.loads((RECON / "original_outputs.json").read_text())
    trained = json.loads((RECON / "trained_outputs.json").read_text())
    if original != json.loads((REFERENCE / "original_outputs.json").read_text()):
        raise ValueError("all252 released-reference outputs changed")
    if any(a["codes"] != b["codes"] for a,b in zip(original, trained)):
        raise ValueError("decoder training changed target codes")
    write_json(RECON / "representation_equivalence.json", {"all252_original_outputs_exact":True,
        "all252_trained_codes_identical_to_original":True})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pilot","train","measure"])
    args = parser.parse_args()
    configure()
    measure() if args.stage == "measure" else train(args.stage == "pilot")


if __name__ == "__main__":
    main()
