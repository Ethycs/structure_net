"""Fixed author word-selection control on the failed, unchanged source-only model."""
import argparse
import ast
import hashlib
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_grounded_development as engine
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = engine.native, engine.q, engine.ROOT
HYPOTHESIS = "ncm-lm1b-native-word-readout-source-only-DEV-v1"
MODULE = "experiments.structure_net.ncm_lm1b_native_word_readout"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_native_word_readout"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_native_word_readout.py")
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-word-readout-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-native-word-readout.md")
TEST = Path("tests/structure_net/test_ncm_lm1b_native_word_readout.py")
WORKFLOW = ROOT / "20261024_lm1b_native_word_readout_workflow_v1"
PILOT = ROOT / "20261024_lm1b_native_word_readout_inference_pilot_v1"
OUTPUTS = ROOT / "20261024_lm1b_native_word_readout_outputs_v1"
PACKETS = ROOT / "20261024_lm1b_native_word_readout_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_native_word_readout_judge_pilot_v1"
GRADE = ROOT / "20261024_lm1b_native_word_readout_grade_v1"
RESULTS = ROOT / "20261024_lm1b_native_word_readout_results_v1"
BASE_OUTPUTS, BASE_RESULTS, BASE_GRADE, BASE_HYPOTHESIS = engine.OUTPUTS, engine.RESULTS, engine.GRADE, engine.HYPOTHESIS
BASE_CAMPAIGN = "0e6e370631899c746e4fcf44c96120047981e75898e9d441bf79a4162a2c9032"
OLD_CACHE = BASE_GRADE / "judgment_cache.json"
CACHE_SHA = "65d6a4fae40ba141e14f2010b6d7247ef4047f07b94fb4d4b42b2af0805d4771"
AUTHOR_SCRIPT = native.vendor.VENDOR / "scripts/eval_shard.py"
AUTHOR_CONFIG = native.RELEASE / "files/lm1b/small/config.yaml"
AUTHOR_GENERATOR = native.RELEASE / "files/lm1b/small/generator.pt"
POLICY = {"frequency_penalty": 11.5, "frequency_deficit": 2., "adjacency_penalty": [10., 4., .5]}
_producing, _load_selected, _predict, _generate = engine.producing, engine.load_selected, engine.predict, engine.generate
_BASE_ROWS, _ACTIVE_ROOT = None, None


def require_baseline():
    campaign = q.read(BASE_RESULTS / "campaign_results.json")
    receipt = q.read(BASE_RESULTS / (BASE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(BASE_RESULTS / "campaign_results.json") != BASE_CAMPAIGN
            or receipt["record"]["campaign_sha256"] != BASE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != BASE_CAMPAIGN
            or campaign["classification"] != "full128_source_only_DEV_quality_failed"
            or campaign["canonical_first_judgments"] != 2600 or file_hash(OLD_CACHE) != CACHE_SHA):
        raise ValueError("registered failed fixed4096 baseline and latest2600 firsts required")


def base_cache():
    require_baseline()
    cache = q.read(OLD_CACHE)
    if len(cache) != 2600:
        raise ValueError("all2600 first decisions required")
    engine.semantic.audit_cache(cache)
    return cache


def native_selector(allowance):
    """Compile only the unchanged author's pure word-selector function, not its CLI."""
    tree = ast.parse(AUTHOR_SCRIPT.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pick_tokens")
    namespace = {"torch": torch, "freq": POLICY["frequency_penalty"], "mu": POLICY["frequency_deficit"],
        "ADJ": POLICY["adjacency_penalty"], "ALLOW": allowance}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(AUTHOR_SCRIPT), "exec"), namespace)
    return namespace["pick_tokens"]


def allowance():
    if file_hash(AUTHOR_GENERATOR) != native.FILES["lm1b/small/generator.pt"]:
        raise ValueError("pinned native allowance checkpoint changed")
    saved = torch.load(AUTHOR_GENERATOR, map_location="cpu", weights_only=True, mmap=True)
    value = saved["inference"]["token_unigram"].double()
    if value.ndim != 1 or len(value) < 30528 or not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError("complete finite nonnegative native token allowance required")
    return value


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ReadoutProxy:
    def __init__(self, model, select):
        self.model, self.select = model, select

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.no_grad()
    def generate(self, features, mask):
        original = self.model.generate(features, mask)
        logits = self.model.word_logits(features, mask, original["codes"])
        if not torch.equal(logits.argmax(-1), original["raw_word_ids"]):
            raise ValueError("fresh word-only pass differs from original plain argmax")
        self.last_plain_words = original["raw_word_ids"].detach().cpu()
        self.last_codes = original["codes"].detach().cpu()
        self.last_logits = logits.detach().cpu().contiguous()
        return {"codes": original["codes"], "raw_word_ids": self.select(logits)}


def configure():
    engine.configure()
    for name in ("HYPOTHESIS", "MODULE", "AUDITOR", "AUDITOR_FILE", "PLAN", "REPORT", "TEST",
            "WORKFLOW", "PILOT", "OUTPUTS", "PACKETS", "JUDGE_PILOT", "GRADE", "RESULTS", "OLD_CACHE"):
        setattr(engine, name, globals()[name])
    engine.CACHE_ARCHIVE, engine.CACHE_HYPOTHESIS = BASE_RESULTS, BASE_HYPOTHESIS
    engine.CACHE_CAMPAIGN, engine.CACHE_SHA = BASE_CAMPAIGN, CACHE_SHA
    engine.producing, engine.load_selected, engine.predict, engine.base_cache = producing, load_selected, predict, base_cache


def producing(root, stage, inputs=(), config=None):
    require_baseline()
    files = [Path(__file__), PLAN, TEST, AUDITOR_FILE, AUTHOR_SCRIPT, AUTHOR_CONFIG]
    extra = [AUTHOR_GENERATOR, *files, *[BASE_OUTPUTS / n for n in ("outputs.json", "result.json", "worker_terminal_receipt.json")]]
    p = _producing(root, stage, [*inputs, *extra], config)
    p["config"].update(schema="nal.ncm-lm1b-native-word-readout.v1", output_policy=POLICY,
        word_selection_precision="FP64 exactly as author function", neural_precision="FP32",
        canonical_prior_firsts=2600, model_training_executed=False,
        baseline_concepts_and_plain_words_must_match_exactly=True, all_word_logits_saved_for_CPU_replay=True,
        concept_sampling="unchanged deterministic source-only greedy; not full author unconditional policy",
        renderer="unchanged native predicted-leading-CLS renderer", baseline_campaign_sha256=BASE_CAMPAIGN)
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"native_readout_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def load_selected():
    model, state = _load_selected()
    return ReadoutProxy(model, native_selector(allowance().cuda())), state


def predict(model, row, manifest, tokenizer):
    value = _predict(model, row, manifest, tokenizer)
    old = _BASE_ROWS[row["id"]]
    if model.last_plain_words.tolist()[0] != old["raw_word_ids"] or model.last_codes.tolist()[0] != old["codes"]:
        raise ValueError("candidate changed source-generated concepts or unpenalized word logits")
    path = _ACTIVE_ROOT / "word_logits" / (row["id"] + ".pt")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if not torch.equal(torch.load(path, map_location="cpu", weights_only=True), model.last_logits):
            raise ValueError("retained pilot word logits changed")
    else:
        torch.save(model.last_logits, path)
    return {**value, "word_logits_tensor_sha256": tensor_hash(model.last_logits),
        "baseline_plain_word_ids_sha256": q.digest(old["raw_word_ids"]), "baseline_concept_codes_unchanged": True}


def generate(pilot=False):
    global _BASE_ROWS, _ACTIVE_ROOT
    configure()
    require_baseline()
    native.vendor.terminal(BASE_OUTPUTS)
    _BASE_ROWS = {r["id"]: r for r in q.read(BASE_OUTPUTS / "outputs.json")}
    _ACTIVE_ROOT = PILOT if pilot else OUTPUTS
    _generate(pilot)
    values = q.read(_ACTIVE_ROOT / "outputs.json")
    manifest = {v["id"]: {"path": str(_ACTIVE_ROOT / "word_logits" / (v["id"] + ".pt")),
        "sha256": file_hash(_ACTIVE_ROOT / "word_logits" / (v["id"] + ".pt")),
        "tensor_sha256": v["word_logits_tensor_sha256"]} for v in values}
    write_json(_ACTIVE_ROOT / "word_logits_manifest.json", manifest)
    result = q.read(_ACTIVE_ROOT / "result.json")
    if result["model_sha256"] != q.read(BASE_OUTPUTS / "result.json")["model_sha256"]:
        raise ValueError("exact complete neural model changed across word policies")
    result.update(word_readout_policy=POLICY, word_logits_manifest_sha256=file_hash(_ACTIVE_ROOT / "word_logits_manifest.json"))
    write_json(_ACTIVE_ROOT / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("inference-pilot", "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    {"inference-pilot": lambda: generate(True), "generate": generate, "packets": engine.packets,
        "judge-pilot": lambda: engine.grade(True), "grade": engine.grade, "workflow": engine.workflow}[args.stage]()
