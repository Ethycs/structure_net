"""Fixed final task-domain decoder evaluation using unchanged semantic firsts."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_lm1b_domain_decoder as domain
from experiments.structure_net import ncm_lm1b_decoder_repair_eval as engine
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, framed, prior, q, reference, oracle, ROOT = (engine.native, engine.framed, engine.prior,
    engine.q, engine.reference, engine.oracle, engine.ROOT)
repair = domain.base
HYPOTHESIS = "ncm-lm1b-domain-decoder-heldout-semantics-v1"
MODULE = "experiments.structure_net.ncm_lm1b_domain_decoder_eval"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_domain_decoder"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-domain-decoder-evaluation-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-domain-decoder.md")
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_domain_decoder.py")
WORKFLOW = ROOT / "20261024_lm1b_domain_decoder_eval_workflow_v1"
PILOT = ROOT / "20261024_lm1b_domain_decoder_reconstruction_pilot_v1"
RECON = ROOT / "20261024_lm1b_domain_decoder_reconstruction_v1"
PACKETS = ROOT / "20261024_lm1b_domain_decoder_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_domain_decoder_judge_pilot_v1"
GRADE = ROOT / "20261024_lm1b_domain_decoder_grade_v1"
RESULTS = ROOT / "20261024_lm1b_domain_decoder_results_v1"
ARMS = engine.ARMS
BASE_CACHE, BASE_RESULTS, BASE_HYPOTHESIS = domain.FIRST_CACHE, domain.PREVIOUS, domain.PREVIOUS_HYPOTHESIS
BASE_PACKETS, BASE_CACHE_SHA = domain.SEMANTIC_PACKETS, domain.FIRST_CACHE_SHA
_configure, _producing = engine.configure, engine.producing


def configure():
    domain.configure()
    engine.repair = domain.base
    engine.repair.PARENT_SHA = domain.PREVIOUS_SHA
    for name in ("HYPOTHESIS", "MODULE", "AUDITOR", "PLAN", "REPORT", "AUDITOR_FILE", "WORKFLOW", "PILOT", "RECON",
            "PACKETS", "JUDGE_PILOT", "GRADE", "RESULTS", "ARMS", "BASE_CACHE", "BASE_RESULTS", "BASE_HYPOTHESIS",
            "BASE_PACKETS", "BASE_CACHE_SHA"):
        setattr(engine, name, globals()[name])
    engine.producing, engine.base_cache, engine.configure = producing, base_cache, configure
    _configure()


def base_cache():
    campaign = q.read(BASE_RESULTS / "campaign_results.json")
    receipt = q.read(BASE_RESULTS / (BASE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(BASE_CACHE) != BASE_CACHE_SHA or file_hash(BASE_RESULTS / "campaign_results.json") != domain.PREVIOUS_SHA
            or campaign["artifacts"].get(str(BASE_CACHE)) != BASE_CACHE_SHA or campaign["canonical_first_judgments"] != 2266
            or receipt["record"]["campaign_sha256"] != domain.PREVIOUS_SHA
            or receipt["exact"]["campaign_sha256"] != domain.PREVIOUS_SHA
            or receipt["status"] != "registered_and_read_back"):
        raise ValueError("latest2266 registered first decisions required")
    cache = q.read(BASE_CACHE)
    if len(cache) != 2266: raise ValueError("complete2266 canonical cache required")
    prior.audit_cache(cache)
    return cache


def producing(root, stage, inputs=(), config=None):
    sources = [Path(__file__), PLAN, AUDITOR_FILE, Path(domain.__file__), domain.PLAN,
        Path("tests/structure_net/test_ncm_lm1b_domain_decoder_eval.py")]
    extra = [domain.DATA / n for n in ("producer.json", "result.json", "split.json", "selection.json", "origins.json", "text_coverage.json")]
    p = _producing(root, stage, [*inputs, *extra, *sources], config)
    p["config"].update(hypothesis_id=HYPOTHESIS, schema="nal.ncm-lm1b-domain-decoder-heldout.v1",
        canonical_prior_firsts=2266, semantic_heldout_cases=64, decoder_fit_cases=626,
        application_fit_targets_used_for_training=True, semantic_heldout_targets_used_for_gradient=False,
        training_source="626 task TRAIN sources/targets plus author TRAIN replay",
        evaluation_role="previously observed, gradient-disjoint component development",
        independent_final_test=False, source_only_task_accuracy=None)
    p["hypothesis_id"] = HYPOTHESIS
    for i, source in enumerate(sources):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / f"domain_eval_{i}_{source.name}").write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "measure", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    {"pilot": lambda: native.measure(True), "measure": native.measure, "packets": engine.packets,
        "judge-pilot": lambda: prior.grade(True), "grade": prior.grade, "workflow": engine.workflow}[args.stage]()
