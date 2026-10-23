"""Use fresh destinations with the unchanged generated-history NCM mechanism."""
import argparse
from pathlib import Path
import sys

from experiments.structure_net import ncm_reviewed_draft_data as data
from experiments.structure_net import ncm_generated_history_study as study
from experiments.structure_net import ncm_generated_history_eval as evaluation
from experiments.structure_net import ncm_generated_history_workflow as word_workflow
from experiments.neural_architecture_lab import finalize_ncm_generated_history_word as development
from experiments.structure_net import ncm_sealed_final_quality as final
from experiments.structure_net import ncm_sealed_final_workflow as final_workflow
from experiments.neural_architecture_lab import finalize_ncm_sealed_final_quality as final_audit

PLAN = data.PLAN
ROOT = data.base.ROOT
WORD_WORKFLOW = ROOT / "20261023_reviewed_draft_history_workflow_v1"
FINAL_WORKFLOW = ROOT / "20261023_reviewed_draft_final_workflow_v1"
MODULE = "experiments.structure_net.ncm_reviewed_draft_recovery"
_configured = False


def command(scope, stage=None):
    result = [sys.executable, "-u", "-m", MODULE, scope]
    return result + ([stage] if stage is not None else [])


def translate_final_command(args):
    scopes = {"experiments.neural_architecture_lab.finalize_ncm_generated_history_word": "development",
        "experiments.structure_net.ncm_sealed_final_quality": "final",
        "experiments.neural_architecture_lab.finalize_ncm_sealed_final_quality": "final-audit"}
    if len(args) != 5 or args[1:3] != ["-u", "-m"] or args[3] not in scopes:
        raise ValueError("unexpected conditional final child command")
    return command(scopes[args[3]], args[4])


def configure():
    global _configured
    if _configured:
        return
    study.HYPOTHESIS = "ncm-reviewed-draft-history-word-v1"
    study.PLAN = PLAN
    study.expanded.DEST = data.DEST
    for attribute, suffix in (("CACHE_PILOT", "cache_pilot"), ("CACHE", "cache"),
            ("PILOT", "train_pilot"), ("TRAIN", "train")):
        setattr(study, attribute, ROOT / f"20261023_reviewed_draft_history_{suffix}_v1")
    for attribute, suffix in (("PILOT", "inference_pilot"), ("OUTPUTS", "outputs"),
            ("PACKETS", "packets"), ("GRADE", "grade")):
        setattr(evaluation, attribute, ROOT / f"20261023_reviewed_draft_history_{suffix}_v1")
    word_workflow.DEST = WORD_WORKFLOW
    development.DEST = ROOT / "20261023_reviewed_draft_history_results_v1"
    development.REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-reviewed-draft-history-word.md")
    development.PLAN = PLAN
    final.HYPOTHESIS = "ncm-reviewed-draft-history-final-quality-v1"
    final.PLAN = PLAN
    for attribute, suffix in (("DATA", "data"), ("FEATURE_PILOT", "feature_pilot"),
            ("FEATURES", "features"), ("PILOT", "inference_pilot"),
            ("OUTPUTS", "outputs"), ("PACKETS", "packets"), ("GRADE", "grade")):
        setattr(final, attribute, ROOT / f"20261023_reviewed_draft_final_{suffix}_v1")
    final_workflow.DEST = FINAL_WORKFLOW
    final_audit.DEST = ROOT / "20261023_reviewed_draft_final_results_v1"
    final_audit.WORKFLOW = FINAL_WORKFLOW
    final_audit.REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-reviewed-draft-final-quality.md")
    original_study_producing = study.producing
    original_final_base = final._base_producing

    def word_producing(root, stage, inputs, config):
        producer = original_study_producing(root, stage, list(inputs)+[Path(data.__file__), Path(__file__)],
            {**config, "recovery_label_source": "first_draft_first_critique", "semantic_rules_unchanged": True})
        return data.capture(root, producer)

    def final_base_producing(root, stage, inputs, config):
        producer = original_final_base(root, stage, list(inputs)+[Path(data.__file__), Path(__file__), PLAN],
            {**config, "recovery_label_source": "first_draft_first_critique", "semantic_rules_unchanged": True})
        return data.capture(root, producer)

    study.producing = word_producing
    final._base_producing = final_base_producing
    original_run_stage = final_workflow.run_stage

    def run_final_stage(name, args, destination, journal, producer, **kwargs):
        return original_run_stage(name, translate_final_command(args), destination, journal, producer, **kwargs)

    final_workflow.run_stage = run_final_stage
    _configured = True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("scope", choices=["data", "study", "eval", "development", "final", "final-audit", "word-workflow", "final-workflow"])
    parser.add_argument("stage", nargs="?")
    args = parser.parse_args()
    configure()
    if args.scope in ("word-workflow", "final-workflow"):
        if args.stage is not None:
            parser.error("workflow takes no stage")
        if args.scope == "word-workflow":
            from experiments.structure_net import ncm_reviewed_draft_workflow
            ncm_reviewed_draft_workflow.main()
        else:
            final_workflow.main()
        return
    if args.scope == "eval":
        evaluation.configure()
    choices = {
        "data": {"pilot": lambda: data.encode(True), "prepare": lambda: data.encode(False)},
        "study": {"cache-pilot": lambda: study.cache(True), "cache": lambda: study.cache(False),
            "pilot": lambda: study.train(True), "train": lambda: study.train(False)},
        "eval": {"pilot": lambda: evaluation.generate(True), "generate": lambda: evaluation.generate(False),
            "packets": lambda: evaluation.reference.prepare_packets(), "grade": lambda: evaluation.reference.grade()},
        "development": {"finalize": development.finalize, "register": development.register},
        "final": {"prepare": final.prepare, "feature-pilot": lambda: final.features(True),
            "features": lambda: final.features(False), "pilot": lambda: final.generate(True),
            "generate": lambda: final.generate(False), "packets": final.packets, "grade": final.grade},
        "final-audit": {"finalize": final_audit.finalize, "register": final_audit.register}}
    if args.stage not in choices[args.scope]:
        parser.error("invalid stage for selected scope")
    choices[args.scope][args.stage]()


if __name__ == "__main__":
    # The word controller imports this adapter too. Configure one canonical
    # module instance so a -m launch cannot wrap producers or dispatch twice.
    from experiments.structure_net import ncm_reviewed_draft_recovery as canonical
    canonical.main()
