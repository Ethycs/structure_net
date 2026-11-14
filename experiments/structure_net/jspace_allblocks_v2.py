"""Versioned rank-gate amendment with explicit immutable v1 artifact reuse.

Only relation fitting changes. All calibration, recovery, evaluation, and
reservation orchestration execute the frozen v1 functions in a scoped context.
Preparing a root never runs a model, selects new examples, or changes v1 files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from threading import RLock

from . import jspace_qwen_parallel as base
from . import jspace_qwen_parallel_evaluate as evaluator

ROOT = base.ROOT
SCHEMA = "jspace.qwen-parallel.v2"
MODULE = "experiments.structure_net.jspace_allblocks_v2"
DEFAULT_ROOT = ROOT / "data/experiments/jspace_qwen_parallel/20261114_v2"
PREVIOUS_ROOT = ROOT / "data/experiments/jspace_qwen_parallel/20261114_v1"
DIAGNOSTIC = ROOT / "data/experiments/jspace_rank_diagnostic/20261114_layer30_v2/result.json"
AMENDMENT = ROOT / "docs/07 - Status Reports/jspace/2026-11-14_jspace-allblocks-rank-gate-amendment.md"
RANK_GATE = "actual_sketch_svd_shape_fp32"
CONFIG = {**base.CONFIG, "rank_gate": RANK_GATE,
          "maximum_accuracy_drop": .05, "family_quality_is_gate": False}
REUSED_LAYERS = (31, 32, 34, 35)
_V1_CONFIG = dict(base.CONFIG)
_V1_SCHEMA = base.SCHEMA
_V1_SOURCES = base.sources
_V1_ACCEPTANCE = evaluator.acceptance_gates
_CONTEXT_LOCK = RLock()


def sources():
    """Explicit amendment closure; no additions to the frozen parallel globs."""
    extra = [Path(__file__), AMENDMENT,
        Path(__file__).with_name("jspace_allblocks_rankfit.py"),
        Path(__file__).with_name("jspace_rank_diagnostic.py"),
        ROOT / "tests/structure_net/test_jspace_allblocks_v2.py",
        ROOT / "tests/structure_net/test_jspace_allblocks_rankfit.py",
        ROOT / "tests/structure_net/test_jspace_rank_diagnostic.py"]
    for path in extra:
        if not path.is_file():
            raise FileNotFoundError("Amendment source is not ready: " + str(path))
    return sorted(set(_V1_SOURCES() + extra))


@contextmanager
def protocol_context():
    """Restore every temporary base binding even when a stage raises."""
    with _CONTEXT_LOCK:
        old = {name: getattr(base, name) for name in ("CONFIG", "SCHEMA", "MODULE", "sources")}
        old_acceptance = evaluator.acceptance_gates
        try:
            base.CONFIG, base.SCHEMA, base.MODULE, base.sources = CONFIG, SCHEMA, MODULE, sources
            evaluator.acceptance_gates = acceptance_gates
            yield
        finally:
            for name, value in old.items():
                setattr(base, name, value)
            evaluator.acceptance_gates = old_acceptance


def acceptance_gates(*args, **kwargs):
    """The authorized 5pp aggregate budget has no hidden per-family veto."""
    config = kwargs["config"]
    if config.get("family_quality_is_gate") is not False or config["maximum_accuracy_drop"] != .05:
        raise ValueError("Unexpected v2 aggregate quality policy")
    result = _V1_ACCEPTANCE(*args, **kwargs)
    quality = args[0] if args else kwargs["quality"]
    result["diagnostics"] = {
        "every_family_quality": result["gates"].pop("every_family_quality"),
        "original_one_percentage_point_quality": quality["overall"]["accuracy_drop"] <= .01 + 1e-12}
    result["thresholds"]["family_quality_is_gate"] = False
    result["primary_quality_scope"] = "aggregate_accuracy"
    result["passed"] = all(result["gates"].values())
    return result


class HashAudit:
    """Hash each immutable source path once during preparation, detect drift."""
    def __init__(self):
        self.seen = {}

    def check(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if path not in self.seen:
            self.seen[path] = (identity, base.sha(path))
        recorded_identity, actual = self.seen[path]
        if identity != recorded_identity or (expected is not None and actual != expected):
            raise ValueError("Immutable input changed: " + str(path))
        return actual


def _read(path):
    return json.loads(Path(path).read_text())


def _relative_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must remain within its campaign root")
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact symlinks cannot be imported")
    if not path.is_file():
        raise ValueError("Artifact is not a regular file")
    return path


def _verify_recorded_sources(root, plan, audit):
    for name, expected in plan["source_sha256"].items():
        path = Path(name)
        relative = path.relative_to(ROOT)
        audit.check(path, expected)
        audit.check(Path(root) / "producer" / relative, expected)


def _assert_no_training_or_evaluation(root):
    root = Path(root)
    forbidden = [root / "evaluation", root / "campaign_results.json"]
    forbidden += list(root.glob("groups/*/recovery*"))
    forbidden += list(root.glob("groups/*/recovered*.pt"))
    forbidden += list(root.glob("logs/train_*.log")) + list(root.glob("logs/evaluate_*.log"))
    if any(path.exists() for path in forbidden):
        raise ValueError("Reuse requires v1 with no training or held-out evaluation")
    for path in root.glob("groups/group_*"):
        if path.name != "group_00":
            raise ValueError("Only the first untrained group may be imported")
    events = root / "coordinator.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            if line and _read_event_is_training_or_evaluation(json.loads(line)):
                raise ValueError("v1 attempted training or held-out evaluation")


def _read_event_is_training_or_evaluation(event):
    return event.get("stage") in {"train", "evaluate"}


def audit_previous(root, audit):
    """Check recorded v1 closure directly, independent of today's glob set."""
    root = Path(root).resolve(strict=True)
    plan = _read(root / "plan.json")
    if (plan["schema"] != _V1_SCHEMA or plan["config"] != _V1_CONFIG
            or plan["systems_only"] or plan["fingerprint"] != base.digest(
                {key: value for key, value in plan.items() if key != "fingerprint"})):
        raise ValueError("Unexpected or changed v1 protocol")
    _assert_no_training_or_evaluation(root)
    _verify_recorded_sources(root, plan, audit)
    for name, expected in plan["input_sha256"].items():
        audit.check(name, expected)
    for name, expected in plan["model"]["files"].items():
        audit.check(Path(plan["model"]["path"]) / name, expected)
    tasks = _read(root / "tasks.json")
    if base.digest(tasks) != plan["task_digest"]:
        raise ValueError("v1 task identities changed")
    records = ["groups/group_00/capture/manifest.json", "groups/group_00/basis/manifest.json"]
    records += [f"groups/group_00/block_{layer:02d}/fit_result/manifest.json" for layer in REUSED_LAYERS]
    actual = set(str(path.relative_to(root)) for path in root.glob("groups/*/block_*/fit_result/manifest.json"))
    if actual != set(records[2:]):
        raise ValueError("Completed v1 relation fits differ from the declared reuse set")
    manifests, artifacts = {}, {}
    for name in records:
        path = _relative_file(root, name)
        manifest = _read(path)
        if manifest["status"] != "completed" or manifest["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Incomplete or mismatched v1 manifest: " + name)
        if not manifest.get("artifacts"):
            raise ValueError("Completed stages must list artifacts")
        for artifact, expected in manifest["artifacts"].items():
            if artifact in artifacts and artifacts[artifact] != expected:
                raise ValueError("Conflicting recorded artifact hashes")
            audit.check(_relative_file(root, artifact), expected)
            artifacts[artifact] = expected
        manifests[name] = {"value": manifest, "sha256": audit.check(path)}
    expected_capture = {f"groups/group_00/capture/{name}" for name in
                        ("metadata.json", "items.json", "accepted_teacher_logits.npy")}
    expected_basis = set()
    for layer in base.group_layers(0):
        expected_capture.update(f"groups/group_00/capture/block_{layer:02d}/{name}" for name in
            ("mlp_calibration.npy", "boundary.npy", "mlp_jacobian_bf16.npy", "statistics.pt", "mlp_development.npy"))
        expected_basis.update(f"groups/group_00/block_{layer:02d}/{name}" for name in
            ("compiled_dev.npy", "output_weight.pt", "fit.json", "interface.json", "fit.pt"))
        for filename in ("fit.json", "interface.json"):
            metadata = _read(root / f"groups/group_00/block_{layer:02d}" / filename)
            for source, expected in metadata["input_sha256"].items():
                audit.check(source, expected)
    if (set(manifests[records[0]]["value"]["artifacts"]) != expected_capture
            or set(manifests[records[1]]["value"]["artifacts"]) != expected_basis):
        raise ValueError("v1 capture or basis artifact inventory is incomplete")
    for layer, name in zip(REUSED_LAYERS, records[2:]):
        expected = {f"groups/group_00/block_{layer:02d}/{filename}" for filename in ("relation.pt", "relations.json")}
        if set(manifests[name]["value"]["artifacts"]) != expected:
            raise ValueError("v1 completed relation artifact inventory is incomplete")
        for source, value in _read(root / f"groups/group_00/block_{layer:02d}/relations.json")["input_sha256"].items():
            audit.check(source, value)
    # Keep the recorded failure as evidence; never import an incomplete fit.
    failure = root / "logs/relation_g0_l30.log"
    if "Insufficient independent task-Jacobian directions for the anchor bank" not in failure.read_text():
        raise ValueError("Expected rank-gate failure evidence is missing")
    return plan, tasks, manifests, artifacts, failure


def audit_diagnostic(path, previous_root, previous_plan, audit):
    path = Path(path).resolve(strict=True)
    root = path.parent
    plan, report = _read(root / "plan.json"), _read(path)
    if plan["fingerprint"] != base.digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Diagnostic plan changed")
    expected_interface = Path(previous_root) / "groups/group_00/block_30/interface.json"
    if (report.get("status") != "completed" or report["fingerprint"] != plan["fingerprint"]
            or plan["layer"] != 30 or report["layer"] != 30
            or Path(plan["interface"]).resolve() != expected_interface.resolve()
            or plan["teacher_fingerprint"] != base.digest(previous_plan["model"])):
        raise ValueError("Diagnostic is not the completed original layer30 spectrum")
    expected_settings = dict(svd_rank=64, oversample=16, power_iterations=1, chunk_rows=256,
                             anchors=CONFIG["anchors"], old_rows=8192, seed=CONFIG["seed"] + 30)
    if plan["settings"] != expected_settings or report["seed"] != expected_settings["seed"]:
        raise ValueError("Diagnostic must use the exact failed fit's sketch settings")
    _verify_recorded_sources(root, plan, audit)
    for name, expected in plan["input_sha256"].items():
        audit.check(name, expected)
    audit.check(root / "spectral_state.pt", report["spectral_state_sha256"])
    cutoffs = report["rank_cutoffs"]
    if (report["requested_anchors"] != CONFIG["anchors"]
            or cutoffs["frozen_original_shape"]["rank"] >= CONFIG["anchors"]
            or cutoffs["actual_svd_shape"]["rank"] < CONFIG["anchors"]):
        raise ValueError("Spectrum does not support the declared numerical-rank amendment")
    return {str(p): audit.check(p) for p in (root / "plan.json", path, root / "spectral_state.pt")}


def hardlink_reuse(previous_root, root, artifacts, manifests, fingerprint, previous_fingerprint):
    """Only artifact bytes are shared; wrapper manifests have new provenance."""
    previous_root, root = Path(previous_root), Path(root)
    for name in sorted(artifacts):
        source = _relative_file(previous_root, name)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        if not source.samefile(target):
            raise OSError("Artifact reuse did not create a hardlink")
    for name, record in manifests.items():
        old = record["value"]
        wrapper = {"schema": SCHEMA, "status": "completed", "fingerprint": fingerprint,
            "artifacts": old["artifacts"], "provenance_kind": "verified_immutable_reuse",
            "reused_from": {"root": str(previous_root), "source_fingerprint": previous_fingerprint,
                "manifest": name, "manifest_sha256": record["sha256"]},
            "original_stage_metadata": {k: v for k, v in old.items() if k != "artifacts"}}
        base.write(root / name, wrapper)


def prepare(root=DEFAULT_ROOT, *, previous_root=PREVIOUS_ROOT, diagnostic_path=DIAGNOSTIC):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires hidden CUDA")
    root, previous_root = Path(root).resolve(), Path(previous_root).resolve(strict=True)
    if root.exists():
        raise FileExistsError(root)
    if root.is_relative_to(previous_root) or previous_root.is_relative_to(root):
        raise ValueError("New and previous campaign roots must be separate")
    source_paths = sources()  # Amendment/helper/test must exist before any mutation.
    audit = HashAudit()
    old, tasks, manifests, artifacts, failure = audit_previous(previous_root, audit)
    diagnostic_hashes = audit_diagnostic(diagnostic_path, previous_root, old, audit)
    inputs = dict(old["input_sha256"])
    for path in (previous_root / "plan.json", previous_root / "tasks.json", failure, AMENDMENT):
        inputs[str(path)] = audit.check(path)
    inputs.update(diagnostic_hashes)
    inputs.update({str(previous_root / name): record["sha256"] for name, record in manifests.items()})
    plan = {**old, "schema": SCHEMA, "config": CONFIG,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path): audit.check(path) for path in source_paths},
        "input_sha256": inputs,
        "reuse": {"root": str(previous_root), "source_fingerprint": old["fingerprint"],
            "policy": "Hardlink verified immutable artifacts only; wrapper manifests preserve original production provenance",
            "capture_groups": [0], "basis_groups": [0], "completed_relation_layers": list(REUSED_LAYERS),
            "remaining_relation_layers_in_group0": [30, 33], "training_or_heldout_scoring_in_previous": False,
            "manifest_sha256": {name: record["sha256"] for name, record in manifests.items()},
            "artifact_count": len(artifacts), "task_file_sha256": audit.check(previous_root / "tasks.json"),
            "prior_failure": {"layer": 30, "path": str(failure), "sha256": inputs[str(failure)],
                "reason": "Insufficient independent task-Jacobian directions for the anchor bank"}},
        "amendment": {"path": str(AMENDMENT), "sha256": inputs[str(AMENDMENT)],
            "diagnostic_sha256": diagnostic_hashes, "rank_gate": RANK_GATE,
            "config_changes": {k: {"previous": old["config"].get(k), "current": value}
                for k, value in CONFIG.items() if old["config"].get(k) != value}}}
    plan.pop("fingerprint", None)
    plan["fingerprint"] = base.digest(plan)
    root.mkdir(parents=True)
    for path in source_paths:
        target = root / "producer" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        audit.check(target, plan["source_sha256"][str(path)])
    shutil.copy2(previous_root / "tasks.json", root / "tasks.json")
    audit.check(root / "tasks.json", plan["reuse"]["task_file_sha256"])
    base.write(root / "plan.json", plan)
    hardlink_reuse(previous_root, root, artifacts, manifests, plan["fingerprint"], old["fingerprint"])
    return {"status": "prepared", "fingerprint": plan["fingerprint"], "reused_artifact_count": len(artifacts),
        "reused_completed_relation_layers": list(REUSED_LAYERS), "no_model_or_evaluation_executed": True}


def relation(root, index, layer):
    from .jspace_allblocks_rankfit import fit_relation
    import torch
    plan, lease, guard = base.gpu_setup(root)
    if layer not in base.group_layers(index):
        raise ValueError("Layer outside group")
    group = base.group_path(root, index)
    base.checked(root, group / "basis")
    path = group / f"block_{layer:02d}"
    fit_relation(path, device="cuda:0", guard=guard, seed=CONFIG["seed"] + layer)
    artifacts = [path / name for name in ("relation.pt", "relations.json", "rank_diagnostic.json", "rank_spectrum.pt")]
    base.write(path / "fit_result/manifest.json", {"schema": SCHEMA, "status": "completed",
        "fingerprint": plan["fingerprint"], "rank_gate": RANK_GATE,
        "artifacts": {str(p.relative_to(Path(root))): base.sha(p) for p in artifacts},
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "lease_id": lease["id"],
        "gpu": torch.cuda.get_device_name(0)})
    return {"status": "completed", "layer": layer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "basis", "relation", "train", "evaluate", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--previous-root", type=Path, default=PREVIOUS_ROOT)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    with protocol_context():
        if args.stage == "prepare":
            result = prepare(args.root, previous_root=args.previous_root, diagnostic_path=args.diagnostic)
        elif args.stage == "relation":
            if args.layer is None:
                parser.error("relation requires --layer")
            result = relation(args.root, args.group, args.layer)
        elif args.stage == "evaluate":
            from .jspace_qwen_parallel_evaluate import evaluate
            result = evaluate(args.root)
        elif args.stage == "run":
            result = base.run(args.root)
        else:
            result = getattr(base, args.stage)(args.root, args.group)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
