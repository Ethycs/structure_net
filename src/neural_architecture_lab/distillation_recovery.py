"""NAL lifecycle and evidence contract for response-complete distillation.

This is a worker adapter, not a scheduler. Reservation-backed launchers set
physical visibility before Python; NAL sees only logical ordinals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from .core import Experiment, ExperimentResult, ExperimentStatus
from .local_scheduler import ExperimentResultLedger, experiment_fingerprint

SCHEMA = "nal.distillation-recovery.v1"
STANDARD = "NAL-STD-DISTILLATION-v1"
SYSTEM_CHECKS = ("teacher_unchanged", "fixed_geometry_unchanged", "fresh_initializer",
                 "all_trainables_updated", "save_reload_exact", "full_vocab_path_parity")


def strict_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(strict_json(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(strict_json(value) + "\n")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class RecoveryArm:
    name: str
    stage: str = "teacher_forced"
    objective: str = "forward_kl"
    temperature: float = 1.
    ce_weight: float = 0.
    rollout_policy: str = "not_applicable"
    runtime_json: str = "{}"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("Safe unique arm name required")
        if self.stage not in ("teacher_forced", "student_rollout") or self.objective not in ("forward_kl", "generalized_jsd"):
            raise ValueError("Unknown stage/objective")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not math.isfinite(self.ce_weight) or self.ce_weight < 0:
            raise ValueError("Explicit finite temperature and CE coefficient required")
        allowed = ("not_applicable",) if self.stage == "teacher_forced" else ("fresh_on_policy", "offline_student", "stale_policy")
        if self.rollout_policy not in allowed:
            raise ValueError("Explicit matching rollout provenance policy required")
        runtime = json.loads(self.runtime_json, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(runtime, dict) or strict_json(runtime) != self.runtime_json:
            raise ValueError("Arm runtime_json must be a canonical strict JSON object containing the full adapter recipe")


@dataclass(frozen=True)
class QualityGate:
    name: str
    maximum_drop_pp: float
    minimum_student_rate: float = 0.
    task: str | None = None

    @property
    def key(self):
        return self.name if self.task is None else self.name + ":" + self.task

    def __post_init__(self):
        if (not self.name or not math.isfinite(self.maximum_drop_pp) or not 0 <= self.maximum_drop_pp <= 100
                or not math.isfinite(self.minimum_student_rate) or not 0 <= self.minimum_student_rate <= 1):
            raise ValueError("Quality margins use percentage points and rates use fractions")


@dataclass(frozen=True)
class RecoveryStudy:
    """Frozen recipe and gates; the supported Hoeffding bound is conservative.

    min_cases is an admission floor, not a power guarantee. Other valid paired
    interval designs need a separately versioned validator, not a silent swap.
    """
    study_id: str
    teacher_sha256: str
    initializer_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    data_sha256: str
    geometry_sha256: str
    arms: tuple[RecoveryArm, ...]
    seeds: tuple[int, ...]
    intended_tasks: tuple[str, ...]
    precision: str
    deployment_hardware: str
    max_input_tokens: int
    max_output_tokens: int
    max_context_tokens: int
    supervised_token_budget: int
    max_updates: int
    profile: str = "systems"
    gates: tuple[QualityGate, ...] = ()
    confirmation_min_cases: int = 1000
    confirmation_min_seeds: int = 5
    confirmation_alpha: float = .05
    paired_interval: str = "hoeffding_one_sided"
    confirmation_data_sha256: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.study_id):
            raise ValueError("Safe study ID required")
        for key in ("teacher", "initializer", "tokenizer", "template", "data", "geometry"):
            if not _sha(getattr(self, key + "_sha256")):
                raise ValueError("SHA256 identity required: " + key)
        if any(type(x) is not tuple for x in (self.arms, self.seeds, self.intended_tasks, self.gates)):
            raise ValueError("Frozen collections must be tuples")
        if not self.arms or len({x.name for x in self.arms}) != len(self.arms):
            raise ValueError("Distinct arms required")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(x) is not int or x < 0 for x in self.seeds):
            raise ValueError("Distinct nonnegative seeds required")
        if not self.intended_tasks or not self.precision or not self.deployment_hardware:
            raise ValueError("Tasks, precision and hardware must be declared")
        if any(type(x) is not int or x < 1 for x in (self.max_input_tokens, self.max_output_tokens,
                self.max_context_tokens, self.supervised_token_budget, self.max_updates,
                self.confirmation_min_cases, self.confirmation_min_seeds)):
            raise ValueError("Positive integer budgets required")
        if self.max_input_tokens + self.max_output_tokens > self.max_context_tokens:
            raise ValueError("Input plus output exceeds validated context")
        if self.profile not in ("systems", "exploratory", "confirmatory"):
            raise ValueError("Unknown evidence profile")
        if self.profile != "systems" and not {"task_success", "generation_success"} <= {g.name for g in self.gates if g.task is None}:
            raise ValueError("Task and generation gates are both required")
        if len({g.key for g in self.gates}) != len(self.gates) or not 0 < self.confirmation_alpha < 1:
            raise ValueError("Unique gates and finite confidence level required")
        if any(g.task is not None and g.task not in self.intended_tasks for g in self.gates):
            raise ValueError("Gate task must be declared")
        if self.profile != "systems" and any(a.runtime_json == "{}" for a in self.arms):
            raise ValueError("Scientific arms must freeze the full adapter/optimizer recipe in runtime_json")
        if self.paired_interval != "hoeffding_one_sided":
            raise ValueError("Unsupported preregistered paired interval")
        if self.profile == "confirmatory" and not _sha(self.confirmation_data_sha256):
            raise ValueError("Confirmatory TEST identity must be frozen in the study")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["arms"] = tuple(RecoveryArm(**x) for x in value["arms"])
        value["gates"] = tuple(QualityGate(**x) for x in value.get("gates", []))
        for key in ("seeds", "intended_tasks"):
            value[key] = tuple(value[key])
        return cls(**value)

    @property
    def fingerprint(self):
        return digest(asdict(self))


def make_experiments(study):
    return [Experiment(id=f"{study.study_id}.{arm.name}.seed{seed}", hypothesis_id=study.study_id,
        name=f"{arm.name}, seed {seed}", parameters={"schema": SCHEMA, "study": asdict(study),
        "study_fingerprint": study.fingerprint, "arm": arm.name}, seed=seed)
        for arm in study.arms for seed in study.seeds]


def paired_gates(study, quality):
    rows = quality["records"]
    if (not rows or len({r["example_id"] for r in rows}) != len(rows)
            or any(not r.get("group_id") or r.get("task") not in study.intended_tasks
                   or not isinstance(r.get("teacher_output"), str) or not isinstance(r.get("student_output"), str) for r in rows)):
        raise ValueError("Unique paired raw outputs with declared tasks required")
    if quality.get("split") not in ("development", "test") or not quality.get("judging_provenance"):
        raise ValueError("Explicit split and judging provenance required")
    # Differences lie in [-1,1]. Hoeffding supplies a conservative paired bound;
    # Bonferroni covers all prespecified gates, without a normal approximation.
    result = {}
    for gate in study.gates:
        subset = [r for r in rows if gate.task is None or r["task"] == gate.task]
        if not subset:
            raise ValueError("No paired evidence for declared gate: " + gate.key)
        radius = math.sqrt(2 * math.log(len(study.gates) / study.confirmation_alpha) / len(subset))
        independent_groups = len({r["group_id"] for r in subset}) == len(subset)
        pairs = [(r["teacher"][gate.name], r["student"][gate.name]) for r in subset]
        if any(type(x) is not bool for pair in pairs for x in pair):
            raise ValueError("Binary raw judgments required; aggregate flags are insufficient")
        native = sum(a for a, _ in pairs) / len(subset)
        student = sum(b for _, b in pairs) / len(subset)
        difference = student - native
        result[gate.key] = dict(count=len(subset), teacher_rate=native, student_rate=student,
            difference_pp=100*difference, lower_difference_pp=100*max(-1., difference-radius),
            wins=sum(not a and b for a, b in pairs), losses=sum(a and not b for a, b in pairs),
            point_pass=100*(sum(b for _, b in pairs)-sum(a for a, _ in pairs)) >= -gate.maximum_drop_pp*len(pairs)
                       and student >= gate.minimum_student_rate,
            independent_groups=independent_groups,
            interval_pass=independent_groups and difference-radius >= -gate.maximum_drop_pp/100 and student >= gate.minimum_student_rate)
    return result


def classify(study, evidence):
    strict_json(evidence)
    systems = evidence.get("systems", {})
    if any(systems.get(name) is not True for name in SYSTEM_CHECKS):
        raise ValueError("Incomplete or failed systems evidence")
    if study.profile == "systems":
        if evidence.get("quality") is not None:
            raise ValueError("A systems shakedown cannot carry model-quality evidence")
        return dict(classification="systems_passed", confirmed=False, gates=None)
    quality = evidence["quality"]
    gates = paired_gates(study, quality)
    passed = all(g["point_pass"] for g in gates.values())
    improved = any(type(r.get("initializer", {}).get("generation_success")) is bool
                   and r["student"]["generation_success"] > r["initializer"]["generation_success"] for r in quality["records"])
    if improved:
        if any(type(r.get("initializer", {}).get("generation_success")) is not bool for r in quality["records"]):
            raise ValueError("Complete paired initializer judgments required for an improvement claim")
        improved = sum(r["student"]["generation_success"] - r["initializer"]["generation_success"] for r in quality["records"]) > 0
    verdict = ("exploratory_improvement" if improved else "exploratory_quality_pass") if passed else "quality_gate_failed"
    confirmed = False
    if study.profile == "confirmatory" and quality["split"] == "test":
        selection = evidence.get("selection", {})
        # This receipt is an authenticated artifact; chronology and selected
        # checkpoint binding are also checked by the worker before admission.
        valid = (quality.get("previously_exposed") is False and len(quality["records"]) >= study.confirmation_min_cases
                 and len(study.seeds) >= study.confirmation_min_seeds
                 and selection.get("checkpoint_sha256") == evidence["checkpoint_sha256"]
                 and selection.get("study_fingerprint") == study.fingerprint
                 and selection.get("test_data_sha256") == quality.get("data_sha256")
                 and quality.get("data_sha256") == study.confirmation_data_sha256
                 and all(g["interval_pass"] for g in gates.values()))
        if valid:
            try:
                valid = datetime.fromisoformat(selection["frozen_at"]) < datetime.fromisoformat(quality["opened_at"])
            except (KeyError, TypeError, ValueError):
                valid = False
        if valid:
            verdict, confirmed = "confirmed_recovery", True
    # Deployment has a separate measured protocol and is deliberately not
    # inferred from training wall time or a trainer-supplied boolean here.
    return dict(classification=verdict, confirmed=confirmed, gates=gates)


def prepare(root, study, source_paths):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    sources = {str(Path(p).resolve()): sha(p) for p in source_paths}
    if not sources:
        raise ValueError("Producing source inventory required")
    plan = dict(schema=SCHEMA, standard=STANDARD, study=asdict(study), study_fingerprint=study.fingerprint,
                source_sha256=sources, status="prepared", confirmed=False)
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    (root / "producer").mkdir()
    for path, checksum in sources.items():
        target = root / "producer" / (checksum + "_" + Path(path).name)
        if not target.exists():
            shutil.copyfile(path, target)
    write(root / "plan.json", plan)
    return plan


def verify_plan(root, *, live=False):
    root = Path(root)
    plan = read(root / "plan.json")
    payload = dict(plan); fingerprint = payload.pop("fingerprint")
    study = RecoveryStudy.from_dict(plan["study"])
    if plan["schema"] != SCHEMA or digest(payload) != fingerprint or study.fingerprint != plan["study_fingerprint"]:
        raise ValueError("Study fingerprint mismatch")
    for path, checksum in plan["source_sha256"].items():
        if (live and sha(path) != checksum) or sha(root / "producer" / (checksum + "_" + Path(path).name)) != checksum:
            raise ValueError("Producing source/archive changed")
    return plan, study


def require_device(device_id, lease_checker=None):
    if device_id == -1:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU work requires CUDA hidden before Python")
        return lambda: None
    if type(device_id) is not int or device_id < 0 or not os.environ.get("IC_LEASE"):
        raise ValueError("Launch with inference-ctl lease run before Python")
    if lease_checker is None:
        from structure_net_resources.client import require_gpu_lease
        lease_checker = require_gpu_lease
    initial = lease_checker([device_id])
    def guard():
        current = lease_checker([device_id])
        if (current["id"], current["allocation"]) != (initial["id"], initial["allocation"]):
            raise RuntimeError("Reservation identity/allocation changed")
    guard()
    return guard


def run_directory(root, experiment):
    return Path(root) / "runs" / experiment.parameters["arm"] / f"seed_{experiment.seed}"


def checked(root, experiment):
    plan, _ = verify_plan(root)
    directory = run_directory(root, experiment)
    result = read(directory / "result.json")
    if (result.get("schema") != SCHEMA or result.get("status") != "completed"
            or result.get("scientific_fingerprint") != experiment_fingerprint(experiment)
            or result.get("producer_fingerprint") != plan["fingerprint"]
            or result.get("study_fingerprint") != plan["study_fingerprint"]):
        raise ValueError("Incomplete or mismatched replicate")
    evidence = read(directory / "evidence.json")
    required = set(evidence["required_artifacts"]) | {evidence["checkpoint"], "evidence.json", "progress.jsonl"}
    if set(result["artifacts"]) != required or result["artifacts"][evidence["checkpoint"]] != evidence["checkpoint_sha256"]:
        raise ValueError("Complete required artifact inventory/checkpoint digest required")
    for relative, checksum in result["artifacts"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != checksum:
            raise ValueError("Required artifact changed: " + relative)
    study = RecoveryStudy.from_dict(experiment.parameters["study"])
    if classify(study, evidence) != result["verdict"]:
        raise ValueError("Stored classification disagrees with raw evidence")
    ledger = ExperimentResultLedger(root).load(experiment)
    primary = result["verdict"]["gates"]["generation_success"]["student_rate"] if result["verdict"]["gates"] else 0.
    if (ledger is None or ledger.experiment_id != experiment.id or ledger.hypothesis_id != experiment.hypothesis_id
            or ledger.model_checkpoint != str(directory / evidence["checkpoint"]) or ledger.metrics != result["metrics"]
            or ledger.primary_metric != primary or ledger.model_parameters != evidence["model_parameters"]
            or ledger.model_architecture != evidence["model_architecture"]
            or ledger.training_time != evidence["progress"]["training_seconds"]):
        raise ValueError("Successful matching NAL ledger readback required")
    return result, ledger


class RecoveryWorker:
    """Adapt a study-specific producer to the existing NAL worker/ledger."""
    def __init__(self, root, adapter, *, lease_checker=None):
        self.root, self.adapter, self.lease_checker = Path(root).resolve(), adapter, lease_checker

    def __call__(self, experiment, device_id):
        plan, study = verify_plan(self.root)
        if experiment.parameters.get("study_fingerprint") != study.fingerprint or experiment_fingerprint(experiment) not in {
                experiment_fingerprint(e) for e in make_experiments(study)}:
            raise ValueError("Experiment does not belong to frozen study")
        directory = run_directory(self.root, experiment)
        if (directory / "result.json").exists():
            return checked(self.root, experiment)[1]
        if directory.exists():
            raise FileExistsError("Incomplete/failed run preserved; use an explicit new root")
        verify_plan(self.root, live=True)
        directory.mkdir(parents=True)
        started = time.monotonic()
        def progress(**event):
            event = dict(event, elapsed_seconds=time.monotonic()-started)
            with (directory / "progress.jsonl").open("a") as stream:
                stream.write(strict_json(event) + "\n")
            print(strict_json(event), flush=True)
        try:
            guard = require_device(device_id, self.lease_checker)
            progress(phase="initialization", optimizer_updates=0, supervised_tokens=0, logical_device=device_id)
            arm = next(a for a in study.arms if a.name == experiment.parameters["arm"])
            evidence = self.adapter(study=study, arm=arm, seed=experiment.seed, device_id=device_id,
                                    output_dir=directory, progress=progress, guard=guard)
            guard()
            strict_json(evidence)
            expected_identities = {key: getattr(study, field) for key, field in (
                ("teacher", "teacher_sha256"), ("student_initializer", "initializer_sha256"),
                ("tokenizer", "tokenizer_sha256"), ("chat_template", "template_sha256"), ("data", "data_sha256"))}
            if evidence.get("identities") != expected_identities:
                raise ValueError("Worker model/data/tokenizer identities differ from frozen study")
            counts = evidence["progress"]
            if (not 0 < counts["optimizer_updates"] <= study.max_updates
                    or not 0 < counts["supervised_tokens"] <= study.supervised_token_budget
                    or not counts["input_tokens"] >= counts["response_tokens"] >= counts["supervised_tokens"]
                    or any(type(counts[k]) is not int for k in ("optimizer_updates", "input_tokens", "response_tokens", "supervised_tokens"))):
                raise ValueError("Invalid real optimizer/token progress")
            if any(type(counts.get(k)) not in (float, int) or counts[k] < 0 for k in ("initialization_seconds", "training_seconds")):
                raise ValueError("Initialization and training latency must be reported separately")
            required = set(evidence["required_artifacts"]) | {evidence["checkpoint"]}
            if not required:
                raise ValueError("Real checkpoints and evidence artifacts required")
            artifacts = {}
            for relative in required:
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Artifact escapes replicate directory")
                artifacts[relative] = sha(path)
            if artifacts[evidence["checkpoint"]] != evidence["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint hash mismatch")
            if evidence.get("selection") is not None:
                selection_path = evidence["selection_artifact"]
                if selection_path not in required or read(directory / selection_path) != evidence["selection"]:
                    raise ValueError("Frozen selection receipt required")
            verdict = classify(study, evidence)
            write(directory / "evidence.json", evidence)
            artifacts["evidence.json"] = sha(directory / "evidence.json")
            metrics = {k: float(v) for k, v in counts.items() if type(v) in (int, float)}
            metrics["quality_gate_passed"] = float(verdict["gates"] is not None and all(g["point_pass"] for g in verdict["gates"].values()))
            primary = (verdict["gates"]["generation_success"]["student_rate"] if verdict["gates"] else 0.)
            result = ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
                metrics=metrics, primary_metric=primary, model_architecture=evidence["model_architecture"],
                model_parameters=evidence["model_parameters"], training_time=counts["training_seconds"],
                model_checkpoint=str(directory / evidence["checkpoint"]),
                observations=[str(directory / "evidence.json"), verdict["classification"],
                              "Systems measurements are not model quality; training time is not inference latency."])
            progress(phase="completed", optimizer_updates=counts["optimizer_updates"], supervised_tokens=counts["supervised_tokens"])
            artifacts["progress.jsonl"] = sha(directory / "progress.jsonl")
            write(directory / "result.json", dict(schema=SCHEMA, status="completed", study_fingerprint=study.fingerprint,
                scientific_fingerprint=experiment_fingerprint(experiment), producer_fingerprint=plan["fingerprint"],
                artifacts=artifacts, verdict=verdict, metrics=metrics))
            ExperimentResultLedger(self.root).save(experiment, result, attempts=1, device_id=device_id)
            return checked(self.root, experiment)[1]
        except Exception as error:
            failure = dict(schema=SCHEMA, status="failed", classification="execution_failure", confirmed=False,
                           scientific_fingerprint=experiment_fingerprint(experiment), error=repr(error))
            if not (directory / "failure.json").exists():
                write(directory / "failure.json", failure)
            return ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
                metrics={}, primary_metric=0., model_architecture=[], model_parameters=0,
                training_time=time.monotonic()-started, error=repr(error), status=ExperimentStatus.FAILED,
                observations=[str(directory / "failure.json")])


def aggregate(root):
    plan, study = verify_plan(root)
    rows = []
    for experiment in make_experiments(study):
        directory = run_directory(root, experiment)
        if (directory / "result.json").exists():
            result, _ = checked(root, experiment)
            rows.append(dict(experiment_id=experiment.id, arm=experiment.parameters["arm"], seed=experiment.seed,
                             status="completed", **result["verdict"], result_sha256=sha(directory / "result.json")))
        elif (directory / "failure.json").exists():
            failure = read(directory / "failure.json")
            if failure["scientific_fingerprint"] != experiment_fingerprint(experiment):
                raise ValueError("Failure fingerprint mismatch")
            rows.append(dict(experiment_id=experiment.id, status="failed", confirmed=False, error=failure["error"]))
        else:
            rows.append(dict(experiment_id=experiment.id, status="running" if directory.exists() else "prepared", confirmed=False))
    complete = all(row["status"] in ("completed", "failed") for row in rows)
    return dict(schema=SCHEMA, study_fingerprint=study.fingerprint, producer_fingerprint=plan["fingerprint"],
                status="completed" if complete else "partial", confirmed=bool(complete and all(r["confirmed"] for r in rows)),
                runs=rows, scope="Per-arm/seed evidence; no pooled rescue of a failed gate, no inference-speed claim")
