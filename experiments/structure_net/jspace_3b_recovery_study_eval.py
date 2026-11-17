"""Pinned paired CPU evaluation, separate from recovery gradients and generation.

prepare freezes decoding, graders, weights and power assumptions without opening
confirmation. select seals checkpoint selection before confirmation generation;
run --responses FILE grades paired rows; check authenticates a saved result.
Responses require example_id/group_id/task, checkpoint, native|student role,
response/output_ids/stop_reason, exact prompt input_ids, decoding, grader_identity.
Missing, unsupported and unavailable evidence never become incorrect answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

from . import jspace_3b_recovery_study_data as corpus
from . import jspace_3b_response_task_metrics as tasks
from . import jspace_3b_response_judge as semantic
from . import jspace_3b_response_eval as generation

data = corpus.data
SCHEMA = "jspace.recovery-reset.evaluation.v1"


def checkpoint_identity(value):
    identity = value.get("sha256") if isinstance(value, dict) else value
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", identity):
        raise ValueError("Pinned checkpoint hash/revision required")
    if isinstance(value, dict) and value.get("path") and corpus.sha(value["path"]) != identity:
        raise ValueError("Checkpoint payload hash mismatch")
    return identity


def authenticate(path, expected):
    if corpus.sha(path) != expected:
        raise ValueError("Artifact changed: " + str(path))


def load_plan(root):
    plan = json.loads((Path(root)/"plan.json").read_text())
    if data.digest({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("Evaluation plan fingerprint mismatch")
    authenticate(plan["spec_path"], plan["spec_sha256"])
    for path, value in plan["implementation_sha256"].items():
        authenticate(path, value)
    authenticate(plan["corpus_manifest"], plan["corpus_manifest_sha256"])
    return plan


def prepare(root, spec_path, *, max_new_tokens=512, judge_repetitions=2, judge_config=None):
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    manifest_path = Path(spec["corpus_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema"] != corpus.SCHEMA or not manifest["training_admitted"] or
        data.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]):
        raise ValueError("Authenticated admitted corpus required")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens < 2048 or judge_repetitions < 2:
        raise ValueError("Bounded output budget and >=2 semantic judge repetitions required")
    config = semantic.JudgeConfig(expected_returned_model=semantic.JudgeConfig().model) if judge_config is None else judge_config
    if not config.expected_returned_model:
        raise ValueError("Returned semantic judge identity must be pinned before results")
    implementations = {str(Path(module.__file__).resolve()): corpus.sha(module.__file__)
                       for module in (tasks, semantic, generation, corpus, corpus.old, data, generation.training)}
    implementations[str(Path(__file__).resolve())] = corpus.sha(__file__)
    implementations.update({str((Path(spec["native_model_path"])/name).resolve()): expected
                            for name, expected in manifest["tokenizer_files"].items()})
    if manifest["spec_sha256"] != corpus.sha(spec_path):
        raise ValueError("Corpus belongs to a different frozen specification")
    grader = dict(implementation_sha256=implementations, semantic_configuration=asdict(config),
        semantic_rubric_sha256=data.digest(semantic.RUBRIC), semantic_repetitions=judge_repetitions,
        primary_semantic_repetition=0, apps_metric="Published TRAIN tests, exact function result / whitespace-token-normalized stdout; not official APPS benchmark accuracy",
        mbpp_metric="Published assertions executed in existing fail-closed bubblewrap sandbox",
        math_metric="Existing pinned GSM8K final-number extraction and exact Decimal comparison",
        reference_policy="Code reference must pass the same sandbox/tests; otherwise unavailable, never silently dropped")
    plan = dict(schema=SCHEMA, spec_path=str(spec_path), spec_sha256=corpus.sha(spec_path), spec_snapshot=spec,
        corpus_manifest=str(manifest_path), corpus_manifest_sha256=corpus.sha(manifest_path),
        corpus_fingerprint=manifest["fingerprint"], implementation_sha256=implementations,
        grader=grader, grader_identity=data.digest(grader),
        decoding=dict(max_new_tokens=max_new_tokens, temperature=0., do_sample=False, eos_token_ids=spec["eos_token_ids"]),
        max_total_length=2048, output_budget_policy="min(declared max_new_tokens, 2048-prompt_length); never truncate prompt",
        family_weights={family:.25 for family in corpus.FAMILIES}, margin=.05, alpha=.05, power=.8,
        planned_discordance=1., required_pairs=corpus.sample_size(),
        power_guaranteed_for_reported_interval=False,
        power_note=manifest["confirmation"]["power_note"],
        hoeffding_power_bound_pairs=manifest["confirmation"]["hoeffding_power_bound_pairs"],
        interval="Two-sided distribution-free weighted Hoeffding CI over independent source-group paired differences in [-1,1]",
        endpoint="Equal-family-weighted task success; useful completed task success reported separately",
        unknown_policy="Unsupported, unavailable and missing stay separate; complete endpoint required for noninferiority",
        confirmation=manifest["confirmation"], expected_counts=manifest["counts"],
        benchmark_scope="Public TRAIN-derived held-out groups; not official published benchmark reproduction; pretraining/semantic leakage unresolved",
        created_at=datetime.now(timezone.utc).isoformat())
    plan["fingerprint"] = data.digest(plan)
    root.mkdir(parents=True, exist_ok=False)
    corpus.save(root/"plan.json", plan)
    return plan


def select(root, student_checkpoint):
    root = Path(root)
    plan = load_plan(root)
    value = dict(schema="jspace.recovery-reset.selection.v1", plan_fingerprint=plan["fingerprint"],
        student_checkpoint=checkpoint_identity(student_checkpoint),
        native_checkpoint=plan["spec_snapshot"]["native_model_revision"],
        created_at=datetime.now(timezone.utc).isoformat(), frozen_before_confirmation_open=True)
    corpus.save(root/"selection.json", value)
    return dict(status="selection_sealed", selection_sha256=corpus.sha(root/"selection.json"), **value)


def apps_program(response, context):
    """Construct test harness text; NOTHING generated executes in this process."""
    tests = context.get("apps_tests")
    if not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list) or not isinstance(tests.get("outputs"), list):
        raise LookupError("APPS test payload unavailable")
    inputs, outputs = tests["inputs"], tests["outputs"]
    if not inputs or len(inputs) != len(outputs):
        raise LookupError("APPS tests empty or unpaired")
    code = tasks.extract_code(response)
    name = tests.get("fn_name")
    if name:
        if not isinstance(name, str) or not name.isidentifier() or any(not isinstance(v, list) for v in inputs):
            raise NotImplementedError("Unsupported APPS callable schema")
        # APPS stores some strings as JSON string literals, including recursively
        # in argument lists. Decode only explicit quoted JSON strings.
        def unquote(value):
            if isinstance(value, list):
                return [unquote(v) for v in value]
            if isinstance(value, dict):
                return {k: unquote(v) for k, v in value.items()}
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                return json.loads(value)
            return value
        harness = ("import json\nfrom typing import *\n" + code +
                   "\n__apps_fn = getattr(Solution(), " + repr(name) + ") if 'Solution' in globals() else globals()[" + repr(name) + "]\n")
        assertions = ["assert json.loads(json.dumps(__apps_fn(*" + repr(unquote(a)) + "))) == " + repr(unquote(b))
                      for a, b in zip(inputs, outputs)]
    else:
        if any(not isinstance(v, str) for v in inputs+outputs):
            raise NotImplementedError("Unsupported APPS stdin/stdout schema")
        harness = """import io, sys
__apps_source = SOURCE
def __apps_run(value):
    old_in, old_out = sys.stdin, sys.stdout
    captured = io.BytesIO()
    output = io.TextIOWrapper(captured, encoding='utf-8', write_through=True)
    sys.stdin = io.TextIOWrapper(io.BytesIO(value.encode('utf-8')), encoding='utf-8')
    sys.stdout = output
    try:
        try:
            exec(compile(__apps_source, '<candidate>', 'exec'), {'__name__': '__main__'})
        except SystemExit as error:
            if error.code not in (None, 0): raise
        output.flush()
        return captured.getvalue().decode('utf-8').split()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
""".replace("SOURCE", repr(code), 1)
        assertions = ["assert __apps_run(" + repr(a) + ") == " + repr(b.split()) for a, b in zip(inputs, outputs)]
    return harness, assertions


def grade_code(response, context, **sandbox_options):
    if context["family"] == "mbpp":
        return tasks.run_mbpp(response, context, **sandbox_options)
    try:
        program, assertions = apps_program(response, context)
    except LookupError as error:
        return dict(status="unavailable", available=False, passed=None, reason=str(error))
    except NotImplementedError as error:
        return dict(status="unsupported", available=False, passed=None, reason=str(error))
    except (ValueError, TypeError) as error:
        return dict(status="incorrect", available=True, passed=False, reason=str(error))
    probe = tasks.sandbox_probe(sandbox_options.get("runtime"), sandbox_options.get("bwrap", "/usr/bin/bwrap"))
    if not probe["available"]:
        return dict(status="unavailable", available=False, passed=None, reason=probe["reason"])
    return tasks._sandbox_run(program, assertions, **sandbox_options)


def normalize_grade(value):
    if value.get("available") is True and type(value.get("passed")) is bool:
        return dict(value, status="correct" if value["passed"] else "incorrect")
    return dict(value, status="unsupported" if value.get("status") == "unsupported" else "unavailable", available=False, passed=None)


def grade_pair(native, student, context, **sandbox_options):
    family = context["family"]
    if family in {"mbpp", "apps"}:
        reference = normalize_grade(grade_code(context["response"], context, **sandbox_options))
        if not reference["available"] or not reference["passed"]:
            result = dict(status="unsupported" if reference["status"] == "unsupported" else "unavailable",
                available=False, passed=None, reason="Published reference could not qualify under frozen tests", reference=reference)
            return result, result
        return tuple(normalize_grade(grade_code(row["response"], context, **sandbox_options)) for row in (native, student))
    if family == "gsm8k":
        return tuple(normalize_grade(tasks.grade_answer(row["response"], context)) for row in (native, student))
    if family in {"dolly", "xsum"}:
        return (dict(status="unavailable", available=False, passed=None, reason="Semantic judgment required"),) * 2
    return (dict(status="unsupported", available=False, passed=None, reason="No pinned task grader"),) * 2


def validate_response(row, record, plan, *, selection_sha256=None):
    required = {"example_id", "group_id", "task", "checkpoint", "role", "response", "output_ids", "stop_reason", "input_ids", "decoding", "grader_identity"}
    if not required <= row.keys() or row["role"] not in {"native", "student"}:
        raise ValueError("Response row lacks declared provenance")
    checkpoint_identity(row["checkpoint"])
    if row["role"] == "native" and checkpoint_identity(row["checkpoint"]) != plan["spec_snapshot"]["native_model_revision"]:
        raise ValueError("Native response checkpoint differs from pinned teacher")
    if row["example_id"] != record.example_id or row["group_id"] != record.group_id or row["task"] != record.task or tuple(row["input_ids"]) != record.prompt_ids:
        raise ValueError("Response prompt/source lineage mismatch")
    if row["grader_identity"] != plan["grader_identity"] or row["decoding"] != plan["decoding"]:
        raise ValueError("Unpinned decoding/grader identity")
    output = row["output_ids"]
    if not isinstance(row["response"], str) or not isinstance(output, list) or any(type(x) is not int or x < 0 for x in output):
        raise ValueError("Actual response text and token IDs required")
    eos = set(plan["decoding"]["eos_token_ids"])
    budget = min(plan["decoding"]["max_new_tokens"], 2048-len(record.prompt_ids))
    if len(output) > budget or any(token in eos for token in output[:-1]):
        raise ValueError("Output exceeds budget or continues after EOS")
    if row["stop_reason"] == "eos":
        if not output or output[-1] not in eos:
            raise ValueError("Natural EOS requires observed terminal EOS")
    elif row["stop_reason"] == "length":
        if len(output) != budget or output and output[-1] in eos:
            raise ValueError("Length termination must exhaust budget without fabricated EOS")
    elif row["stop_reason"] != "stop" or output and output[-1] in eos:
        raise ValueError("Unknown/conflicting termination")
    if selection_sha256 is not None and row.get("selection_sha256") != selection_sha256:
        raise ValueError("Confirmation generation not bound to prospective selection")


def paired_interval(values, weights, *, alpha=.05):
    """Finite-sample bounded CI; values are source-group means by family."""
    if any(not values.get(family) for family in weights):
        return dict(available=False, reason="Missing family endpoint")
    delta = sum(weight*sum(values[family])/len(values[family]) for family, weight in weights.items())
    sum_squared_weights = sum(weight**2/len(values[family]) for family, weight in weights.items())
    radius = math.sqrt(2*math.log(2/alpha)*sum_squared_weights)
    return dict(available=True, delta=delta, low=max(-1., delta-radius), high=min(1., delta+radius),
        confidence=1-alpha, groups=sum(len(values[f]) for f in weights), method="weighted_paired_group_Hoeffding")


def summarize(rows, plan, split):
    statuses = {role: Counter(row[role]["status"] for row in rows) for role in ("native", "student")}
    by_family, differences, useful, grouped = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["group_id"])].append(row)
    for (family, _), entries in grouped.items():
        if all(entry[role]["available"] for entry in entries for role in ("native", "student")):
            differences[family].append(sum(int(r["student"]["passed"])-int(r["native"]["passed"]) for r in entries)/len(entries))
            useful[family].append(sum(int(r["student"]["useful_completed"])-int(r["native"]["useful_completed"]) for r in entries)/len(entries))
    for family in plan["family_weights"]:
        subset = [r for r in rows if r["category"] == family]
        both = [r for r in subset if all(r[a]["available"] for a in ("native", "student"))]
        by_family[family] = dict(pairs=len(subset), available_pairs=len(both),
            unavailable_pairs=len(subset)-len(both),
            native_correct=sum(r["native"]["passed"] for r in both), student_correct=sum(r["student"]["passed"] for r in both),
            discordant_pairs=sum(r["native"]["passed"] != r["student"]["passed"] for r in both),
            native_accuracy=sum(r["native"]["passed"] for r in both)/len(both) if both else None,
            student_accuracy=sum(r["student"]["passed"] for r in both)/len(both) if both else None,
            paired_difference=paired_interval(differences, {family:1.}, alpha=plan["alpha"]),
            useful_completed_difference=paired_interval(useful, {family:1.}, alpha=plan["alpha"]),
            metrics={role: dict(natural_eos=sum(r[role].get("natural_eos", False) for r in subset),
                useful_completed=sum(r[role].get("useful_completed", False) for r in subset),
                repetition=[r[role].get("diagnostics", {}).get("repetition") for r in subset]) for role in ("native", "student")})
    interval = paired_interval(differences, plan["family_weights"], alpha=plan["alpha"])
    complete = all(r[a]["available"] for r in rows for a in ("native", "student")) and bool(rows)
    enough = interval.get("groups", 0) >= plan["required_pairs"]
    confirmation = split == "confirmation"
    eligible = confirmation and complete and enough and plan["confirmation"]["sample_size_ready"]
    if not eligible:
        verdict = "inconclusive"
    elif interval["low"] > -plan["margin"]:
        verdict = "noninferior"
    elif interval["high"] < -plan["margin"]:
        verdict = "inferior_beyond_margin"
    else:
        verdict = "inconclusive"
    return dict(statuses=statuses, families=by_family, task_accuracy_difference=interval,
        useful_completed_difference=paired_interval(useful, plan["family_weights"], alpha=plan["alpha"]),
        complete_endpoint=complete, planned_sample_size_met=enough, noninferiority_eligible=eligible,
        margin=plan["margin"], verdict=verdict,
        scope="DEV is diagnostic only; confirmation requires frozen selection, full endpoint and paired interval")


def run(root, responses_path, *, split="development", label="paired", sandbox_options=None):
    from transformers import AutoTokenizer
    root, responses_path = Path(root), Path(responses_path).resolve()
    plan = load_plan(root)
    manifest = json.loads(Path(plan["corpus_manifest"]).read_text())
    if split not in {"development", "confirmation"} or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Declared evaluation split and safe unique result label required")
    output = root/"runs"/label
    output.mkdir(parents=True, exist_ok=False)
    responses_sha256 = corpus.sha(responses_path)
    corpus.save(output/"inputs.json", dict(plan_fingerprint=plan["fingerprint"], responses=str(responses_path),
        responses_sha256=responses_sha256, split=split))
    try:
        selection_sha = None
        selection = None
        if split == "confirmation":
            selection = json.loads((root/"selection.json").read_text())
            if selection["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Selection belongs to another protocol")
            selection_sha = corpus.sha(root/"selection.json")
            corpus.save(output/"unseal.json", dict(selection_sha256=selection_sha, opened_at=datetime.now(timezone.utc).isoformat()))
        for key in (split+"_records", split+"_contexts"):
            authenticate(manifest[key], manifest["sha256"][manifest[key]])
        records = {r.example_id: r for r in data.stream_records(manifest[split+"_records"],
                   allowed_splits=("test" if split == "confirmation" else "development",))}
        contexts = {r["example_id"]: r for r in corpus.jsonl(manifest[split+"_contexts"])}
        if records.keys() != contexts.keys():
            raise ValueError("Evaluation contexts/records mismatch")
        tokenizer = AutoTokenizer.from_pretrained(plan["spec_snapshot"]["native_model_path"], local_files_only=True, trust_remote_code=False)
        outputs, identities = {}, defaultdict(set)
        for row in corpus.jsonl(responses_path):
            if row["example_id"] not in records:
                raise ValueError("Response source not admitted to selected evaluation split")
            validate_response(row, records[row["example_id"]], plan, selection_sha256=selection_sha)
            if tokenizer.decode(row["output_ids"], skip_special_tokens=True).strip() != row["response"].strip():
                raise ValueError("Raw response differs from decoded output IDs")
            key = row["example_id"], row["role"]
            if key in outputs:
                raise ValueError("Duplicate evaluation role/example")
            outputs[key] = row
            identities[row["role"]].add(checkpoint_identity(row["checkpoint"]))
        if any(len(values) != 1 for values in identities.values()):
            raise ValueError("One frozen checkpoint per role/run required")
        if selection and identities.get("student") != {selection["student_checkpoint"]}:
            raise ValueError("Confirmation student differs from sealed selection")
        # Pin native responses across checkpoint comparisons; immutable per-ID
        # receipts permit partial runs without discarding or regenerating pairs.
        bindings = root/"native_bindings"/split
        bindings.mkdir(parents=True, exist_ok=True)
        for (example, role), row in outputs.items():
            if role != "native":
                continue
            binding = dict(example_id=example, native_response_sha256=data.digest(row),
                context_sha256=data.digest(contexts[example]), prompt_sha256=data.digest(row["input_ids"]),
                decoding_sha256=data.digest(row["decoding"]))
            path = bindings/(data.digest(example)+".json")
            if path.exists():
                if json.loads(path.read_text()) != binding:
                    raise ValueError("Native response or judging context changed across paired comparisons")
            else:
                corpus.save(path, binding)
        semantic_pairs = [example for example in records if contexts[example]["family"] in {"dolly", "xsum"}
                          and all((example, role) in outputs for role in ("native", "student"))]
        judgment_runs = []
        if semantic_pairs:
            native, student, judge_contexts = [], [], {}
            for example in semantic_pairs:
                for role, target in (("native", native), ("student", student)):
                    row = outputs[example, role]
                    target.append(dict(row, max_new_tokens=row["decoding"]["max_new_tokens"], eos_token_ids=row["decoding"]["eos_token_ids"]))
                c = contexts[example]
                judge_contexts[example] = dict(prompt=c["instruction"], source=c["context"], reference=c["response"], task=c["task"])
            config = semantic.JudgeConfig(**plan["grader"]["semantic_configuration"])
            for repetition in range(plan["grader"]["semantic_repetitions"]):
                judgment_runs.append(semantic.judge_pairs(native, student, output/("semantic_"+str(repetition)), config=config, contexts=judge_contexts))
        judged = [{r["example_id"]: r for r in run["judgments"]} for run in judgment_runs]
        rows = []
        for example, record in records.items():
            context = contexts[example]
            row = dict(example_id=example, group_id=record.group_id, task=record.task, split=split, category=context["category"],
                       context_sha256=data.digest(context), grader_identity=plan["grader_identity"])
            present = all((example, role) in outputs for role in ("native", "student"))
            if present:
                n, s = outputs[example, "native"], outputs[example, "student"]
                grades = grade_pair(n, s, context, **(sandbox_options or {}))
                if judged and example in judged[0]:
                    verdict = judged[0][example]
                    if verdict["status"] == "completed":
                        grades = tuple(dict(status="correct" if verdict[role]["task_success"] else "incorrect",
                            available=True, passed=verdict[role]["task_success"], semantic=verdict[role]) for role in ("native", "student"))
                row["pair_sha256"] = data.digest(dict(native=data.digest(n), student=data.digest(s), context=row["context_sha256"], grader=plan["grader_identity"]))
            else:
                grades = (dict(status="missing", available=False, passed=None, reason="Paired response missing"),) * 2
            for role, grade in zip(("native", "student"), grades):
                response = outputs.get((example, role))
                if response:
                    diagnostic = generation.generation_diagnostics(response, reference_text=context["response"], source_text=context["context"])
                    diagnostic["token_limit"] = response["stop_reason"] == "length"
                    natural = response["stop_reason"] == "eos"
                    semantic_good = grade.get("semantic", {}).get("generation_success", True)
                    grade = dict(grade, checkpoint=response["checkpoint"], response_sha256=data.digest(response),
                        natural_eos=natural, useful_completed=bool(grade["available"] and grade["passed"] and natural and response["response"].strip() and semantic_good), diagnostics=diagnostic)
                else:
                    grade = dict(grade, natural_eos=False, useful_completed=False)
                row[role] = grade
            rows.append(row)
        variability = Counter()
        for repeat in judged[1:]:
            for example in semantic_pairs:
                a, b = judged[0][example], repeat[example]
                if a["status"] == b["status"] == "completed":
                    variability["available_repeated_pairs"] += 1
                    for role in ("native", "student"):
                        variability[role+"_task_disagreements"] += a[role]["task_success"] != b[role]["task_success"]
                        variability[role+"_generation_disagreements"] += a[role]["generation_success"] != b[role]["generation_success"]
                else:
                    variability["unavailable_repeated_pairs"] += 1
        authenticate(responses_path, responses_sha256)
        result = dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], responses_sha256=responses_sha256,
            split=split, checkpoints={k: sorted(v) for k, v in identities.items()}, selection_sha256=selection_sha,
            rows=rows, summary=summarize(rows, plan, split), judge_variability=dict(variability),
            teacher_pairing_policy="Exact native response/context/prompt/decoding bindings reused; repeated judge outcomes measured, not cherry-picked",
            semantic_provenance=[r["judge_provenance"] for r in judgment_runs])
        result["spec_sha256"] = plan["spec_sha256"]
        result["evidence_sha256"] = {str(path.resolve()): corpus.sha(path) for path in output.rglob("*.json")}
        result["evidence_sha256"][str(responses_path)] = responses_sha256
        for response in outputs.values():
            checkpoint = response["checkpoint"]
            if isinstance(checkpoint, dict) and checkpoint.get("path"):
                result["evidence_sha256"][str(Path(checkpoint["path"]).resolve())] = checkpoint["sha256"]
        result["fingerprint"] = data.digest(result)
        corpus.save(output/"result.json", result)
        return result
    except Exception as error:
        corpus.save(output/"failure.json", dict(status="failed", error_type=type(error).__name__, reason=str(error)))
        raise


def check(root, label=None):
    plan = load_plan(root)
    manifest = json.loads(Path(plan["corpus_manifest"]).read_text())
    for path, expected in manifest["sha256"].items():
        authenticate(path, expected)
    results = [Path(root)/"runs"/label/"result.json"] if label else sorted((Path(root)/"runs").glob("*/result.json"))
    summaries = {}
    for path in results:
        value = json.loads(path.read_text())
        if value["plan_fingerprint"] != plan["fingerprint"] or data.digest({k: v for k, v in value.items() if k != "fingerprint"}) != value["fingerprint"]:
            raise ValueError("Evaluation result authentication failed")
        for artifact, expected in value["evidence_sha256"].items():
            authenticate(artifact, expected)
        summaries[path.parent.name] = value["summary"]
    return dict(status="authenticated", plan_fingerprint=plan["fingerprint"], completed=summaries)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("prepare", "select", "run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--label")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--student-checkpoint")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.spec is None:
            parser.error("prepare requires --spec")
        value = prepare(args.root, args.spec, max_new_tokens=args.max_new_tokens)
        result = {k: value[k] for k in ("fingerprint", "grader_identity", "decoding", "required_pairs")}
    elif args.phase == "select":
        if args.student_checkpoint is None:
            parser.error("select requires --student-checkpoint SHA256")
        result = select(args.root, args.student_checkpoint)
    elif args.phase == "run":
        if args.responses is None:
            parser.error("run requires --responses")
        value = run(args.root, args.responses, split=args.split, label=args.label or "paired")
        result = dict(fingerprint=value["fingerprint"], summary=value["summary"], judge_variability=value["judge_variability"])
    else:
        result = check(args.root, args.label)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
