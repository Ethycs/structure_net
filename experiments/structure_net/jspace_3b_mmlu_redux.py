"""Frozen MMLU-Redux choice-logit reference comparison, not a Qwen reproduction.

Native weights live on the host and execute one block at a time on the leased
GPU. One native pass supplies both exact suffix boundaries and the teacher.
No answer, annotation or contamination field is accepted by the model runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_3b_final_evaluate as execution
from . import jspace_3b_packed_response_model as packed
from . import jspace_3b_response_cache as cache
from . import jspace_3b_response_model as response
from . import jspace_qwen_parallel_model as native

SCHEMA = "jspace.3b-mmlu-redux.v1"
MODEL_NAMES = ("teacher", "blocks08", "blocks03")
CUTS = {"blocks08": 28, "blocks03": 33}
BOOTSTRAP_SEED = 20261117
BOOTSTRAP_DRAWS = 10000
ALLOCATION_MIB = 3584
ALLOCATOR_MIB = 3328
read, write, sha = cache.read, cache.write, cache.sha


@contextmanager
def host_weights(module, device, guard):
    """Temporarily transfer frozen storage; restore even a partial transfer."""
    values = [(value, value.detach()) for value in (*module.parameters(), *module.buffers())]
    if any(value.device.type != "cpu" or value.requires_grad for value, _ in values):
        raise ValueError("Streaming requires frozen host weights")
    guard()
    try:
        for value, original in values:
            value.data = original.to(device)
        guard()
        yield
        guard()
    finally:
        for value, original in values:
            value.data = original


class NativeStream:
    """Reusable original blocks; teacher uses the original final norm and E."""
    def __init__(self, blocks, embedding, norm, rotary, *, guard, cuts=(28, 33)):
        self.blocks, self.embedding, self.norm, self.rotary = blocks, embedding, norm, rotary
        self.guard, self.cuts = guard, tuple(cuts)
        if any(cut <= 0 or cut >= len(blocks) for cut in self.cuts):
            raise ValueError("Native boundaries must precede the final block")
        self.calls = 0
        self.host_to_device_bytes = 0

    @torch.inference_mode()
    def forward(self, input_ids, *, capture=True):
        self.guard()
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.embedding.device)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("One nonempty unpadded token sequence required")
        ids = ids[None]
        hidden = F.embedding(ids, self.embedding)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        allowed = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).tril()[None, None]
        mask = torch.zeros(allowed.shape, device=ids.device, dtype=hidden.dtype)
        mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        rope = self.rotary(hidden, positions)
        boundaries = {}
        for index, block in enumerate(self.blocks):
            if capture and index in self.cuts:
                # Keep the actual residual, not a refitted or recomputed prefix.
                boundaries[index] = hidden.detach().clone()
            with host_weights(block, ids.device, self.guard):
                hidden = block(hidden, attention_mask=mask, position_ids=positions,
                    position_embeddings=rope, cache_position=positions[0], use_cache=False)[0]
                if ids.device.type == "cuda":
                    self.host_to_device_bytes += sum(v.numel()*v.element_size() for v in (*block.parameters(), *block.buffers()))
        self.guard()
        self.calls += 1
        return self.norm(hidden[:, -1]), boundaries


def choice_scores(hidden, readout, label_ids):
    if len(label_ids) != 4 or len(set(label_ids)) != 4:
        raise ValueError("Four distinct A-D token IDs required")
    logits = F.linear(hidden, readout[list(label_ids)])[0].float().cpu()
    if logits.shape != (4,) or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite or malformed choice logits")
    return dict(choice_logits=logits.tolist(), prediction=int(logits.argmax()))


class Runner:
    def __init__(self, plan, guard, *, device="cuda:0"):
        self.plan, self.guard = plan, guard
        cfg = native._config(plan["model_path"])
        if cfg.num_hidden_layers != 36 or not cfg.tie_word_embeddings:
            raise ValueError("Original tied 36-layer Qwen required")
        guard()
        embedding, norm, rotary = native.load_readout_norm(plan["model_path"], device="cpu", config=cfg)
        blocks = response._prefix_layers(plan["model_path"], 36, cfg, torch.device("cpu"), None)
        self.host_embedding = embedding
        self.students, self.reload_receipts = {}, {}
        for name, cut in CUTS.items():
            model, receipt = packed.load_packed_snapshot(plan["models"][name]["export"], device="cpu")
            if (model.cut != cut or plan["models"][name]["cut"] != cut or model.max_length != 416
                    or not torch.equal(model.embedding, embedding)):
                raise ValueError("Frozen suffix cut, support or original embedding mismatch")
            # Remove each loaded duplicate before transferring the shared E.
            model.embedding = embedding
            self.students[name], self.reload_receipts[name] = model, receipt
        guard()
        self.embedding = embedding.to(device)
        self.stream = NativeStream(blocks, self.embedding, norm.to(device), rotary.to(device), guard=guard)
        for model in self.students.values():
            model.embedding = self.embedding
            model.guard = guard
            model.stack.enabled = False  # Ordinary packed Torch BF16 on Turing.
        guard()

    @contextmanager
    def resident_student(self, model):
        # E is already shared on-device; never move the whole model or copy E.
        with host_weights(model.stack, self.embedding.device, self.guard):
            with host_weights(model.norm, self.embedding.device, self.guard):
                with host_weights(model.rotary, self.embedding.device, self.guard):
                    yield

    @torch.inference_mode()
    def student(self, name, input_ids, boundary, label_ids):
        if len(input_ids)+1 > self.plan["max_length"]:
            raise ValueError("Unsupported student prompt; no position extension or clamping")
        model = self.students[name]
        with self.resident_student(model):
            ids = torch.tensor([input_ids], device=self.embedding.device)
            hidden = model.hidden_forward(dict(input_ids=ids, boundary_hidden=boundary))[:, -1]
            self.guard()
            return choice_scores(hidden, self.embedding, label_ids)

    @torch.inference_mode()
    def evaluate(self, input_ids, *, eligible, label_ids):
        if eligible != (len(input_ids)+1 <= self.plan["max_length"]):
            raise ValueError("Frozen eligibility disagrees with prompt support")
        hidden, boundaries = self.stream.forward(input_ids, capture=eligible)
        self.guard()
        result = {"teacher": choice_scores(hidden, self.embedding, label_ids)}
        for name, cut in CUTS.items():
            result[name] = self.student(name, input_ids, boundaries[cut], label_ids) if eligible else None
        self.guard()
        return result

    @torch.inference_mode()
    def pilot(self, control):
        evidence = control["evidence"]
        if not evidence or any(sha(path) != digest for path, digest in evidence.items()):
            raise ValueError("Authenticated native DEV evidence changed")
        ids, labels = control["input_ids"], control["label_ids"]
        hidden, boundaries = self.stream.forward(ids)
        actual = choice_scores(hidden, self.embedding, labels)
        expected = torch.tensor(control["expected_choice_logits"], dtype=torch.float32)
        observed = torch.tensor(actual["choice_logits"])
        native_exact = torch.equal(observed, expected)
        native_control = dict(identity=control["identity"], actual=actual,
            expected_choice_logits=expected.tolist(), bit_exact=native_exact,
            max_absolute_error=float((observed-expected).abs().max()), evidence=evidence)
        if not native_exact:
            raise ValueError("Streamed native logits differ from authenticated existing DEV cache: " + json.dumps(native_control))
        reloads = {}
        for name, cut in CUTS.items():
            before = self.student(name, ids, boundaries[cut], labels)
            original = self.students.pop(name)
            del original
            gc.collect()
            fresh, receipt = packed.load_packed_snapshot(self.plan["models"][name]["export"], device="cpu")
            if not torch.equal(fresh.embedding, self.host_embedding):
                raise ValueError("Fresh packed embedding differs from native")
            fresh.embedding, fresh.guard = self.embedding, self.guard
            fresh.stack.enabled = False
            self.students[name] = fresh
            after = self.student(name, ids, boundaries[cut], labels)
            exact = before == after and receipt["packed_state_exact"]
            # Independently exercise the production entrypoint on the same
            # actual native residual, without a second native prefix pass.
            with self.resident_student(fresh):
                fresh.native_prefix = lambda _ids, _valid, b=boundaries[cut]: b
                try:
                    tensor = torch.tensor([ids], device=self.embedding.device)
                    production = fresh.hidden_forward(dict(input_ids=tensor))[:, -1]
                    production_scores = choice_scores(production, self.embedding, labels)
                finally:
                    fresh.native_prefix = None
            if not exact or production_scores != after:
                raise ValueError("Packed reload or actual-prefix entrypoint is not exact")
            reloads[name] = dict(bit_exact=True, packed_state_exact=True,
                choice_scores=after, production_entrypoint_exact=True, receipt=receipt)
        return dict(passed=True, native_control=native_control, packed_reload=reloads,
            benchmark_answers_observed=0, scope="Only pre-existing authenticated DEV control")


def _accuracy(cases, outputs, name, label):
    eligible = [c for c in cases if c[label] is not None and outputs[c["identity"]][name] is not None]
    correct = sum(outputs[c["identity"]][name]["prediction"] == c[label] for c in eligible)
    return dict(count=len(eligible), correct=correct, accuracy=correct/len(eligible) if eligible else None)


def paired(cases, outputs, name, label, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Exact paired counts and subject-stratified empirical bootstrap of delta."""
    groups = {}
    wins = losses = both_correct = both_wrong = 0
    for case in cases:
        values = outputs[case["identity"]]
        if case[label] is None or values[name] is None:
            continue
        student = values[name]["prediction"] == case[label]
        teacher = values["teacher"]["prediction"] == case[label]
        difference = int(student)-int(teacher)
        groups.setdefault(case["subject"], []).append(difference)
        wins += difference == 1
        losses += difference == -1
        both_correct += student and teacher
        both_wrong += not student and not teacher
    count = wins+losses+both_correct+both_wrong
    interval = None
    if count:
        rng = np.random.default_rng(seed)
        totals = np.zeros(draws, dtype=np.int64)
        for subject in sorted(groups):
            differences = np.asarray(groups[subject])
            counts = np.array([(differences == value).sum() for value in (-1, 0, 1)])
            samples = rng.multinomial(len(differences), counts/len(differences), size=draws)
            totals += samples[:, 2]-samples[:, 0]
        interval = np.quantile(totals/count, [.025, .975]).tolist()
    return dict(count=count, wins=int(wins), losses=int(losses), both_correct=int(both_correct),
        both_wrong=int(both_wrong), accuracy_delta=(wins-losses)/count if count else None,
        bootstrap_95_ci=interval, bootstrap_draws=draws, bootstrap_seed=seed,
        bootstrap_method="Paired within-subject empirical resampling; fixed subject sample counts")


def summarize(cases, rows):
    outputs = {row["identity"]: row["models"] for row in rows}
    if len(outputs) != len(rows) or set(outputs) != {c["identity"] for c in cases}:
        raise ValueError("Output case identities must exactly cover the frozen inventory")
    def section(subset):
        return dict(count=len(subset), models={name: dict(
            evaluated=sum(outputs[c["identity"]][name] is not None for c in subset),
            unsupported=sum(outputs[c["identity"]][name] is None for c in subset),
            original=_accuracy(subset, outputs, name, "answer_original"),
            corrected=_accuracy(subset, outputs, name, "answer_corrected")) for name in MODEL_NAMES},
            paired_vs_teacher={name: {kind: paired(subset, outputs, name, "answer_"+kind)
                for kind in ("original", "corrected")} for name in CUTS})
    return dict(all=section(cases), paired_eligible=section([c for c in cases if c["eligible"]]),
        subjects={subject: section([c for c in cases if c["subject"] == subject])
            for subject in sorted({c["subject"] for c in cases})},
        contamination={label: section([c for c in cases if c["contamination"]["overlap"] is overlap])
            for label, overlap in (("known_overlap", True), ("disjoint_from_audited_sources", False))},
        interpretation="Unsupported students are missing, never scored wrong. Corrected accuracy excludes null labels. "
            "Overlap strata refer only to audited local sources, not unknown pretraining. "
            "Bootstrap describes this benchmark; no population-generalization or leakage-free claim.",
        published_context=dict(model="Qwen2.5-3B-Instruct", benchmark="MMLU-redux", score_percent=64.4,
            url="https://qwenlm.github.io/blog/qwen2.5-llm/", matched_protocol=False,
            note="Official instruct protocol is not fully specified. Our zero-shot chat A-D final-token "
                "choice-logit argmax is a reference comparison, not a reproduction or generative-compliance metric."))


def _plan(root):
    from . import jspace_3b_mmlu_redux_data as data
    return data.verify(root)


def _reservation():
    lease, guard = cache.reservation(ALLOCATOR_MIB)
    if list(lease["allocation"].values()) != [ALLOCATION_MIB]:
        raise ValueError("Exactly one 3584 MiB reservation required")
    name = torch.cuda.get_device_name(0)
    if "2060" not in name or "SUPER" in name.upper():
        raise ValueError("The predeclared plain RTX2060 worker is required")
    guard()
    return lease, guard


def _receipt(directory, fingerprint):
    manifest = read(directory/"manifest.json")
    if manifest.get("status") != "completed" or manifest.get("fingerprint") != fingerprint:
        raise ValueError("Completed phase fingerprint differs")
    if "result.json" not in manifest["artifacts"]:
        raise ValueError("Result is not authenticated by the phase manifest")
    for name, digest in manifest["artifacts"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError("Phase artifact changed: " + name)
    result = read(directory/"result.json")
    if result.get("status") != "completed" or result.get("fingerprint") != fingerprint:
        raise ValueError("Phase result differs")
    return result


def _passed_pilot(root, plan):
    result = _receipt(root/"pilot", plan["fingerprint"])
    if (result.get("passed") is not True or not result["native_control"]["bit_exact"]
            or any(not result["packed_reload"][name]["bit_exact"] for name in CUTS)
            or result["peak_reserved_bytes"] > ALLOCATOR_MIB*1024**2):
        raise ValueError("A passed exact, bounded existing-DEV pilot is mandatory")
    return result


def _validate_rows(cases, rows, plan):
    if len(cases) != len(rows):
        raise ValueError("Incomplete benchmark outputs")
    for case, row in zip(cases, rows):
        if row["identity"] != case["identity"] or row["eligible"] != case["eligible"]:
            raise ValueError("Frozen output ordering or eligibility differs")
        if case["eligible"] != (len(case["input_ids"])+1 <= plan["max_length"]):
            raise ValueError("Invalid support accounting")
        if set(row["models"]) != set(MODEL_NAMES):
            raise ValueError("All three model results must be represented")
        for name, result in row["models"].items():
            if result is None:
                if name == "teacher" or case["eligible"]:
                    raise ValueError("Missing supported model result")
                continue
            if name != "teacher" and not case["eligible"]:
                raise ValueError("Unsupported student was evaluated")
            logits = np.asarray(result["choice_logits"])
            if logits.shape != (4,) or not np.isfinite(logits).all() or int(logits.argmax()) != result["prediction"]:
                raise ValueError("Invalid saved choice scores or decision")


def _cases(root):
    with (root/"cases.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


@torch.inference_mode()
def run(root, pilot=False):
    root = Path(root).resolve()
    plan = _plan(root)
    if plan["max_length"] != 416 or set(plan["models"]) != set(CUTS):
        raise ValueError("Frozen two-suffix 416-position plan required")
    if not pilot:
        _passed_pilot(root, plan)
    directory = root/("pilot" if pilot else "run")
    directory.mkdir()  # Preserve every prior or partial response artifact.
    started, lease = time.monotonic(), None
    try:
        lease, guard = _reservation()
        torch.set_num_threads(2)
        with execution.arithmetic_policy():
            runner = Runner(plan, guard)
            if pilot:
                result = runner.pilot(plan["native_control"])
            else:
                cases = _cases(root)
                if len(cases) != 3000 or len({c["subject"] for c in cases}) != 30:
                    raise ValueError("Full MMLU-Redux v1 inventory of 3000/30 required")
                rows = []
                with (directory/"records.jsonl").open("x") as stream:
                    for index, case in enumerate(cases):
                        # Deliberately pass no labels or other case metadata.
                        values = runner.evaluate(case["input_ids"], eligible=case["eligible"], label_ids=plan["label_ids"])
                        row = dict(identity=case["identity"], subject=case["subject"], eligible=case["eligible"], models=values)
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                        stream.flush()
                        rows.append(row)
                        print(json.dumps(dict(phase="benchmark", completed=index+1, total=len(cases),
                            seconds=time.monotonic()-started)), flush=True)
                _validate_rows(cases, rows, plan)
                result = dict(record_count=len(rows), metrics=summarize(cases, rows),
                    pilot_manifest_sha256=sha(root/"pilot/manifest.json"))
            guard()
            result.update(schema=SCHEMA, status="completed", fingerprint=plan["fingerprint"],
                phase="pilot" if pilot else "run", lease_id=lease["id"], allocation=lease["allocation"],
                gpu_name=torch.cuda.get_device_name(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), allocator_cap_mib=ALLOCATOR_MIB,
                native_forward_calls=runner.stream.calls, native_weight_transfer_bytes=runner.stream.host_to_device_bytes,
                native_weight_storage="host", compute_device="cuda:0", shared_native_prefix=True,
                original_native_final_norm=True, shared_original_embedding=True,
                inference_backend="packed_bf16_torch", seconds=time.monotonic()-started,
                confirmed=False)
            write(directory/"result.json", result)
            write(directory/"manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
                artifacts={p.name: sha(p) for p in directory.iterdir() if p.is_file()}))
            return result
    except BaseException as error:
        write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"],
            error_type=type(error).__name__, error=str(error), lease=lease, partial_evidence_preserved=True))
        raise


def check(root):
    root = Path(root).resolve()
    plan = _plan(root)
    if (root/"run/manifest.json").exists():
        _passed_pilot(root, plan)
        result = _receipt(root/"run", plan["fingerprint"])
        if result["pilot_manifest_sha256"] != sha(root/"pilot/manifest.json"):
            raise ValueError("Benchmark pilot binding changed")
        cases = _cases(root)
        with (root/"run/records.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        _validate_rows(cases, rows, plan)
        if result["metrics"] != summarize(cases, rows) or result["record_count"] != len(rows):
            raise ValueError("Saved aggregate differs from actual case outputs")
        if result["peak_reserved_bytes"] > ALLOCATOR_MIB*1024**2:
            raise ValueError("Completed run exceeded the declared allocator cap")
        return result
    for phase in ("run", "pilot"):
        if (root/phase/"failure.json").exists():
            return read(root/phase/"failure.json")
    if (root/"pilot/manifest.json").exists():
        return _passed_pilot(root, plan)
    return dict(status="prepared_or_active", fingerprint=plan["fingerprint"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "run", "check"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.root) if args.phase == "check" else run(args.root, pilot=args.phase == "pilot")
    print(json.dumps({k: result[k] for k in ("status", "fingerprint", "record_count", "passed", "seconds") if k in result}), flush=True)


if __name__ == "__main__":
    main()
