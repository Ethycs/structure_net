"""Complete-response loss and generation checks for the response recovery study.

The caller authenticates data/checkpoints and owns GPU reservations. This module
does not discover TEST files or promote a checkpoint. Numeric/repetition checks
are diagnostics, never semantic-summary accuracy. Small executable canaries are
reported separately from held-out corpus semantics.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
import math
import re
from typing import Callable

import torch
from torch.nn import functional as F

from experiments.neural_architecture_lab.distillation_recovery import module

data = module("distillation_data")
training = module("distillation_training")
SCHEMA = "jspace.response-evaluation.v1"


@contextmanager
def evaluation_mode(model):
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training is not None:
            model.train(was_training)


def _logits(value, vocabulary_size=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError("A full-vocabulary logit vector is required")
    if value.numel() < 2 or (vocabulary_size is not None and value.numel() != vocabulary_size):
        raise ValueError("Wrong full-vocabulary dimension")
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite full-vocabulary inference logits")
    return value


@torch.no_grad()
def full_vocabulary_path_parity(actual_next: Callable, reference_next: Callable,
                               prompt_ids, *, vocabulary_size, generated_prefixes=2,
                               max_length=416, atol=0., rtol=0., device="cpu"):
    """Compare two paths of the SAME checkpoint at prompt and generated prefixes.

    Comparing the compressed model with the native teacher is not path parity.
    The frozen study supplies two independently constructed execution paths and
    tolerance. All vocabulary entries and greedy choices must agree.
    """
    if (not prompt_ids or generated_prefixes < 1 or atol < 0 or rtol < 0
            or len(prompt_ids) + generated_prefixes > max_length):
        raise ValueError("Prompt and at least one generated prefix must fit unchanged")
    prefix = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    rows = []
    for step in range(generated_prefixes + 1):
        actual = _logits(actual_next(prefix), vocabulary_size).float().cpu()
        reference = _logits(reference_next(prefix), vocabulary_size).float().cpu()
        difference = (actual - reference).abs()
        passed = bool(torch.all(difference <= atol + rtol * reference.abs()))
        same_choice = int(actual.argmax()) == int(reference.argmax())
        rows.append(dict(prefix_tokens=prefix.shape[1], maximum_absolute_error=float(difference.max()),
                         full_vocabulary_passed=passed, greedy_choice_equal=same_choice))
        if not passed or not same_choice:
            raise ValueError("Full-vocabulary inference paths disagree: " + json.dumps(rows[-1]))
        if step < generated_prefixes:
            prefix = torch.cat((prefix, torch.tensor([[int(reference.argmax())]], device=device)), dim=1)
    return dict(schema=SCHEMA, passed=True, vocabulary_size=vocabulary_size,
                generated_prefixes=generated_prefixes, atol=atol, rtol=rtol, records=rows,
                scope="Same-checkpoint execution parity; not teacher/student agreement or quality")


def response_losses(model, records, *, pad_token_id, max_length=416,
                    teacher_provider=None, prepare_batch=None, token_chunk_size=16,
                    device="cpu"):
    """Exact held-out response NLL and optional KL at every target including EOS.

    Token chunking limits vocabulary tensors, with normalization over the entire
    vocabulary. ``prepare_batch`` can attach authenticated native-prefix cache
    rows for a suffix model. The provider must target those exact full prefixes.
    No training or checkpoint selection occurs here.
    """
    if type(token_chunk_size) is not int or token_chunk_size < 1:
        raise ValueError("Positive token chunk size required")
    rows, seen = [], set()
    with evaluation_mode(model):
        for record in records:
            if record.split != "development" or record.example_id in seen:
                raise ValueError("Unique explicitly admitted DEV response records required")
            seen.add(record.example_id)
            batch = data.response_batch([record], pad_token_id=pad_token_id,
                                        max_length=max_length, device=device)
            if prepare_batch is not None:
                result = prepare_batch(batch)
                if result is not None:
                    batch = result
            if teacher_provider is not None and hasattr(teacher_provider, "prepare"):
                teacher_provider.prepare(batch)
            hidden = model.hidden_forward(batch)
            positions = batch["loss_mask"].nonzero(as_tuple=False)
            if len(positions) != len(record.response_ids):
                raise ValueError("All response targets, including observed EOS, must be evaluated")
            nll, divergence, teacher_nll = 0., 0., 0.
            for start in range(0, len(positions), token_chunk_size):
                selected = positions[start:start + token_chunk_size]
                local = selected.to(hidden.device)
                logits = model.readout(hidden[local[:, 0], local[:, 1]]).float()
                targets = batch["targets"][selected[:, 0], selected[:, 1]].to(logits.device)
                if logits.ndim != 2 or logits.shape[0] != len(selected) or not torch.isfinite(logits).all():
                    raise ValueError("Finite full-vocabulary response logits required")
                nll += float(F.cross_entropy(logits, targets, reduction="sum"))
                if teacher_provider is not None:
                    teacher = teacher_provider(batch, selected).to(logits.device).float()
                    if teacher.shape != logits.shape or not torch.isfinite(teacher).all():
                        raise ValueError("Teacher full-vocabulary rows disagree with student shape")
                    teacher_logp = F.log_softmax(teacher, dim=-1)
                    student_logp = F.log_softmax(logits, dim=-1)
                    divergence += float((teacher_logp.exp() * (teacher_logp - student_logp)).sum())
                    teacher_nll += float(F.cross_entropy(teacher, targets, reduction="sum"))
            count = len(positions)
            rows.append(dict(example_id=record.example_id, group_id=record.group_id, task=record.task,
                             response_tokens=count, eos_supervised=record.finish_reason == "eos",
                             nll_sum=nll, nll=nll/count,
                             kl_sum=divergence if teacher_provider is not None else None,
                             kl=divergence/count if teacher_provider is not None else None,
                             teacher_nll_sum=teacher_nll if teacher_provider is not None else None))
            if teacher_provider is not None and hasattr(teacher_provider, "clear_cache"):
                teacher_provider.clear_cache()
    if not rows:
        raise ValueError("DEV evaluation cannot be empty")
    tokens = sum(row["response_tokens"] for row in rows)
    mean_nll = sum(row["nll_sum"] for row in rows)/tokens
    # Strict JSON cannot carry inf; a null with an overflow flag is explicit.
    perplexity = math.exp(mean_nll) if mean_nll < 700 else None
    return dict(schema=SCHEMA, count=len(rows), response_tokens=tokens,
                nll=mean_nll, perplexity=perplexity, perplexity_overflow=perplexity is None,
                kl=sum(row["kl_sum"] for row in rows)/tokens if teacher_provider is not None else None,
                teacher_nll=sum(row["teacher_nll_sum"] for row in rows)/tokens if teacher_provider is not None else None,
                temperature=1., denominator="all response tokens including observed EOS", records=rows)


@torch.no_grad()
def generate_prompt(next_logits, prompt_ids, *, eos_token_ids, max_new_tokens=128,
                    max_length=416, vocabulary_size=None, device="cpu"):
    eos = {eos_token_ids} if type(eos_token_ids) is int else set(eos_token_ids)
    if (not eos or any(type(x) is not int or x < 0 for x in eos)
            or not prompt_ids or type(max_new_tokens) is not int or max_new_tokens < 1
            or len(prompt_ids) + max_new_tokens > max_length):
        raise ValueError("Pinned EOS and entire untruncated prompt/output budget must fit")
    prefix = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    output = []
    for _ in range(max_new_tokens):
        logits = _logits(next_logits(prefix), vocabulary_size)
        token = int(logits.argmax())
        output.append(token)
        if token in eos:
            break
        prefix = torch.cat((prefix, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
    return dict(input_ids=list(prompt_ids), output_ids=output,
                stop_reason="eos" if output[-1] in eos else "token_limit", use_cache=False,
                max_new_tokens=max_new_tokens, max_length=max_length,
                eos_token_ids=sorted(eos), decoding="greedy full vocabulary")


_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w|\.\d)")


def number_spans(text):
    values = set()
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        percent = raw.endswith("%")
        try:
            number = Decimal(raw.rstrip("%").replace(",", ""))
            value = format(number.normalize(), "f")
        except InvalidOperation:
            continue
        values.add(value + ("%" if percent else ""))
    return sorted(values)


def generation_diagnostics(row, *, reference_text=None, source_text=None):
    text = row["response"]
    output = row["output_ids"]
    lexical = output[:-1] if row["stop_reason"] == "eos" else output
    repetition = training.repetition_report(lexical)
    result = dict(empty=not text.strip(), repetition=repetition,
                  terminated=row["stop_reason"] == "eos",
                  token_limit=row["stop_reason"] == "token_limit",
                  bare_option_label=bool(re.fullmatch(r"\s*[A-D][.)]?\s*", text)),
                  scope="Automatic diagnostics only; not semantic correctness or summary accuracy")
    if source_text is not None:
        source, produced = set(number_spans(source_text)), set(number_spans(text))
        result["numbers_absent_from_source"] = sorted(produced - source)
        result["source_numbers"] = sorted(source)
        result["generated_numbers"] = sorted(produced)
    if reference_text is not None:
        reference, produced = set(number_spans(reference_text)), set(number_spans(text))
        result["reference_numbers"] = sorted(reference)
        result["reference_number_recall"] = len(reference & produced)/len(reference) if reference else None
    return result


def generate_record(model, record, tokenizer, *, max_new_tokens=128, max_length=416,
                    device="cpu", source_text=None, eos_token_ids=None):
    if record.split != "development":
        raise ValueError("Generation DEV evaluation requires explicit development record")
    with evaluation_mode(model):
        row = generate_prompt(model.next_logits, record.prompt_ids,
                              eos_token_ids=record.eos_token_id if eos_token_ids is None else eos_token_ids,
                              max_new_tokens=max_new_tokens,
                              max_length=max_length, device=device)
    row.update(example_id=record.example_id, group_id=record.group_id, task=record.task,
               response=tokenizer.decode(row["output_ids"], skip_special_tokens=True),
               reference=tokenizer.decode(list(record.response_ids), skip_special_tokens=True),
               prompt=tokenizer.decode(list(record.prompt_ids), skip_special_tokens=False))
    row["diagnostics"] = generation_diagnostics(row, reference_text=row["reference"], source_text=source_text)
    return row


# These constructed probes are intentionally small, executable and separate from
# held-out corpus semantic evaluation. They never constitute general accuracy.
CANARIES = (
    dict(id="addition", kind="text", expected="42",
         prompt="What is 17 + 25? Reply with only the integer result."),
    dict(id="sort", kind="json", expected=[2, 5, 9],
         prompt="Sort these integers in ascending order: 9, 2, 5. Reply with only a JSON array."),
    dict(id="replace", kind="text", expected="alpha-beta",
         prompt="Replace the underscore in alpha_beta with a hyphen. Reply with only the changed string."),
    dict(id="invoice", kind="json", expected={"quantity": 12, "unit_price": 7, "total": 84},
         prompt="An invoice contains 12 units at 7 dollars each, totaling 84 dollars. Return only a JSON object with integer fields quantity, unit_price, and total."),
    dict(id="boolean", kind="text", expected="yes",
         prompt="Mira had five tokens and gave away two. Does she now have three tokens? Reply with only yes or no."),
    dict(id="date", kind="json", expected={"date": "2031-04-09", "delay_days": 3},
         prompt="The launch took place on 2031-04-09 after a delay of three days. Return only a JSON object with date as an ISO date string and delay_days as an integer."),
)


def canary_inputs(tokenizer, *, max_new_tokens=128, max_length=416):
    result = []
    for case in CANARIES:
        ids = list(tokenizer.apply_chat_template([dict(role="user", content=case["prompt"])],
                                                tokenize=True, add_generation_prompt=True))
        if not ids or len(ids) + max_new_tokens > max_length:
            raise ValueError("Untruncated canary and generation budget exceed fitted geometry")
        result.append(dict(case, input_ids=ids))
    return result


def check_canary(case, response):
    if case["kind"] == "text":
        return response.strip().lower() == case["expected"].lower()
    if case["kind"] != "json":
        raise ValueError("Unknown canary constraint")
    try:
        parsed = json.loads(response)
    except (TypeError, ValueError):
        return False
    # JSON True == 1 in Python; type-preserving canonical JSON prevents that leak.
    return json.dumps(parsed, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)


def evaluate_canaries(model, tokenizer, *, cases=None, max_new_tokens=128,
                      max_length=416, device="cpu", eos_token_ids=None):
    cases = canary_inputs(tokenizer, max_new_tokens=max_new_tokens, max_length=max_length) if cases is None else cases
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Nonempty distinct frozen canaries required")
    rows = []
    with evaluation_mode(model):
        for case in cases:
            row = generate_prompt(model.next_logits, case["input_ids"],
                                  eos_token_ids=tokenizer.eos_token_id if eos_token_ids is None else eos_token_ids,
                                  max_new_tokens=max_new_tokens, max_length=max_length, device=device)
            row.update(id=case["id"], prompt=case["prompt"], expected=case["expected"],
                       response=tokenizer.decode(row["output_ids"], skip_special_tokens=True))
            row["diagnostics"] = generation_diagnostics(row)
            row["constraint_passed"] = check_canary(case, row["response"])
            row["generation_success"] = bool(row["constraint_passed"] and row["stop_reason"] == "eos"
                                               and not row["diagnostics"]["repetition"]["flagged"])
            rows.append(row)
    return dict(schema=SCHEMA, count=len(rows), successes=sum(row["generation_success"] for row in rows),
                constraints_passed=sum(row["constraint_passed"] for row in rows), records=rows,
                scope="Constructed deterministic generation canaries; not general semantic accuracy")


def generation_gate(native, student, *, maximum_drop_pp=5., minimum_student_rate=.5):
    """Executable canary gate; corpus semantic judgment is an additional gate."""
    if not 0 <= maximum_drop_pp <= 100 or not 0 <= minimum_student_rate <= 1:
        raise ValueError("Valid frozen gate thresholds required")
    left, right = native["records"], student["records"]
    if (not left or len(left) != len(right) or len({row["id"] for row in left}) != len(left)
            or [row["id"] for row in left] != [row["id"] for row in right]
            or any(a["input_ids"] != b["input_ids"] or a["max_new_tokens"] != b["max_new_tokens"]
                   or a["eos_token_ids"] != b["eos_token_ids"] for a, b in zip(left, right))):
        raise ValueError("Identical paired native/student prompts, budgets and EOS required")
    if any(type(row.get("generation_success")) is not bool for row in [*left, *right]):
        raise ValueError("Executable per-case binary judgments required")
    native_rate = sum(row["generation_success"] for row in left)/len(left)
    student_rate = sum(row["generation_success"] for row in right)/len(right)
    passed = student_rate >= minimum_student_rate and 100*(native_rate-student_rate) <= maximum_drop_pp + 1e-10
    return dict(passed=passed, native_success_rate=native_rate, student_success_rate=student_rate,
                maximum_drop_pp=maximum_drop_pp, minimum_student_rate=minimum_student_rate,
                count=len(left), confirmed=False, semantic_evaluation_required=True,
                scope="Small deterministic canary screen; no population accuracy bound")


def semantic_gate(native_rows, student_rows, judgments, *, judge_provenance,
                  maximum_drop_pp=5., minimum_student_rate=.5):
    """Admit explicit reviewed semantic judgments, never infer them from overlap.

    A judge error/missing judgment fails eligibility. Raw paired outputs and
    verdicts must be saved by the study. This deliberately does not invoke an
    external service or treat exact reference wording as summary correctness.
    """
    required = {"judge_id", "judge_version", "rubric", "prompt_sha256", "settings"}
    if not required <= set(judge_provenance) or not all(judge_provenance[k] for k in required - {"settings"}):
        raise ValueError("Complete semantic judge identity/rubric/settings provenance required")
    if not isinstance(judge_provenance["settings"], dict):
        raise ValueError("Judge settings must be recorded")
    if not native_rows or len(native_rows) != len(student_rows) or len(judgments) != len(native_rows):
        raise ValueError("Complete paired semantic outputs and raw judgments required")
    ids = [row["example_id"] for row in native_rows]
    if len(set(ids)) != len(ids) or ids != [row["example_id"] for row in student_rows] or ids != [row["example_id"] for row in judgments]:
        raise ValueError("Semantic judgment identities/order differ")
    valid = all(row.get("status") == "completed" and all(type(row.get(arm, {}).get(key)) is bool
               for arm in ("native", "student") for key in ("task_success", "generation_success")) for row in judgments)
    if not valid:
        return dict(passed=False, missing_or_failed_judgments=True, confirmed=False)
    rates = {}
    for key in ("task_success", "generation_success"):
        n = sum(row["native"][key] for row in judgments)/len(judgments)
        s = sum(row["student"][key] for row in judgments)/len(judgments)
        rates[key] = dict(native=n, student=s, passed=s >= minimum_student_rate and 100*(n-s) <= maximum_drop_pp + 1e-10)
    return dict(passed=all(x["passed"] for x in rates.values()), rates=rates, confirmed=False,
                count=len(judgments), maximum_drop_pp=maximum_drop_pp, minimum_student_rate=minimum_student_rate,
                judge_provenance=judge_provenance, scope="Paired descriptive DEV semantic screen")
