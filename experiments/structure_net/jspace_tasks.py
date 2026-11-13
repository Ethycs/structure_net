"""Bounded task and intervention helpers for the Jacobian-lens pilot.

The data adapter consumes the released ``data/experiments/probe-swap.json``
from https://github.com/anthropics/jacobian-lens (Apache-2.0). It does not fit
the lens. In the experiment, concept directions must be actual rows of
``W_U @ J_bar`` at the intervention layer, not independently trained probes.

All direction banks below have shape [number_of_directions, hidden_width],
and activations have shape [..., hidden_width]. A coefficient-coordinate swap
and a dot-product-score clamp are deliberately different operations when the
bank is nonorthogonal. The former reproduces the paper's intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Callable, Mapping, Sequence

import torch


SCHEMA_VERSION = "jspace.tasks.v1"
UPSTREAM_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
UPSTREAM_DATA = (
    "https://github.com/anthropics/jacobian-lens/blob/"
    + UPSTREAM_REVISION + "/data/experiments/probe-swap.json"
)


@dataclass(frozen=True)
class TwoHopCase:
    case_id: str
    category: str
    prompt: str
    intermediate: str
    answer: str
    swap_to: str
    swap_answer: str

    @property
    def pair_group(self) -> tuple[str, str]:
        """Both directions and all questions about a concept pair stay together."""
        return tuple(sorted((self.intermediate.casefold(), self.swap_to.casefold())))

    def to_dict(self) -> dict:
        return asdict(self)


def load_cases(path: str | Path) -> list[TwoHopCase]:
    """Load the pinned published file; retain exact prompts and answer labels.

    Filtering is a separate, recorded operation. The file contains ground-truth
    swapped answers, not clean donor prompts. No counterfactual wording is guessed.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    cases = []
    for row in rows:
        values = {key: row[key] for key in (
            "category", "prompt", "intermediate", "answer", "swap_to", "swap_answer"
        )}
        if not all(isinstance(v, str) and v.strip() for v in values.values()):
            raise ValueError("Every case needs nonempty prompt, concept and answer strings")
        case = TwoHopCase(case_id=row["name"], **values)
        if not case.case_id or case.intermediate.casefold() == case.swap_to.casefold():
            raise ValueError("Case identifiers and distinct intermediate concepts are required")
        if normalize_answer(case.answer) == normalize_answer(case.swap_answer):
            raise ValueError("A functional swap must change the ground-truth answer")
        cases.append(case)
    _unique_cases(cases)
    return cases


def _unique_cases(cases: Sequence[TwoHopCase]) -> None:
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Repeated case identifiers would duplicate evidence")


def lexical_contamination(case: TwoHopCase) -> tuple[str, ...]:
    """Flag literal concept/answer names in a source prompt, by whole words.

    This only rules out direct lexical copying; it cannot prove that a model
    actually computes an intermediate rather than recalling an association.
    """
    present = []
    for field in ("intermediate", "swap_to", "answer", "swap_answer"):
        value = getattr(case, field)
        if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", case.prompt, re.IGNORECASE):
            present.append(field)
    return tuple(present)


def eligible_cases(
    cases: Sequence[TwoHopCase],
    encode: Callable[[str], Sequence[int]] | None = None,
    *,
    reject_lexical_contamination: bool = True,
) -> tuple[list[TwoHopCase], list[dict]]:
    """Filter before any model outcomes, returning an explicit exclusion ledger.

    ``encode`` must disable special tokens. Concepts use their leading-space
    tokenization. Answers may be multi-token: the runner must evaluate actual
    greedy text and can additionally report sequence likelihoods.
    """
    _unique_cases(cases)
    accepted, excluded = [], []
    for case in cases:
        reasons = []
        if reject_lexical_contamination:
            reasons.extend("literal_" + name + "_in_prompt" for name in lexical_contamination(case))
        ids = {}
        if encode is not None:
            for field in ("intermediate", "swap_to"):
                token_ids = tuple(encode(" " + getattr(case, field)))
                ids[field] = token_ids
                if len(token_ids) != 1:
                    reasons.append(field + "_not_single_token")
            if ids.get("intermediate") == ids.get("swap_to"):
                reasons.append("concepts_share_token_id")
        if reasons:
            excluded.append({"case_id": case.case_id, "reasons": reasons,
                             "concept_token_ids": {k: list(v) for k, v in ids.items()}})
        else:
            accepted.append(case)
    return accepted, excluded


def split_cases(
    cases: Sequence[TwoHopCase], *, seed: int = 7,
    fit_count: int = 8, development_count: int = 12, evaluation_count: int = 48,
) -> dict[str, list[TwoHopCase]]:
    """Deterministically select one independent item per unordered concept pair.

    Individual concept names can recur across splits; concept *combinations*
    cannot. Extra wording variants of a selected pair are not counted as extra
    independent trials. ``fit`` is for control/PCA fitting, not Jacobian fitting;
    the latter needs a separate generic corpus. Exact requested counts are
    enforced, so tokenizer exclusions cannot silently shrink a reported pilot.
    """
    _unique_cases(cases)
    counts = (fit_count, development_count, evaluation_count)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Split counts must be nonnegative integers")

    def order_key(value) -> str:
        return hashlib.sha256(json.dumps([seed, value], sort_keys=True).encode()).hexdigest()

    groups: dict[tuple[str, str], list[TwoHopCase]] = {}
    for case in cases:
        groups.setdefault(case.pair_group, []).append(case)
    keys = sorted(groups, key=order_key)
    if sum(counts) > len(keys):
        raise ValueError(f"Requested {sum(counts)} independent pairs; only {len(keys)} eligible")
    selected = [min(groups[key], key=lambda c: order_key(c.case_id)) for key in keys]
    a, b, c = fit_count, fit_count + development_count, sum(counts)
    return {"fit": selected[:a], "development": selected[a:b], "evaluation": selected[b:c]}


def split_manifest(splits: Mapping[str, Sequence[TwoHopCase]]) -> dict:
    flat = [case for cases in splits.values() for case in cases]
    _unique_cases(flat)
    groups = [case.pair_group for case in flat]
    if len(set(groups)) != len(groups):
        raise ValueError("Independent splits cannot repeat a concept-pair group")
    payload = {name: [case.to_dict() for case in cases] for name, cases in splits.items()}
    return {"schema_version": SCHEMA_VERSION, "source": UPSTREAM_DATA,
            "independent_unit": "unordered_intermediate_pair",
            "counts": {name: len(cases) for name, cases in splits.items()},
            "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "splits": payload}


_NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    map(str, range(21)),
))


def normalize_answer(text: str) -> str:
    """Conservative whole-answer matching, with spelled small integer equivalence.

    Explanations, substring matches, and partial multi-token completions are not
    accepted. Punctuation and case do not distinguish otherwise equal answers.
    """
    answer = " ".join(text.strip().casefold().split()).strip(" \t\r\n.,:;!?\"'`")
    return _NUMBER_WORDS.get(answer, answer)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Wilson interval needs integer counts with 0 <= successes <= total")
    if not total:
        return {"successes": 0, "total": 0, "accuracy": None, "ci95": None}
    p = successes / total
    scale = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return {"successes": successes, "total": total, "accuracy": p,
            "ci95": [max(0.0, centre - radius), min(1.0, centre + radius)]}


def score_predictions(
    cases: Sequence[TwoHopCase], baseline: Mapping[str, str], predictions: Mapping[str, str],
    *, target: str = "counterfactual",
) -> dict:
    """Report raw and parent-correct-conditioned accuracy without selection bias.

    The same preselected cases must be present in both maps. Counterfactual
    success means the *ground-truth swap answer*, not merely any output change.
    For retained-space/control arms, use ``target='original'``.
    """
    if target not in ("original", "counterfactual"):
        raise ValueError("Target must be original or counterfactual")
    _unique_cases(cases)
    ids = {case.case_id for case in cases}
    if set(baseline) != ids or set(predictions) != ids:
        raise ValueError("Predictions must cover exactly the prespecified case IDs")
    if len({case.pair_group for case in cases}) != len(cases):
        raise ValueError("Repeated pair groups cannot be treated as independent trials")
    rows = []
    for case in cases:
        expected = case.swap_answer if target == "counterfactual" else case.answer
        parent_ok = normalize_answer(baseline[case.case_id]) == normalize_answer(case.answer)
        correct = normalize_answer(predictions[case.case_id]) == normalize_answer(expected)
        rows.append({"case_id": case.case_id, "pair_group": list(case.pair_group),
                     "baseline_correct": parent_ok, "correct": correct,
                     "baseline": baseline[case.case_id], "prediction": predictions[case.case_id],
                     "expected": expected})
    n_parent = sum(row["baseline_correct"] for row in rows)
    return {"target": target, "independent_unit": "unordered_intermediate_pair",
            "baseline_accuracy": wilson_interval(n_parent, len(rows)),
            "raw_accuracy": wilson_interval(sum(row["correct"] for row in rows), len(rows)),
            "parent_correct_accuracy": wilson_interval(
                sum(row["correct"] and row["baseline_correct"] for row in rows), n_parent),
            "cases": rows}


def paired_difference(first: Sequence[bool], second: Sequence[bool], *, seed: int = 7,
                      resamples: int = 2000) -> dict:
    """Exploratory paired bootstrap CI; independent units must be concept pairs."""
    if len(first) != len(second) or not first or resamples < 100:
        raise ValueError("Equal nonempty paired outcomes and at least 100 resamples required")
    if any(type(value) is not bool for value in (*first, *second)):
        raise ValueError("Paired outcomes must be bools")
    delta = [int(b) - int(a) for a, b in zip(first, second)]
    n = len(delta)
    rng = random.Random(seed)
    estimates = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    improved = sum(a is False and b is True for a, b in zip(first, second))
    worsened = sum(a is True and b is False for a, b in zip(first, second))
    discordant = improved + worsened
    exact_p = (min(1.0, 2 * sum(math.comb(discordant, i)
                               for i in range(min(improved, worsened) + 1)) / 2**discordant)
               if discordant else 1.0)
    return {"second_minus_first": sum(delta) / n,
            "bootstrap_ci95": [estimates[int(.025 * (resamples - 1))],
                               estimates[int(.975 * (resamples - 1))]],
            "improved": improved, "worsened": worsened, "n": n,
            "mcnemar_exact_p": exact_p, "resamples": resamples, "seed": seed,
            "confirmatory": False}


def _bank(bank: torch.Tensor) -> torch.Tensor:
    if bank.ndim != 2 or not bank.is_floating_point() or not bool(torch.isfinite(bank).all()):
        raise ValueError("Direction bank must be a finite floating [directions, hidden] tensor")
    if bank.shape[1] < 1:
        raise ValueError("Hidden width must be positive")
    return bank.to(dtype=torch.float64 if bank.dtype == torch.float64 else torch.float32)


def _activation(h: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    if h.ndim < 1 or h.shape[-1] != bank.shape[1] or h.device != bank.device:
        raise ValueError("Activations must share the bank's hidden width and device")
    if not h.is_floating_point() or not bool(torch.isfinite(h).all()):
        raise ValueError("Activations must be finite floating tensors")
    return h.to(bank.dtype)


def orthonormal_basis(bank: torch.Tensor, *, rtol: float = 1e-6) -> torch.Tensor:
    """Return orthonormal *rows* spanning the numerical row space of a J bank."""
    matrix = _bank(bank)
    if not matrix.shape[0]:
        return matrix
    _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((s > rtol * s[0]).sum()) if s[0] > 0 else 0
    return vh[:rank]


def project_to_span(h: torch.Tensor, bank: torch.Tensor, *, rtol: float = 1e-6) -> torch.Tensor:
    basis = orthonormal_basis(bank, rtol=rtol)
    work = _activation(h, basis)
    return ((work @ basis.T) @ basis).to(h.dtype)


def coordinate_swap(h: torch.Tensor, bank: torch.Tensor, source: int = 0, target: int = 1,
                    *, alpha: float = 1.0, rtol: float = 1e-6) -> torch.Tensor:
    """Paper swap: c=V†h; h'=h+alpha*V*(swap(c)-c), V=bank.T.

    Only the two named directions participate. Reject a dependent pair because
    their two coefficients are not independently identifiable. The orthogonal
    complement is preserved, including for nonorthogonal direction pairs.
    """
    full = _bank(bank)
    if source == target or not 0 <= source < len(full) or not 0 <= target < len(full):
        raise ValueError("Swap needs two distinct valid bank indices")
    if not math.isfinite(alpha):
        raise ValueError("Swap strength must be finite")
    pair = full[[source, target]]
    if len(orthonormal_basis(pair, rtol=rtol)) != 2:
        raise ValueError("Swap directions are linearly dependent at the declared tolerance")
    work = _activation(h, pair)
    coordinates = work @ torch.linalg.pinv(pair, rtol=rtol)
    delta = (coordinates.flip(-1) - coordinates) @ pair
    return (work + alpha * delta).to(h.dtype)


def replace_scores(h: torch.Tensor, desired: torch.Tensor, bank: torch.Tensor,
                   *, rtol: float = 1e-6, atol: float = 1e-6) -> torch.Tensor:
    """Minimum-norm change to enforce dot-product scores h @ bank.T = desired.

    This is a score clamp, NOT the paper's coefficient-coordinate swap.
    Inconsistent targets in a rank-deficient bank are rejected explicitly.
    """
    matrix = _bank(bank)
    work = _activation(h, matrix)
    desired = desired.to(device=matrix.device, dtype=matrix.dtype)
    if desired.shape != (*work.shape[:-1], matrix.shape[0]) or not bool(torch.isfinite(desired).all()):
        raise ValueError("Target scores must match the activation batch and bank length")
    correction = (desired - work @ matrix.T) @ torch.linalg.pinv(matrix, rtol=rtol).T
    result = work + correction
    if not torch.allclose(result @ matrix.T, desired, atol=atol, rtol=rtol):
        raise ValueError("Requested scores are inconsistent with this direction bank")
    return result.to(h.dtype)


def complement_swap(source: torch.Tensor, donor: torch.Tensor, bank: torch.Tensor,
                    *, rtol: float = 1e-6) -> torch.Tensor:
    """Replace the orthogonal complement while preserving source J-bank scores."""
    matrix = _bank(bank)
    src, dst = _activation(source, matrix), _activation(donor, matrix)
    if src.shape != dst.shape:
        raise ValueError("Source and donor activations must have equal shapes")
    basis = orthonormal_basis(matrix, rtol=rtol)
    keep = (src @ basis.T) @ basis
    complement = dst - (dst @ basis.T) @ basis
    return (keep + complement).to(source.dtype)


def random_basis(hidden_width: int, rank: int, *, seed: int = 7,
                 dtype: torch.dtype = torch.float32, device="cpu") -> torch.Tensor:
    """Fixed-seed orthonormal row control with the same numerical rank."""
    if not 0 <= rank <= hidden_width or hidden_width < 1:
        raise ValueError("Control rank must lie between zero and hidden width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    sample = torch.randn(hidden_width, rank, generator=generator, dtype=work_dtype)
    q, _ = torch.linalg.qr(sample, mode="reduced")
    return q.T.to(device=device, dtype=dtype)


def pca_basis(fit_activations: torch.Tensor, rank: int, *, rtol: float = 1e-6) -> torch.Tensor:
    """PCA control fitted ONLY on a disjoint fit split, with exact requested rank.

    Centering is used to estimate axes; runtime projection remains origin-based,
    matching J-bank projection. The caller must record the fit split identity.
    """
    matrix = _bank(fit_activations)
    if len(matrix) < 2 or not 0 <= rank <= min(matrix.shape):
        raise ValueError("PCA requires at least two fit activations and a valid rank")
    _, s, vh = torch.linalg.svd(matrix - matrix.mean(0), full_matrices=False)
    available = int((s > rtol * s[0]).sum()) if s[0] > 0 else 0
    if rank > available:
        raise ValueError(f"Requested PCA rank {rank}, but calibration rank is {available}")
    return vh[:rank].to(fit_activations.dtype)
