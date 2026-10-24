"""Reject changed noise, objective routing, populations and semantic gate shortcuts."""
from copy import deepcopy
import json

import pytest
import torch

from experiments.structure_net import ncm_lm1b_word_gradient_development as run
from experiments.neural_architecture_lab import finalize_ncm_lm1b_word_gradient_development as audit


def fixture_trace(path):
    rows = [{"id": str(i)} for i in range(690)]
    manifest = {r["id"]: {"source_tokens": 3 + i % 13} for i, r in enumerate(rows)}
    codes = {r["id"]: [i % 8192] * 641 for i, r in enumerate(rows)}
    sampler = torch.Generator().manual_seed(run.continuation.Config().sampler_seed)
    neighbor = torch.Generator().manual_seed(run.continuation.Config().neighbor_seed)
    trace = []
    for step in range(1, 3):
        micros = []
        for index in range(4):
            selected = [rows[i] for i in torch.randint(690, (4,), generator=sampler).tolist()]
            clean = torch.tensor([codes[r["id"]] for r in selected], dtype=torch.long)
            _, proof = run.continuation._original_objective.corrupt_history(clean, 4096 + step, neighbor)
            loss = {k: 1. for k in ("clean_concept_loss", "clean_word_loss", "noisy_concept_loss", "noisy_word_loss")}
            loss.update(corruption=proof, corruption_schedule_step=4096 + step, word_gradient_auxiliary_executed=index == 0)
            if index == 0:
                loss.update(self_concept_loss=1., self_word_loss=1., source_only_history_codes=clean.tolist(),
                    temperature=1., auxiliary_concept_weight=.25, auxiliary_word_weight=1., word_gradient_into_concept_attention=True)
            micros.append({"case_ids": [r["id"] for r in selected],
                "source_positions": max(manifest[r["id"]]["source_tokens"] for r in selected), "losses": loss})
        trace.append({"step": step, "lr": run.continuation.rate(step), "gradient_norm": 1., "microbatches": micros})
    saved = {"sampler_rng": sampler.get_state(), "neighbor_rng": neighbor.get_state()}
    path.write_text("".join(json.dumps(r) + "\n" for r in trace))
    return rows, manifest, codes, saved, trace


def test_actual_sampler_noise_and_first_micro_history_replay(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows, manifest, codes, saved, _ = fixture_trace(path)
    result = audit.audit_trace(rows, manifest, codes, saved, path, updates=2)
    assert result["sampled_case_presentations"] == 32 and result["source_only_history_microbatches"] == 2
    with pytest.raises(ValueError, match="complete fixed1024"):
        audit.audit_trace(rows, manifest, codes, saved, path)


@pytest.mark.parametrize("change", ("schedule", "routing", "weight", "source", "sample"))
def test_auditor_rejects_mechanism_and_source_trace_tampering(tmp_path, change):
    path = tmp_path / "trace.jsonl"
    rows, manifest, codes, saved, trace = fixture_trace(path)
    micro = trace[0]["microbatches"][0]
    if change == "schedule":
        micro["losses"]["corruption_schedule_step"] = 1
    elif change == "routing":
        micro["losses"]["word_gradient_auxiliary_executed"] = False
    elif change == "weight":
        micro["losses"]["auxiliary_word_weight"] = .25
    elif change == "source":
        micro["source_positions"] -= 1
    else:
        micro["case_ids"][0] = "not-the-sampled-source"
    path.write_text("".join(json.dumps(r) + "\n" for r in trace))
    with pytest.raises(ValueError):
        audit.audit_trace(rows, manifest, codes, saved, path, updates=2)


def test_same_cases_under_both_judges_and_source_only_boundary():
    scores = [{"id": str(i), "valid": True, "passed": i >= 3} for i in range(64)]
    other = deepcopy(scores)
    for i, row in enumerate(other):
        row["passed"] = i not in (3, 4, 5)
    assert run.joint_gate(scores, scores)["qualified"]
    assert run.joint_gate(scores, other)["joint_passed"] == 58
    assert not run.joint_gate(scores, other)["qualified"]
    for field in run.FORBIDDEN:
        with pytest.raises(ValueError, match="source metadata only"):
            run.predict(None, {field: []}, {}, None)


def test_latest2728_cache_and_unchanged_source_population():
    run.configure()
    assert len(run.base_cache()) == 2728
    assert run.validate_sources(run.q.read(run.SOURCE / "cases.json"))
    assert run.file_hash(run.SOURCE / "cases.json") == run.SOURCE_SHA
    assert run.file_hash(run.FEATURES / "manifest.json") == run.FEATURE_SHA
