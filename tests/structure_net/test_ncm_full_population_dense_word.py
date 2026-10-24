from dataclasses import asdict, replace
import pytest
import torch

from experiments.structure_net import ncm_full_population_dense_word as run
from experiments.neural_architecture_lab import finalize_ncm_full_population_dense_word as audit


def test_full_population_budget_rates_and_seed_separation():
    cfg = run.Config()
    assert cfg.updates == 32768 and cfg.model_seed != cfg.sampler_seed
    assert cfg.latent_weight == 1 and cfg.word_weight == .1
    assert run.rate(cfg.warmup) == cfg.max_lr
    assert run.rate(cfg.updates) == pytest.approx(cfg.min_lr)
    with pytest.raises(ValueError, match="configuration"):
        replace(cfg, model_seed=cfg.sampler_seed)


def test_invalid_config_and_pending_model_cannot_enter_cuda():
    with pytest.raises(ValueError, match="configuration"):
        replace(run.Config(), batch=2)
    with pytest.raises(ValueError, match="CPU only"):
        run.model(device="cuda", pending=True)


def test_completed_requires_every_observed_exit_and_exact_registry(tmp_path):
    workflow, results = tmp_path / "workflow", tmp_path / "results"
    workflow.mkdir(); results.mkdir()
    assert run.completed(workflow, results, "test", "done", ("train", "register")) is False
    path = results / "campaign_results.json"
    run.write_json(path, {"component_gate": False})
    receipt = {"status": "registered_and_read_back", "record": {"campaign_sha256": run.file_hash(path)},
        "exact": {"campaign_sha256": run.file_hash(path)}}
    run.write_json(results / "test_registry_readback.json", receipt)
    done = {"status": "done", "stages": [{"stage": s, "exit_code": 0} for s in ("train", "register")]}
    run.write_json(workflow / "result.json", done)
    assert run.completed(workflow, results, "test", "done", ("train", "register")) is True
    run.write_json(workflow / "result.json", {**done, "stages": done["stages"][:-1]})
    with pytest.raises(ValueError, match="lifecycle"):
        run.completed(workflow, results, "test", "done", ("train", "register"))
    run.write_json(workflow / "failure.json", {"status": "failed"})
    with pytest.raises(RuntimeError, match="failure preserved"):
        run.completed(workflow, results, "test", "done", ("train", "register"))


def test_registered_latent_source_word_pairing_and_finite_shape(tmp_path):
    row = {"id": "case", "source_sha256": "source", "target_text": "Summary", "word_ids": [2, 50256] + [50257] * 190}
    path, measured = tmp_path / "label.pt", tmp_path / "measurement.json"
    torch.save({"refined": torch.zeros(1, 192, 384)}, path)
    run.write_json(measured, {k: row[k] for k in ("id", "source_sha256", "target_text")}
        | {"target_word_ids": row["word_ids"]})

    def bind():
        entry = {"state": str(path), "measurement": str(measured), "state_sha256": run.file_hash(path),
            "measurement_sha256": run.file_hash(measured), "updates": 256,
            "target_word_sha256": run.clean.recovery.study.base.tensor_digest({"target": torch.tensor([row["word_ids"]])})}
        campaign = {"artifacts": {str(path): entry["state_sha256"], str(measured): entry["measurement_sha256"]}}
        return entry, campaign

    labels = run.validate_label(row, *bind())
    assert labels["refined"].shape == (1, 192, 384) and not labels["refined"].requires_grad
    with pytest.raises(ValueError, match="source and summary"):
        run.validate_label({**row, "source_sha256": "other"}, *bind())
    for tensor in (torch.zeros(1, 48, 384), torch.zeros(1, 192, 384, dtype=torch.float16),
            torch.full((1, 192, 384), float("inf"))):
        torch.save({"refined": tensor}, path)
        with pytest.raises(ValueError, match="FP32"):
            run.validate_label(row, *bind())


def test_labels_are_detached_and_actual_weighted_loss_is_audited():
    latent = torch.zeros(1, 2, 3, requires_grad=True)
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    teacher = torch.ones_like(latent, requires_grad=True)
    loss, parts = run.core.objective(latent, logits, teacher, torch.tensor([[1, 2]]))
    loss.backward()
    assert teacher.grad is None
    assert latent.grad.abs().sum() > 0 and logits.grad.abs().sum() > 0
    record = {**parts, "objective_loss": float(loss.detach())}
    audit.objective_receipts([record], run.Config())
    with pytest.raises(ValueError, match="weighted"):
        audit.objective_receipts([{**record, "objective_loss": record["objective_loss"] + 1}], run.Config())


def test_every_training_sample_history_and_optimizer_step_replays(monkeypatch):
    config = replace(run.Config(), updates=8, warmup=1, snapshot_interval=4)
    monkeypatch.setattr(run, "Config", lambda: config)
    monkeypatch.setattr(audit.reference, "run", run)
    rows = [{"id": str(i), "source_ids": [i + 1] * (i + 1), "source_frame": 256 * (i + 1),
        "word_ids": [i + 1, 50256] + [50257] * 190} for i in range(2)]
    histories = {r["id"]: {"coarse_codes": [int(r["id"])] * 608} for r in rows}
    labels = {r["id"]: {"refined": torch.zeros(1, 192, 384)} for r in rows}
    rng = torch.Generator().manual_seed(config.sampler_seed)
    trace = []
    for step in range(1, 9):
        row = rows[int(torch.randint(2, (1,), generator=rng))]
        hashes = {"actual_history_sha256": run.clean.recovery.study.base.tensor_digest({"codes": torch.tensor([histories[row["id"]]["coarse_codes"]])}),
            "target_word_sha256": run.clean.recovery.study.base.tensor_digest({"words": torch.tensor([row["word_ids"]])}),
            "teacher_latent_sha256": run.clean.recovery.study.base.tensor_digest({"latent": labels[row["id"]]["refined"]})}
        trace.append({"step": step, "id": row["id"], "source_frame": row["source_frame"], "lr": run.rate(step),
            **hashes, "latent_MSE": .5, "word_CE": 2., "objective_loss": .7, "gradient_norm": 1.})
    name = "source_head.source_projection.weight"
    state = {"config": asdict(config), "step": 8, "sampler_rng": rng.get_state(),
        "parameter_names": [name], "trainable_parameter_names": [name], "model": {name: torch.zeros(1)},
        "optimizer": {"param_groups": [{"params": [0], "betas": (config.beta1, config.beta2),
            "eps": config.optimizer_eps, "weight_decay": config.weight_decay, "lr": run.rate(8)}],
            "state": {0: {"step": torch.tensor(8.), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}}}}
    replay = audit.reference.replay_trace(rows, histories, labels, trace, state)
    assert set(replay["sample_counts"]) == {"0", "1"}
    altered = [dict(r) for r in trace]
    altered[0]["actual_history_sha256"] = "changed"
    with pytest.raises(ValueError, match="history"):
        audit.reference.replay_trace(rows, histories, labels, altered, state)


def test_pending_producer_binds_fixed_inputs_without_hashing_future_outputs(tmp_path, monkeypatch):
    config = run.Config()
    first = run.producing(tmp_path / "first", "systems_producer_contract", pending=True, identities={})
    monkeypatch.setattr(run, "Config", lambda: replace(config, sampler_seed=57142))
    changed = run.producing(tmp_path / "changed", "systems_producer_contract", pending=True, identities={})
    assert first["fingerprint"] != changed["fingerprint"]
    assert first["config"]["TRAIN_cases"] == 690
    assert first["config"]["initial_word_checkpoint_sha256"] == run.INITIAL_WORD
    assert first["input_hashes"][str(run.INITIAL_PATH)] == run.INITIAL_WORD
    assert str(run.forecaster.CACHE / "histories.json") not in first["input_hashes"]
    assert str(run.forecaster.TRAIN / "model.pt") not in first["input_hashes"]
    assert str(run.WORKFLOW / "admission.json") not in first["input_hashes"]
    run.clean.recovery.study.data.verify(first)
