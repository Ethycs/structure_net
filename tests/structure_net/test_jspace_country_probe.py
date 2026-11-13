"""Known-signal controls for the cached-activation probe experiment."""
import numpy as np
import pytest
import torch

from experiments.structure_net.jspace_country_probe import (
    RidgeFamily, features, pack, paired_difference, predict, score_predictions, select_layer, unpack,
)


def test_ridge_matches_direct_solution_and_uses_training_mean():
    rng = np.random.default_rng(4)
    train, dev = rng.normal(size=(40, 7)), rng.normal(size=(11, 7))+2
    train_y, dev_y = np.arange(40) % 4, np.arange(11) % 4
    penalty = .3
    head, _ = RidgeFamily(train, dev).select(train_y, dev_y, [penalty])
    centered = train-train.mean(0)
    y = np.eye(4)[train_y]
    scale = np.sum(centered**2)/len(train)
    weight = np.linalg.solve(centered.T@centered+penalty*scale*np.eye(7), centered.T@(y-y.mean(0)))
    np.testing.assert_allclose(head["mean"], train.mean(0))
    np.testing.assert_allclose(head["weight"], weight, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(predict(head, dev), (dev-train.mean(0))@weight+y.mean(0))


def test_known_country_signal_is_removed_by_complement():
    rng = np.random.default_rng(19)
    y = np.tile(np.arange(4), 80)
    hidden = np.column_stack((np.eye(4)[y]*8, rng.normal(size=(320, 6))))
    fitted = {"jacobian": np.eye(10)[:4]}
    projected = features(hidden, fitted, "J4")
    residual = features(hidden, fitted, "remainder")
    np.testing.assert_allclose(residual[:, :4], 0)
    head, _ = RidgeFamily(projected[:240], projected[240:280]).select(y[:240], y[240:280])
    assert score_predictions(predict(head, projected[280:]), y[280:])["accuracy"] == 1
    assert np.max(np.abs(residual+projected@fitted["jacobian"]-hidden)) == 0


def test_layer_selection_uses_full_head_development_with_tiebreaks():
    def row(acc, mse):
        return {"full": {"selected": {"accuracy": acc, "mse": mse}}}
    assert select_layer({"11": row(.8, .2), "24": row(.9, .3), "12": row(.9, .1)}) == "12"
    assert select_layer({"24": row(.9, .1), "12": row(.9, .1)}) == "12"


def test_confusion_and_paired_comparison():
    truth = np.tile(np.arange(4), 20)
    scores = np.eye(4)[truth]
    metrics = score_predictions(scores, truth)
    assert metrics["correct"] == 80
    assert np.diag(metrics["confusion_true_rows_predicted_columns"]).tolist() == [20]*4
    assert metrics["wilson_95"][0] > .9
    contrast = paired_difference(truth, (truth+1) % 4, truth)
    assert contrast["difference"] == 1
    assert contrast["bootstrap_95"] == [1, 1]


def test_saved_heads_reproduce_predictions(tmp_path):
    rng = np.random.default_rng(13)
    x = rng.normal(size=(32, 12))
    y = np.arange(32) % 4
    head, _ = RidgeFamily(x[:24], x[24:]).select(y[:24], y[24:])
    path = tmp_path/"head.pt"
    torch.save(pack({"heads": [head]}), path)
    restored = unpack(torch.load(path, map_location="cpu", weights_only=True))["heads"][0]
    np.testing.assert_array_equal(predict(head, x), predict(restored, x))


def test_invalid_arrays_labels_and_penalties_fail():
    with pytest.raises(ValueError, match="finite"):
        RidgeFamily(np.full((4, 2), np.nan), np.ones((4, 2)))
    with pytest.raises(ValueError, match="Degenerate"):
        RidgeFamily(np.ones((4, 2)), np.ones((4, 2)))
    x = np.arange(24).reshape(8, 3)
    family = RidgeFamily(x, x)
    with pytest.raises(ValueError, match="integer country"):
        family.select(np.ones(8)*4, np.arange(8) % 4)
    with pytest.raises(ValueError, match="positive"):
        family.select(np.arange(8) % 4, np.arange(8) % 4, [-1])
