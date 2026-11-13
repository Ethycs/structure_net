"""CPU fitting of paired semantic directions and matched-input task heads.

The low-residual polynomial head is gradient-normalized and class conditional.
It is an approximate-vanishing-feature experiment, not an exact Gröbner basis
or a replication of VI-Net's full training pipeline.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from .jspace_active_math import encode, head_scores, quadratic, tensor_bytes
from .jspace_country_probe import RidgeFamily
from .jspace_residual_expansion import nonnegative_pursuit, orthogonal_rows
from .jspace_relation_study import file_hash, save_tensor, write_json
from .jspace_relation_tasks import COUNTRIES, COLOURS


def target_labels(pairs, side, field):
    classes = COUNTRIES if field == "intermediate" else COLOURS
    return np.array([classes.index(pair[side][field]) for pair in pairs], dtype=np.int64)


def paired_concepts(source, donor, pairs):
    eye = np.eye(4)
    country = eye[target_labels(pairs, "donor", "intermediate")]-eye[target_labels(pairs, "source", "intermediate")]
    colour = eye[target_labels(pairs, "donor", "answer")]-eye[target_labels(pairs, "source", "answer")]
    design = np.concatenate((country, colour), axis=1)
    if np.linalg.matrix_rank(design) != 6:
        raise ValueError("Country and colour contrasts are not identifiable on the declared six-dimensional contrast space")
    effects = np.linalg.lstsq(design, donor-source, rcond=1e-8)[0]
    effects[:4] -= effects[:4].mean(0)
    effects[4:] -= effects[4:].mean(0)
    return effects[:4], effects[4:], {"design_rank": 6,
        "paired_difference_energy_explained": float(1-np.sum((donor-source-design@effects)**2)/np.sum((donor-source)**2))}


def common_encoder(values, country, colour, dimension=8):
    mean = values.mean(0)
    centered = values-mean
    basis = orthogonal_rows(np.concatenate((country, colour)))
    if len(basis) > dimension:
        raise ValueError("Semantic basis exceeds frozen encoder width")
    semantic_rank = len(basis)
    residual = centered-(centered@basis.T)@basis
    eigenvalues, vectors = np.linalg.eigh(residual@residual.T)
    for i in range(len(eigenvalues)-1, -1, -1):
        if len(basis) == dimension:
            break
        if eigenvalues[i] <= 1e-10:
            continue
        vector = vectors[:, i]@residual/np.sqrt(eigenvalues[i])
        for _ in range(2):
            vector -= (vector@basis.T)@basis
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            basis = np.concatenate((basis, (vector/norm)[None, :]))
    if len(basis) != dimension:
        raise ValueError("Insufficient training rank for declared encoder")
    scale = (centered@basis.T).std(0)
    if np.min(scale) <= 1e-8:
        raise ValueError("Degenerate encoder scale")
    return {"mean": torch.tensor(mean, dtype=torch.float32), "basis": torch.tensor(basis, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32)}, {"semantic_rank": semantic_rank,
            "width": dimension, "training_energy_fraction": float(np.sum((centered@basis.T)**2)/np.sum(centered**2))}


def polynomial_features(values, labels, *, per_class=4, candidates=8):
    """Generalized low-residual eigenvectors with unit class gradient RMS."""
    values = np.asarray(values, dtype=np.float64)
    n, width = values.shape
    phi = quadratic(torch.from_numpy(values)).numpy()
    gradients = np.zeros((n, phi.shape[1]-1, width))
    gradients[:, :width] = np.eye(width)[None, :]
    offset = width
    for i in range(width):
        for j in range(i, width):
            gradients[:, offset, i] += values[:, j]
            gradients[:, offset, j] += values[:, i]
            offset += 1
    polynomials, diagnostics = [], []
    for label in range(4):
        mask = labels == label
        if mask.sum() < phi.shape[1]:
            raise ValueError("Need more class examples than polynomial terms")
        mean = phi[mask, 1:].mean(0)
        features = phi[mask, 1:]-mean
        a = features.T@features/mask.sum()
        b = np.einsum("nki,nli->kl", gradients[mask], gradients[mask])/mask.sum()
        eigenvalues, vectors = np.linalg.eigh(b)
        keep = eigenvalues > max(1e-12, eigenvalues.max()*1e-10)
        whiten = vectors[:, keep]/np.sqrt(eigenvalues[keep])[None, :]
        _, eigenvectors = np.linalg.eigh(whiten.T@a@whiten)
        coefficients = (whiten@eigenvectors[:, :candidates]).T
        coefficients = np.column_stack((-coefficients@mean, coefficients))
        candidates_scored = []
        for index, polynomial in enumerate(coefficients):
            own = float(np.sqrt(np.mean((phi[mask]@polynomial)**2)))
            other = float(np.sqrt(np.mean((phi[~mask]@polynomial)**2)))
            grad = float(np.sqrt(polynomial[1:]@b@polynomial[1:]))
            if not np.isfinite(grad) or abs(grad-1) > 1e-6:
                raise ValueError("Gradient normalization failed")
            candidates_scored.append({"class": label, "index": index, "class_gradient_rms": grad,
                                      "class_residual_rms": own, "other_class_residual_rms": other,
                                      "contrast": other/max(own, 1e-8)})
        selected = sorted(candidates_scored, key=lambda row: (-row["contrast"], row["class_residual_rms"], row["index"]))[:per_class]
        polynomials.extend(coefficients[row["index"]] for row in selected)
        diagnostics.extend(selected)
    return torch.tensor(np.asarray(polynomials), dtype=torch.float32), diagnostics


def fit_ridge(training, development, train_labels, dev_labels):
    fitted, selection = RidgeFamily(training.double().numpy(), development.double().numpy()).select(train_labels, dev_labels)
    return {"feature_mean": torch.from_numpy(fitted["mean"]).float(),
            "weight": torch.from_numpy(fitted["weight"]).float(), "prior": torch.from_numpy(fitted["prior"]).float()}, selection


def fit_mlp(training, development, train_labels, dev_labels, seed):
    train_y, dev_y = torch.from_numpy(train_labels), torch.from_numpy(dev_labels)
    best, logs = None, []
    for decay in (0., .001, .01):
        torch.manual_seed(seed)
        network = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.GELU(), torch.nn.Linear(64, 4))
        optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=decay)
        for epoch in range(1, 301):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(network(training), train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP training loss")
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0:
                with torch.no_grad():
                    logits = network(development)
                    correct = int((logits.argmax(1) == dev_y).sum())
                    nll = float(torch.nn.functional.cross_entropy(logits, dev_y))
                logs.append({"weight_decay": decay, "epoch": epoch, "correct": correct, "nll": nll})
                key = (-correct, nll, epoch, decay)
                if best is None or key < best[0]:
                    best = (key, copy.deepcopy(network.state_dict()), logs[-1])
    return best[1], {"selected": best[2], "candidates": logs}


def fit(root):
    from .jspace_active_study import CONFIG, verify, checked_manifest, cpu_only
    cpu_only()
    torch.set_num_threads(4)
    plan = verify(root)
    manifest = checked_manifest(root, "training")
    if (root/"fit.pt").exists():
        raise FileExistsError("Do not overwrite frozen fits")
    started = time.perf_counter()
    tasks = json.loads((root/"tasks.json").read_text())
    training = torch.load(root/"training/train.pt", map_location="cpu", weights_only=True)
    development = torch.load(root/"training/development.pt", map_location="cpu", weights_only=True)
    for split, values in (("train", training), ("development", development)):
        records = json.loads((root/f"training/{split}.json").read_text())
        pairs = tasks["pairs"][split]
        for side in ("source", "donor"):
            if [r[side]["case_id"] for r in records] != [p[side]["case_id"] for p in pairs]:
                raise ValueError("Fitting record/case alignment changed")
            if any(len(tensor) != len(pairs) for tensor in values[side].values()):
                raise ValueError("Fitting feature count changed")
    dictionaries = torch.load(root/"training/dictionary.pt", map_location="cpu", weights_only=True)
    fits, heads, diagnostics, selections = {}, {}, {}, {}
    train_y = target_labels(tasks["pairs"]["train"], "source", "answer")
    dev_y = target_labels(tasks["pairs"]["development"], "source", "answer")
    for key in CONFIG["sites"]:
        layer = int(key)
        source = training["source"][key].double().numpy()
        donor = training["donor"][key].double().numpy()
        country, colour, diagnostic = paired_concepts(source, donor, tasks["pairs"]["train"])
        dictionary = dictionaries[layer].double().numpy()
        norms = np.linalg.norm(dictionary, axis=1)
        if np.min(norms) <= 1e-12:
            raise ValueError("Zero task-Jacobian dictionary row")
        dictionary /= norms[:, None]
        initial = [plan["token_ids"].index(i) for i in plan["country_token_ids"]]
        sparse = nonnegative_pursuit(country, dictionary, initial, budgets=(4, 25, len(dictionary)))
        jpart = sparse["coefficients"][25]@dictionary
        generator = torch.Generator().manual_seed(CONFIG["random_seed"]+layer)
        fits[key] = {"full": torch.tensor(country).float(), "colour": torch.tensor(colour).float(),
                     "J": torch.tensor(jpart).float(), "remainder": torch.tensor(country-jpart).float(),
                     "country_inverse": torch.linalg.pinv(torch.tensor(country), rtol=1e-6).float(),
                     "colour_inverse": torch.linalg.pinv(torch.tensor(colour), rtol=1e-6).float(),
                     "permutation": torch.randperm(source.shape[1], generator=generator),
                     "sign": (2*torch.randint(2, (source.shape[1],), generator=generator)-1).float()}
        encoder, encoded_diagnostic = common_encoder(source, country, colour)
        train_tensor = training["source"][key].float()
        dev_tensor = development["source"][key].float()
        x, d = encode(encoder, train_tensor), encode(encoder, dev_tensor)
        polynomials, poly_diagnostic = polynomial_features(x.double().numpy(), train_y)
        heads[key], selections[key] = {}, {}
        for kind in CONFIG["heads"]:
            head = {"kind": kind}
            if kind != "linear_full":
                head["encoder"] = encoder
            if kind == "mlp":
                head["state"], selection = fit_mlp(x, d, train_y, dev_y, CONFIG["head_seed"]+layer)
            else:
                a, b = (train_tensor, dev_tensor) if kind == "linear_full" else (x, d)
                if kind == "quadratic":
                    a, b = quadratic(a), quadratic(b)
                elif kind == "vanishing":
                    head["polynomials"] = polynomials
                    a, b = (quadratic(a)@polynomials.T).abs(), (quadratic(b)@polynomials.T).abs()
                fitted, selection = fit_ridge(a, b, train_y, dev_y)
                head.update(fitted)
            scores = head_scores(head, dev_tensor)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite frozen head")
            heads[key][kind] = head
            selections[key][kind] = {"development_correct": int((scores.argmax(1).numpy() == dev_y).sum()),
                                      "development_total": len(dev_y), "numeric_tensor_bytes": tensor_bytes(head),
                                      "selection": selection}
        diagnostics[key] = {**diagnostic, "encoder": encoded_diagnostic, "polynomials": poly_diagnostic,
                            "country_rank": int(np.linalg.matrix_rank(country, tol=1e-6)),
                            "colour_rank": int(np.linalg.matrix_rank(colour, tol=1e-6)),
                            "J_energy_fraction": (np.sum(jpart*jpart, 1)/np.sum(country*country, 1)).tolist(),
                            "sparse_metrics": sparse["metrics"]}
        print(json.dumps({"fitted_layer": layer, "development_correct": {k: v["development_correct"] for k, v in selections[key].items()}}), flush=True)
    chosen = min(((key, kind) for key in heads for kind in heads[key]),
                 key=lambda item: (-selections[item[0]][item[1]]["development_correct"], int(item[0]),
                                   selections[item[0]][item[1]]["numeric_tensor_bytes"], CONFIG["heads"].index(item[1])))
    save_tensor(root/"fit.pt", {"components": fits, "heads": heads})
    reloaded = torch.load(root/"fit.pt", map_location="cpu", weights_only=True)
    for key in heads:
        for kind in heads[key]:
            if not torch.equal(head_scores(heads[key][kind], development["source"][key].float()),
                               head_scores(reloaded["heads"][key][kind], development["source"][key].float())):
                raise ValueError("Frozen head reload changed predictions")
    write_json(root/"fit.json", {"study_id": plan["study_id"], "scientific_fingerprint": plan["scientific_fingerprint"],
        "frozen_before_evaluation_capture": True, "fit_sha256": file_hash(root/"fit.pt"),
        "training_manifest_sha256": file_hash(root/"training_manifest.json"),
        "diagnostics": diagnostics, "heads": selections,
        "benchmark_candidate": {"layer": int(chosen[0]), "head": chosen[1]},
        "elapsed_seconds": time.perf_counter()-started,
        "training_runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"}})
    return {"status": "fitted", "candidate": chosen, "development": {l: {k: v["development_correct"] for k, v in rows.items()} for l, rows in selections.items()}}
