"""Posthoc paired full-state perturbation energy; no new selection or fitting."""
import argparse
import json
from pathlib import Path

import torch

from .jspace_attention_runtime import boundaries, forward
from .jspace_attention_study import DEFAULT_ROOT, Stage, checked, verify
from .jspace_relation_study import ROOT, file_hash, write_json
from .jspace_tasks import orthonormal_basis
from .jspace_causal_components import cpu_only


PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-attention-full-trace-audit.md"


def freeze(root):
    cpu_only()
    plan = verify(root)
    checked(root, "evaluation")
    value = {"scope": "posthoc_reused_all_128_evaluation_worlds_full_state_energy_only",
             "scientific_fingerprint": plan["scientific_fingerprint"], "result_sha256": file_hash(root/"result.json"),
             "producer_sha256": {str(Path(__file__).resolve()):file_hash(__file__), str(PROTOCOL):file_hash(PROTOCOL)}}
    write_json(root/"full_trace_plan.json", value)
    for p in (Path(__file__).resolve(), PROTOCOL):
        target = root/"producer"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(p.read_bytes())
    return value


def run(root):
    plan = json.loads((root/"full_trace_plan.json").read_text())
    if file_hash(root/"result.json") != plan["result_sha256"] or any(file_hash(p) != h for p,h in plan["producer_sha256"].items()):
        raise ValueError("Full trace audit inputs changed")
    checked(root, "evaluation")
    stage = Stage(root, "full_trace")
    bank = torch.load(root/"dictionary.pt",map_location="cpu",weights_only=True)["rows"]
    q = {k:orthonormal_basis(v) for k,v in bank.items()}
    selection = json.loads((root/"selection.json").read_text())
    layer, method = selection["layer"], selection["method"]
    previous = json.loads((root/"evaluation/records.json").read_text())
    def capture(case, start, donor=None):
        full = {}
        def observe(key, h):
            if h.shape[1] > 1:
                full[key] = h[0].detach().float().cpu().clone()
        with boundaries(stage.model, 11, observe):
            output = forward(stage.model,stage.tokenizer,case,start,bank,keep_attention=(layer,),
                             patch_layer=layer if donor is not None else None, donor=donor,
                             method=method if donor is not None else None)
        # Outer observers run before the inner editing hook at this single site.
        # Use the inner hook's explicitly post-edit output for that attention node.
        full[f"{layer}.attention"] = output["attention"][layer].float()
        return output, full
    sums = {node:{"delta_energy":0., "J_energy":0., "source_energy":0., "elements":0} for node in stage.nodes}
    records = []
    for pair, frozen in zip(stage.tasks["pairs"]["evaluation"],previous):
        source, h = capture(pair["source"], pair["question_start"])
        donor = forward(stage.model,stage.tokenizer,pair["donor"],pair["question_start"],bank,keep_attention=(layer,))
        changed, hp = capture(pair["source"], pair["question_start"], donor["attention"][layer])
        for current, old in ((source["row"],frozen["source"]),(donor["row"],frozen["donor"]),(changed["row"],frozen["edits"][method])):
            if current["token_ids"] != old["token_ids"]:
                raise ValueError("Replayed generation changed")
        values = {}
        for node in stage.nodes:
            delta = hp[node]-h[node]
            de = float(delta.double().square().sum())
            je = float((delta@q[node].T).double().square().sum())
            se = float(h[node].double().square().sum())
            if je > de*1.0001+1e-8:
                raise ValueError("Projected energy exceeds full energy")
            values[node] = {"delta_energy":de, "J_energy":je, "source_energy":se, "elements":delta.numel()}
            for key,v in values[node].items():
                sums[node][key] += v
        records.append({"case_id":pair["source"]["case_id"], "nodes":values, "generation_replay_exact":True})
        stage.event("full_state_world", {"case_id":pair["source"]["case_id"], "generation_replay_exact":True})
    write_json(stage.path/"records.json", records)
    curve = [{"node":node, **v, "delta_l2":v["delta_energy"]**.5,
              "J_energy_fraction":v["J_energy"]/v["delta_energy"] if v["delta_energy"] else None,
              "relative_delta_l2":(v["delta_energy"]/v["source_energy"])**.5} for node,v in sums.items()]
    write_json(stage.path/"result.json", {"scope":plan["scope"], "confirmed":False, "worlds":len(records), "curve":curve,
               "generation_replay_exact":True, "plan_sha256":file_hash(root/"full_trace_plan.json"),
               "interpretation":"Whole-prefill paired hidden-state perturbation, not a proof of retained semantic identity; layer-specific eight-row J spans. Raw states are processed per world, not retained."})
    return stage.finish(("full_trace_plan.json",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](DEFAULT_ROOT), allow_nan=False))
