"""Exploratory teacher-forced error tracing after the primary quality decision."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from experiments.structure_net.llama_neighbourhood_tensorization import (
    Config, install_factors, load_teacher, normalized_error, run_blocks, seal)
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    training_path = args.training / "training.json"
    training = json.loads(training_path.read_text())
    cache = json.loads((args.cache / "cache.json").read_text())
    fixtures = json.loads(args.fixtures.read_text())
    dev = [e for e in fixtures["examples"] if e["split"] == "development"]
    selected = [dev[0], next(e for e in dev if len(e["fragments"]) == 8)]
    inventory = {e["id"]: e for e in cache["examples"]}
    for e in selected:
        for f in inventory[e["id"]]["files"]:
            if sha256(f["path"]) != f["sha256"]:
                raise ValueError("teacher boundary cache changed")
    seal(args.output, Config(), [Path(__file__), training_path, args.fixtures, args.cache / "cache.json"],
         "exploratory_teacher_forced_accumulation_not_quality_rescue")
    model = load_teacher("cpu")
    for record in training["groups"]:
        path = Path(record["selected_checkpoint"])
        if sha256(path) != record["selected_sha256"]:
            raise ValueError("factor checkpoint changed")
        group = record["group"]
        install_factors(model.model.layers[2*group:2*group+2], torch.load(path, weights_only=True), "cpu")
    model.cuda().eval().requires_grad_(False)
    rows = []
    for e in selected:
        cached = [torch.load(f["path"], weights_only=True, map_location="cpu") for f in inventory[e["id"]]["files"]]
        local = []
        for group in range(14):
            x, y = cached[group]["input"].cuda(), cached[group]["target"].cuda()
            prediction = run_blocks(model.model.layers[2*group:2*group+2], x, model.model.rotary_emb)
            local.append(float(normalized_error(prediction, y)))
            del prediction, x, y
        traces, hooks = [], []
        answer_start = len(e["prompt_ids"]) - 1
        for group in range(14):
            def hook(module, inputs, result, group=group):
                target = cached[group]["target"].cuda()
                actual = result[0]
                if not torch.isfinite(actual).all():
                    raise FloatingPointError("nonfinite assembled hidden state")
                traces.append({"group": group, "local_same_input_nmse": local[group],
                               "assembled_prefix_nmse": float(normalized_error(actual, target)),
                               "assembled_answer_positions_nmse": float(normalized_error(actual[:, answer_start:-1], target[:, answer_start:-1]))})
            hooks.append(model.model.layers[2*group+1].register_forward_hook(hook))
        output = model.model(torch.tensor([e["training_ids"]], device="cuda"), use_cache=False)
        for h in hooks:
            h.remove()
        student = model.lm_head(output.last_hidden_state[:, answer_start:-1]).float()
        teacher_hidden = model.model.norm(cached[-1]["target"][:, answer_start:-1].cuda())
        teacher = model.lm_head(teacher_hidden).float()
        teacher_log = F.log_softmax(teacher, -1)
        student_log = F.log_softmax(student, -1)
        kl = (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()
        agreement = (teacher.argmax(-1) == student.argmax(-1)).float().mean()
        row = {"id": e["id"], "prompt_tokens": len(e["prompt_ids"]), "traces": traces,
               "reference_response_positions": student.shape[1], "teacher_student_top1_agreement": float(agreement),
               "teacher_to_student_mean_kl_nats": float(kl),
               "boundary": "teacher-forced reference response; next-token agreement is not free-generation task accuracy"}
        rows.append(row)
        write_json(args.output / "diagnostic.json", {"status": "running", "rows": rows})
        print(json.dumps(row), flush=True)
        del output, student, teacher, teacher_hidden, teacher_log, student_log, cached
    write_json(args.output / "diagnostic.json", {"status": "completed", "profile": "exploratory_not_quality_rescue",
               "rows": rows, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
    write_json(args.output / "result.json", {"status": "completed", "diagnostic_sha256": sha256(args.output / "diagnostic.json")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "cache", "fixtures", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
