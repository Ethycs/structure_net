"""Regenerate labels locally after the saved-summary quality gate failed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, parent_inputs
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256, verify_cases
from experiments.structure_net.ncm_llama import GPT2, label_ids, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_recipe_control import PLAN, ROOT
from experiments.structure_net.ncm_recipe_data import PARENT, QUALITY as FAILED_QUALITY

TEACHER = ROOT / "20261022_teacher_v1"
QUALITY = ROOT / "20261022_quality_v2"
NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md")


def verify_public(cases):
    if file_hash(PUBLIC_FILE) != PUBLIC_SHA256:
        raise ValueError("public dataset SHA mismatch")
    receipt = q.read(PARENT / "public_data_readback.json")
    if receipt["public_file_metadata"]["lfs"]["oid"] != PUBLIC_SHA256:
        raise ValueError("published SHA proof missing")
    return verify_cases(q.read(PUBLIC_FILE), cases)


def freeze(root, inputs):
    if root.exists():
        raise ValueError("fresh teacher/quality root required")
    root.mkdir(parents=True)
    sources = [Path(__file__), NOTE, PLAN,
        Path("experiments/structure_net/ncm_frozen_body_public_data.py"),
        Path("experiments/structure_net/ncm_llama.py"),
        Path("experiments/structure_net/ncm_memory_accuracy.py"),
        Path("experiments/structure_net/ncm_memory_accuracy_v2.py")]
    copies = root / "producer_source"
    copies.mkdir()
    for i, p in enumerate(sources):
        shutil.copyfile(p, copies / f"{i:03d}_{p.name}")
    producer = {"schema": "nal.ncm-recipe-teacher.v1", "hypothesis_id": "ncm-recipe-update-v1",
        "source_hashes": {str(p): file_hash(p) for p in sources},
        "input_hashes": {str(p): file_hash(p) for p in inputs},
        "teacher": str(MODEL_PATH), "batch": 4, "max_new_tokens": 96, "do_sample": False,
        "word_request": 24, "judge": q.JUDGES["primary"], "scope": "training label preparation"}
    producer["fingerprint"] = q.digest(producer)
    write_json(root / "producer.json", producer)
    return producer


def verify(producer):
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("teacher producer changed")


def generate():
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    cases = q.read(FAILED_QUALITY / "cases.json")[:64]
    verify_public(cases)
    inputs = [FAILED_QUALITY / "cases.json", FAILED_QUALITY / "result.json", PUBLIC_FILE]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]
    inputs += [p for p in GPT2.iterdir() if p.is_file()]
    producer = freeze(TEACHER, inputs)
    gpt, tokenizer = tokenizers()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    rows = []
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        records = []
        for case in chunk:
            ids = tokenizer.apply_chat_template([
                {"role": "system", "content": "Summarize the passage in one coherent paragraph of about 24 words. "
                 "Preserve the main point and key facts. Do not invent facts. Return only the summary."},
                {"role": "user", "content": case["source"]}], add_generation_prompt=True)
            records.append({"parent_ids": ids})
        x = parent_inputs(records, tokenizer, "cuda")
        with torch.inference_mode():
            output = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        for case, generated in zip(chunk, output[:, 512:].tolist()):
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                ids = label_ids(text, gpt)
                eligible = bool(text)
            except ValueError:
                ids, eligible = None, False
            rows.append({**case, "target": text, "target_sha256": text_hash(text),
                "original_saved_target_sha256": case["target_sha256"],
                "target_gpt2_tokens": len(gpt.encode(text, add_special_tokens=False)),
                "teacher_eligible": eligible, "native_target_ids": ids,
                "teacher_output_ids": generated})
        write_json(TEACHER / "progress.json", {"cases": len(rows)})
        print(json.dumps({"stage": "local_teacher_labels", "completed": len(rows), "total": 64}), flush=True)
    verify(producer)
    write_json(TEACHER / "cases.json", rows)
    write_json(TEACHER / "result.json", {"status": "completed", "cases": len(rows),
        "native_eligible": sum(r["teacher_eligible"] for r in rows), "producer_fingerprint": producer["fingerprint"],
        "trained": False, "teacher_frozen": True})


def grade():
    import tiktoken
    cases = [r for r in q.read(TEACHER / "cases.json") if r["teacher_eligible"]]
    verify_public(cases)
    producer = freeze(QUALITY, [TEACHER / "producer.json", TEACHER / "cases.json", TEACHER / "result.json", PUBLIC_FILE])
    for name in ("attempts", "rows"):
        (QUALITY / name).mkdir()
    packets = [q.packet(r["id"], r["source"], [{"candidate_id": "generated-"+r["id"][:16], "text": r["target"]}]) for r in cases]
    write_json(QUALITY / "packets.json", packets)
    enc = tiktoken.get_encoding("cl100k_base")
    with ThreadPoolExecutor(max_workers=4) as pool:
        judgments = []
        for row in pool.map(lambda p: q.request(p, "primary", QUALITY), packets):
            judgments.append(row)
            write_json(QUALITY / "rows" / (row["case_id"]+".json"), row)
            if len(judgments) % 16 == 0:
                print(json.dumps({"stage": "teacher_target_quality", "completed": len(judgments)}), flush=True)
    accepted = []
    for case, packet, row in zip(cases, packets, judgments):
        if row["status"] != "valid":
            continue
        raw = q.read(row["attempts"][-1])
        parsed = q.parse(raw["response"]["choices"][0]["message"]["content"], packet)
        if parsed != row["evaluations"] or any(q.read(p)["valid"] for p in row["attempts"][:-1]):
            raise ValueError("teacher-label raw judgment audit failed")
        if q.passes(parsed[0], case["target"], len(enc.encode(case["target"], disallowed_special=()))):
            accepted.append(case)
    verify(producer)
    write_json(QUALITY / "judgments.json", judgments)
    write_json(QUALITY / "selected.json", {"train": accepted[:32], "selection": "first 32 accepted generated labels in fixed source order"})
    result = {"status": "passed" if len(accepted) >= 32 else "target_quality_gate_failed", "candidates": len(cases),
        "accepted": len(accepted), "selected": min(32, len(accepted)),
        "invalid": sum(r["status"] != "valid" for r in judgments),
        "retried": sum(len(r["attempts"]) > 1 for r in judgments),
        "producer_fingerprint": producer["fingerprint"], "heldout_accuracy": False}
    write_json(QUALITY / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["generate", "grade"])
    globals()[parser.parse_args().command]()
