"""Conservatively audit the dual-source-attention mixed-precision lifecycle."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_grounded_pilot as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

RESULTS=run.ROOT/"20261024_lm1b_grounded_cuda_pilot_results_v1"
REPORT=Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-grounded-cuda-pilot.md")


def tensor_elements(value):
    if isinstance(value,torch.Tensor):return value.numel()
    if isinstance(value,dict):return sum(tensor_elements(v) for v in value.values())
    raise ValueError("attention tensor dictionaries only")


def finalize():
    if torch.cuda.is_initialized():raise ValueError("CPU audit only")
    run.native.vendor.terminal(run.DEST)
    if RESULTS.exists() or REPORT.exists():raise ValueError("fresh dual-attention aggregate required")
    result=run.q.read(run.DEST/"result.json");producer=run.q.read(run.DEST/"producer.json")
    expected={"generator":"60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a",
        "codec":"569bf1ad07bc4d2cc762098b7f7e9fa0f6c68fccaa954f65a191d5a3b77f9c16"}
    if (result["status"]!="systems_dual_source_attention_BF16_gradients_and_reload_passed_not_quality"
            or result["both_native_states_unchanged"]!=expected or result["new_trainable_parameters"]!=11796480
            or result["microbatch"]!=4 or result["updates"]!=2 or result["peak_allocated_bytes"]>7_500_000_000
            or not result["both_stages_zero_output_native_parity_exact"] or not result["both_source_attention_gradients_nonzero"]
            or not result["teacher_and_source_only_reload_exact"] or not result["optimizer_RNG_reload_exact"]
            or result["task_accuracy"] is not None or result["speed_tested"] or result["final_test_opened"]
            or file_hash(run.DEST/"attention.pt")!=result["checkpoint_sha256"]):
        raise ValueError("dual-stage systems evidence differs")
    trace=run.q.read(run.DEST/"trace.json")
    if len(trace)!=2:raise ValueError("two actual update rows required")
    for step,row in enumerate(trace,1):
        if row["step"]!=step or any(not math.isfinite(row[k]) for k in ("concept_loss","word_loss","gradient_norm")):
            raise ValueError("finite dual-stage gradient trace differs")
        prefixes=("word_adapters.","concepts.adapters.") if step==1 else (
            "word_adapters.","concepts.adapters.","word_memory.","concepts.source_memory.")
        if any(not any(n.startswith(p) for n in row["nonzero_gradient_names"]) for p in prefixes):
            raise ValueError("both learned source grounding paths required")
    saved=torch.load(run.DEST/"attention.pt",map_location="cpu",weights_only=True)
    if set(saved["attention"])!={"concept_attention","word_memory","word_adapters"} or tensor_elements(saved["attention"])!=11796480:
        raise ValueError("attention-only checkpoint parameter count differs")
    if any(int(v["step"])!=2 for v in saved["optimizer"]["state"].values()):raise ValueError("actual2step optimizer state required")
    outputs=torch.load(run.DEST/"reload_outputs.pt",map_location="cpu",weights_only=True)
    shapes={"concept_logits":(1,641,8192),"word_logits":(1,128,30528)}
    if any(outputs["teacher"][k].shape!=shape or not torch.isfinite(outputs["teacher"][k]).all() for k,shape in shapes.items()):
        raise ValueError("complete two-stage native teacher logits required")
    if outputs["source_only"]["codes"].shape!=(1,641) or outputs["source_only"]["raw_word_ids"].shape!=(1,128):
        raise ValueError("label-free full concept and word generation required")
    artifacts={**producer["source_hashes"],**producer["input_hashes"],str(Path(__file__)):file_hash(__file__)}
    artifacts.update({str(p):file_hash(p) for p in run.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    campaign={"hypothesis_id":run.HYPOTHESIS,"classification":"systems_dual_source_attention_BF16_lifecycle_passed_not_quality",
        "result":result,"trace":trace,"task_accuracy":None,"source_only_task_accuracy_measured":False,
        "underpowered":True,"confirmed":False,"final_test_opened":False,"speed_tested":False,"artifacts":artifacts}
    RESULTS.mkdir();path=RESULTS/"campaign_results.json";write_json(path,campaign)
    REPORT.write_text(f"""# Source attention in native NCM concept and word decoding

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **systems_dual_source_attention_BF16_lifecycle_passed_not_quality**.
Campaign `{path}`, SHA256 `{file_hash(path)}`.

New source attention now runs in both the released concept generator and all
four word-decoder blocks. Its 11,796,480 new parameters start with zero output
projections; initial FP32 concept AND word logits match the existing model exactly.
The 97M generator and measured task-domain codec remain frozen and unchanged.
The word decoder executes directly so its no_grad convenience API cannot block
gradients to new attention. No original Llama weights or MLPs were loaded.

Two real updates at microbatch four used BF16 autocast with FP32 weights, on
complete sources of 486 and 2744 aligned feature positions. Both source K/V and
both attention branches received gradients. Peak allocation was
{result['peak_allocated_bytes']} bytes on physical GPU0 RTX3060. A fresh load of
the frozen states plus new attention/optimizer/RNG reproduced FP32 teacher logits
and label-free concept/word generation exactly. Request memories cleared.

This is systems evidence only, not summarization quality or a speed measurement.
The source features are SHA-verified cached full-source frozen-body outputs.
The existing domain codec's 52/64 semantic result is below its qualification
gate; this experiment does not rescue it. The previous conditional student
remains 0/128. The pilot's two-update checkpoint is not a qualified summarizer
and must not silently become full-training initialization. Full 95% task
accuracy remains unproven; all final sources stayed closed.
""")
    print(json.dumps({"classification":campaign["classification"],"campaign_sha256":file_hash(path),"task_accuracy":None}),flush=True)


def register():
    register_records(RESULTS/"campaign_results.json",REPORT,RESULTS,{run.HYPOTHESIS:{
        "name":"Native NCM concept and word source-attention BF16 lifecycle",
        "description":"Both native stages retain exact initial logits, frozen weights and full sources;11.8Mnew attention weights receive gradients underBF16,microbatch4,with exact fresh model/optimizer/RNG and label-free generation reload. Systems-only,not model quality.",
        "classification":"systems_dual_source_attention_BF16_lifecycle_passed_not_quality",
        "source_only_task_accuracy_measured":False,"systems_lifecycle_only":True}})


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("stage",choices=("finalize","register"));args=p.parse_args()
    (finalize if args.stage=="finalize" else register)()
