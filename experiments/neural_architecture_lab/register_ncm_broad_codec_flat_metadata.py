"""Finish local registration after Chroma rejected nested metric metadata."""
import json
from pathlib import Path

from experiments.neural_architecture_lab import finalize_ncm_broad_learned_codec as final
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash,write_json


def main():
    root=final.study.ROOT/"20261023_broad_learned_codec_registry_recovery_v1"
    root.mkdir(exist_ok=False)
    failed=final.study.ROOT/"20261023_broad_learned_codec_register_v1.stdout.log"
    if "Expected metadata value to be a str, int, float or bool" not in failed.read_text():
        raise ValueError("unexpected registration failure")
    campaign=final.DEST/"campaign_results.json"
    x=json.loads(campaign.read_text())
    record={"name":"Broad unpaired learned-codec transfer with excluded complete-summary reconstruction",
        "description":"Released learned NCM codec,2048FP32updates on31109windows;252source-disjoint complete targets reconstructed with own oraclecodes. Raw words/codes/counts replayed. Originalparentnotloaded; no sourceprediction, taskaccuracy orspeedclaim.",
        "confirmation_status":x["classification"],"task_accuracy_measured":False,
        "oracle_target_codes":True,"source_prediction":False,"NCM_success_established":False,
        "representation_gate":x["reconstruction"]["representation_gate"]}
    for arm,metrics in x["components"].items():
        record.update({arm+"_"+key:value for key,value in metrics.items()})
    if any(not isinstance(v,(str,int,float,bool)) for v in record.values()):
        raise ValueError("registry metadata must remain scalar")
    write_json(root/"producer.json",{"kind":"registry_format_recovery_only","failed_exec_session":86451,
        "failed_exit_code":1,"reason":"Chroma rejects dictionary metadata; flatten component metric names.",
        "failed_log_sha256":file_hash(failed),"source_sha256":file_hash(__file__),
        "campaign_sha256":file_hash(campaign),"original_finalizer_sha256":file_hash(final.__file__),
        "scientific_results_changed":False,"new_model_or_judge_runs":False})
    register_records(campaign,final.REPORT,final.DEST,{"ncm-broad-learned-codec-reconstruction-v1":record})
    write_json(root/"result.json",{"status":"registered_and_read_back_after_flat_metadata_recovery",
        "original_failed_attempt_preserved":True,"scientific_results_changed":False})
    print(json.dumps({"status":"registered_and_read_back_after_flat_metadata_recovery"}),flush=True)


if __name__=="__main__":
    main()
