"""Continue the observed pilot draft, then run the fixed512-source label stages."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_additional_whole_labels as labels
from experiments.structure_net.ncm_memory_data import file_hash,write_json

DEST=labels.prior.data.ROOT/"20261023_additional_whole_labels_workflow_v1"


def main():
    draft=labels.root(True,"draft")
    labels.wide.terminal(draft)
    if labels.prior.q.read(draft/"result.json")["status"]!="generated":
        raise ValueError("existing observed valid pilot draft required; never regenerate it")
    files=[Path(__file__),Path(labels.__file__),labels.PLAN,Path(labels.wide.__file__)]
    inputs=[draft/n for n in ("producer.json","result.json","labels.json","worker_terminal_receipt.json")]
    inputs += [labels.PREP/n for n in ("producer.json","result.json","public_payload_proof.json","train_sources.json")]
    p=labels.prior.data.producing(DEST,"fixed_additional_whole_summary_label_workflow",inputs+files,
        {"cases":512,"pilot_cases":4,"existing_pilot_draft_restarted":False,
            "stages":["pilot_critique","pilot_revision","pilot_audit","main_draft","main_critique","main_revision","main_audit"],
            "teacher":labels.wide.q.JUDGES["primary"],"auditor":labels.wide.q.JUDGES["secondary"],
            "student_training_admitted":False,"max_attempts":3,"one_semantic_revision":True})
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"label_workflow_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=labels.prior.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p)
    journal={"status":"running","observer_pid":os.getpid(),"stages":[],"producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"progress.json",journal)
    try:
        schedule=[(True,s) for s in ("critique","revision","audit")]
        schedule += [(False,s) for s in ("draft","critique","revision","audit")]
        for pilot,stage in schedule:
            if not pilot:
                labels.wide.terminal(labels.root(True,"audit"))
                if not labels.prior.q.read(labels.root(True,"audit")/"result.json")["pilot_gate"]:
                    raise ValueError("fixed4/4pilot quality gate failed; main not admitted")
            destination=labels.root(pilot,stage)
            if destination.exists():raise ValueError("existing stage must not be duplicated: "+str(destination))
            labels.prior.data.verify(p)
            args=[sys.executable,"-u","-m","experiments.structure_net.ncm_additional_whole_labels",stage]
            if not pilot:args.append("--main")
            child=subprocess.Popen(args)
            journal.update(active_stage=("pilot_" if pilot else "main_")+stage,child_pid=child.pid)
            write_json(DEST/"progress.json",journal)
            code=child.wait()
            if destination.exists():
                path=destination/"worker_terminal_receipt.json"
                if path.exists():raise ValueError("observed child receipt must not be overwritten")
                write_json(path,{"status":"observed_worker_terminal","child_pid":child.pid,
                    "observer_pid":os.getpid(),"observer":"subprocess.wait","exit_code":code})
            journal["stages"].append({"pilot":pilot,"stage":stage,"child_pid":child.pid,"exit_code":code})
            write_json(DEST/"progress.json",journal)
            if code!=0:raise RuntimeError("label stage failed: "+stage)
            result=labels.prior.q.read(destination/"result.json")
            if result["status"] not in ("generated","completed"):
                raise ValueError("invalid teacher/schema endpoint stops workflow")
            expected=4 if pilot else 512
            if stage in ("critique","audit") and (result["endpoint"]["cases"]!=expected
                    or result["endpoint"]["valid"]!=expected):
                raise ValueError("invalid or incomplete judgments stop workflow without semantic rerolls")
        labels.prior.data.verify(p)
        endpoint=labels.prior.q.read(labels.root(False,"audit")/"result.json")
        journal.update(status="completed_training_label_preparation",active_stage=None,
            accepted_labels=endpoint["accepted_labels"],accepted_by_bin=endpoint["accepted_by_bin"],
            teacher_endpoint=endpoint["endpoint"],student_training_admitted=False,
            task_accuracy=None,NCM_success_established=False,final_test_opened=False,speed_tested=False)
        write_json(DEST/"result.json",journal);print(json.dumps(journal),flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved",error_type=type(error).__name__,error=str(error),
            student_training_admitted=False,NCM_success_established=False)
        write_json(DEST/"failure.json",journal)
        raise


if __name__=="__main__":main()
