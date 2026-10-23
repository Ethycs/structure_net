"""Fresh, time-guarded infrastructure recovery; preserve the failed v1 pilot."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiments.structure_net import ncm_native_label_quality as old
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT

AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-native-label-quota-recovery.md")
NOT_BEFORE = datetime(2026,10,23,9,50,tzinfo=timezone.utc)
BASE_PRODUCING = old.producing


def producing(root, stage, inputs, config):
    p = BASE_PRODUCING(root,stage,inputs,{**config,"quota_reset_not_before":NOT_BEFORE.isoformat()})
    for path in (Path(__file__),AMENDMENT):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("quota_recovery_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root/"producer.json",p)
    return p


def main(command):
    if datetime.now(timezone.utc) < NOT_BEFORE:
        raise ValueError("documented provider reset has not occurred; do not request yet")
    prior = q.read(old.PILOT/"result.json")
    terminal = q.read(old.PILOT/"worker_terminal_receipt.json")
    if prior["status"] != "failed_serial_pilot" or terminal["exit_code"] != 0:
        raise ValueError("documented terminal v1 service failure required")
    for name in ("PILOT","GENERATE","JUDGE_PILOT","AUDIT"):
        setattr(old,name,ROOT/str(getattr(old,name).name).replace("_v1","_v2"))
    old.producing = producing
    if command in ("pilot","generate"):
        old.teacher(command == "pilot")
    else:
        from experiments.structure_net import ncm_paper_grade
        run_grade = ncm_paper_grade.grade
        def wrapped(cases,destination,inputs,**kwargs):
            return run_grade(cases,destination,[*inputs,Path(__file__),AMENDMENT],**kwargs)
        ncm_paper_grade.grade = wrapped
        old.grade(command == "judge-pilot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["pilot","generate","judge-pilot","audit"])
    main(parser.parse_args().command)
