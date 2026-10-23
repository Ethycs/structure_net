"""Grade the accepted codec on the same fixed curated-label audit cohort."""
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash
from experiments.structure_net.ncm_paper_curate_v3 import CURATED
from experiments.structure_net.ncm_paper_data import ROOT
from experiments.structure_net.ncm_paper_grade import grade


def main():
    codec = ROOT / "20261022_codec_training_v2"
    state = q.read(codec / "result.json")
    if state["nonpad_reconstruction"] < .98 or state["model_sha256"] != file_hash(codec / "model.pt"):
        raise ValueError("accepted trained codec required")
    audit = q.read(ROOT / "20261022_curated_audit_v3/result.json")
    if audit["status"] != "completed" or audit["passed"] < 31:
        raise ValueError("curated training-label audit required")
    selected = sorted(q.read(CURATED / "train.json"), key=lambda c: text_hash("curated-independent-audit-v2:"+c["id"]))[:32]
    outputs = {r["id"]: r for r in q.read(codec / "outputs.json")}
    cases = []
    for case in selected:
        row = outputs[case["id"]]
        if row["source_sha256"] != case["source_sha256"] or row["target_sha256"] != case["target_sha256"]:
            raise ValueError("oracle/source/label mismatch")
        cases.append(row)
    grade(cases, ROOT / "20261022_oracle_grade_secondary_v2",
        [codec / "outputs.json", codec / "model.pt", codec / "result.json", CURATED / "train.json", Path(__file__)], "secondary")


if __name__ == "__main__":
    main()
