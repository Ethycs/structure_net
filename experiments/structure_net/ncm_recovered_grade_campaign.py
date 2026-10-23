"""Finish the noisy-pair development evaluation with the recovered original judges."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT
from experiments.structure_net.ncm_paper_grade import grade

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-recovered-judge-plan.md")


def main(role):
    pilot = ROOT / ("20261022_recovered_judge_pilot_"+role+"_v1/result.json")
    result = q.read(pilot)
    if (result["status"] != "completed" or result["cases"] != 4 or result["invalid"] != 0
            or not result["systems_only"] or result["judge"] != q.JUDGES[role]):
        raise ValueError("matching4-concurrent-request recovery pilot required")
    sources = [Path(__file__), PLAN]
    results = {}
    for arm in ("native", "hybrid"):
        source = ROOT / ("20261022_noisy_"+arm+"_v1/development_step2048.json")
        output = ROOT / ("20261022_noisy_"+arm+"_grade_"+role+"_v1")
        results[arm] = grade(q.read(source), output, [source, pilot, *sources], role=role)
    receipt = ROOT / ("20261022_noisy_quality_"+role+"_v1.json")
    if receipt.exists():
        raise ValueError("fresh grading receipt required")
    write_json(receipt, {"role": role, "judge": q.JUDGES[role], "results": results,
        "source_hashes": {str(path): file_hash(path) for path in sources},
        "final_test_opened": False, "speed_tested": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("role", choices=q.JUDGES)
    main(parser.parse_args().role)
