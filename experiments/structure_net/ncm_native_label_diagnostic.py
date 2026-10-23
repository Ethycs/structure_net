"""Judge all unchanged candidates from a failed quote-format pilot."""
from pathlib import Path
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_label_quality as labels
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_paper_grade import grade

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-label-diagnostic-plan.md")
SOURCE = ROOT / "20261023_native_labels_pilot_v2"
DEST = ROOT / "20261023_native_labels_diagnostic_v1"


def main():
    labels.dependencies()
    verify(q.read(SOURCE / "producer.json"))
    receipt = q.read(SOURCE / "worker_terminal_receipt.json")
    previous = q.read(SOURCE / "result.json")
    if (receipt["exit_code"] != 0 or previous["status"] != "failed_systems_pilot"
            or previous["cases"] != 10 or previous["valid_batches"] != 5):
        raise ValueError("exact terminal failed pilot required")
    cases = q.read(SOURCE / "labels.json")
    if len(cases) != 10:
        raise ValueError("all ten unchanged candidates required")
    grade(cases, DEST, [SOURCE / n for n in ("producer.json", "result.json", "labels.json")]
          + [Path(__file__), PLAN], role="secondary", systems_only=False)


if __name__ == "__main__":
    main()
