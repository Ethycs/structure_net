"""Audit recovered failures and preserve links to all original raw responses."""
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.neural_architecture_lab import finalize_ncm_memory_accuracy as analysis

_audit = analysis.audit


def audit(root, packets, role):
    rows, result = _audit(root, packets, role)
    for row in rows:
        for path in row["attempts"]:
            raw = q.read(path)
            if "imported_from" in raw:
                source = q.read(raw["imported_from"])
                if q.file_hash(raw["imported_from"]) != raw["origin_sha256"]:
                    raise ValueError("imported original response changed")
                if any(raw[k] != source[k] for k in ("request", "request_sha256", "response")):
                    raise ValueError("request or judge reply changed during parser recovery")
    return rows, result


analysis.q = q
analysis.audit = audit


if __name__ == "__main__":
    analysis.main()
