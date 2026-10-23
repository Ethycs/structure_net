"""Run the fixed lifecycle under fresh identity-preserving inference semantics."""
from pathlib import Path
from experiments.structure_net import ncm_intact_memory_pilot as pilot
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_intact_memory_identity import IdentityMemoryCorrection
from experiments.structure_net.ncm_memory_data import file_hash
from experiments.structure_net.ncm_paper_data import ROOT

AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-intact-memory-identity-amendment.md")
BASE_FREEZE = pilot.freeze
EXTRA = [Path(__file__), Path("experiments/structure_net/ncm_intact_memory_identity.py"),
    Path("experiments/structure_net/ncm_intact_memory.py"), pilot.PLAN, AMENDMENT]


def freeze(root, stage, inputs, config):
    p = BASE_FREEZE(root, stage, inputs+EXTRA, {**config, "zero_inference_original_tensor_identity": True})
    for path in EXTRA:
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("identity_"+path.name)).write_bytes(path.read_bytes())
    return p


if __name__ == "__main__":
    failed = ROOT / "20261023_intact_memory_pilot_v1"
    if q.read(failed/"worker_terminal_receipt.json")["exit_code"] != 1:
        raise ValueError("original failed pilot must be terminal")
    pilot.NativeMemoryCorrection = IdentityMemoryCorrection
    pilot.DEST = ROOT / "20261023_intact_memory_pilot_v2"
    pilot.PLAN = AMENDMENT
    pilot.freeze = freeze
    pilot.main()
