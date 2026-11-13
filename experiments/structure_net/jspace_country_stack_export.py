"""Export the frozen country-stack artifacts with a verified member manifest."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from .jspace_country_stack import DEFAULT_ROOT, verify, checked
from .jspace_causal_components import cpu_only
from .jspace_relation_study import ROOT, file_hash, write_json


REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-13_jspace-country-stack.md"


def main(root=DEFAULT_ROOT):
    cpu_only()
    plan = verify(root)
    checked(root, "fit")
    checked(root, "capture")
    destination = root / "country_stack_bundle.zip"
    if destination.exists() or (root / "export_manifest.json").exists():
        raise FileExistsError(destination)
    names = ["country_quotient.json", "continuous_stack_degree1.json", "continuous_stack_degree2.json",
             "compressed_readout.pt", "linear_readout_reference.pt", "uncompressed_readouts.pt",
             "result.json", "campaign_results.json", "identifiability_audit.json",
             "attention_pullback_audit.json", "presentation_metrics.json", "country_stack.png",
             "country_stack.svg", "plan.json"]
    names += [f"{cohort}_countries.{mode}.bin" for cohort in ("source", "donor", "compact_J", "dense")
              for mode in ("raw", "compressed")]
    members = {name: (root / name).read_bytes() for name in names}
    members["REPORT.md"] = REPORT.read_bytes()
    for name in ("jspace_country_stack_algebra.py", "jspace_country_stack_pullback.py"):
        members["reference_code/" + name] = (ROOT / "experiments/structure_net" / name).read_bytes()
    members["README.txt"] = (
        "JSPACE-COUNTRY-STACK-GROEBNER-01; exploratory, confirmed=false.\n"
        "The exact country quotient and observed-label codec are lossless within their stated scope.\n"
        "The selected quadratic continuous decoder failed its neural fidelity gate.\n"
        "Readout payload savings exclude the unchanged Qwen backbone. No full-model speedup was measured.\n"
        "The tiny-Qwen pullback control verifies projected readouts, not a trained attention replacement.\n\n"
        "Use the accompanying StructureNet checkout and its locked Pixi environment.\n"
        "Decode label files with jspace_country_stack_algebra.unpack_labels(blob).\n"
        "Load the .pt dictionaries with torch.load(..., map_location='cpu', weights_only=True).\n"
        "decode_portable(payload, root_projection, attribute) needs the original layer-27 J projections\n"
        "and known task attribute, as described in REPORT.md; it cannot reconstruct full hidden states.\n"
        "Reference code uses modules from the checkout; this archive is not a standalone installation.\n"
        "Report links refer to the repository layout. MANIFEST.json hashes every other archive member.\n"
    ).encode()
    manifest = {"study_id": plan["study_id"], "scientific_fingerprint": plan["scientific_fingerprint"],
                "members": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}}
    members["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = ZipInfo(name)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(members):
            raise ValueError("Export archive membership or CRC mismatch")
        if any(archive.read(name) != data for name, data in members.items()):
            raise ValueError("Export archive readback mismatch")
    value = {"status": "completed", "source_sha256": file_hash(__file__),
             "report_sha256": file_hash(REPORT), "archive_sha256": file_hash(destination),
             "archive_bytes": destination.stat().st_size, "member_count": len(members),
             "readback_verified": True, "manifest": manifest}
    write_json(root / "export_manifest.json", value)
    return {k: v for k, v in value.items() if k != "manifest"}


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
