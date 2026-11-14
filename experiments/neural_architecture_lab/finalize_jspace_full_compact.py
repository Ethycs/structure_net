"""Hash-check and archive every attempt in the full-model compact campaign."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

from experiments.structure_net import jspace_full_capture as base
from experiments.structure_net.jspace_full_affine_v2 import DEFAULT_ROOT
from experiments.structure_net.jspace_bench_data import sha,write


def main():
    base.cpu_only();artifact_base=base.ROOT/"data/experiments/jspace_full_compact"
    specification={
        "20261113_capture_shakedown_v1":["capture"],"20261113_capture_v1":["capture"],
        "20261113_fit_v1":[],"20261113_fit_v2":["fit"],
        "20261113_training_shakedown_v1":["shakedown"],"20261113_v1":["controls","train"],
        "20261113_attribution_v1":["mlp_only","attention_only","attention_full_input","attention_full_head"],
        "20261113_export_v1":["reload"],"20261113_refinement_v1":["attention_identity","mlp_runtime"],
        "20261113_affine_shakedown_v2":["shakedown"],"20261113_affine_v2":["controls","train"],
        "20261113_affine_export_v2":["reload"]}
    registry=base.ROOT/"data/hypothesis_registry/jspace_full_compact_20261113.json"
    stored=json.loads(registry.read_text());readback=json.loads((DEFAULT_ROOT/"registry_readback.json").read_text())
    if readback.get("readback_verified") is not True or readback["result_hash"]!=stored["storage"]["result_hash"]:
        raise ValueError("Fresh registry readback missing")
    expected={Path(p):v for p,v in stored["source_sha256"].items()};campaigns=[]
    expected[base.ROOT/"experiments/neural_architecture_lab/readback_jspace_full_compact.py"]=readback["reader_source_sha256"]
    for name,stages in specification.items():
        root=artifact_base/name;plan=base.verify(root);entries={}
        for stage in stages:
            manifest=json.loads((root/f"{stage}_manifest.json").read_text())
            if manifest["status"]!="completed" or manifest["fingerprint"]!=plan["scientific_fingerprint"]:
                raise ValueError("Incomplete stage: "+name+"/"+stage)
            for p,value in manifest["artifact_sha256"].items():
                full=root/p
                if full in expected and expected[full]!=value:raise ValueError("Conflicting hashes")
                expected[full]=value
            entries[stage]=manifest
        campaigns.append({"root":str(root),"fingerprint":plan["scientific_fingerprint"],"stages":entries})
    if (artifact_base/"20261113_fit_v1/fit_manifest.json").exists():raise ValueError("Failed fit was relabeled")
    for name in ("20261113_export_v1","20261113_affine_export_v2"):
        exported=json.loads((artifact_base/name/"export.json").read_text())
        if exported["all_state_tensors_exact"] is not True or exported["export_has_embedding_alias"] is not True:
            raise ValueError("Checkpoint export is not exact")
        expected[artifact_base/name/"compact_model_tied.pt"]=exported["export_sha256"]
    extras=[Path(__file__),base.ROOT/"experiments/neural_architecture_lab/store_jspace_full_compact.py",
        base.ROOT/"experiments/neural_architecture_lab/readback_jspace_full_compact.py",
        base.ROOT/"src/neural_architecture_lab/jspace_full_compact_record.py",
        base.ROOT/"experiments/structure_net/jspace_full_analysis.py",
        base.ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-full-compact.md",
        base.ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-full-compact-notes.md",
        base.ROOT/"docs/07 - Status Reports/jspace/research-log.jsonl"]
    archive=DEFAULT_ROOT/"postprocessing_source";archive.mkdir(exist_ok=False)
    for path in extras:
        target=archive/path.relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    paths=set(expected)|{registry}
    for name in specification:
        paths.update(p for p in (artifact_base/name).rglob("*") if p.is_file() and p.name!="finalization.json")
    diagnostic=artifact_base/"20261113_projection_diagnostic_v1"
    diagnostic_plan=json.loads((diagnostic/"plan.json").read_text())
    expected.update({Path(p):v for p,v in diagnostic_plan["source_sha256"].items()})
    expected.update({Path(p):v for p,v in diagnostic_plan["input_sha256"].items()})
    paths.update(expected);paths.update(p for p in diagnostic.rglob("*") if p.is_file())
    paths=sorted(paths)
    def check(path):
        value=sha(path)
        if path in expected and value!=expected[path]:raise ValueError("Artifact changed: "+str(path))
        return str(path),value
    with ThreadPoolExecutor(max_workers=4) as pool:artifacts=dict(pool.map(check,paths))
    value={"study_id":base.STUDY_ID,"status":"completed","confirmed":False,"created_at":datetime.now(timezone.utc).isoformat(),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=base.ROOT,text=True).strip(),
        "campaigns":campaigns,"artifact_sha256":artifacts,"registry_readback":readback,
        "integrity":{"reservation_stopped_fit_attempts":1,"completed_fit_retries":1,"trained_variants":2,
            "adaptation_steps_per_variant":3456,"training_questions":512,"development_questions":64,"evaluation_questions":256,
            "independent_confirmations":0,"all_gpu_stages_reservation_supervised":True,
            "checkpoint_alias_correction_changes_tensor_values":False},
        "scope":"One exploratory campaign with preserved failures, a corrected affine implementation, component diagnostics and matched runtime controls"}
    write(DEFAULT_ROOT/"finalization.json",value)
    print(json.dumps({"status":"finalized","artifact_count":len(artifacts),"source_commit":value["source_commit"],
                      "sha256":sha(DEFAULT_ROOT/"finalization.json")},allow_nan=False))


if __name__=="__main__":main()
