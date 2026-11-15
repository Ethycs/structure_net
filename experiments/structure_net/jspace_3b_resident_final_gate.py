"""Explicit resident-campaign admission to unchanged full-model final checks.

Only admission and preparation wiring are adapted. Numerical export, fresh
reload, cache tests, untouched confirmation scoring, bracketed runtime and final
gates execute the existing frozen final-gate implementation without monkeypatches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from . import jspace_3b_resident_campaign as campaign
from . import jspace_3b_final_gate as numerical

base,old,read=numerical.base,numerical.old,numerical.read
ADMISSION_SCHEMA="jspace.3b-resident-final-admission.v1"
PHASES=numerical.PHASES


def admit_campaign(root):
    root=Path(root).resolve();plan=campaign.verify_root(root)
    outcome=read(root/"outcome.json")
    if (outcome.get("status")!="all_blocks_recovered" or outcome.get("last_accepted_cut")!=0 or
            outcome.get("compressed_blocks")!=36):
        raise ValueError("Accepted full36 resident campaign required before confirmation")
    previous_sha,previous_correct=plan["seed_checkpoint_sha256"],plan["seed_correct"]
    chain=[];inputs=dict(plan["input_sha256"])
    for cut in range(plan["seed_cut"]-1,-1,-1):
        directory=campaign.stage_dir(root,cut);stage=read(directory/"plan.json");campaign.check_plan(stage)
        if (stage.get("schema")!=campaign.SCHEMA or stage.get("root_fingerprint")!=plan["fingerprint"] or
                stage.get("cut")!=cut or stage.get("parent_checkpoint_sha256")!=previous_sha or
                stage.get("parent_correct")!=previous_correct):
            raise ValueError("Broken accepted resident-parent chain")
        for phase in campaign.PHASES:
            phase_dir=directory/phase;receipt=read(phase_dir/"manifest.json")
            if (receipt.get("status")!="completed" or receipt.get("fingerprint")!=stage["fingerprint"] or
                    receipt.get("artifacts",{}).get("result.json")!=base.sha(phase_dir/"result.json")):
                raise ValueError("Complete authenticated six-phase receipts required")
            numerical._pin(inputs,phase_dir/"manifest.json")
            numerical._pin(inputs,phase_dir/"result.json")
        trained=campaign.checked(directory,"train",stage);verified=campaign.checked(directory,"verify",stage)
        if not trained["quality_pass"] or not verified["passed"]:
            raise ValueError("Rejected stage cannot enter final confirmation")
        for path in (directory/"plan.json",directory/"train/manifest.json",directory/"train/result.json",
                     directory/"verify/manifest.json",directory/"verify/result.json"):
            numerical._pin(inputs,path)
        inputs[str((directory/"train/selected.pt").resolve())]=trained["selected_checkpoint_sha256"]
        chain.append(dict(cut=cut,fingerprint=stage["fingerprint"],checkpoint_sha256=trained["selected_checkpoint_sha256"],
            correct=trained["selected"]["correct"],quality=trained["quality"]))
        previous_sha,previous_correct=trained["selected_checkpoint_sha256"],trained["selected"]["correct"]
    selected=root/"stages/block_00/train/selected.pt"
    reference=root/"stages/block_00/train"/f"epoch_{trained['selected_epoch']:02d}_development.json"
    for path in (root/"plan.json",root/"outcome.json",reference):numerical._pin(inputs,path)
    return dict(root=str(root),plan=plan,stage=stage,training=trained,verification=verified,chain=chain,
        seed_gate_receipt=plan["qualification"],checkpoint=str(selected),selected_reference=str(reference),input_sha256=inputs)


def prepare(root,campaign_root,confirmation_root=numerical.confirmation.DEFAULT_ROOT):
    numerical._cpu();root=Path(root).resolve();confirmation_root=Path(confirmation_root).resolve()
    if root.exists():raise FileExistsError(root)
    accepted=admit_campaign(campaign_root)  # No confirmation reads before accepted full36.
    confirmation_receipt=numerical.confirmation.verify(confirmation_root)
    conf_plan=read(confirmation_root/"plan.json");tasks=read(confirmation_root/"tasks.json")
    cases=tasks["splits"]["confirmation"]
    used,usage_paths=numerical.historical_usage(base.ROOT/"data/experiments",confirmation_root)
    items=read(accepted["plan"]["items_path"])["items"]
    used.update(c["identity"] for c in items)
    separation=numerical.validate_confirmation(cases,used)
    if conf_plan["tokenizer_model"]["revision"]!=accepted["plan"]["model"]["revision"]:
        raise ValueError("Confirmation tokenization differs from pinned native model")
    inputs=dict(accepted["input_sha256"])
    for path in [*[confirmation_root/n for n in ("plan.json","tasks.json","exclusions.json","manifest.json")],
                 accepted["plan"]["items_path"],*usage_paths]:numerical._pin(inputs,path)
    model=accepted["plan"]["model"]
    for name,digest in model["files"].items():numerical._pin(inputs,Path(model["path"])/name,digest)
    for directory in accepted["stage"]["fits"].values():
        for name in ("fit.pt","relation.pt","fit.json","relations.json","interface.json"):
            numerical._pin(inputs,Path(directory)/name)
    boundary=Path(accepted["plan"]["native_cache"])/"boundary_00.npy";numerical._pin(inputs,boundary)
    sources=dict(accepted["plan"]["source_sha256"])
    extras=[Path(__file__),Path(numerical.__file__),Path(numerical.evaluator.__file__),
        Path(__file__).with_name("jspace_allblocks_artifact.py"),Path(__file__).with_name("jspace_qwen_parallel_evaluate.py"),
        Path(numerical.confirmation.__file__),numerical.PROTOCOL,base.ROOT/"pyproject.toml",base.ROOT/"pixi.lock",
        base.ROOT/"tests/structure_net/test_jspace_3b_resident_final_gate.py",
        base.ROOT/"tests/structure_net/test_jspace_3b_final_gate.py",
        base.ROOT/"tests/structure_net/test_jspace_3b_final_evaluate.py"]
    for path in campaign.inherited.local_import_closure(extras):sources[str(path)]=base.sha(path)
    plan=dict(schema=numerical.SCHEMA,study_id=numerical.STUDY,config=numerical.CONFIG,
        admission_schema=ADMISSION_SCHEMA,campaign_root=accepted["root"],
        campaign_fingerprint=accepted["plan"]["fingerprint"],stage_fingerprint=accepted["stage"]["fingerprint"],
        campaign_schema=accepted["plan"]["schema"],seed_gate_receipt=accepted["seed_gate_receipt"],
        accepted_chain=accepted["chain"],selected_checkpoint=accepted["checkpoint"],
        selected_checkpoint_sha256=accepted["training"]["selected_checkpoint_sha256"],
        selected_reference=accepted["selected_reference"],fits=accepted["stage"]["fits"],model=model,
        confirmation_root=str(confirmation_root),confirmation_receipt=confirmation_receipt,separation=separation,
        label_ids=accepted["plan"]["label_ids"],native_boundary_zero=str(boundary),input_sha256=inputs,
        input_identity={p:old.file_identity(p) for p in inputs},source_sha256=sources,confirmed=False)
    frozen=dict(confirmation=cases,development=[c for c in items if c["split"]=="development"],
        all_original_items=items,benchmark=numerical.benchmark_cases(items))
    plan["tasks_digest"]=base.digest(frozen);plan["fingerprint"]=base.digest(plan)
    root.mkdir(parents=True)
    for path,digest in sources.items():
        if base.sha(path)!=digest:raise ValueError("Source changed during final preparation")
        destination=root/"producer"/Path(path).relative_to(base.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    base.write(root/"tasks.json",frozen);base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],confirmation_count=512,compressed_blocks=36)


def verify(root):
    plan=numerical.verify(root)
    if plan.get("admission_schema")!=ADMISSION_SCHEMA or plan.get("campaign_schema")!=campaign.SCHEMA:
        raise ValueError("Wrong resident final-gate admission")
    return plan


def run(root,phase):
    verify(root)
    return numerical.run(root,phase)


def finalize(root):
    verify(root)
    return numerical.finalize(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","finalize",*PHASES))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--campaign",type=Path)
    parser.add_argument("--confirmation-root",type=Path,default=numerical.confirmation.DEFAULT_ROOT)
    args=parser.parse_args()
    result=prepare(args.root,args.campaign,args.confirmation_root) if args.phase=="prepare" else finalize(args.root) if args.phase=="finalize" else run(args.root,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
