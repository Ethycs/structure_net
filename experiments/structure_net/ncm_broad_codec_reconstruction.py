"""Paired original/final codec reconstruction; correct codes are oracle inputs."""
import gc
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_broad_learned_codec as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def summarize(rows):
    total=sum(r["total"] for r in rows)
    if not rows or total<1:
        raise ValueError("complete reconstruction denominator required")
    correct=sum(r["correct"] for r in rows)
    return {"cases":len(rows),"correct":correct,"total":total,"token_recovery":correct/total,
        "exact_texts":sum(r["exact_text"] for r in rows),
        "first_EOS_correct":sum(r["first_EOS_correct"] for r in rows),
        "active_invalid_ids":sum(r["active_invalid_ids"] for r in rows),
        "changed_number_cases":sum(r["changed_numbers"] for r in rows)}


def gate(original,trained):
    if original["cases"]!=252 or trained["cases"]!=252:
        raise ValueError("all252 matched diagnostic targets required")
    return (trained["token_recovery"]>=.995 and trained["first_EOS_correct"]/252>=.99
        and trained["active_invalid_ids"]==0
        and trained["changed_number_cases"]<=original["changed_number_cases"])


def main():
    study.require_gpu();study.terminal(study.corpus.DEST);study.terminal(study.TRAIN)
    trained=json.loads((study.TRAIN/"result.json").read_text())
    if (trained["status"]!="trained_pending_reconstruction_diagnostic" or trained["updates"]!=2048
            or file_hash(study.TRAIN/"model.pt")!=trained["checkpoint_sha256"]):
        raise ValueError("terminal fixed final2048 codec checkpoint required")
    p=study.producing(study.RECON,False,[Path(__file__)]+[study.TRAIN/n for n in
        ("producer.json","result.json","model.pt","initial_probe.json","reload_probe.json","worker_terminal_receipt.json")])
    p["source_hashes"][str(Path(__file__))]=file_hash(__file__)
    (study.RECON/"producer_source"/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    p["config"].update(oracle_reconstruction_only=True,source_prediction=False,cases=252)
    p["fingerprint"]=study.corpus.quality.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(study.RECON/"producer.json",p)
    cases=json.loads((study.corpus.DEST/"summary_diagnostic.json").read_text())
    origins=json.loads((study.corpus.DEST/"fit_origins.json").read_text())
    for key in ("source_id","group_id","source_sha256"):
        if {c[key] for c in cases}&{o[key] for r in origins for o in r["origins"]}:
            raise ValueError("codec fitting and reconstruction diagnostic overlap")
    if len(cases)!=252 or len({c["id"] for c in cases})!=252:
        raise ValueError("all252 complete targets required")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(study.corpus.GPT2,local_files_only=True)
    tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    saved=torch.load(study.TRAIN/"model.pt",map_location="cpu",weights_only=True)
    if saved["step"]!=2048 or saved["producer_fingerprint"]!=trained["producer_fingerprint"]:
        raise ValueError("final codec checkpoint provenance differs")
    summaries={};hashes={}
    for arm,state in (("original",None),("trained",saved["model"])):
        codec=study.load_codec(state)
        if state is not None:codec.vq._fd_step.copy_(saved["fine_dropout_step"])
        before=study.module_hash(codec)
        if arm=="original" and before!=trained["initial_model_sha256"]:
            raise ValueError("original codec initialization differs")
        canonical=json.loads((study.TRAIN/("initial_probe.json" if arm=="original" else "reload_probe.json")).read_text())
        canonical={r["id"]:r for r in canonical}
        rows=[]
        for i,case in enumerate(cases,1):
            row=study.reconstruction(codec,[case],tokenizer)[0]
            if case["id"] in canonical and row!=canonical[case["id"]]:
                raise ValueError("canonical first/max-target reconstruction changed")
            rows.append(row)
            if i%16==0 or i==252:
                write_json(study.RECON/(arm+"_outputs.json"),rows)
                print(json.dumps({"arm":arm,"reconstructed":i,"total":252}),flush=True)
        if study.module_hash(codec)!=before:
            raise ValueError("oracle evaluation mutated codec state")
        if json.loads((study.RECON/(arm+"_outputs.json")).read_text())!=rows:
            raise ValueError("raw-code/text serialization differs")
        summaries[arm]=summarize(rows);hashes[arm]=before
        del codec;gc.collect();torch.cuda.empty_cache()
    passed=gate(summaries["original"],summaries["trained"])
    study.verify(p)
    result={"status":"completed_oracle_reconstruction_diagnostic",
        "classification":"codec_reconstruction_gate_passed_semantic_audit_required" if passed else "codec_reconstruction_fidelity_failed",
        "original":summaries["original"],"trained":summaries["trained"],
        "token_recovery_improvement":summaries["trained"]["token_recovery"]-summaries["original"]["token_recovery"],
        "representation_gate":passed,"codec_hashes":hashes,"cases":252,"source_disjoint":True,
        "all_targets_complete":True,"oracle_target_codes":True,"source_prediction":False,
        "parent_loaded":False,"task_accuracy":None,"NCM_success_established":False,
        "final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"],
        "output_hashes":{arm:file_hash(study.RECON/(arm+"_outputs.json")) for arm in summaries}}
    write_json(study.RECON/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
