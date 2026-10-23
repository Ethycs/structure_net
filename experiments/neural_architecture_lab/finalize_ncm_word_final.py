"""Replay fixed word-final NCM development evidence before conservative indexing."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_word_final_eval as study
from experiments.structure_net import ncm_word_final_generation_gap as gap
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST=study.data.ROOT/"20261023_word_final_results_v1"
REPORT=Path("docs/08 - Analysis/2026-10-23_ncm-word-final-frozen-parent.md")


def replay_training():
    training=study.recovery.TRAIN
    measured=study.q.read(training/"result.json")
    rows=[json.loads(line) for line in (training/"trace.jsonl").read_text().splitlines()]
    if (len(rows)!=512 or measured["updates"]!=512 or measured["accumulation"]!=4
            or measured["status"]!="word_final_trained_pending_source_only_task_evaluation"
            or not measured["reload_exact"] or not measured["source_only_reload_probe_exact"]
            or measured["parent_loaded"] or not measured["original_parent_weights_frozen"]
            or measured["peak_allocated_bytes"]>8_000_000_000
            or file_hash(training/"model.pt")!=measured["checkpoint_sha256"]):
        raise ValueError("fixed512training and frozen-parent/reload/memory checks required")
    targets=study.q.read(study.data.DATA/"cases.json")["train"]
    lookup={r["id"]:r for r in targets}
    if len(targets)!=252 or len(lookup)!=252:raise ValueError("fixed252training labels required")
    neighbors=torch.load(training/"neighbors.pt",map_location="cpu",weights_only=True)
    config=study.core.Config()
    sampler=torch.Generator().manual_seed(config.sampler_seed)
    noise=torch.Generator().manual_seed(config.noise_seed)
    levels=[4*n for n in (4,6,10,14,20,26,32,40,48)]
    for step,trace in enumerate(rows,1):
        if trace["step"]!=step or trace["lr"]!=study.core.rate(step) or len(trace["samples"])!=4:
            raise ValueError("update count,LR or accumulation differs")
        for sample in trace["samples"]:
            target=targets[int(torch.randint(len(targets),(1,),generator=sampler))]
            changed=study.core.perturb(torch.tensor([target["concept_ids"]]),levels,neighbors,noise,config.probability)
            if (sample["id"]!=target["id"] or sample["source_frame"]!=target["source_frame"]
                    or sample["noise_sha256"]!=study.recovery.base.tensor_digest({"codes":changed})):
                raise ValueError("sample/source-frame/neighborhood replay differs")
    return measured


def replay_scores(rows,packets,role,cache,recorded,endpoint,encoding):
    scores=[]
    for row,packet in zip(rows,packets):
        key=study.first.packet_key(packet,role)
        entry=cache[key];study.first.audit_row(packet,role,entry["row"])
        tokens=len(encoding.encode(row["text"],disallowed_special=()))
        valid=entry["row"]["status"]=="valid"
        scores.append({"id":row["id"],"source_id":row["source_id"],"text":row["text"],"tokens":tokens,
            "valid":valid,"passed":study.q.passes(entry["row"]["evaluations"][0],row["text"],tokens) if valid else None,
            "evaluations":entry["row"].get("evaluations"),"attempts":entry["row"]["attempts"],
            "packet_key":key,"first_origin":entry["origin"]})
    if len(scores)!=64 or recorded!=scores:raise ValueError("all64 first-valid case decisions must replay")
    summary=study.previous.study.teacher.old.summarize_scores(scores)
    if any(endpoint[key]!=value for key,value in summary.items()):
        raise ValueError("complete-summary counts/accuracy/interval differ")
    return summary,scores


def replay_generation_gap():
    study.data.terminal(gap.DEST)
    result=study.q.read(gap.DEST/"result.json")
    measured=study.q.read(gap.DEST/"case_measurements.json")
    if (result["status"]!="TRAIN_only_generation_gap_measured" or result["cases"]!=16
            or result["task_accuracy"] is not None or result["complete_summary_quality_measured"]
            or file_hash(gap.DEST/"case_measurements.json")!=result["case_measurements_sha256"]):
        raise ValueError("complete TRAIN-only component diagnostic required")
    selected=gap.selection(study.q.read(study.data.DATA/"cases.json")["train"])
    neighbors=torch.load(study.recovery.TRAIN/"neighbors.pt",map_location="cpu",weights_only=True)
    if len(measured)!=16:raise ValueError("all16component cases required")
    for case,row in zip(selected,measured):
        if (any(case[k]!=row[k] for k in ("id","source_id","source_sha256","source_cl100k_tokens","source_frame"))
                or row["target_word_ids"]!=case["word_ids"]):
            raise ValueError("TRAIN-only selection/source/word target changed")
        target=torch.tensor([row["target_word_ids"]]);mask=target!=50257
        for name,ids in (("gold_conditioned_words","gold_conditioned_word_ids"),
                ("source_only_words","raw_word_ids"),("own_conditioning_forward_words","own_conditioning_forward_word_ids")):
            actual=torch.tensor([row[ids]])
            if actual.shape!=(1,192) or (actual<0).any() or (actual>=50257).any():
                raise ValueError("complete valid diagnostic word frame required")
            counted={"correct":int(((actual==target)&mask).sum()),"total":int(mask.sum())}
            if row[name]!=counted:raise ValueError("component word recovery differs")
        agreement={"correct":sum(a==b for a,b in zip(row["raw_word_ids"],row["own_conditioning_forward_word_ids"])),"total":192}
        if row["generation_forward_agreement"]!=agreement:raise ValueError("generation/forward agreement differs")
        active=mask.reshape(4,48).any(-1);position=0
        for level,width in enumerate([4*n for n in (4,6,10,14,20,26,32,40)]):
            labels=torch.tensor([case["concept_ids"][position:position+width]])
            actual=torch.tensor([row["source_only_codes"][level]])
            teacher=torch.tensor([row["gold_conditioned_codes"][level]])
            if actual.shape!=labels.shape or teacher.shape!=labels.shape:
                raise ValueError("component concept shape differs")
            valid=active.repeat_interleave(width//4)[None];wrong=(actual!=labels)&valid
            covered=(neighbors[level][labels]==actual[...,None]).any(-1)&wrong
            counted={"level":level,"total":labels.numel(),"active_total":int(valid.sum()),
                "gold_conditioned_correct":int((teacher==labels).sum()),"source_only_correct":int((actual==labels).sum()),
                "gold_conditioned_active_correct":int(((teacher==labels)&valid).sum()),
                "source_only_active_correct":int(((actual==labels)&valid).sum()),
                "source_only_active_wrong":int(wrong.sum()),"wrong_within_eight_neighbors":int(covered.sum())}
            if row["concepts"][level]!=counted:raise ValueError("active-block concept/error-neighborhood counts differ")
            position+=width
    if gap.merge_counts(measured)!=result["components"]:raise ValueError("component aggregate differs")
    return result


def finalize():
    if DEST.exists() or REPORT.exists():raise ValueError("fresh append-only finalization required")
    study.data.prior.dependencies();study.profile.install()
    roots=(study.data.LONG,study.data.DATA,study.data.PILOT,study.data.FEATURES,
        study.recovery.base.initialization.DEST,study.recovery.PILOT,study.recovery.TRAIN,
        study.SOURCES,study.PILOT,study.OUTPUTS,study.PACKETS,study.GRADE)
    for root in roots:study.data.terminal(root)
    failed=study.q.read(study.recovery.FAILED/"worker_terminal_receipt.json")
    if failed["exit_code"]!=1:raise ValueError("preserved initial systems failure required")
    trained=replay_training()
    mechanism=replay_generation_gap()
    generated=study.q.read(study.OUTPUTS/"result.json")
    grade=study.q.read(study.GRADE/"result.json")
    source_sets=study.sources()
    rows=study.q.read(study.OUTPUTS/"outputs.json")
    study.require_outputs(rows,[r for cohort in source_sets.values() for r in cohort])
    if (file_hash(study.OUTPUTS/"outputs.json")!=generated["output_sha256"]
            or generated["checkpoint_sha256"]!=trained["checkpoint_sha256"]
            or rows[:2]!=study.q.read(study.PILOT/"outputs.json")):
        raise ValueError("final checkpoint,raw output hash or retained first two outputs changed")
    from transformers import AutoTokenizer
    import tiktoken
    tokenizer=AutoTokenizer.from_pretrained(study.data.GPT2,local_files_only=True)
    storage=tiktoken.get_encoding("cl100k_base")
    for row in rows:
        text,end=study.decode_words(row["raw_word_ids"],tokenizer)
        if row["text"]!=text or row["global_EOS_position"]!=(end if end<192 else None):
            raise ValueError("global first-EOS/full-frame text replay differs")
        widths=[4*n for n in (4,6,10,14,20,26,32,40)]
        if len(row["coarse_codes"])!=8 or any(len(c)!=n or any(not 0<=v<4096 for v in c)
                for c,n in zip(row["coarse_codes"],widths)):
            raise ValueError("predicted hierarchy shape or vocabulary invalid")
    study.verify_public(rows)
    prior=study.prior_cache();cache=study.q.read(study.GRADE/"judgment_cache.json")
    if grade["prior_first_judgments_preserved"]!=802 or any(cache.get(k)!=v for k,v in prior.items()):
        raise ValueError("all802 earlier first judgments must remain exact")
    packets=study.q.read(study.PACKETS/"exact_packets.json")
    components={};failures={}
    for split in study.COHORTS:
        subset=[r for r in rows if r["split"]==split]
        expected=[study.q.packet(r["id"],r["source"],[{"candidate_id":"summary","text":r["text"]}]) for r in subset]
        if packets[split]!=expected:raise ValueError("public single-candidate packet changed")
        results=grade["results"][split];components[split]={};failures[split]={}
        if (results["primary"] is not None)!=study.previous.study.qualifies(results["secondary"]):
            raise ValueError("secondary-then-primary quality gate order changed")
        for role in ("secondary","primary"):
            endpoint=results[role]
            if endpoint is None:continue
            scores=study.q.read(study.GRADE/f"{split}_{role}_case_scores.json")
            summary,replayed=replay_scores(subset,expected,role,cache,scores,endpoint,storage)
            components[split][role]=summary
            failures[split][role]=[{"id":r["id"],"source":r["source"],"summary":r["text"],
                "tokens":s["tokens"],"valid":s["valid"],"evaluations":s["evaluations"],
                "first_origin":s["first_origin"]} for r,s in zip(subset,replayed) if s["passed"] is not True]
        passed=results["primary"] is not None and study.previous.study.qualifies(results["primary"])
        if results["passed"]!=passed:raise ValueError("cohort quality gate differs")
    passed=all(r["passed"] for r in grade["results"].values())
    classification="development_gates_passed_final_evaluation_required" if passed else "word_final_NCM_development_quality_failed"
    if grade["classification"]!=classification or grade["development_gate"]!=passed or grade["NCM_success_established"]:
        raise ValueError("final/unseen quality scope must remain unproven")
    paths=[Path(__file__),Path(study.__file__),study.data.PLAN,study.data.AMENDMENT,
        study.recovery.AMENDMENT,Path(gap.__file__),gap.PLAN,Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        study.previous.GRADE/"judgment_cache.json",study.recovery.FAILED/"producer.json",
        study.recovery.FAILED/"worker_terminal_receipt.json",
        study.data.ROOT/"20261023_word_final_train_pilot_v1.worker.log"]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    paths += [p for p in gap.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir();write_json(DEST/"case_failures.json",failures)
    paths.append(DEST/"case_failures.json")
    aggregate={"schema":"nal.ncm-word-final-development.v1","hypothesis_id":study.data.HYPOTHESIS,
        "classification":classification,"training":trained,"generation":generated,"grading":grade,
        "components":components,"TRAIN_only_generation_gap":mechanism,"training_samples_noise_LRs_replayed":True,
        "raw_words_texts_and_first_judgments_replayed":True,"prior_802_judgments_preserved":True,
        "initial_systems_failure_preserved":True,"task_accuracy_measured":True,"NCM_adaptation":True,
        "NCM_success_established":False,"confirmed":False,"final_test_opened":False,"speed_tested":False,
        "artifacts":{str(p):file_hash(p) for p in dict.fromkeys(paths)}}
    campaign=DEST/"campaign_results.json";write_json(campaign,aggregate)
    table=[]
    for split,roles in components.items():
        for role,m in roles.items():
            c=m["components"]
            table.append(f"| {split} | {role} | {m['passed']}/64 | {m['valid']}/64 | {c['faithful']} | {c['salient']} | {c['readable']} | {m['stored_overflow']} |")
    m=mechanism["components"]
    REPORT.write_text(f'''# Word-final NCM attached to a frozen Llama body

Date:2026-10-23. Hypothesis `{study.data.HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Development cohort | Judge role | Complete-summary passes | Valid | Faithful | Salient | Readable | Over128tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
'''+'\n'.join(table)+f'''

These are source-only complete-summary development results, requiring supported
material claims, useful salient content, readability, nonempty text and at most
128stored cl100k tokens. Each cohort must independently score at least61/64
with64valid Terra judgments before the same texts reach Sonnet; Sonnet requires
the same gate. The two development gates {'passed' if passed else 'failed'}.
Per-cohort Wilson intervals remain in the aggregate. The reused short cohort
is a development diagnostic, not a fresh final test. Original256and longer64
final cohorts remain closed. No final95%quality, compression or speedup claim.

The model keeps eight learned concept levels and predicts final GPT-2 words
directly through a trainable parallel projection/decoder. It is an NCM
adaptation, not a replication of the published final VQ mechanism. Full frozen
28-layer Llama3.2-3B source features form a zero-initialized residual over the
released raw GPT-2 source embedding path. Every original parent weight and
MLP remains frozen; its source pass still computes all original blocks.

Training used252complete vetted summaries and exactly512updates, microbatch1
and accumulation4. Released generator weights were copied, root queries
repeated for four48-position blocks, and geometry rebuilt. New word projection
and decoder copy released weights;110,298,376new-module parameters may train.
The original codec remains exact. FP32master/AdamW with BF16autocast and FP32
losses/distances used clean wordCE+.1noisywordCE+.1meancoarseCE. Noise replaces
5%of earlier concepts using eight nearest non-self codebook neighbors. This is
supervised corruption, not a replication of CLLM fixed-point training. Only
the final512checkpoint was selected. All2048sample identities, frames, noise
hashes and learning rates replayed. No adaptive budget extension occurred.

The initial CUDA pilot stopped before updates because released RoPE frequencies
were onCPU while positions were onCUDA. Its exit1and traceback remain intact.
A separate device-only recovery calculates reference RoPE onCPU and transfers
it, passing exact CPU/CUDA geometry checks and the actual longest-frame
two-update gradient/memory/checkpoint/optimizer/RNG/inference reload pilot.
Full training started fresh after its observed exit0. Peak training allocation
was{trained['peak_allocated_bytes']:,}bytes; this is not a latency comparison.

All64short and64longer sources are disjoint from252training cases by
case/session/group/hash. Source metadata for inference contains no target
summaries or concepts. Every source token and all192raw word positions were
retained, with eight predicted coarse levels and one word level. Text is
decoded once through the first global EOS; no oracle concepts, previous-word
labels, autoregressive fallback, per-block EOS repair, reroll or best-of
selection. The first two outputs were retained exactly after deterministic
checkpoint-reload qualification. Raw IDs, full texts and all first valid
judgments replayed. All802prior judgments, including failures, stayed exact.
Public full-source payload proofs were fixed before external judging. Detailed
failures are retained in `{DEST/'case_failures.json'}`. All earlier codec and
native-hybrid failures remain failures. Quality must precede speed claims.

The separately planned16-TRAIN-case mechanism diagnostic recovered
{m['gold_conditioned_words']['correct']}/{m['gold_conditioned_words']['total']}
non-PAD target words with correct earlier concepts, versus
{m['source_only_words']['correct']}/{m['source_only_words']['total']} with
source-only generated concepts. Generation agreed with the full forward at its
own concepts on{m['generation_forward_agreement']['correct']}/3072raw word IDs.
The first concept level recovered only{m['concepts'][0]['source_only_active_correct']}/
{m['concepts'][0]['active_total']}labels in blocks containing target text.
These are TRAIN-only literal component measurements, not task accuracy or a
new codec qualification. They support investigating source-conditioned concept
learning and the mismatch between training and generated histories. Alternate
wording can be valid; this diagnostic alone cannot assign semantic failure.
''')
    print(json.dumps({"status":"finalized","classification":classification,"components":components}),flush=True)


def register():
    aggregate=study.q.read(DEST/"campaign_results.json")
    spec={"name":"Parallel word-final NCM on complete frozen-Llama source features",
        "description":"Fixed512updates on252complete summaries;8learned coarse levels plus direct final words;originalLlama3B weights andMLPs frozen. Source-disjoint64short+64long development, qualifiedTerra thenSonnet, all802priorfirstjudgments preserved. InitialCUDAdevice failure preserved. Finalcohortsclosed;nospeedclaim.",
        "confirmation_status":aggregate["classification"],"task_accuracy_measured":True,
        "NCM_adaptation":True,"NCM_success_established":False,
        "development_gate":aggregate["grading"]["development_gate"],"training_updates":512,
        "original_parent_weights_frozen":True,"final_test_opened":False,"speed_tested":False}
    for split,roles in aggregate["components"].items():
        for role,m in roles.items():
            for key in ("cases","passed","valid","invalid","stored_overflow","empty"):
                spec[f"{split}_{role}_{key}"]=m[key]
            if m["task_accuracy"] is not None:spec[f"{split}_{role}_task_accuracy"]=m["task_accuracy"]
    register_records(DEST/"campaign_results.json",REPORT,DEST,{study.data.HYPOTHESIS:spec})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["finalize","register"])
    globals()[parser.parse_args().stage]()
