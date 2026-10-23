"""Replay codec reconstruction counts without labeling them task accuracy."""
import argparse
import json
from pathlib import Path
import re

from experiments.structure_net import ncm_broad_codec_reconstruction as measurement
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

study=measurement.study
DEST=study.ROOT/"20261023_broad_learned_codec_results_v1"
REPORT=Path("docs/08 - Analysis/2026-10-23_ncm-broad-learned-codec-reconstruction.md")


def replay(cases,rows,tokenizer):
    if len(rows)!=252 or [r["id"] for r in rows]!=[c["id"] for c in cases]:
        raise ValueError("all252 ordered complete targets required")
    numbers=lambda text:re.findall(r"\b\d+(?:[,.]\d+)*\b",text)
    for c,r in zip(cases,rows):
        raw=r["raw_ids"];gold=c["target_ids"]
        if (len(raw)!=192 or any(not isinstance(t,int) or not 0<=t<50304 for t in raw)
                or len(r["codes"])!=4 or any(len(v)!=200 or any(not 0<=t<4096 for t in v) for v in r["codes"])
                or r["target_ids"]!=gold or r["target"]!=c["text"]):
            raise ValueError("raw full-frame word/code representation differs")
        end=next((i for i,t in enumerate(raw) if t==50256),192)
        text=tokenizer.decode(raw[:end],skip_special_tokens=True)
        expected={"text":text,"correct":sum(a==b for a,b in zip(raw,gold)),"total":len(gold),
            "exact_text":text==c["text"],"first_EOS_correct":end==len(gold)-1,
            "active_invalid_ids":sum(t>=len(tokenizer) for t in raw[:end]),
            "changed_numbers":numbers(text)!=numbers(c["text"])}
        if any(r[k]!=v for k,v in expected.items()):
            raise ValueError("raw reconstruction score/text replay differs")
    return measurement.summarize(rows)


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("completed codec finalization is append-only")
    roots=(study.corpus.DEST,study.PILOT,study.TRAIN,study.RECON)
    for root in roots:study.terminal(root)
    result=json.loads((study.RECON/"result.json").read_text())
    trained=json.loads((study.TRAIN/"result.json").read_text())
    pilot=json.loads((study.PILOT/"result.json").read_text())
    corpus=json.loads((study.corpus.DEST/"result.json").read_text())
    if (trained["updates"]!=2048 or not trained["reload_exact"] or not pilot["reload_exact"]
            or trained["initial_model_sha256"]!=pilot["initial_model_sha256"]
            or trained["parent_loaded"] or file_hash(study.TRAIN/"model.pt")!=trained["checkpoint_sha256"]
            or corpus["fit_windows"]!=31109 or corpus["summary_diagnostic_cases"]!=252
            or not corpus["source_disjoint"] or corpus["summary_truncated"]):
        raise ValueError("fixed broad codec/corpus/reload controls required")
    cases=json.loads((study.corpus.DEST/"summary_diagnostic.json").read_text())
    origins=json.loads((study.corpus.DEST/"fit_origins.json").read_text())
    for key in ("source_id","group_id","source_sha256"):
        if {c[key] for c in cases}&{o[key] for r in origins for o in r["origins"]}:
            raise ValueError("codec fit and target source identities overlap")
    if {text_hash(c["text"]) for c in cases}&{r["text_sha256"] for r in origins}:
        raise ValueError("exact diagnostic target text entered codec fitting")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(study.corpus.GPT2,local_files_only=True)
    tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    components={};failures={}
    for arm in ("original","trained"):
        path=study.RECON/(arm+"_outputs.json")
        if file_hash(path)!=result["output_hashes"][arm]:
            raise ValueError("frozen reconstruction outputs differ")
        rows=json.loads(path.read_text());components[arm]=replay(cases,rows,tokenizer)
        if components[arm]!=result[arm]:raise ValueError("codec aggregate differs")
        failures[arm]=[r for r in rows if not r["exact_text"]]
    passed=measurement.gate(components["original"],components["trained"])
    expected="codec_reconstruction_gate_passed_semantic_audit_required" if passed else "codec_reconstruction_fidelity_failed"
    if (result["representation_gate"]!=passed or result["classification"]!=expected
            or result["source_prediction"] or not result["oracle_target_codes"]
            or result["task_accuracy"] is not None or result["NCM_success_established"]):
        raise ValueError("representation gate or task-accuracy scope changed")
    paths=[Path(__file__),Path(study.__file__),Path(measurement.__file__),study.PLAN,study.corpus.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate={"schema":"nal.ncm-broad-learned-codec-results.v1","classification":expected,
        "corpus":corpus,"training":trained,"pilot":pilot,"reconstruction":result,"components":components,
        "all252raw_codes_words_and_counts_replayed":True,"fit_diagnostic_source_disjoint":True,
        "oracle_target_codes":True,"source_prediction":False,"task_accuracy_measured":False,
        "NCM_success_established":False,"confirmed":False,"final_test_opened":False,"speed_tested":False,
        "artifacts":{str(p):file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir();write_json(DEST/"nonexact_reconstructions.json",failures)
    aggregate["artifacts"][str(DEST/"nonexact_reconstructions.json")]=file_hash(DEST/"nonexact_reconstructions.json")
    campaign=DEST/"campaign_results.json";write_json(campaign,aggregate)
    table=[]
    for arm,m in components.items():
        table.append(f"| {arm} | {m['correct']}/{m['total']} ({100*m['token_recovery']:.3f}%) | {m['exact_texts']}/252 | {m['first_EOS_correct']}/252 | {m['changed_number_cases']} | {m['active_invalid_ids']} |")
    REPORT.write_text(f'''# Broad learned codec reconstruction on excluded summary texts

Date: 2026-10-23. Hypothesis `ncm-broad-learned-codec-reconstruction-v1`.
Classification **{expected}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences | Active invalid IDs |
| --- | --- | --- | --- | --- | --- |
'''+"\n".join(table)+f'''

Both codecs receive their own full correct target-derived codes. These are
oracle reconstruction results, not source-to-summary task accuracy. The
representation gate **{'passed' if passed else 'failed'}**: at least 99.5% token
recovery, at least 99% first-EOS agreement, zero active invalid IDs, and no
increase in changed-number cases. Exact text matching is descriptive. Numeric
sequence differences are a diagnostic and are not independently judged factual
errors. Passing would require a separate semantic audit before any generator
quality claim. The original parent is absent from this component study.

The complete 252 targets were excluded from fitting by source session, group,
exact input hash and exact target text. They remain training labels for a later
conditional summarizer, not a final application test. Every target fits four
48-position GPT-2 blocks without clipping; the longest needs 136 tokens with
EOS. Raw predicted IDs are concatenated before the first global EOS and text
decoding. There is no per-block EOS insertion, output repair or choice between
arms. All raw codes, word IDs, stopping decisions and counts replayed exactly.

The new codec copy started from the released 60.63M-parameter XSum codec. It
completed exactly 2,048 FP32 updates on GPU2, microbatch four and accumulation
eight, using 31,109 windows from 7,088 unpaired texts in 1,024 training groups.
The corpus contains 1,303,913 valid token positions including genuine EOS.
Saved atomic summaries supplied unpaired text, not complete-passage labels.
Encoder, decoder, quantization modules and EMA books used the released forward
with reconstruction CE plus its returned VQ loss and .5 fine-level dropout.
AdamW excludes EMA books. All original parent weights, including its MLPs,
remain outside this training process. The released checkpoint file is intact.

The actual two-update gradient/memory pilot passed before the fixed run. The
full run restored fresh released weights, not pilot weights, and evaluated
only its final checkpoint. Model, optimizer, CPU/CUDA/sampler RNG, nonpersistent
dropout step state and probe outputs reloaded exactly. Peak allocated memory
was {trained['peak_allocated_bytes']:,} bytes. Original and final evaluation
states remained unchanged. Frozen code/config/input identities and observed
terminal-success receipts are retained in the aggregate's artifact map.

This is a transfer experiment around released NCM modules. The official release
does not include its trainer; the returned all-level VQ loss is not asserted to
reproduce the paper's sampled-cutoff objective. Four-block text representation
also differs from the original single 48-token target. No conditional concept
generator was trained here. The clean native teacher's separate 56/64 (87.5%)
development result remains below the goal. The original and longer-source final
task cohorts remain sealed, and no latency, compression or tokens/s benefit is
established by this component experiment.
''')
    print(json.dumps({"status":"finalized","classification":expected,"components":components}),flush=True)


def register():
    x=json.loads((DEST/"campaign_results.json").read_text())
    register_records(DEST/"campaign_results.json",REPORT,DEST,{"ncm-broad-learned-codec-reconstruction-v1":{
        "name":"Broad unpaired learned-codec transfer with excluded complete-summary reconstruction",
        "description":"Released learned NCM codec,2048FP32updates on31109windows;252source-disjoint complete targets reconstructed with own oraclecodes. Raw words/codes/counts replayed. Originalparentnotloaded; no sourceprediction, taskaccuracy orspeedclaim.",
        "confirmation_status":x["classification"],"task_accuracy_measured":False,
        "oracle_target_codes":True,"source_prediction":False,"NCM_success_established":False,
        "components":x["components"],"representation_gate":x["reconstruction"]["representation_gate"]}})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("command",choices=["finalize","register"])
    globals()[parser.parse_args().command]()
