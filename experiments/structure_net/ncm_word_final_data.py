"""Complete public sources and character-aligned frozen-parent token features."""
import argparse
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_isolated_quality_data as prior
from experiments.structure_net import ncm_frozen_source_head as source
from experiments.structure_net.ncm_llama import GPT2, INITIAL, VENDOR
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash
from experiments.structure_net.ncm_native_parent_control import SYSTEM
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE,PUBLIC_SHA256
from experiments.structure_net.ncm_prefix_geometry import source_frame
from experiments.structure_net.ncm_blockwise_codec import to_level_major

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-word-final-frozen-parent-plan.md")
DATA=ROOT/"20261023_word_final_data_v1"
PILOT=ROOT/"20261023_word_final_feature_pilot_v1"
FEATURES=ROOT/"20261023_word_final_features_v1"
ORACLE=ROOT/"20261023_frozen_representation_decoder_reconstruction_v1"
HYPOTHESIS="ncm-word-final-frozen-parent-v1"
LONG=ROOT/"20261023_word_final_long_diagnostic_v1"
AMENDMENT=Path("docs/07 - Status Reports/2026-10-23_ncm-word-final-diagnostic-expansion.md")


def overlaps(query_offsets,native_offsets,start):
    native=sorted((a,b,i) for i,(a,b) in enumerate(native_offsets) if b>a)
    result=[];first=0
    for qa,qb in query_offsets:
        if qb<=qa:
            raise ValueError("empty source token character span")
        a,b=qa+start,qb+start
        while first<len(native) and native[first][1]<=a:
            first+=1
        values=[];cursor=first
        while cursor<len(native) and native[cursor][0]<b:
            na,nb,i=native[cursor];weight=max(0,min(nb,b)-max(na,a))
            if weight:values.append((i,weight))
            cursor+=1
        total=sum(w for _,w in values)
        if not total:
            raise ValueError("source token has no native feature overlap")
        result.append([[i,w/total] for i,w in values])
    return result


def align(hidden,alignment):
    if hidden.ndim!=2 or not alignment:
        raise ValueError("full native feature matrix and alignment required")
    rows=[];columns=[];values=[]
    for row,entries in enumerate(alignment):
        if not entries or abs(sum(v for _,v in entries)-1)>1e-6:
            raise ValueError("each token needs normalized feature weights")
        for column,value in entries:
            if not 0<=column<len(hidden) or value<=0:
                raise ValueError("invalid overlap feature reference")
            rows.append(row);columns.append(column);values.append(value)
    matrix=torch.sparse_coo_tensor(torch.tensor([rows,columns]),torch.tensor(values,dtype=torch.float32),
        (len(alignment),len(hidden))).coalesce()
    return torch.sparse.mm(matrix,hidden.detach().cpu().float())


def producing(root,stage,inputs,config):
    files=[Path(__file__),PLAN,AMENDMENT,Path(source.__file__),Path("experiments/structure_net/ncm_prefix_geometry.py")]
    p=freeze(root,stage,list(inputs)+files,{**config,"hypothesis_id":HYPOTHESIS,
        "source_embedding":"GPT2raw+zero_initialized_parent_residual","max_source_frame":8192,
        "source_right_padding":True,"parent_precision":"BF16","aligned_features":"FP32",
        "parent_layers":28,"parent_weights_frozen":True,"original_MLPs_frozen":True,
        "source_prediction":False,"task_accuracy":None,"final_test_opened":False,"speed_tested":False})
    p["hypothesis_id"]=HYPOTHESIS
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"word_data_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def terminal(root):
    if prior.run.q.read(root/"worker_terminal_receipt.json")["exit_code"]!=0:
        raise ValueError("observed terminal success required: "+str(root))
    verify(prior.run.q.read(root/"producer.json"))


def prepare():
    prior.dependencies()
    from transformers import AutoTokenizer
    import tiktoken
    prior.terminal(prior.DEST)
    terminal(LONG)
    native=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    gpt=AutoTokenizer.from_pretrained(GPT2,local_files_only=True)
    gpt.add_special_tokens({"pad_token":"<|pad|>"});gpt.model_max_length=1_000_000
    storage=tiktoken.get_encoding("cl100k_base")
    parts={"train":prior.run.q.read(prior.DEST/"cases.json"),
        "short_development":prior.run.q.read(PREP/"development_sources.json"),
        "long_diagnostic":prior.run.q.read(LONG/"sources.json")}
    if [len(parts[k]) for k in parts]!=[252,64,64]:
        raise ValueError("fixed252/64/64cohorts required")
    for a,b in (("train","short_development"),("train","long_diagnostic"),("short_development","long_diagnostic")):
        prior.check_disjoint(parts[a],parts[b])
    all_rows=[r for rows in parts.values() for r in rows]
    public_proof=verify_public(all_rows)
    metadata=prior.run.q.read(prior.POOL_META)
    sealed=metadata["sealed_final"]
    for key in prior.IDENTITY:
        if {r[key] for r in all_rows}&{r[key] for r in sealed}:
            raise ValueError("long sealed cohort identity overlap")
    original_final=prior.original_final_metadata(prior.run.q.read(PREP/"public_proof.json"),
        prior.run.q.read(PREP/"train_sources.json"),parts["short_development"])
    if set(original_final)&{r["id"] for r in all_rows} or set(original_final.values())&{r["source_sha256"] for r in all_rows}:
        raise ValueError("original final identity overlap")
    oracle={r["id"]:r for r in prior.run.q.read(ORACLE/"original_outputs.json")}
    inputs=[prior.DEST/n for n in ("producer.json","result.json","cases.json","split_audit.json","worker_terminal_receipt.json")]
    inputs += [PREP/n for n in ("development_sources.json","train_sources.json","public_proof.json")]
    inputs += [PUBLIC_FILE,prior.POOL_META,ORACLE/"original_outputs.json",ORACLE/"result.json",ORACLE/"worker_terminal_receipt.json"]
    inputs += [LONG/n for n in ("producer.json","result.json","sources.json","worker_terminal_receipt.json")]
    inputs += [MODEL_PATH/n for n in ("config.json","tokenizer.json","tokenizer_config.json")]
    inputs += [GPT2/n for n in ("tokenizer.json","tokenizer_config.json") if (GPT2/n).exists()]
    p=producing(DATA,"complete_public_metadata_and_token_alignment",inputs,{"cases":380,"training_labels":252,
        "short_development":64,"long_diagnostic":64,"original_final_text_read":False,"long_final_text_read":False})
    prepared={}
    for split,rows in parts.items():
        output=[]
        for row in rows:
            context=native.apply_chat_template([{"role":"system","content":SYSTEM},
                {"role":"user","content":"<passage>\n"+row["source"]+"\n</passage>"}],add_generation_prompt=True,tokenize=False)
            encoded=native(context,add_special_tokens=False,return_offsets_mapping=True)
            direct=native.apply_chat_template([{"role":"system","content":SYSTEM},
                {"role":"user","content":"<passage>\n"+row["source"]+"\n</passage>"}],add_generation_prompt=True)
            if encoded["input_ids"]!=direct or (split=="train" and direct!=row["context_ids"]):
                raise ValueError("canonical native prompt changed")
            start=context.index("<passage>\n")+len("<passage>\n")
            if context[start:start+len(row["source"])]!=row["source"]:
                raise ValueError("literal complete passage boundary differs")
            raw=gpt(row["source"],add_special_tokens=False,return_offsets_mapping=True)
            if gpt.decode(raw["input_ids"],skip_special_tokens=False)!=row["source"]:
                raise ValueError("source GPT2 round trip failed")
            mapping=overlaps(raw["offset_mapping"],encoded["offset_mapping"],start)
            item={**{k:row[k] for k in (*prior.IDENTITY,"source","spans")},"split":split,
                "context_ids":direct,"source_ids":raw["input_ids"],"alignment":mapping,
                "alignment_sha256":prior.run.q.digest(mapping),"source_frame":source_frame(len(raw["input_ids"]),maximum=8192),
                "source_cl100k_tokens":len(storage.encode(row["source"],disallowed_special=())),
                "native_prompt_sha256":prior.run.q.digest(direct)}
            if split=="train":
                target=oracle[row["id"]]
                ids=gpt.encode(row["target_text"],add_special_tokens=False)+[50256]
                if (target["target"]!=row["target_text"] or target["target_ids"]!=ids
                        or len(ids)>192 or len(storage.encode(row["target_text"],disallowed_special=()))>128):
                    raise ValueError("complete word target identity/capacity differs")
                codes=to_level_major(torch.tensor(target["codes"],dtype=torch.long),(4,6,10,14,20,26,32,40,48),4)[0].tolist()
                item.update(target_text=row["target_text"],word_ids=ids+[50257]*(192-len(ids)),concept_ids=codes)
            output.append(item)
        prepared[split]=output
    write_json(DATA/"cases.json",prepared)
    write_json(DATA/"public_proof.json",{"public_file_sha256":PUBLIC_SHA256,"exact_sources":public_proof,
        "matched":len(public_proof),"outgoing_private_data":False})
    first=prepared["train"][0]
    longest=sorted([r for rows in prepared.values() for r in rows],key=lambda r:(-len(r["context_ids"]),r["id"]))[0]
    longest_train=sorted(prepared["train"],key=lambda r:(-len(r["source_ids"]),r["id"]))[0]
    write_json(DATA/"selection.json",{"feature_pilot_ids":[first["id"],longest["id"]],
        "training_pilot_ids":[first["id"],longest_train["id"]]})
    verify(p)
    result={"status":"complete_word_final_metadata_prepared","counts":{k:len(v) for k,v in prepared.items()},
        "max_native_prompt":max(len(r["context_ids"]) for rows in prepared.values() for r in rows),
        "max_GPT2_source":max(len(r["source_ids"]) for rows in prepared.values() for r in rows),
        "max_source_frame":max(r["source_frame"] for rows in prepared.values() for r in rows),
        "max_GPT2_target":max(sum(t!=50257 for t in r["word_ids"]) for r in prepared["train"]),
        "source_disjoint":True,"all_sources_complete":True,"task_accuracy":None,
        "cases_sha256":file_hash(DATA/"cases.json"),"producer_fingerprint":p["fingerprint"]}
    write_json(DATA/"result.json",result);print(json.dumps(result),flush=True)


def require_gpu():
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0]<8_600_000_000):
        raise ValueError("planned GPU0 with8.6GBfree required")
    torch.backends.cuda.matmul.allow_tf32=False


def feature_cache(pilot=False):
    require_gpu();terminal(DATA)
    prepared=prior.run.q.read(DATA/"cases.json")
    if file_hash(DATA/"cases.json")!=prior.run.q.read(DATA/"result.json")["cases_sha256"]:
        raise ValueError("prepared metadata changed")
    all_rows=[r for rows in prepared.values() for r in rows]
    selected=prior.run.q.read(DATA/"selection.json")["feature_pilot_ids"]
    if not pilot:terminal(PILOT)
    rows=[next(r for r in all_rows if r["id"]==identifier) for identifier in selected] if pilot else all_rows
    root=PILOT if pilot else FEATURES
    inputs=[DATA/n for n in ("producer.json","result.json","cases.json","selection.json","worker_terminal_receipt.json")]
    inputs += [MODEL_PATH/n for n in ("config.json","model.safetensors.index.json","model-00001-of-00002.safetensors","model-00002-of-00002.safetensors")]
    if not pilot:inputs += [PILOT/n for n in ("producer.json","result.json","manifest.json","worker_terminal_receipt.json")]
    p=producing(root,"frozen_parent_feature_pilot" if pilot else "frozen_parent_source_features",inputs,
        {"pilot":pilot,"physical_gpu":0,"cases":len(rows),"reuse_pilot_cases":0 if pilot else 2})
    from transformers import AutoModelForCausalLM
    parent=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers!=28 or module_hash(parent.model)!=EXPECTED_BODY:
        raise ValueError("original28-layer parent body identity differs")
    torch.cuda.reset_peak_memory_stats()
    (root/"features").mkdir();manifest={}
    prior_manifest={} if pilot else prior.run.q.read(PILOT/"manifest.json")
    for i,row in enumerate(rows,1):
        if row["id"] in prior_manifest:
            entry=prior_manifest[row["id"]]
            if file_hash(entry["path"])!=entry["sha256"]:
                raise ValueError("retained feature pilot changed")
            manifest[row["id"]]=entry
        else:
            ids=torch.tensor([row["context_ids"]],device="cuda",dtype=torch.long)
            mask=torch.ones_like(ids)
            hidden=source.frozen_parent_features(parent,ids,mask)[0]
            features=align(hidden,row["alignment"])
            if pilot:
                repeated=source.frozen_parent_features(parent,ids,mask)[0]
                if not torch.equal(hidden,repeated) or not torch.equal(features,align(repeated,row["alignment"])):
                    raise ValueError("frozen feature extraction is not exactly repeatable")
            if features.shape!=(len(row["source_ids"]),3072) or not torch.isfinite(features).all():
                raise ValueError("incomplete/nonfinite aligned source features")
            path=root/"features"/(row["id"]+".pt")
            torch.save({"id":row["id"],"source_sha256":row["source_sha256"],"features":features,
                "alignment_sha256":row["alignment_sha256"],"native_prompt_sha256":row["native_prompt_sha256"]},path)
            manifest[row["id"]]={"path":str(path),"sha256":file_hash(path),"source_tokens":len(features)}
        write_json(root/"manifest.json",manifest)
        if pilot or i%16==0 or i==len(rows):print(json.dumps({"cached":i,"total":len(rows)}),flush=True)
    if module_hash(parent.model)!=EXPECTED_BODY or any(v.grad is not None for v in parent.parameters()):
        raise ValueError("frozen parent changed or received gradients")
    if torch.cuda.max_memory_allocated()>8_600_000_000:
        raise ValueError("feature cache allocation exceeds plan")
    verify(p)
    result={"status":"systems_feature_identity_passed_not_quality" if pilot else "frozen_source_features_prepared",
        "cases":len(manifest),"parent_body_hash_unchanged":EXPECTED_BODY,"all_source_tokens_retained":True,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"task_accuracy":None,
        "manifest_sha256":file_hash(root/"manifest.json"),"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["prepare","feature-pilot","features"])
    stage=parser.parse_args().stage
    prepare() if stage=="prepare" else feature_cache(stage=="feature-pilot")
