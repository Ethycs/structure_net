"""Fixed accepted-summary labels and source-disjoint native language windows."""
import json
from pathlib import Path
import torch
from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_control as accepted
from experiments.structure_net.ncm_frozen_body import MODEL_PATH
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_paper_data import ROOT,PREP,freeze,verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_source_readout_grade import audit_row,packet_key

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-query-neighborhood-training-plan.md")
DEST = ROOT/"20261023_native_query_neighborhood_data_v1"
SOURCE = ROOT/"20261022_expanded_data_v2/train_sources.json"
CRITIQUE = ROOT/"20261023_revised_labels_main_critique_v2"
DRAFT = ROOT/"20261023_revised_labels_main_draft_v1"
TEACHER_PREP = ROOT/"20261023_native_labels_prepare_v1/sources.json"
BOOK_WINDOWS = ROOT/"20261023_native_concepts_prepare_v1/windows.json"


def source_inputs(cases,pad_id,device):
    if not cases or len({c["source_frame"] for c in cases}) != 1:
        raise ValueError("one homogeneous task frame per source batch required")
    frame = cases[0]["source_frame"]
    ids = torch.full((len(cases),frame),pad_id,dtype=torch.long,device=device)
    mask = torch.zeros_like(ids)
    for i,case in enumerate(cases):
        value = case["context_ids"]
        if not 2 <= len(value) <= frame: raise ValueError("source context must fit without clipping")
        ids[i,-len(value):] = torch.tensor(value,device=device)
        mask[i,-len(value):] = 1
    return {"input_ids":ids,"attention_mask":mask}


def target_tensor(cases,device):
    result = torch.full((len(cases),128),-100,dtype=torch.long,device=device)
    for i,c in enumerate(cases):
        if not 1 <= len(c["target_ids"]) <= 128: raise ValueError("native target outside128frame")
        result[i,:len(c["target_ids"])] = torch.tensor(c["target_ids"],device=device)
    return result


def identity(case):
    return {k:case[k] for k in ("id","source_id","group_id","source_sha256")}


def selected_summaries():
    # Replay the older labels under their ORIGINAL request rubric, then the
    # new labels under128-token grading. Neither first decision is rewritten.
    inherited = q.RUBRIC
    original = profile.RUBRIC.replace(profile.NEW_PHRASE,profile.OLD_PHRASE)
    q.RUBRIC = q.original.RUBRIC = original
    try:
        previous,extra = accepted.selected_cases()
    finally:
        q.RUBRIC = q.original.RUBRIC = inherited
    profile.install()
    verify(q.read(CRITIQUE/"producer.json"))
    if q.read(CRITIQUE/"worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("new first judgments must be terminal")
    cache = q.read(CRITIQUE/"judgment_cache.json")
    drafts = q.read(DRAFT/"labels.json")
    selected = []
    for case in drafts:
        packet = q.packet(case["id"],case["source"],[{"candidate_id":"summary","text":case["text"]}])
        key = packet_key(packet,"secondary")
        entry = cache[key]; row = entry["row"]
        audit_row(packet,"secondary",row)
        if row["status"] == "valid" and q.passes(row["evaluations"][0],case["text"],case["stored_tokens"]):
            selected.append({**case,"target":case["text"],"label_judge":q.JUDGES["secondary"],
                "label_profile":"budget128-v1","label_evidence":key})
    if len(selected) != 111: raise ValueError("fixed111new accepted labels changed")
    selected += [{**c,"text":c["target"],"label_judge":q.JUDGES["primary"],
                  "label_profile":"original32","label_evidence":str(accepted.QUALITY)} for c in previous]
    if len(selected) != 143 or len({c["group_id"] for c in selected}) != 143:
        raise ValueError("fixed143accepted source groups required")
    return selected,extra


def prepare():
    from experiments.structure_net.ncm_native_label_quality import dependencies
    dependencies()
    from transformers import AutoTokenizer
    import tiktoken
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    stored = tiktoken.get_encoding("cl100k_base")
    summaries,audits = selected_summaries()
    verify(q.read(SOURCE.parent/"producer.json"))
    sources = q.read(SOURCE)
    development = q.read(PREP/"development_sources.json")
    summary_rows = []
    for case in summaries:
        target = tokenizer.encode(case["target"],add_special_tokens=False)+[tokenizer.eos_token_id]
        if (len(target)>128 or len(stored.encode(case["target"],disallowed_special=()))>128
                or tokenizer.decode(target[:-1],skip_special_tokens=True) != case["target"]):
            raise ValueError("accepted target no longer fits/round-trips")
        _,context = native_inputs([case],tokenizer,"cpu")
        summary_rows.append({**identity(case),"kind":"summary","source_frame":768,"context_ids":context[0],
            "target_ids":target,"target_text":case["target"],"window_eos_injected":False,
            "label_judge":case["label_judge"],"label_profile":case["label_profile"],"label_evidence":case["label_evidence"]})
    teacher_sources = q.read(TEACHER_PREP)
    forbidden_groups = {c["group_id"] for c in [*teacher_sources,*summaries]}
    forbidden_hashes = {c["source_sha256"] for c in [*teacher_sources,*summaries]}
    book_groups = {c["group_id"] for c in q.read(BOOK_WINDOWS)}
    tokenized = {}
    def ids_for(case):
        if case["id"] not in tokenized:
            tokenized[case["id"]] = tokenizer.encode(case["source"],add_special_tokens=True)
        return tokenized[case["id"]]
    def choose(prefix,count,groups,hashes):
        result = []
        for case in sorted(sources,key=lambda c:text_hash(prefix+c["id"])):
            if case["group_id"] in groups or case["source_sha256"] in hashes: continue
            ids = ids_for(case)
            if len(ids)<255: continue
            result.append(case)
            groups.add(case["group_id"]); hashes.add(case["source_sha256"])
            if len(result)==count: return result
        raise ValueError("not enough disjoint native-language windows")
    diagnostic = choose("native-query-language-diagnostic-v1:",64,
        forbidden_groups|book_groups,set(forbidden_hashes))
    language = choose("native-query-language-train-v1:",1024,
        forbidden_groups|{c["group_id"] for c in diagnostic},
        forbidden_hashes|{c["source_sha256"] for c in diagnostic})
    verify_public([*summaries,*language,*diagnostic])
    def language_rows(cases):
        return [{**identity(c),"kind":"language","source_frame":128,"context_ids":ids_for(c)[:128],
            "target_ids":ids_for(c)[128:255]+[tokenizer.eos_token_id],"window_eos_injected":True,
            "native_window":{"context_start":0,"context_end":128,"target_start":128,"target_end":255}}
            for c in cases]
    cohorts = {"language_train":language_rows(language),"language_diagnostic":language_rows(diagnostic),
               "summary_train":summary_rows}
    for key in ("id","source_id","group_id","source_sha256"):
        seen = {c[key] for c in development}
        for rows in cohorts.values():
            value = {c[key] for c in rows}
            if len(value)!=len(rows) or seen&value: raise ValueError("source-disjoint split contract failed: "+key)
            seen |= value
    inputs = [SOURCE,SOURCE.parent/"producer.json",CRITIQUE/"producer.json",CRITIQUE/"judgment_cache.json",
        CRITIQUE/"result.json",DRAFT/"labels.json",TEACHER_PREP,BOOK_WINDOWS,PREP/"development_sources.json"]+audits
    inputs += [accepted.QUALITY/n for n in ("producer.json","result.json","selected.json","packets.json")]
    for role in ("primary","secondary"):
        qual = ROOT/("20261023_budget128_judge_"+role+"_v1")
        verify(q.read(qual/"producer.json"))
        if q.read(qual/"result.json")["status"] != "passed": raise ValueError("qualified budget128judges required")
        inputs += [qual/"producer.json",qual/"result.json"]
    inputs += [MODEL_PATH/n for n in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","config.json")]
    paths = [Path(__file__),PLAN,Path(profile.__file__),profile.PLAN,Path(accepted.__file__),
             Path("experiments/structure_net/ncm_source_readout_grade.py")]
    p = freeze(DEST,"selected_accepted_labels_and_language_windows",inputs+paths,
        {"summary_labels":143,"old_primary_accepted":32,"new_secondary_accepted":111,
         "language_train":1024,"language_diagnostic":64,"context_tokens":128,"next_words":127,
         "word_frame":128,"native_tokenizer_revision":MODEL_PATH.name,"selected_labels_not_population_accuracy":True})
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST/"producer_source"/("neighborhood_data_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(DEST/"producer.json",p)
    write_json(DEST/"cases.json",cohorts)
    verify(p)
    result = {"status":"prepared","counts":{k:len(v) for k,v in cohorts.items()},
        "source_disjointness_verified":True,"selected_labels_not_teacher_population_accuracy":True,
        "old_and_new_label_profiles_recorded":True,"final_test_opened":False,"task_accuracy":None,
        "producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    prepare()
