"""Stack country readouts, certify surrogate reduction, and test a compact codec."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import zlib

import numpy as np
import torch

from .jspace_attention_study import DEFAULT_ROOT as PARENT, verify as parent_verify, checked as parent_checked
from .jspace_attention_runtime import forward
from .jspace_attention_transport import predict
from .jspace_causal_components import cpu_only
from .jspace_lens import load_small_qwen
from .jspace_relation_study import ROOT, digest, file_hash, write_json, save_tensor
from .jspace_relation_tasks import COUNTRIES
from .jspace_country_stack_algebra import DENOMINATOR, fit_family, decode, decode_portable, graph_basis, country_quotient, pack_labels, unpack_labels


STUDY_ID = "JSPACE-COUNTRY-STACK-GROEBNER-01"
DEFAULT_ROOT = ROOT/"data/experiments/jspace_country_stack/20261113_v1"
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-country-stack-plan.md"
CONFIG = {"layers":list(range(27,36)), "families":[1,2], "denominator":DENOMINATOR,
          "symbolic_seconds":30, "minimum_world_agreement":.95, "maximum_score_nrmse":.25,
          "minimum_payload_ratio":2., "capture_worlds":128, "scope":"exploratory_reused_parent_worlds"}


def inputs():
    return [PARENT/n for n in ("plan.json","tasks.json","dictionary.pt","transport.pt","fit.json","result.json",
            "training/features.pt","discovery/features.pt","evaluation/features.pt","evaluation/edited_features.pt",
            "evaluation/records.json","training_manifest.json","discovery_manifest.json","evaluation_manifest.json")]


def sources():
    parent = json.loads((PARENT/"plan.json").read_text())
    return sorted(set([Path(p) for p in parent["producer_sha256"]]+[Path(__file__).resolve(),
        ROOT/"experiments/structure_net/jspace_country_stack_algebra.py", PROTOCOL,
        ROOT/"tests/structure_net/test_jspace_country_stack.py"]))


def prepare(root):
    cpu_only()
    if root.exists():
        raise FileExistsError(root)
    parent = parent_verify(PARENT)
    for stage in ("discovery","training","evaluation"):
        parent_checked(PARENT,stage)
    value = {"study_id":STUDY_ID,"schema_version":1,"config":CONFIG,"model":parent["model"],
             "parent_fingerprint":parent["scientific_fingerprint"],"parent_root":str(PARENT),
             "sources":{str(p):file_hash(p) for p in sources()},"inputs":{str(p):file_hash(p) for p in inputs()},
             "conformance":"EXPLORATORY_REUSED_EVALUATION", "created_at":datetime.now(timezone.utc).isoformat()}
    value["scientific_fingerprint"] = digest(value)
    root.mkdir(parents=True)
    for p in sources():
        target = root/"producer"/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    write_json(root/"plan.json",value)
    return {"scientific_fingerprint":value["scientific_fingerprint"],"root":str(root)}


def verify(root):
    plan = json.loads((root/"plan.json").read_text())
    if plan["config"] != CONFIG or plan["study_id"] != STUDY_ID or digest({k:v for k,v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Scientific configuration changed")
    for section in ("sources","inputs"):
        for name,expected in plan[section].items():
            if file_hash(name) != expected:
                raise ValueError(f"Frozen {section} changed: {name}")
            if section == "sources" and file_hash(root/"producer"/Path(name).relative_to(ROOT)) != expected:
                raise ValueError("Archived producer changed")
    return plan


def manifest(root, stage, names, **extra):
    plan = verify(root)
    value = {"study_id":STUDY_ID,"stage":stage,"status":"completed","scientific_fingerprint":plan["scientific_fingerprint"],
             "artifacts":{name:file_hash(root/name) for name in names},**extra}
    write_json(root/f"{stage}_manifest.json",value)
    return value


def checked(root, stage):
    plan = verify(root); value = json.loads((root/f"{stage}_manifest.json").read_text())
    if value["status"] != "completed" or value["scientific_fingerprint"] != plan["scientific_fingerprint"]:
        raise ValueError("Incomplete or mismatched stage")
    for name,expected in value["artifacts"].items():
        if file_hash(root/name) != expected:
            raise ValueError(f"Stage artifact changed: {name}")
    return value


def stack_scores(capture, side, heads):
    value = capture[side].double().numpy()
    result = []
    for layer in CONFIG["layers"]:
        node = f"{layer}.block"; i = heads["nodes"].index(node)
        x = value[:,capture["nodes"].index(node)].reshape(len(value),16)
        z = (x-heads["mean"][i].numpy())/heads["scale"][i].numpy()
        result.append(predict(heads["probes"][i],z))
    result = np.stack(result,1)
    if np.max(np.abs(result.sum(-1)-1)) > 1e-8:
        raise ValueError("Affine four-class score identity does not hold")
    return result


def observed(split, heads):
    parent_stage = {"train":"training","development":"discovery","evaluation":"evaluation"}[split]
    capture = torch.load(PARENT/f"{parent_stage}/features.pt",map_location="cpu",weights_only=True)
    pairs = json.loads((PARENT/"tasks.json").read_text())["pairs"][split]
    values = np.concatenate([stack_scores(capture,side,heads) for side in ("source","donor")],0)
    attribute = np.array([float(p[side]["attribute"] == "nationality") for side in ("source","donor") for p in pairs])
    country = np.array([COUNTRIES.index(p[side]["intermediate"]) for side in ("source","donor") for p in pairs])
    return values,attribute,country


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value,dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def fit(root):
    cpu_only(); verify(root); started=time.perf_counter(); torch.set_num_threads(4)
    if (root/"capture").exists():
        raise ValueError("Freeze decoder before new dense-swap trace")
    heads = torch.load(PARENT/"transport.pt",map_location="cpu",weights_only=True)
    train,attr,_ = observed("train",heads); dev,ad,_ = observed("development",heads)
    models,metrics = {},{}
    mean,scale = train.mean(0),np.maximum(train.std(0),1e-6)
    for degree in CONFIG["families"]:
        model,metric = fit_family(train,attr,dev,ad,degree)
        prediction = decode(model,dev[:,0,:3],ad)
        metric["development_standardized_mse"] = float(np.mean(((prediction-dev)/scale)**2))
        metric["development_world_agreement"] = float(np.all(prediction.argmax(-1) == dev.argmax(-1),axis=1).mean())
        models[str(degree)],metrics[str(degree)] = model,metric
    selected = min(models,key=lambda k:(metrics[k]["development_standardized_mse"],int(k)))
    bank = torch.load(PARENT/"dictionary.pt",map_location="cpu",weights_only=True)["rows"]
    original = {}
    for layer in CONFIG["layers"]:
        node=f"{layer}.block"; i=heads["nodes"].index(node)
        original[node]={"bank":bank[node],"mean":heads["mean"][i],"scale":heads["scale"][i],"probe":heads["probes"][i]}
    payload = {"encoder":original["27.block"],"layers":CONFIG["layers"],"powers":models[selected]["powers"],
               "denominator":DENOMINATOR,"decoder_numerators":models[selected]["numerators"][1:,:3].clone()}
    save_tensor(root/"uncompressed_readouts.pt",original)
    save_tensor(root/"compressed_readout.pt",payload)
    save_tensor(root/"fit.pt",{"models":models,"selected":selected,"mean":torch.from_numpy(mean),"scale":torch.from_numpy(scale)})
    names=["uncompressed_readouts.pt","compressed_readout.pt","fit.pt"]
    symbolic = {}
    for degree,model in models.items():
        try:
            symbolic[degree] = graph_basis(model,CONFIG["layers"],CONFIG["symbolic_seconds"])
        except TimeoutError as exc:
            symbolic[degree] = {"status":"unresolved_timeout","reason":str(exc)}
        name=f"continuous_stack_degree{degree}.json"; write_json(root/name,symbolic[degree]); names.append(name)
    quotient = country_quotient(CONFIG["layers"],CONFIG["symbolic_seconds"])
    write_json(root/"country_quotient.json",quotient); names.append("country_quotient.json")
    info={"selected_degree":int(selected),"selection":"development standardized score MSE, then lower degree",
          "metrics":metrics,"frozen_before_new_dense_capture":True,"training_worlds":384,"training_paired_observations":len(train),
          "uncompressed_numerical_bytes":tensor_bytes(original),"compressed_numerical_bytes":tensor_bytes(payload),
          "uncompressed_file_bytes":(root/"uncompressed_readouts.pt").stat().st_size,
          "compressed_file_bytes":(root/"compressed_readout.pt").stat().st_size,
          "symbolic_status":{k:v["status"] for k,v in symbolic.items()},"elapsed_seconds":time.perf_counter()-started}
    write_json(root/"fit.json",info); names.append("fit.json")
    manifest(root,"fit",names,elapsed_seconds=info["elapsed_seconds"])
    return info


def capture(root):
    from structure_net_resources.client import require_gpu_lease
    lease=require_gpu_lease([0]); plan=verify(root); checked(root,"fit")
    parent_verify(PARENT,model=True)
    path=root/"capture"; path.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    model,tokenizer=load_small_qwen(plan["model"]["path"],device="cuda:0",dtype=torch.bfloat16)
    bank=torch.load(PARENT/"dictionary.pt",map_location="cpu",weights_only=True)["rows"]
    pairs=json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    previous=json.loads((PARENT/"evaluation/records.json").read_text())
    natural=torch.load(PARENT/"evaluation/features.pt",map_location="cpu",weights_only=True)
    nodes=natural["nodes"]; features_out=[]; records=[]
    write_json(path/"lease.json",{k:lease[k] for k in ("id","holder","state","ready","allocation","end")})
    for index,(pair,old) in enumerate(zip(pairs,previous)):
        donor=forward(model,tokenizer,pair["donor"],pair["question_start"],bank,keep_attention=(26,))
        donor_features=torch.stack([donor["scores"][node] for node in nodes])
        if donor["row"]["token_ids"] != old["donor"]["token_ids"] or not torch.equal(donor_features,natural["donor"][index]):
            raise ValueError("Donor generation/coordinate replay changed")
        output=forward(model,tokenizer,pair["source"],pair["question_start"],bank,patch_layer=26,
                       donor=donor["attention"][26],method="dense")
        if output["row"]["token_ids"] != old["edits"]["dense"]["token_ids"]:
            raise ValueError("Dense generation replay changed")
        features_out.append(torch.stack([output["scores"][node] for node in nodes])); records.append(output["row"])
        with (path/"events.jsonl").open("a") as f:
            f.write(json.dumps({"world":index+1,"case_id":pair["source"]["case_id"],"exact_replay":True},allow_nan=False)+"\n")
        if index == 0 or (index+1)%16 == 0:
            require_gpu_lease([0]); print(json.dumps({"worlds":index+1,"elapsed_seconds":time.perf_counter()-started}),flush=True)
    save_tensor(path/"features.pt",{"nodes":nodes,"dense":torch.stack(features_out)})
    write_json(path/"records.json",records)
    require_gpu_lease([0])
    peak=torch.cuda.max_memory_reserved()
    if peak > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("GPU reservation exceeded")
    value=manifest(root,"capture",[str(p.relative_to(root)) for p in path.iterdir() if p.is_file()],
                   worlds=len(records),forward_count=2*len(records),lease_id=lease["id"],peak_reserved_bytes=peak,
                   elapsed_seconds=time.perf_counter()-started,generation_and_donor_features_replay_exact=True)
    return {k:value[k] for k in ("worlds","elapsed_seconds","peak_reserved_bytes","lease_id")}


def prediction_metrics(prediction, actual, country, mean, scale):
    error=(prediction-actual)/scale; reference=(actual-mean)/scale
    agreement=np.all(prediction.argmax(-1)==actual.argmax(-1),axis=1)
    return {"worlds":len(actual),"worlds_all_nine_labels_agree":int(agreement.sum()),"world_agreement":float(agreement.mean()),
            "country_correct_by_layer":[int(v) for v in (prediction.argmax(-1)==country[:,None]).sum(0)],
            "score_nrmse":float(np.sqrt(np.sum(error**2)/np.sum(reference**2))),
            "score_rmse":float(np.sqrt(np.mean((prediction-actual)**2)))}


def analyze(root):
    cpu_only(); verify(root); checked(root,"fit"); checked(root,"capture")
    heads=torch.load(PARENT/"transport.pt",map_location="cpu",weights_only=True)
    fitted=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    fitinfo=json.loads((root/"fit.json").read_text()); models=fitted["models"]; selected=fitted["selected"]
    tasks=json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    natural=torch.load(PARENT/"evaluation/features.pt",map_location="cpu",weights_only=True)
    changed=torch.load(PARENT/"evaluation/edited_features.pt",map_location="cpu",weights_only=True)
    dense=torch.load(root/"capture/features.pt",map_location="cpu",weights_only=True)
    portable=torch.load(root/"compressed_readout.pt",map_location="cpu",weights_only=True)
    families={"source":stack_scores(natural,"source",heads),"donor":stack_scores(natural,"donor",heads),
              "compact_J":stack_scores({"nodes":changed["nodes"],"scores":changed["scores"]},"scores",heads),
              "dense":stack_scores(dense,"dense",heads)}
    attr=np.array([float(p["source"]["attribute"]=="nationality") for p in tasks])
    mean,scale=fitted["mean"].numpy(),fitted["scale"].numpy()
    metrics={}; names=[]; all_predictions={}
    raw_inputs={"source":(natural,"source"),"donor":(natural,"donor"),
                "compact_J":({"nodes":changed["nodes"],"scores":changed["scores"]},"scores"),"dense":(dense,"dense")}
    for name,actual in families.items():
        country=np.array([COUNTRIES.index(p["source"]["swap_to"] if name in ("donor","dense") else p["source"]["intermediate"]) for p in tasks])
        results={}
        predictions={"carry_anchor":np.repeat(actual[:,0:1],len(CONFIG["layers"]),axis=1)}
        predictions.update({degree:decode(model,actual[:,0,:3],attr) for degree,model in models.items()})
        capture_input,key=raw_inputs[name]
        jvalues=capture_input[key][:,capture_input["nodes"].index("27.block")].double().numpy().reshape(len(actual),16)
        portable_prediction=decode_portable(portable,jvalues,attr)
        portable_error=float(np.max(np.abs(portable_prediction-predictions[selected])))
        if portable_error > 1e-10:
            raise ValueError("Standalone compressed readout differs from selected surrogate")
        for label,predicted in predictions.items():
            results[label]=prediction_metrics(predicted,actual,country,mean,scale)
        labels=actual.argmax(-1)
        raw,_=pack_labels(labels,False); packed,packing=pack_labels(labels,True)
        if not np.array_equal(unpack_labels(packed),labels) or not np.array_equal(unpack_labels(raw),labels):
            raise ValueError("Lossless country codec roundtrip failed")
        for suffix,payload in (("raw",raw),("compressed",packed)):
            filename=f"{name}_countries.{suffix}.bin"
            with (root/filename).open("xb") as f:f.write(payload)
            names.append(filename)
        groups={}
        for group,mask in (("seen",np.array([not p["source"]["composition_heldout"] for p in tasks])),
                           ("heldout",np.array([p["source"]["composition_heldout"] for p in tasks]))):
            groups[group]=prediction_metrics(predictions[selected][mask],actual[mask],country[mask],mean,scale)
        metrics[name]={"actual_country_correct_by_layer":[int(v) for v in (labels==country[:,None]).sum(0)],
                       "predictors":results,"selected_groups":groups,"portable_roundtrip_max_absolute_error":portable_error,
                       "country_codec":{**packing,"raw_bytes":len(raw),"compressed_bytes":len(packed),"ratio":len(raw)/len(packed),
                           "zlib_raw_bytes":len(zlib.compress(raw,9)),"roundtrip_exact":True}}
        all_predictions[name]={"actual":torch.from_numpy(actual), **{k:torch.from_numpy(v) for k,v in predictions.items()}}
    save_tensor(root/"predictions.pt",all_predictions); names.append("predictions.pt")
    symbolic=fitinfo["symbolic_status"][selected]=="certified_for_rational_surrogate"
    neural=all(metrics[name]["predictors"][selected]["world_agreement"]>=CONFIG["minimum_world_agreement"] and
               metrics[name]["predictors"][selected]["score_nrmse"]<=CONFIG["maximum_score_nrmse"] for name in ("source","dense"))
    ratio=fitinfo["uncompressed_numerical_bytes"]/fitinfo["compressed_numerical_bytes"]
    result={"study_id":STUDY_ID,"status":"completed","confirmed":False,"conformance":"EXPLORATORY_REUSED_EVALUATION",
            "scientific_fingerprint":verify(root)["scientific_fingerprint"],"selected_degree":int(selected),"metrics":metrics,
            "classification":"country_stack_surrogate_compressed_with_neural_fidelity" if symbolic and neural and ratio>=2 else "exact_country_codec_continuous_stack_not_quality_qualified",
            "gates":{"symbolic_certificate":symbolic,"continuous_neural_fidelity":neural,"readout_payload_ratio":ratio,"payload_at_least_2x":ratio>=2},
            "fit":fitinfo,"categorical_compression":"exact finite quotient with mixed-trajectory fallback; an ordinary repeated-label codec can achieve the same compression",
            "limitations":["single_frozen_Qwen_3B","parent_worlds_reused_posthoc","only_country_readouts_not_full_hidden_states",
                "stacked_anchor_decoder_not_composed_transformer_dynamics","no_attention_module_replaced","no_tok_per_second_benchmark",
                "exact_surrogate_algebra_does_not_make_neural_scores_exact","known_attribute_bit_supplied"],
            "artifacts":{n:file_hash(root/n) for n in names+['country_quotient.json','continuous_stack_degree1.json','continuous_stack_degree2.json','fit_manifest.json','capture_manifest.json']}}
    write_json(root/"result.json",result)
    write_json(root/"campaign_results.json",{**result,"result_path":"result.json","result_sha256":file_hash(root/"result.json")})
    return {"classification":result["classification"],"selected_degree":int(selected),"gates":result["gates"],"metrics":metrics}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","fit","capture","analyze"))
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    try:
        result=globals()[args.stage](args.output)
    except BaseException as exc:
        path=args.output/f"failure_{args.stage}.json"
        if args.output.exists() and not path.exists() and not isinstance(exc,FileExistsError):
            write_json(path,{"status":"invalid_stage","exception":type(exc).__name__,"message":str(exc)})
        raise
    print(json.dumps(result,allow_nan=False))
