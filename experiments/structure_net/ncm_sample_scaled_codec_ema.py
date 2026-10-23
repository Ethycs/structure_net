"""Matched follow-up changing EMA sample units, with unchanged model forwards."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_broad_learned_codec as core
from experiments.structure_net.ncm_memory_data import file_hash,write_json

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-sample-scaled-codec-ema-plan.md")
PILOT=core.ROOT/"20261023_sample_scaled_codec_ema_pilot_v1"
TRAIN=core.ROOT/"20261023_sample_scaled_codec_ema_train_v1"
RECON=core.ROOT/"20261023_sample_scaled_codec_ema_reconstruction_v1"
FAILED_TRAIN=core.TRAIN
FAILED_RECON=core.RECON
ORIGINAL_PILOT=core.PILOT
FAILED_RESULTS=core.ROOT/"20261023_broad_learned_codec_results_v1"
_load=core.load_codec
_producing=core.producing
_reconstruction=core.reconstruction


def scale_ema(quantizer,microbatch=4,reference_batch=4096,scale_buffers=True):
    if microbatch<1 or reference_batch<1 or getattr(quantizer,"ema_scale_by_world_size",False):
        raise ValueError("positive single-device EMA sample units required")
    if getattr(quantizer,"_ncm_sample_scaled_ema",False):
        raise ValueError("EMA normalization must be applied once")
    scale=microbatch/reference_batch
    original_decay=float(quantizer.base_decay)
    original_epsilon=float(quantizer.epsilon)
    if not 0<original_decay<1 or original_epsilon<=0:
        raise ValueError("valid original EMA decay/smoothing required")
    quantizer.base_decay=original_decay**scale
    if quantizer.base_decay_per_level is not None:
        quantizer.base_decay_per_level=[float(d)**scale for d in quantizer.base_decay_per_level]
    quantizer.epsilon=original_epsilon*scale
    if scale_buffers:
        with torch.no_grad():
            for i in range(len(quantizer.embeddings)):
                getattr(quantizer,f"ema_cluster_size_{i}").mul_(scale)
                getattr(quantizer,f"ema_w_{i}").mul_(scale)
    quantizer._ncm_sample_scaled_ema=True
    return {"actual_microbatch":microbatch,"reference_batch":reference_batch,"statistic_scale":scale,
        "original_decay":original_decay,"per_forward_decay":quantizer.base_decay,
        "original_epsilon":original_epsilon,"epsilon":quantizer.epsilon,
        "buffers_rescaled":scale_buffers}


class FixedVocabularySize:
    def __init__(self,tokenizer):
        self.inner=tokenizer
        self.size=len(tokenizer)

    def __len__(self):
        return self.size

    def __getattr__(self,name):
        return getattr(self.inner,name)


def fast_reconstruction(codec,cases,tokenizer):
    return _reconstruction(codec,cases,FixedVocabularySize(tokenizer))


def load_codec(state=None,device="cuda"):
    codec=_load(state,device)
    codec.ncm_ema_settings=scale_ema(codec.vq,scale_buffers=state is None)
    return codec


def producing(root,pilot,extra=()):
    controls=[FAILED_RECON/"result.json",FAILED_TRAIN/"trace.jsonl",ORIGINAL_PILOT/"initial_probe.json",
        FAILED_RESULTS/"campaign_results.json",
        FAILED_RESULTS/"ncm-broad-learned-codec-reconstruction-v1_registry_readback.json"]
    p=_producing(root,pilot,list(extra)+controls+[Path(__file__),PLAN])
    p["hypothesis_id"]="ncm-sample-scaled-codec-ema-v1"
    p["config"].update(EMA_count_scale=4/4096,EMA_reference_batch=4096,
        EMA_decay_per_forward=.99**(4/4096),EMA_epsilon=1e-5*(4/4096),
        initial_codebook_vectors_unchanged=True,vocabulary_size_cached_per_reconstruction=True,
        runtime_overrides=["PLAN","PILOT","TRAIN","RECON","producing","load_codec","reconstruction"])
    for path in (Path(__file__),PLAN):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/("sample_scaled_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"]=core.corpus.quality.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def configure():
    receipt=json.loads((FAILED_RESULTS/"ncm-broad-learned-codec-reconstruction-v1_registry_readback.json").read_text())
    if (receipt["status"]!="registered_and_read_back"
            or receipt["record"]["campaign_sha256"]!=file_hash(FAILED_RESULTS/"campaign_results.json")
            or receipt["record"]["confirmation_status"]!="codec_reconstruction_fidelity_failed"):
        raise ValueError("prior failed codec experiment must be finalized and registered")
    core.PLAN=PLAN;core.PILOT=PILOT;core.TRAIN=TRAIN;core.RECON=RECON
    core.load_codec=load_codec;core.producing=producing;core.reconstruction=fast_reconstruction


def train(pilot):
    if not pilot:
        prior=json.loads((PILOT/"result.json").read_text())
        if not prior.get("initial_inference_matches_original_exactly"):
            raise ValueError("normalized zero-update inference must match original release")
    core.train(pilot)
    root=PILOT if pilot else TRAIN
    initial=json.loads((root/"initial_probe.json").read_text())
    if initial!=json.loads((ORIGINAL_PILOT/"initial_probe.json").read_text()):
        raise ValueError("EMA unit normalization changed initial inference")
    result=json.loads((root/"result.json").read_text())
    result.update(initial_inference_matches_original_exactly=True,EMA_reference_batch=4096,
        EMA_count_scale=4/4096,EMA_decay_per_forward=.99**(4/4096),EMA_epsilon=1e-5*(4/4096))
    if not pilot:
        before=[json.loads(x) for x in (FAILED_TRAIN/"trace.jsonl").read_text().splitlines()]
        after=[json.loads(x) for x in (TRAIN/"trace.jsonl").read_text().splitlines()]
        if (len(before)!=2048 or len(after)!=2048
                or any((a["window_indices"],a["lr"])!=(b["window_indices"],b["lr"]) for a,b in zip(before,after))):
            raise ValueError("matched corpus sampling or optimizer schedule changed")
        result["all_sampled_windows_and_LRs_match_unscaled_run"]=True
    write_json(root/"result.json",result)
    print(json.dumps({"EMA_followup_checks_passed":True,"pilot":pilot}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["pilot","train","measure"])
    args=parser.parse_args();configure()
    if args.stage=="measure":
        from experiments.structure_net import ncm_broad_codec_reconstruction as measurement
        measurement.main()
        original=json.loads((RECON/"original_outputs.json").read_text())
        if original!=json.loads((FAILED_RECON/"original_outputs.json").read_text()):
            raise ValueError("all252 original reference outputs must remain identical")
        write_json(RECON/"original_reference_equivalence.json",{"all252_raw_outputs_codes_and_text_exact":True})
    else:train(args.stage=="pilot")


if __name__=="__main__":
    main()
