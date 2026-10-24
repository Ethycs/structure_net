"""Real mixed-precision lifecycle for source attention in both native NCM stages."""
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_grounded_head as grounded
from experiments.structure_net import ncm_lm1b_source_attention_pilot as old
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = old.native, old.q, old.ROOT
DEST = ROOT / "20261024_lm1b_grounded_cuda_pilot_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-grounded-cuda-pilot.md")
HYPOTHESIS = "ncm-lm1b-dual-source-attention-BF16-lifecycle-v1"
FRAMED = ROOT / "20261024_lm1b_framed_codec_data_v1/cases.json"


def optimizer(model):
    return torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-3, betas=(.9,.95), weight_decay=0.)


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < 8_000_000_000):
        raise ValueError("physicalGPU0 RTX3060 nativeBF16 with8GBfree required")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rows, manifest, targets = q.read(old.CASES)["train"], q.read(old.MANIFEST), q.read(old.TARGETS)
    targets={r["id"]:r for r in targets}
    eligible=[r for r in rows if targets[r["id"]]["fits_one_native_frame"]]
    selected=[eligible[0],min(eligible,key=lambda r:(-manifest[r["id"]]["source_tokens"],r["id"]))]
    sources=[Path(__file__),PLAN,Path(grounded.__file__),Path(old.__file__),
        Path("tests/structure_net/test_ncm_lm1b_grounded_head.py")]
    inputs=[old.CASES,old.MANIFEST,old.TARGETS,FRAMED,grounded.DOMAIN_CHECKPOINT,
        ROOT/"20261024_lm1b_domain_decoder_results_v1/campaign_results.json",
        ROOT/"20261024_lm1b_domain_decoder_backend_v1.json",
        *[Path(manifest[r["id"]]["path"]) for r in selected],*sources]
    p=native.producing(DEST,"systems_dual_source_attention_BF16_lifecycle",inputs,{"physical_gpu":0})
    p["hypothesis_id"]=HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS,schema="nal.ncm-lm1b-grounded-native-pilot.v1",
        classification="systems_lifecycle_only_not_quality_evidence",model_training_executed=True,
        precision="FP32weights/BF16autocast updates;FP32parity/reload probes",microbatch=4,updates=2,
        seed=91142,new_trainable_parameters=11796480,original_Llama_loaded=False,
        native_generator_frozen=True,measured_domain_codec_frozen=True,
        domain_codec_semantic_quality="52/64 below61gate; exploratory systems-only extension",
        source_truncated=False,probes=[r["id"] for r in selected],task_accuracy=None,speed_tested=False)
    for i,path in enumerate(sources):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"grounded_pilot_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=q.digest({k:v for k,v in p.items() if k!="fingerprint"});write_json(DEST/"producer.json",p)
    model=grounded.from_domain(device="cuda")
    if sum(v.numel() for v in model.parameters() if v.requires_grad)!=11796480:raise ValueError("only expected new attention may train")
    frozen={"generator":native.vendor.module_hash(model.concepts.generator),"codec":native.vendor.module_hash(model.codec)}
    if frozen!={"generator":"60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a","codec":grounded.DOMAIN_CODEC_HASH}:
        raise ValueError("measured frozen native states differ")
    framed={r["id"]:r for r in q.read(FRAMED)}
    samples=[old.sample(r,manifest,targets) for r in selected]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for features,mask,codes in samples:
            native_c=old.native_logits(model.concepts,codes)
            native_w=model.codec.decoder(model.codec.vq.idx_to_z_q(codes))
            actual=model(features,mask,codes)
            if not torch.equal(native_c,actual["concept_logits"]) or not torch.equal(native_w,actual["word_logits"]):
                raise ValueError("new zero-output attention must exactly preserve BOTH native stages")
    del actual,native_c,native_w
    length=max(f.shape[1] for f,m,c in samples)
    features=torch.stack([F.pad(samples[i%2][0][0],(0,0,0,length-samples[i%2][0].shape[1])) for i in range(4)])
    mask=torch.stack([F.pad(samples[i%2][1][0],(0,length-samples[i%2][1].shape[1])) for i in range(4)])
    codes=torch.cat([samples[i%2][2] for i in range(4)])
    words=torch.tensor([framed[selected[i%2]["id"]]["packed_ids"][0] for i in range(4)],device="cuda",dtype=torch.long)
    opt=optimizer(model);torch.manual_seed(91142);torch.cuda.manual_seed_all(91142);model.train()
    trace=[]
    for step in (1,2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):out=model(features,mask,codes)
        concept_loss=F.cross_entropy(out["concept_logits"].float().flatten(0,1),codes.flatten())
        word_loss=F.cross_entropy(out["word_logits"].float().flatten(0,1),words.flatten(),ignore_index=0)
        loss=concept_loss+word_loss
        if not torch.isfinite(loss):raise ValueError("nonfinite combined source-attention loss")
        loss.backward()
        if any(v.grad is not None for m in (model.codec,model.concepts.generator) for v in m.parameters()):
            raise ValueError("frozen pretrained weights received gradients")
        active=[n for n,v in model.named_parameters() if v.requires_grad and v.grad is not None and v.grad.abs().sum()>0]
        if (not any(n.startswith("word_adapters.") for n in active)
                or not any(n.startswith("concepts.adapters.") for n in active)
                or (step==2 and not all(any(n.startswith(prefix) for n in active)
                    for prefix in ("word_memory.","concepts.source_memory.")))):
            raise ValueError("both new source-attention gradient paths required")
        norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
        if not torch.isfinite(norm):raise ValueError("nonfinite combined gradients")
        opt.step();trace.append({"step":step,"concept_loss":float(concept_loss.detach()),"word_loss":float(word_loss.detach()),
            "gradient_norm":float(norm),"nonzero_gradient_names":active})
    model.eval();probe_f,probe_m,probe_c=samples[-1]
    with torch.no_grad():expected={k:v.cpu() for k,v in model(probe_f,probe_m,probe_c).items()}
    generated={k:v.cpu() for k,v in model.generate(probe_f,probe_m).items()}
    saved={"attention":model.attention_state(),"optimizer":opt.state_dict(),"CPU_rng":torch.get_rng_state(),"CUDA_rng":torch.cuda.get_rng_state_all()}
    torch.save(saved,DEST/"attention.pt")
    if any(native.vendor.module_hash(m)!=frozen[k] for k,m in (("generator",model.concepts.generator),("codec",model.codec))):
        raise ValueError("frozen dual-stage state changed")
    model.close();del model,opt,out,loss,concept_loss,word_loss,saved;gc.collect();torch.cuda.empty_cache()
    model=grounded.from_domain(device="cuda")
    saved=torch.load(DEST/"attention.pt",map_location="cpu",weights_only=True)
    model.load_attention_state(saved["attention"]);opt=optimizer(model);opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["CPU_rng"]);torch.cuda.set_rng_state_all(saved["CUDA_rng"]);model.eval()
    from experiments.structure_net.ncm_lm1b_decoder_repair import same
    with torch.no_grad():actual={k:v.cpu() for k,v in model(probe_f,probe_m,probe_c).items()}
    actual_generated={k:v.cpu() for k,v in model.generate(probe_f,probe_m).items()}
    if (not same(actual,expected) or not same(actual_generated,generated) or not same(opt.state_dict(),saved["optimizer"])
            or not torch.equal(torch.get_rng_state(),saved["CPU_rng"]) or not same(torch.cuda.get_rng_state_all(),saved["CUDA_rng"])
            or model._word_bound is not None or model.concepts._bound is not None):
        raise ValueError("combined model/optimizer/RNG/source-only reload or memory cleanup failed")
    peak=torch.cuda.max_memory_allocated()
    if peak>7_500_000_000:raise ValueError("representative combined microbatch4 exceeds7.5GBplan")
    native.vendor.verify(p)
    write_json(DEST/"trace.json",trace)
    torch.save({"teacher":expected,"source_only":generated},DEST/"reload_outputs.pt")
    result={"status":"systems_dual_source_attention_BF16_gradients_and_reload_passed_not_quality",
        "hypothesis_id":HYPOTHESIS,"new_trainable_parameters":11796480,"microbatch":4,"updates":2,
        "both_stages_zero_output_native_parity_exact":True,"both_source_attention_gradients_nonzero":True,
        "both_native_states_unchanged":frozen,"teacher_and_source_only_reload_exact":True,"optimizer_RNG_reload_exact":True,
        "peak_allocated_bytes":peak,"unique_source_tokens":[manifest[r["id"]]["source_tokens"] for r in selected],
        "checkpoint_sha256":file_hash(DEST/"attention.pt"),"task_accuracy":None,"speed_tested":False,
        "final_test_opened":False,"producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
