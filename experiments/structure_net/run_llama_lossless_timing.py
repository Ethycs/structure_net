"""Explicitly counterbalanced complete-model lossless correction timing."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net.llama_lossless_residual import ExactResidualLlama
from experiments.structure_net.llama_tensorized_model import generate
from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json
from experiments.structure_net.run_llama_lossless import FACTORS,FIXTURES,PROTOCOL,HYPOTHESIS_ID


def paired(arms,samples=5):
    for fn in arms.values():
        value=fn()
        del value
    names=list(arms)
    records={name:[] for name in names}
    orders=[]
    for index in range(samples):
        order=names if index%2==0 else list(reversed(names))
        orders.append(order.copy())
        for name in order:
            timing,value=common.timed(arms[name])
            records[name].append(timing)
            del value
    return {name:{"samples":rows,"wall_median_ms":statistics.median(r["wall_ms"] for r in rows),
                  "cuda_median_ms":statistics.median(r["cuda_ms"] for r in rows)} for name,rows in records.items()},orders


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    common.setup()
    common.HYPOTHESIS_ID=HYPOTHESIS_ID
    extra=[Path(__file__),PROTOCOL,FIXTURES,Path("experiments/structure_net/llama_lossless_residual.py"),
           Path("experiments/structure_net/run_llama_lossless.py"),Path("docs/07 - Status Reports/2026-10-10_llama32-lossless-timing-amendment.md")]
    producing=common.producer(args.output,"counterbalanced_full_model_timing",extra,
                              {"checkpoint":str(args.checkpoint),"checkpoint_sha256":sha256(args.checkpoint),"factors_sha256":sha256(FACTORS),"samples":5,"replicas":1})
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False,mmap=True)
    factors=torch.load(FACTORS,map_location="cpu",weights_only=False,mmap=True)
    model=ExactResidualLlama(payload,factors)
    original=model.reference_view()
    fixtures=json.loads(FIXTURES.read_text())
    rows=[]
    for length,steps in ((128,0),(512,0),(128,32)):
        ids=torch.tensor([fixtures["benchmark_ids"][0][:length]],device="cuda")
        slack_bytes=(1024-length-max(0,steps-1))*28*2*8*128*2
        slack=torch.empty(slack_bytes,dtype=torch.uint8,device="cuda")
        arms=({"original":lambda:generate(original,ids,steps,False),"exact_residual":lambda:generate(model,ids,steps,False)} if steps else
              {"original":lambda:original(ids,use_cache=True),"exact_residual":lambda:model(ids,use_cache=True)})
        timings,orders=paired(arms)
        row={"length":length,"decode_steps":steps,"timings":timings,"orders":orders,"reserved_unused_kv_bytes":slack_bytes,
             "speedup":timings["original"]["wall_median_ms"]/timings["exact_residual"]["wall_median_ms"]}
        rows.append(row)
        print(json.dumps(row),flush=True)
        del slack
    after=common.gpu_state()
    write_json(args.output/"result.json",{"status":"completed","producer_fingerprint":producing["fingerprint"],"paired_timings":rows,
                                         "gpu_after":after,"uncontended":common.uncontended(producing["gpu_before"],after)})


if __name__=="__main__":
    main()
