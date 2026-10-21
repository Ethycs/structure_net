"""Summarize one complete source with a selected NCM distillation checkpoint."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.structure_net.ncm_llama import BASE, infer_ncm, load_ncm, tokenizers, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text");source.add_argument("--input",type=Path)
    p.add_argument("--model",type=Path,default=BASE/"20261021_generator_v1/model.pt")
    p.add_argument("--codec",type=Path,default=BASE/"20261021_codec_v1/model.pt")
    p.add_argument("--runtime",choices=("eager","graph"),default="graph")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.output and args.output.exists():raise ValueError("new output path required")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("select physical GPU0 before starting Python")
    text=args.text if args.text is not None else args.input.read_text()
    gpt,llama=tokenizers();ids=gpt.encode(text,add_special_tokens=False)
    if not 0<len(ids)<=512:raise ValueError("NCM requires a nonempty source of at most512 GPT-2 tokens; no truncation")
    vae,net,_=load_ncm(args.codec,args.model,adaptation=True)
    runtime=None
    if args.runtime=="graph":
        from experiments.structure_net.ncm_llama_graph import GraphRuntime
        runtime=GraphRuntime(vae,net)
    torch.cuda.synchronize();start=time.perf_counter()
    padded=ids+[50257]*(512-len(ids))
    if runtime is None:
        prefix=torch.tensor([padded],device="cuda")
        result=infer_ncm(vae,net,prefix,gpt)[0]
    else:result=runtime.generate(padded,gpt)
    torch.cuda.synchronize();result["wall_seconds"]=time.perf_counter()-start
    result.update(runtime=args.runtime,graph_setup_seconds=runtime.setup_seconds if runtime else 0.)
    result.update(source_gpt2_tokens=len(ids),common_output_tokens=len(llama.encode(result["text"],add_special_tokens=False)))
    if args.output:write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
