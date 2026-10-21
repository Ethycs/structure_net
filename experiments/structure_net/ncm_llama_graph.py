"""Lossless execution candidate: capture the unchanged NCM generation schedule."""
import torch
from types import MethodType

from experiments.structure_net.ncm_llama import autocast, decode_ids, final_latent
from experiments.structure_net.ncm_capture_generate import generate as capture_generate


def validate_prefix(ids):
    if len(ids)!=512 or any(type(t) is not int or not 0<=t<=50257 for t in ids):
        raise ValueError("512 valid padded GPT-2 source IDs required")
    if all(t==50257 for t in ids):raise ValueError("nonempty source required")


def unpack_output(values,gpt):
    if len(values)!=49 or values[-1]!=1:raise ValueError("finite48-position text output required")
    ids=values[:-1];text,termination=decode_ids(ids,gpt)
    return {"text":text,"termination":termination,"native_output_ids":ids,"generator_steps":9}


def top1_categorical(logits):
    """Inverse CDF for the original top_k=1 law, including uniform max ties."""
    tied=logits.eq(logits.amax(dim=-1,keepdim=True))
    cdf=tied.long().cumsum(dim=-1)
    count=cdf[...,-1]
    ordinal=(torch.rand(count.shape,device=logits.device,dtype=torch.float32)*count).long()
    return (cdf>ordinal[...,None]).long().argmax(dim=-1)


def capture_sample(self,logits,temperature,top_k,top_p):
    if temperature!=1. or top_k!=1 or top_p!=0.:raise ValueError("fixed native top1 sampling policy required")
    self._graph_sampling_finite=self._graph_sampling_finite & torch.isfinite(logits).all()
    return top1_categorical(logits)


class GraphRuntime:
    @torch.inference_mode()
    def __init__(self,vae,net):
        import time
        self.vae,self.net=vae,net
        self.original_generate=net.generate
        self.original_sample=net._sample_level_raw
        net.generate=MethodType(capture_generate,net)
        net._sample_level_raw=MethodType(capture_sample,net)
        self.prefix=torch.full((1,512),50257,device="cuda",dtype=torch.long)
        self.prefix[:,0]=50256
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        start=time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):self.raw()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=self.raw()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.setup_seconds=time.perf_counter()-start

    def raw(self):
        # Keep original native sampling, arithmetic and level order intact.
        with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
            self.net._graph_sampling_finite=torch.ones((),device="cuda",dtype=torch.bool)
            codes=self.net.generate(None,self.prefix,temperature=1.,top_k=1,top_p=0.,cfg_scale=0.,use_kv_cache=False)
            logits=self.vae.decoder(final_latent(self.vae,codes))
            finite=torch.isfinite(logits).all() & self.net._graph_sampling_finite
            return torch.cat((logits.argmax(-1).flatten(),finite.long()[None]))

    @torch.inference_mode()
    def generate(self,ids,gpt):
        validate_prefix(ids)
        self.prefix.copy_(torch.tensor([ids],device="cuda",dtype=torch.long))
        self.graph.replay()
        return unpack_output(self.output.tolist(),gpt)
