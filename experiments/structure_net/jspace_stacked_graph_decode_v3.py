"""Preserve original dynamic prefill; copy its KV state into the decode graph.

v2 measured 41.10 tok/s but static padded prefill scored 206/256 versus packed
209/256. This version changes no weights or decode kernel. It restores ordinary
dynamic-cache prefill and copies its exact BF16 KV tensors into the static cache.
The copies and resets are included in timing. Preserve v2 as measured evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from transformers import DynamicCache

from . import jspace_stacked_graph_decode_v2 as v2
from . import jspace_stacked_graph_decode as original
from . import jspace_bench_tensor as base
from .jspace_bench_data import sha, write


DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_bench_tensor/20261113_graph_v3"
original.CONFIG={**original.CONFIG,"prefill":"original dynamic cache, followed by exact BF16 KV copy to static decode cache",
                 "known_v2":"41.1035 tok/s; static prefill 206/256 vs packed 209/256"}


class DynamicPrefillGraphDecoder(v2.RepetitionGraphDecoder):
    @torch.inference_mode()
    def prefill(self,ids):
        self.seen.zero_();self.seen.index_fill_(0,ids.reshape(-1),True)
        dynamic=DynamicCache()
        hidden=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                                past_key_values=dynamic,use_cache=True).last_hidden_state[:,-1]
        self.cache.reset()
        for i in range(len(self.cache.key_cache)):
            self.cache.key_cache[i][:,:,:ids.shape[1]].copy_(dynamic.key_cache[i])
            self.cache.value_cache[i][:,:,:ids.shape[1]].copy_(dynamic.value_cache[i])
        return hidden


original.GraphDecoder=DynamicPrefillGraphDecoder


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(v2.DEFAULT_ROOT);base.checked(v2.DEFAULT_ROOT,"graph")
    value={**parent,"graph_config":original.CONFIG,
           "sources":{**parent["sources"],str(Path(__file__)):sha(__file__)},
           "input_sha256":{**parent["input_sha256"],str(v2.DEFAULT_ROOT/"result.json"):sha(v2.DEFAULT_ROOT/"result.json")}}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        path=Path(name);target=root/"producer"/path.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(original.PACKED/"tasks.json",root/"tasks.json");write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"graph_config":original.CONFIG}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":original.run}[args.stage](args.root),allow_nan=False),flush=True)
