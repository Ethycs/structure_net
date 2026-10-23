"""Verify actual released generator migration before the CUDA head pilot."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_word_final_data as data
from experiments.structure_net import ncm_word_final_training as training
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.structure_net.ncm_native_word_mechanism import tensor_digest

DEST=data.ROOT/"20261023_word_final_initialization_v1"


def main():
    data.terminal(data.DATA)
    files=[Path(__file__),Path(training.__file__),Path("experiments/structure_net/ncm_word_final_head.py"),
        Path("experiments/structure_net/ncm_blockwise_codec.py"),Path("experiments/structure_net/ncm_native_word_mechanism.py")]
    files += sorted(data.VENDOR.rglob("*.py"))
    inputs=[data.DATA/n for n in ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]
    inputs += [data.INITIAL/n for n in ("tokenizer.pt","generator.pt","config.yaml")]
    p=data.producing(DEST,"CPU_full_size_released_weight_initialization_not_quality",inputs+files,
        {"CPU_only":True,"parent_loaded":False,"conditional_training":False})
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"initialization_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=data.prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p)
    torch.manual_seed(training.Config().model_seed)
    net,codec,flags=training.make_model("cpu")
    result={"status":"full_size_initialization_passed_not_quality",**flags,
        "all_new_parameter_sha256":tensor_digest(dict(net.named_parameters())),
        "frozen_codec_sha256":training.released.module_hash(codec),
        "word_vocab_limit":net.word_vocab_limit,"source_residual_mode":net.source_head.residual_prefix,
        "parent_loaded":False,"source_prediction":False,"task_accuracy":None,
        "producer_fingerprint":p["fingerprint"]}
    data.verify(p)
    write_json(DEST/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
