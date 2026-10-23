"""Independent zero-initialized word adapter over the frozen NCM body/bank."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_native_word_mechanism as mechanism
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_query_cache import QueryLoraLinear
from experiments.structure_net.ncm_native_query_concepts import query_hidden
from experiments.structure_net.ncm_native_query_distillation import pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_neighborhood import predicted_concepts
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-separated-word-bank-plan.md")
PILOT = ROOT / "20261023_separated_word_bank_pilot_v1"


class SeparatedQueryLora(QueryLoraLinear):
    def __init__(self, old, word_rank=8):
        super().__init__(old.base, old.a.shape[0])
        with torch.no_grad():
            self.a.copy_(old.a); self.b.copy_(old.b)
        self.a.requires_grad_(False); self.b.requires_grad_(False)
        self.word_a = nn.Parameter(torch.empty(word_rank, old.base.in_features, device=old.a.device, dtype=torch.float32))
        self.word_b = nn.Parameter(torch.zeros(old.base.out_features, word_rank, device=old.b.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.word_a, a=5**.5)
        self.bank = "coarse"
        self.enabled = old.enabled

    def forward(self, x):
        if self.bank == "coarse":
            return super().forward(x)
        if self.bank != "word":
            raise ValueError("unknown adapter bank")
        if not self.enabled or (not torch.is_grad_enabled() and torch.count_nonzero(self.word_b).item() == 0):
            return self.base(x)
        value = self.base(x)
        return value + F.linear(F.linear(x.float(), self.word_a), self.word_b).to(value.dtype)


@contextmanager
def word_bank(backbone):
    before = [module.bank for module in backbone.adapters]
    try:
        for module in backbone.adapters:
            module.bank = "word"
        yield
    finally:
        for module, previous in zip(backbone.adapters, before):
            module.bank = previous


def separate(net, rank=8):
    net.requires_grad_(False)
    torch.manual_seed(5142)
    for layer in net.backbone.language.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            attention = layer.self_attn
            setattr(attention, name, SeparatedQueryLora(getattr(attention, name), rank))
    net.backbone.adapters = [m for m in net.backbone.language.modules() if isinstance(m, SeparatedQueryLora)]
    return net


def word_logits(net, memory, gold):
    query = net.backbone.predecessor_embeddings(memory, gold[:, :-1])
    with word_bank(net.backbone):
        hidden = query_hidden(net.backbone, memory, query, torch.arange(query.shape[1], device=query.device))
        return net.backbone.language.lm_head(hidden)


@torch.no_grad()
def parent_logits(net, memory, gold):
    query = net.backbone.predecessor_embeddings(memory, gold[:, :-1])
    batch, width = query.shape[:2]
    mask = torch.cat((memory.source_mask, torch.ones(batch, width, device=query.device, dtype=memory.source_mask.dtype)), 1)
    positions = memory.source_mask.sum(-1, keepdim=True) + torch.arange(width, device=query.device)[None]
    cache = DynamicCache.from_legacy_cache(memory.keys_values)
    with net.backbone.query_adapters(False):
        hidden = net.backbone.language.model(inputs_embeds=query, attention_mask=mask, position_ids=positions,
            past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state
        return net.backbone.language.lm_head(hidden)


def pilot():
    run.require_gpu()
    cache = run.cache_preparation.CACHE
    if (run.q.read(cache / "worker_terminal_receipt.json")["exit_code"] != 0
            or file_hash(cache / "manifest.json") != run.q.read(cache / "result.json")["manifest_sha256"]):
        raise ValueError("terminal immutable source-cache manifest required")
    manifest = run.q.read(cache / "manifest.json")
    ids = manifest["cohorts"]["summary_train"][:2]
    entries = [manifest["cases"][i] for i in ids]
    for entry in entries:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("selected cache differs")
    if file_hash(run.TRAIN / "model.pt") != run.q.read(run.TRAIN / "result.json")["checkpoint_sha256"]:
        raise ValueError("original trained forecaster checkpoint differs")
    paths = [Path(p) for p in run.q.read(run.TRAIN / "producer.json")["source_hashes"]]
    paths += [Path(__file__), PLAN, Path(mechanism.__file__)]
    paths = list(dict.fromkeys(paths))
    inputs = [run.TRAIN / n for n in ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [cache / n for n in ("manifest.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [run.cache_preparation.BOOKS / "concepts.pt"] + [Path(e["path"]) for e in entries]
    producer = freeze(PILOT, "systems_separated_word_identity_and_training_only", inputs + paths,
        {"hypothesis_id": "ncm-separated-word-bank-identity-v1", "cases": ids, "word_rank": 8,
         "updates": 2, "lr": 1e-4, "weight_decay": .01, "clip": 1., "batch": 2,
         "new_initialization_seed": 5142, "old_forecaster_frozen": True, "original_body_frozen": True,
         "precision": "BF16native/FP32new", "physical_gpu": 0, "task_accuracy": None})
    for path in paths:
        producer["source_hashes"][str(path)] = file_hash(path)
        (PILOT / "producer_source" / ("separated_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(PILOT / "producer.json", producer)
    examples = [torch.load(e["path"], map_location="cpu", weights_only=True) for e in entries]
    net, tokenizer = run.model_and_tokenizer()
    restore_trainable(net, torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True)["trainable"])
    memory = pack_prefix([e["prefix"] for e in examples], "cuda")
    targets = torch.stack([e["targets"] for e in examples]).cuda()
    gold = targets.masked_fill(targets.eq(-100), tokenizer.eos_token_id)
    with torch.no_grad():
        old_codes, old_prior = predicted_concepts(net, memory)
    separate(net)
    with torch.no_grad():
        codes, prior = predicted_concepts(net, memory)
        if not all(torch.equal(a, b) for a, b in zip(old_codes, codes)) or not torch.equal(old_prior, prior):
            raise ValueError("splitting adapters changed old concept generation")
        reference = parent_logits(net, memory, gold)
        initial = word_logits(net, memory, gold)
        if not torch.equal(reference, initial):
            raise ValueError("zero word bank differs from original cached native logits")
        del reference, initial
    frozen = frozen_hash(net)
    books = mechanism.tensor_digest({str(i): b for i, b in enumerate(net.books)})
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    trace = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = word_logits(net, memory, gold)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite word training loss")
        loss.backward()
        if any(p.grad is not None for p in net.parameters() if not p.requires_grad):
            raise ValueError("original body or forecaster received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if not float(norm) > 0:
            raise ValueError("missing independent word gradients")
        optimizer.step()
        trace.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        del logits, loss
    with torch.no_grad():
        after = word_logits(net, memory, gold)
        if not torch.isfinite(after).all():
            raise ValueError("nonfinite new word logits")
    checkpoint = {"word": trainable_state(net), "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    torch.save(checkpoint, PILOT / "model.pt")
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    optimizer.state.clear(); torch.manual_seed(0)
    saved = torch.load(PILOT / "model.pt", map_location="cpu", weights_only=True)
    restore_trainable(net, saved["word"]); optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    with torch.no_grad():
        if not torch.equal(word_logits(net, memory, gold), after):
            raise ValueError("word output reload differs")
        codes, prior = predicted_concepts(net, memory)
    if not all(torch.equal(a, b) for a, b in zip(old_codes, codes)) or not torch.equal(old_prior, prior):
        raise ValueError("word training changed old concept generation")
    actual = trainable_state(net)
    if any(not torch.equal(actual[name], value) for name, value in saved["word"].items()):
        raise ValueError("word parameter reload differs")
    actual_optimizer = optimizer.state_dict()
    if actual_optimizer["param_groups"] != saved["optimizer"]["param_groups"]:
        raise ValueError("word optimizer groups differ")
    for key, state in saved["optimizer"]["state"].items():
        if any(not torch.equal(value.cpu(), actual_optimizer["state"][key][k].cpu()) for k, value in state.items()):
            raise ValueError("word optimizer moments differ")
    if not torch.equal(torch.get_rng_state(), saved["torch_rng"]) or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng"])):
        raise ValueError("word RNG reload differs")
    if frozen_hash(net) != frozen or mechanism.tensor_digest({str(i): b for i, b in enumerate(net.books)}) != books:
        raise ValueError("frozen forecaster/body/books mutated")
    verify(producer)
    write_json(PILOT / "trace.json", trace)
    result = {"status": "systems_identity_and_word_training_passed_not_quality_evidence",
        "zero_word_bank_parent_logits_exact": True, "old_concept_generation_unchanged": True,
        "original_body_forecaster_books_cache_unchanged": True, "only_new_word_bank_trained": True,
        "word_parameters": sum(p.numel() for p in parameters), "checkpoint_optimizer_RNG_output_reload_exact": True,
        "updates": 2, "trace": trace, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "NCM_success_established": False, "new_full_training_admitted": False,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": producer["fingerprint"],
        "checkpoint_sha256": file_hash(PILOT / "model.pt")}
    write_json(PILOT / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    pilot()
