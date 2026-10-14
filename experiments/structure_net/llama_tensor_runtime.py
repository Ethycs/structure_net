"""GPU replay and verified within-request lookup for the frozen tensor student."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from transformers import AutoTokenizer, DynamicCache

from experiments.structure_net.llama_neighbourhood_tensorization import digest, generate
from experiments.structure_net.llama_noise_tensor_continuation import assembled, FIXTURES
from experiments.structure_net.llama_summary_throughput import make_fixture
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json

HYPOTHESIS_ID = "llama32-3b-tensor-runtime-v1"
SCHEMA_VERSION = "nal.llama-tensor-runtime.v1"
PLAN = Path("docs/07 - Status Reports/2026-10-14_llama32-tensor-runtime-plan.md")
CHECKPOINT = Path("data/experiments/llama_neighbourhood_continuation/20261014_noise_global_v2")
ARMS = ("reference", "eager", "graph", "graph_lookup8")


class PromptLookup:
    def __init__(self, prompt, eos):
        self.prompt, self.eos = list(prompt), set(eos)
        self.index = defaultdict(list)
        for n in range(1, 5):
            for i in range(len(prompt) - n):
                self.index[tuple(prompt[i:i+n])].append(i+n)

    def propose(self, history, limit=8):
        for n in range(min(4, len(history)), 0, -1):
            for start in self.index.get(tuple(history[-n:]), ()):
                result = []
                for token in self.prompt[start:start+limit]:
                    if token in self.eos:
                        break
                    result.append(token)
                if result:
                    return result
        return []


def accept(draft, predictions):
    """Predictions follow [pending, *draft]; bonus token is never consumed KV."""
    if len(predictions) < len(draft) + 1:
        raise ValueError("one causal prediction per verifier input required")
    matched = 0
    while matched < len(draft) and draft[matched] == predictions[matched]:
        matched += 1
    return draft[:matched] + [predictions[matched]], matched + 1


def bounded(tokens, remaining, eos, stop):
    result = tokens[:remaining]
    if stop:
        for i, token in enumerate(result):
            if token in eos:
                return result[:i+1], True
    return result, False


class TensorRuntime:
    @torch.inference_mode()
    def __init__(self, model, context=8192):
        from flash_attn import flash_attn_with_kvcache
        self.model, self.context, self.attend = model, context, flash_attn_with_kvcache
        config = model.config
        if (config.num_hidden_layers, config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads) != (28, 3072, 24, 8):
            raise ValueError("pinned Llama 3B architecture required")
        shape = (1, context, 8, 128)
        self.kv = [(torch.zeros(shape, dtype=torch.float16, device="cuda"),
                    torch.zeros(shape, dtype=torch.float16, device="cuda")) for _ in model.model.layers]
        self.length = torch.zeros(1, dtype=torch.int32, device="cuda")
        positions = torch.arange(context, device="cuda")[None]
        cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], positions)
        self.cos, self.sin = cos[0, :, :64].contiguous(), sin[0, :, :64].contiguous()
        self.graphs = {}
        started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for width in (1, 9):
                ids = torch.ones((1, width), dtype=torch.long, device="cuda")
                for _ in range(3):
                    self.forward(ids)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    logits = self.forward(ids)
                    output = torch.cat((logits.argmax(-1).flatten(), torch.isfinite(logits).all().long()[None]))
                self.graphs[width] = (graph, ids, output)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter()-started

    def forward(self, ids):
        hidden = self.model.model.embed_tokens(ids)
        for layer, (kc, vc) in zip(self.model.model.layers, self.kv):
            residual = hidden
            x = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(x).view(1, ids.shape[1], 24, 128)
            k = attn.k_proj(x).view(1, ids.shape[1], 8, 128)
            v = attn.v_proj(x).view(1, ids.shape[1], 8, 128)
            x = self.attend(q, kc, vc, k=k, v=v, cache_seqlens=self.length,
                            rotary_cos=self.cos, rotary_sin=self.sin,
                            rotary_interleaved=False, causal=True, softmax_scale=attn.scaling)
            hidden = residual + attn.o_proj(x.reshape(1, ids.shape[1], 3072))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.lm_head(self.model.model.norm(hidden)).float()

    @torch.inference_mode()
    def prefill(self, ids):
        cache = DynamicCache()
        hidden = self.model.model(ids, past_key_values=cache, use_cache=True).last_hidden_state
        logits = self.model.lm_head(hidden[:, -1:]).float()
        for (kc, vc), k, v in zip(self.kv, cache.key_cache, cache.value_cache):
            kc[:, :ids.shape[1]].copy_(k.transpose(1, 2))
            vc[:, :ids.shape[1]].copy_(v.transpose(1, 2))
        self.length.fill_(ids.shape[1])
        return logits

    @torch.inference_mode()
    def predict(self, ids, captured):
        width = len(ids)
        if captured:
            graph, static, output = self.graphs[width]
            static.copy_(torch.tensor([ids], dtype=torch.long, device="cuda"))
            graph.replay()
            result = output.tolist()
            if result[-1] != 1:
                raise RuntimeError("nonfinite captured logits")
            return result[:-1]
        logits = self.forward(torch.tensor([ids], dtype=torch.long, device="cuda"))
        result = torch.cat((logits.argmax(-1).flatten(), torch.isfinite(logits).all().long()[None])).tolist()
        if result[-1] != 1:
            raise RuntimeError("nonfinite eager logits")
        return result[:-1]

    @torch.inference_mode()
    def generate(self, prompt, cap, arm, stop_eos=True):
        if arm not in ARMS[1:] or len(prompt)+cap+9 > self.context:
            raise ValueError("valid custom arm and bounded context required")
        eos = set(self.model.generation_config.eos_token_id)
        # CPU input through CPU final output: include allocation/indexing and KV transfer.
        torch.cuda.synchronize()
        started = time.perf_counter()
        lookup = PromptLookup(prompt, eos) if arm == "graph_lookup8" else None
        logits = self.prefill(torch.tensor([prompt], device="cuda"))
        result = torch.cat((logits.argmax(-1).flatten(), torch.isfinite(logits).all().long()[None])).tolist()
        if result[-1] != 1:
            raise RuntimeError("nonfinite prefill")
        ttft = time.perf_counter()-started
        output = [result[0]]
        logical_length = len(prompt)
        calls, proposed, accepted = 0, 0, 0
        while len(output) < cap and not (stop_eos and output[-1] in eos):
            remaining = cap-len(output)
            draft = lookup.propose(output, min(8, remaining-1)) if lookup and remaining > 1 else []
            inputs = [output[-1]] + draft
            if draft:
                inputs += [128009] * (9-len(inputs))
            predictions = self.predict(inputs, arm != "eager")
            emitted, consumed = accept(draft, predictions)
            logical_length += consumed
            self.length.fill_(logical_length)
            emitted, ended = bounded(emitted, remaining, eos, stop_eos)
            output.extend(emitted)
            calls += 1
            proposed += len(draft)
            accepted += consumed-1
            if ended:
                break
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
        return {"output_ids": output, "wall_seconds": wall, "ttft_seconds": ttft,
                "decode_seconds": wall-ttft, "verifier_calls": calls,
                "proposed_tokens": proposed, "accepted_tokens": accepted,
                "eos": output[-1] in eos, "cap": cap}


def fresh_fixtures(tokenizer):
    specs = ((1, 500, "raw"), (1, 1980, "raw"), (8, 6220, "raw"), (1, 1000, "hierarchy"))
    examples = []
    for i in range(20):
        e = make_fixture(tokenizer, 24000+i, *specs[i % 4], "heldout")
        labels = ", ".join(f["id"] for f in e["fragments"])
        body = "\n\n".join(f"[{f['id']}]\n{f['source']}" for f in e["fragments"])
        instruction = (f"\n\nSummarize these {len(e['fragments'])} fragments in exactly {len(e['fragments'])} lines labeled {labels}, then stop. "
                       "Include each project's final owner, approved budget and deadline. Ignore superseded proposals and background. "
                       "Maximum 128 tokens per line. Output only summaries.")
        ids = tokenizer.apply_chat_template([
            {"role": "system", "content": "Summarize the supplied evidence accurately and concisely."},
            {"role": "user", "content": body+instruction}], add_generation_prompt=True)
        e.update(prompt_ids=ids, prompt_tokens=len(ids))
        if len(ids) > 6300:
            raise ValueError("prompt exceeds workload")
        examples.append(e)
    old = json.loads(FIXTURES.read_text())["examples"]
    old_ids = {digest(e["prompt_ids"]) for e in old}
    if any(digest(e["prompt_ids"]) in old_ids for e in examples):
        raise ValueError("final requests overlap prior data")
    return examples


def seal(output, fixture, stage, arms, repeats):
    if output.exists():
        raise ValueError("fresh evidence root required")
    output.mkdir(parents=True)
    paths = [Path(__file__), PLAN, fixture, CHECKPOINT / "training.json", FIXTURES,
             Path("experiments/structure_net/llama_neighbourhood_tensorization.py"),
             Path("experiments/structure_net/llama_noise_tensor_continuation.py"),
             Path("experiments/structure_net/llama_summary_throughput.py"),
             Path("experiments/structure_net/prepare_llama_cotenancy.py"),
             Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")]
    import flash_attn.flash_attn_interface as interface
    import flash_attn_2_cuda
    import transformers.models.llama.modeling_llama as llama
    paths += [Path(interface.__file__), Path(llama.__file__), Path(flash_attn_2_cuda.__file__)]
    manifest = json.loads(Path("data/models/Llama-3.2-3B-cotenancy/manifest.json").read_text())
    for name, item in manifest["files"].items():
        if sha256(MODEL_PATH / name) != item["sha256"]:
            raise ValueError("original model identity changed")
    source_hashes = {str(p): sha256(p) for p in paths}
    snapshots = output / "producer_source"
    snapshots.mkdir()
    for i, p in enumerate(paths):
        if p.suffix != ".so":
            shutil.copyfile(p, snapshots / f"{i:03d}_{p.name}")
    factor_hashes = {g["selected_checkpoint"]: g["selected_sha256"]
                     for g in json.loads((CHECKPOINT / "training.json").read_text())["groups"]}
    for p, expected in factor_hashes.items():
        if sha256(p) != expected:
            raise ValueError("factor identity changed")
    producer = {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
                "stage": stage, "arms": arms, "repeats": repeats,
                "context": 8192, "lookup": {"max_ngram": 4, "max_tokens": 8, "source": "current_prompt_only"},
                "source_hashes": source_hashes, "factor_hashes": factor_hashes,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "python": sys.version, "command": sys.argv}
    producer["fingerprint"] = digest(producer)
    write_json(output / "producer.json", producer)
    return producer


@torch.inference_mode()
def pilot(runtime, example):
    prompt = torch.tensor([example["prompt_ids"]], device="cuda")
    cache = DynamicCache()
    runtime.model.model(prompt, past_key_values=cache, use_cache=True)
    token = torch.tensor([[1234]*9], device="cuda")
    runtime.prefill(prompt)
    reference = runtime.model.lm_head(runtime.model.model(token, past_key_values=cache, use_cache=True).last_hidden_state).float()
    actual = runtime.forward(token)
    diff = float((reference-actual).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(reference.flatten(), actual.flatten(), dim=0))
    if not torch.isfinite(actual).all() or diff > .25 or cosine < .999:
        raise RuntimeError(f"numerical cache gate failed: {diff}, {cosine}")
    rows = {arm: runtime.generate(example["prompt_ids"], 32, arm, False) for arm in ARMS[1:]}
    # Eager and replay execute exactly the same arithmetic; lookup may use a different GEMM shape.
    if rows["eager"]["output_ids"] != rows["graph"]["output_ids"]:
        raise RuntimeError("graph/eager mismatch")
    replay = runtime.generate(example["prompt_ids"], 32, "graph", False)
    if replay["output_ids"] != rows["graph"]["output_ids"]:
        raise RuntimeError("fresh request reset mismatch")
    return {"status": "passed", "label": "systems_lifecycle_only_not_quality_evidence",
            "max_abs_logit_difference": diff, "logit_cosine": cosine,
            "rows": rows, "graph_setup_seconds": runtime.setup_seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("fixtures", "pilot", "run"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--split", choices=("development", "heldout"), default="development")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if args.command == "fixtures":
        if args.output.exists():
            raise ValueError("fresh fixtures required")
        write_json(args.output, {"schema": SCHEMA_VERSION, "examples": fresh_fixtures(tokenizer),
            "generator_sha256": sha256(__file__), "parent_generator_sha256": sha256("experiments/structure_net/llama_summary_throughput.py")})
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or args.repeats < 1:
        raise RuntimeError("one selected CUDA GPU and positive repeats required")
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("unique known arms required")
    fixture = args.fixtures or FIXTURES
    examples = [e for e in json.loads(fixture.read_text())["examples"] if e["split"] == args.split]
    producer = seal(args.output, fixture, args.command+"_"+args.split, arms, args.repeats)
    model = assembled(CHECKPOINT)
    runtime = TensorRuntime(model)
    longest = max(examples, key=lambda e: len(e["prompt_ids"]))
    if args.command == "pilot":
        write_json(args.output / "result.json", pilot(runtime, longest))
        return
    for arm in arms:
        (generate(model, longest["prompt_ids"], longest["cap"]) if arm == "reference" else
         runtime.generate(longest["prompt_ids"], longest["cap"], arm))
    rows = []
    for e in examples:
        for rep in range(args.repeats):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                result = (generate(model, e["prompt_ids"], e["cap"]) if arm == "reference" else
                          runtime.generate(e["prompt_ids"], e["cap"], arm))
                result.update(id=e["id"], arm=arm, repeat=rep, order=order,
                              text=tokenizer.decode(result["output_ids"], skip_special_tokens=True))
                rows.append(result)
                write_json(args.output / "rows.json", {"status": "running", "rows": rows})
                print(json.dumps({"id": e["id"], "arm": arm, "repeat": rep,
                                  "seconds": result["wall_seconds"], "tokens": len(result["output_ids"])}), flush=True)
    write_json(args.output / "rows.json", {"status": "completed", "rows": rows})
    for path, expected in producer["source_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError("producer changed during campaign")
    write_json(args.output / "result.json", {"status": "completed_pending_source_review", "rows": len(rows),
               "producer_fingerprint": producer["fingerprint"], "rows_sha256": sha256(args.output / "rows.json"),
               "graph_setup_seconds": runtime.setup_seconds, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
