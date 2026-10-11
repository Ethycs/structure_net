"""Complete Llama decoder with independent model-major tensorized weights."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from safetensors import safe_open

from structure_net.components.layers import model_major_tensor as tensor
from experiments.structure_net.prepare_llama_cotenancy import write_json, sha256


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def convert(model_path, config, setting, output):
    """Stream matrices through CUDA, keeping checkpoint and original weights intact."""
    if output.exists():
        raise ValueError("conversion root exists; preserve prior evidence")
    output.mkdir(parents=True)
    matrices, vectors, records = {}, {}, []
    started = time.perf_counter()
    for shard in sorted(model_path.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name == "lm_head.weight" and config.tie_word_embeddings:
                    continue
                value = handle.get_tensor(name).to(device="cuda", dtype=torch.float16)
                if value.ndim == 2:
                    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                    torch.manual_seed(seed)
                    start = time.perf_counter()
                    spec = tensor.decompose(value, setting)
                    # Column probes avoid a second full vocabulary reconstruction.
                    generator = torch.Generator(device="cuda").manual_seed(seed ^ 1771)
                    probe = torch.randn(1, 16, value.shape[1], generator=generator, device="cuda", dtype=value.dtype)
                    actual = tensor.apply(probe, tensor.pack([spec]))
                    expected = probe @ value.T
                    relative = float((actual.float()-expected.float()).norm()/expected.float().norm())
                    torch.cuda.synchronize()
                    matrices[name] = {**spec, "cores": [v.detach().cpu() for v in spec["cores"]]}
                    records.append({"name": name, "kind": spec["kind"], "shape": list(value.shape), "seed": seed,
                                    "parameters_before": value.numel(), "parameters_after": sum(v.numel() for v in spec["cores"]),
                                    "probe_relative_l2": relative, "seconds": time.perf_counter()-start,
                                    "source_tensor_sha256": tensor_hash(value),
                                    "factor_sha256": [tensor_hash(v) for v in spec["cores"]]})
                    print({"converted": name, "kind": spec["kind"], "relative_l2": relative, "seconds": records[-1]["seconds"]}, flush=True)
                    del spec, probe, actual, expected
                else:
                    vectors[name] = value.detach().cpu()
                del value
    checkpoint = output / "model.pt"
    torch.save({"schema": "structure-net.tensorized-llama.v1", "config": config.to_dict(), "setting": asdict(setting),
                "matrices": matrices, "vectors": vectors}, checkpoint)
    record = {"status": "completed", "setting": asdict(setting), "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
              "matrices": records, "vector_parameters": sum(v.numel() for v in vectors.values()),
              "seconds": time.perf_counter()-started, "trained_parameters": 0,
              "parameter_bytes": 2*(sum(r["parameters_after"] for r in records)+sum(v.numel() for v in vectors.values()))}
    write_json(output / "conversion.json", record)
    return record


class TensorizedLlama:
    def __init__(self, payload, replicas=1, device="cuda", attention="flash"):
        self.config = LlamaConfig(**payload["config"])
        self.replicas = replicas
        self.matrices = {name: tensor.pack([spec]*replicas, device=device) for name, spec in payload["matrices"].items()}
        self.vectors = {name: torch.stack([value]*replicas, 0).to(device=device) for name, value in payload["vectors"].items()}
        self.rope = LlamaRotaryEmbedding(self.config, device=device)
        self.attention = attention

    @classmethod
    def from_hf(cls, model, device="cpu", replicas=1, attention="sdpa"):
        matrices, vectors = {}, {}
        for name, value in model.state_dict().items():
            if name == "lm_head.weight" and model.config.tie_word_embeddings:
                continue
            if value.ndim == 2:
                matrices[name] = {"kind": "dense", "shape": list(value.shape), "cores": [value.detach()]}
            else:
                vectors[name] = value.detach()
        return cls({"config": model.config.to_dict(), "matrices": matrices, "vectors": vectors}, replicas, device, attention)

    def slice(self, index):
        if not 0 <= index < self.replicas:
            raise IndexError(index)
        result = object.__new__(type(self))
        result.config, result.replicas, result.rope, result.attention = self.config, 1, self.rope, self.attention
        result.matrices = {name: {**spec, "cores": [c[index:index+1] for c in spec["cores"]]} for name, spec in self.matrices.items()}
        result.vectors = {name: v[index:index+1] for name, v in self.vectors.items()}
        return result

    def storage_bytes(self):
        return sum(c.numel()*c.element_size() for spec in self.matrices.values() for c in spec["cores"]) + sum(v.numel()*v.element_size() for v in self.vectors.values())

    def linear(self, x, name):
        return tensor.apply(x, self.matrices[name + ".weight"])

    def embed(self, ids):
        return tensor.embedding(ids, self.matrices["model.embed_tokens.weight"])

    def output(self, x):
        head = self.matrices["model.embed_tokens.weight"] if self.config.tie_word_embeddings else self.matrices["lm_head.weight"]
        return tensor.apply(x, head)

    def norm(self, x, name):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)).to(x.dtype) * self.vectors[name + ".weight"][:, None, :]

    def __call__(self, ids, past=None, use_cache=False, all_logits=False):
        if ids.ndim != 2 or ids.shape[0] != self.replicas:
            raise ValueError("one independent token sequence per model is required")
        models, length = ids.shape
        config = self.config
        x = self.embed(ids)
        past_length = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(past_length, past_length+length, device=ids.device)[None, :]
        cos, sin = self.rope(x, positions)
        caches = []
        for index in range(config.num_hidden_layers):
            name = f"model.layers.{index}"
            residual = x
            normalized = self.norm(x, name + ".input_layernorm")
            q = self.linear(normalized, name + ".self_attn.q_proj").view(models, length, config.num_attention_heads, config.head_dim).transpose(1, 2)
            k = self.linear(normalized, name + ".self_attn.k_proj").view(models, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
            v = self.linear(normalized, name + ".self_attn.v_proj").view(models, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past is not None:
                k, v = torch.cat((past[index][0], k), 2), torch.cat((past[index][1], v), 2)
            if use_cache:
                caches.append((k, v))
            if self.attention == "flash":
                from flash_attn import flash_attn_func
                attended = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True).reshape(models, length, -1)
            else:
                groups = config.num_attention_heads // config.num_key_value_heads
                mask = torch.arange(k.shape[2], device=ids.device)[None, :] <= positions[0, :, None]
                attended = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1), attn_mask=mask).transpose(1, 2).reshape(models, length, -1)
            x = residual + self.linear(attended, name + ".self_attn.o_proj")
            normalized = self.norm(x, name + ".post_attention_layernorm")
            gate = F.silu(self.linear(normalized, name + ".mlp.gate_proj"))
            up = self.linear(normalized, name + ".mlp.up_proj")
            x = x + self.linear(gate * up, name + ".mlp.down_proj")
        x = self.norm(x if all_logits else x[:, -1:], "model.norm")
        return self.output(x).float(), caches if use_cache else None


@torch.inference_mode()
def generate(model, ids, steps=96, stop_eos=True):
    logits, past = model(ids, use_cache=True)
    generated = []
    eos = model.config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    for index in range(steps):
        token = logits[:, -1].argmax(-1)
        generated.append(token)
        if stop_eos and all(int(v) in eos for v in token):
            break
        if index + 1 < steps:
            logits, past = model(token[:, None], past=past, use_cache=True)
    return torch.stack(generated, dim=1)
