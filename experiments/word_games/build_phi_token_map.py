"""Build and verify cached Phi GGUF vocabulary maps using CPU-only native APIs."""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import random
import statistics
import time

from .common import digest, read, sha, write
from .phi_token_map import SCHEMA, Vocabulary, TokenMapBundle, build_map

LARGE = "/model-weights/Phi-4-GGUF/phi-4-Q4_K_M.gguf"
SMALL = "/model-weights/Phi-4-mini-instruct-GGUF/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
LARGE_SHA = "009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9"
SMALL_SHA = "01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2"
PLAN = Path("docs/07 - Status Reports/word-games/2026-10-31_word-games-phi-token-map-plan.md")


def snapshot_model(model, name):
    import llama_cpp as lib
    vocab = lib.llama_model_get_vocab(model._model.model)
    pieces, attributes = [], []
    for token in range(model.n_vocab()):
        buffer = ctypes.create_string_buffer(64)
        n = lib.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, True)
        if n < 0:
            buffer = ctypes.create_string_buffer(-n)
            n = lib.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, True)
        if n < 0 or n > len(buffer):
            raise ValueError("native token-to-piece buffer failure")
        pieces.append(bytes(buffer.raw[:n]))
        attributes.append(int(lib.llama_vocab_get_attr(vocab, token)))
    return Vocabulary(name, pieces, attributes, {k: v for k, v in model.metadata.items() if k.startswith("tokenizer.")})


def encoder(model):
    import llama_cpp as lib
    vocab = lib.llama_model_get_vocab(model._model.model)
    def encode(data):
        buffer = (lib.llama_token * (len(data) + 16))()
        n = lib.llama_tokenize(vocab, data, len(data), buffer, len(buffer), False, False)
        if n < 0:
            raise ValueError("native tokenizer exceeded byte-size buffer")
        return list(buffer[:n])
    return encode


def corpus():
    inputs_path = Path("data/experiments/word_games/20261031_speculative_inputs_v1/requests.json")
    if sha(inputs_path) != "f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb":
        raise ValueError("frozen request corpus changed")
    texts, bindings = [], {str(inputs_path): sha(inputs_path)}
    for request in read(inputs_path):
        texts.append(("prompt/" + request["id"], request["prompt"].encode()))
        p = Path("data/experiments/word_games/20261031_native_baseline_full_v1/runs") / request["id"] / "response.json"
        bindings[str(p)] = sha(p)
        texts.append(("response/" + request["id"], read(p)["response"]["content"].encode()))
    return texts, bindings


def check_texts(mapping, src_encode, dst_encode, texts):
    rows = []
    for name, data in texts:
        source_ids = src_encode(data)
        result = mapping.translate(source_ids)
        canonical = dst_encode(data)
        if mapping.source.decode(source_ids) != data or mapping.destination.decode(result) != data:
            raise ValueError("corpus byte preservation failed: " + name)
        rows.append({"id": name, "source_tokens": len(source_ids), "mapped_tokens": len(result),
                     "native_destination_tokens": len(canonical), "byte_equal": True,
                     "canonical_ids_equal": result == canonical,
                     "direct_counterpart_tokens": sum(mapping.direct[i] >= 0 for i in source_ids)})
    return {"texts": len(rows), "byte_exact_texts": len(rows),
            "canonical_equal_texts": sum(r["canonical_ids_equal"] for r in rows),
            "source_tokens": sum(r["source_tokens"] for r in rows),
            "direct_counterpart_tokens": sum(r["direct_counterpart_tokens"] for r in rows),
            "mapped_tokens": sum(r["mapped_tokens"] for r in rows),
            "native_destination_tokens": sum(r["native_destination_tokens"] for r in rows), "rows": rows}


def conversion_benchmark(mapping, src_encode, dst_encode, texts):
    rng = random.Random(73)
    all_ids = [src_encode(data) for _, data in texts]
    results = []
    for width in (2, 4, 8):
        chunks = []
        for ids in all_ids:
            for start in range(0, len(ids), width):
                chunk = ids[start:start + width]
                try:
                    mapping.source.decode(chunk).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if chunk:
                    chunks.append(chunk)
        rng.shuffle(chunks)
        chunks = chunks[:1000]
        def mapped(ids): return mapping.translate(ids)
        def retokenized(ids): return dst_encode(mapping.source.decode(ids))
        # Each candidate is arbitrary; canonical parity determines proposal
        # quality, while byte parity protects conversion fidelity.
        for ids in chunks:
            assert mapping.destination.decode(mapped(ids)) == mapping.source.decode(ids)
            retokenized(ids)
        timing = {"mapped": [], "retokenized": []}
        for repeat in range(5):
            order = [("mapped", mapped), ("retokenized", retokenized)]
            if repeat % 2: order.reverse()
            for name, fn in order:
                start = time.perf_counter()
                for ids in chunks: fn(ids)
                timing[name].append(time.perf_counter() - start)
        native = statistics.median(timing["retokenized"])
        table = statistics.median(timing["mapped"])
        results.append({"candidate_source_tokens": width, "chunks": len(chunks), "timing_seconds": timing,
                        "conversion_speedup": native / table,
                        "mapped_us_per_chunk": table * 1e6 / len(chunks),
                        "retokenized_us_per_chunk": native * 1e6 / len(chunks),
                        "canonical_equal_chunks": sum(mapped(ids) == retokenized(ids) for ids in chunks)})
    return results


def main(args):
    import llama_cpp
    from llama_cpp import Llama
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("phi_token_map.py"),
               Path(__file__).with_name("mapped_phi_draft.py"), Path(__file__).with_name("speculative_draft.py"),
               Path(__file__).with_name("common.py"), PLAN]
    texts, data_hashes = corpus()
    print("Verifying cached GGUF files", flush=True)
    for path, expected in [(LARGE, LARGE_SHA), (SMALL, SMALL_SHA)]:
        if sha(path) != expected: raise ValueError("checkpoint hash mismatch")
    config = {"hypothesis_id": "word-games-phi-token-map-v1", "schema": SCHEMA,
              "model_hashes": {LARGE: LARGE_SHA, SMALL: SMALL_SHA}, "data_hashes": data_hashes,
              "source_hashes": {str(p): sha(p) for p in sources}, "python": platform.python_version(),
              "llama_cpp_python": llama_cpp.__version__, "vocab_only": True, "cuda_visible_devices": "",
              "seed": 73, "timing_repeats": 5}
    write(out / "producer.json", {**config, "fingerprint": digest(config)})
    for p in sources: (out / p.name).write_bytes(p.read_bytes())
    large = small = None
    try:
        large = Llama(LARGE, vocab_only=True, n_gpu_layers=0, n_ctx=16, verbose=False)
        small = Llama(SMALL, vocab_only=True, n_gpu_layers=0, n_ctx=16, verbose=False)
        sv, lv = snapshot_model(small, "phi-4-mini-instruct"), snapshot_model(large, "phi-4-14b")
        se, le = encoder(small), encoder(large)
        print("Building bidirectional ordinary-byte maps", flush=True)
        sl, ls = build_map(sv, lv, le), build_map(lv, sv, se)
        bundle = TokenMapBundle(sl, ls)
        bundle.save(out / "token_map", {"model_hashes": config["model_hashes"], "producer_fingerprint": digest(config)})
        restored = TokenMapBundle.load(out / "token_map")
        restored.validate_tokenizers(sv, lv)
        controls = [("unicode", "Résumé e\u0301 中文 العربية 🙂🧑🏽‍💻".encode()),
                    ("json_numbers", b'{"due":"2027-01-10","revision":-1,"x":1.25e-9}'),
                    ("whitespace", b" " * 256 + b"\r\n\t\t indentation"),
                    ("special_as_literal", b"<|im_end|><|assistant|><|end|><|endoftext|>"),
                    ("nul", b"a\x00b")]
        rng = random.Random(73)
        alphabet = list("abcXYZ 0123\n\t{}[]:,-_é中🙂αع")
        controls += [("fuzz_" + str(i), "".join(rng.choices(alphabet, k=rng.randint(1, 120))).encode()) for i in range(256)]
        evidence = {}
        for name, mapping, src, dst in [("small_to_large", sl, se, le), ("large_to_small", ls, le, se)]:
            check = check_texts(mapping, src, dst, texts)
            controls_checked = check_texts(mapping, src, dst, controls)
            bench = conversion_benchmark(mapping, src, dst, texts)
            write(out / (name + "_corpus.json"), check)
            write(out / (name + "_controls.json"), controls_checked)
            write(out / (name + "_timings.json"), bench)
            evidence[name] = {"vocabulary": mapping.summary(),
                              "corpus": {k: v for k, v in check.items() if k != "rows"},
                              "controls": {k: v for k, v in controls_checked.items() if k != "rows"},
                              "conversion_timings": bench}
        from .mapped_phi_draft import MappedPhiDraft
        adapter = MappedPhiDraft(large, small, 4, out / "token_map")
        # Construction validates the actual runtime vocabularies. No forward pass.
        adapter.begin('Return {"ready":true}.')
        assert adapter.prefix_ids and not adapter.mapped_conditioning
        for p, h in config["source_hashes"].items():
            if sha(p) != h: raise ValueError("producing source changed")
        for p, h in data_hashes.items():
            if sha(p) != h: raise ValueError("input corpus changed")
        result = {"hypothesis_id": config["hypothesis_id"], "schema": SCHEMA, "fingerprint": digest(config),
                  "classification": "byte_exact_maps_verified_not_inference_validated", "confirmed": False,
                  "mapping_gate_passed": True, "target_verification_required": True,
                  "inference_tested": False, "runtime_binding_and_chat_render_checked": True,
                  "evidence": evidence,
                  "map_bytes_on_disk": sum(p.stat().st_size for p in (out / "token_map").iterdir()),
                  "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*.json*")}}
        write(out / "campaign_results.json", result)
        print(json.dumps({"classification": result["classification"], "map_bytes": result["map_bytes_on_disk"],
                          "small_to_large": sl.summary(), "large_to_small": ls.summary()}), flush=True)
    except Exception as exc:
        write(out / "interruption.json", {"error": repr(exc)})
        raise
    finally:
        if small: small.close()
        if large: large.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
