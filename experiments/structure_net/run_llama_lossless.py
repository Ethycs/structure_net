"""Lossless residual storage and complete-network execution measurements."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net import llama_lossless_residual as lossless
from experiments.structure_net.llama_tensorized_model import TensorizedLlama, tensor_hash, generate
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.structure_net.run_llama_cotenancy import quality, aggregate_quality, benchmark
from structure_net.components.layers import model_major_tensor as tensor

HYPOTHESIS_ID = "llama32-3b-lossless-residual-cotenancy-v1"
PROTOCOL = Path("docs/07 - Status Reports/2026-10-10_llama32-lossless-residual-preregistration.md")
FIXTURES = common.ROOT/"inputs_v2/fixtures.json"
FACTORS = common.ROOT/"20261010_campaign_v2/svd2_r215_conversion/checkpoint/model.pt"
ORIGINAL = common.ROOT/"20261010_control_v1/model.pt"


def start(output, stage, config):
    common.setup()
    common.HYPOTHESIS_ID = HYPOTHESIS_ID
    extras = [Path(__file__), Path(lossless.__file__), PROTOCOL, FIXTURES,
              Path("experiments/structure_net/run_llama_cotenancy.py"),
              Path("experiments/structure_net/llama_cotenancy_quality_v2.py"),
              Path("experiments/structure_net/lossless_llama_campaign.py")]
    return common.producer(output, stage, extras, config)


@torch.inference_mode()
def audit_weights(output):
    original = torch.load(ORIGINAL, map_location="cpu", weights_only=False, mmap=True)
    factors = torch.load(FACTORS, map_location="cpu", weights_only=False, mmap=True)
    recovered_matrices, rows = {}, []
    started = time.perf_counter()
    for index, (name, spec) in enumerate(original["matrices"].items()):
        weight = spec["cores"][0].cuda()
        cores = {**factors["matrices"][name], "cores": [c.cuda().float() for c in factors["matrices"][name]["cores"]]}
        prediction = tensor.reconstruct(cores)
        residual, indices, repairs, reconstructed = lossless.matrix_residual(weight, prediction)
        residual_path = output/"residuals"/f"{index:03d}.pt"
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": name, "residual": residual.cpu(), "indices": indices.cpu(), "repairs": repairs.cpu(),
                    "source_weight_sha256": tensor_hash(weight), "prediction_sha256": tensor_hash(prediction)}, residual_path)
        # Verify serialization and actual decoding, not just the in-memory sum.
        loaded = torch.load(residual_path, map_location="cpu", weights_only=True, mmap=True)
        decoded = (prediction+loaded["residual"].cuda()).half()
        decoded.flatten()[loaded["indices"].cuda().long()] = loaded["repairs"].cuda()
        exact = torch.equal(lossless.bits(decoded), lossless.bits(weight))
        if not exact:
            raise RuntimeError("serialized residual reconstruction is not lossless")
        recovered_matrices[name] = {"kind": "dense", "shape": list(weight.shape), "cores": [decoded.cpu()]}
        row = {"name": name, "shape": list(weight.shape), "bitwise_reconstruction": exact,
               "original_bytes": weight.numel()*weight.element_size(), "residual_bytes": residual.numel()*residual.element_size(),
               "repair_entries": indices.numel(), "repair_bytes": indices.numel()*4+repairs.numel()*2,
               "factor_bytes": sum(c.numel()*c.element_size() for c in factors["matrices"][name]["cores"]),
               "source_weight_sha256": tensor_hash(weight), "reconstructed_sha256": tensor_hash(decoded),
               "residual_path": str(residual_path), "residual_sha256": sha256(residual_path)}
        rows.append(row)
        print(json.dumps({"weight_audited": name, "bitwise": exact, "repair_entries": indices.numel()}), flush=True)
        del loaded, prediction, cores, residual, indices, repairs, reconstructed, decoded, weight
    checkpoint = output/"decoded_model.pt"
    torch.save({**original, "matrices": recovered_matrices, "setting": {"name": "lossless_decoded", "kind": "dense"}}, checkpoint)
    vector_bytes = sum(v.numel()*v.element_size() for v in original["vectors"].values())
    dense_bytes = sum(r["original_bytes"] for r in rows)+vector_bytes
    factor_bytes = sum(r["factor_bytes"] for r in rows)
    stored_residual_bytes = sum(r["residual_bytes"]+r["repair_bytes"] for r in rows)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    storage = {"original_dense": dense_bytes, "factor_only": factor_bytes+vector_bytes,
               "stored_matrix_residual": factor_bytes+stored_residual_bytes+vector_bytes,
               "exact_residual_network": dense_bytes+factor_bytes}
    return {"status": "completed", "weights": rows, "all_weights_bitwise": all(r["bitwise_reconstruction"] for r in rows),
            "matrix_count": len(rows), "vector_bytes": vector_bytes, "storage_bytes_per_model": storage,
            "repair_entries": sum(r["repair_entries"] for r in rows), "seconds": time.perf_counter()-started,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "capacity_storage_lower_bound": {name: {str(m): {"bytes": size*m, "fits_free_before_inference": size*m <= free,
                                                               "fits_total_device_memory": size*m <= total}
                                                    for m in (1, 2, 4)} for name, size in storage.items()},
            "gpu_free_bytes": free, "gpu_total_bytes": total,
            "capacity_note": "Hard parameter-storage lower bounds; activations, KV and runtime add more. No infeasible allocation was attempted."}


@torch.inference_mode()
def projection(output):
    original = torch.load(ORIGINAL, map_location="cpu", weights_only=False, mmap=True)
    factors = torch.load(FACTORS, map_location="cpu", weights_only=False, mmap=True)
    name = "model.layers.0.self_attn.q_proj.weight"
    spec = original["matrices"][name]
    factor = factors["matrices"][name]
    dense, packed = tensor.pack([spec, spec], "cuda"), tensor.pack([factor, factor], "cuda")
    torch.manual_seed(917)
    x = torch.randn(2,128,3072,device="cuda",dtype=torch.float16)
    def exact():
        return lossless.exact_output_correction(tensor.apply(x, packed), tensor.apply(x,dense))
    def separate():
        return torch.cat([lossless.exact_output_correction(
            tensor.apply(x[i:i+1], {**packed,"cores":[c[i:i+1] for c in packed["cores"]]}),
            tensor.apply(x[i:i+1], {**dense,"cores":[c[i:i+1] for c in dense["cores"]]})) for i in range(2)])
    target = tensor.apply(x,dense)
    actual = exact()
    prediction = tensor.reconstruct({**factor,"cores":[c.cuda().float() for c in factor["cores"]]})
    residual, indices, repairs, decoded = lossless.matrix_residual(spec["cores"][0].cuda(), prediction)
    direct = tensor.apply(x.float(), {**packed,"cores":[c.float() for c in packed["cores"]]}) + x.float()@residual.T
    timings = common.paired({"original_dense":lambda:tensor.apply(x,dense),"exact_packed":exact,"exact_separate":separate})
    return {"status":"completed", "bitwise_exact_control": torch.equal(lossless.bits(actual),lossless.bits(target)),
            "packed_vs_separate":common.parity(actual,separate()), "timings":timings,
            "direct_residual_sum_bitwise":torch.equal(lossless.bits(direct.half()),lossless.bits(target)),
            "direct_residual_sum_parity":common.parity(direct.half(),target),
            "weight_reconstruction_bitwise":torch.equal(lossless.bits(decoded),lossless.bits(spec["cores"][0].cuda())),
            "weight_repair_entries":indices.numel(), "replicas":2,"length":128,
            "peak_allocated_bytes_including_diagnostics":torch.cuda.max_memory_allocated()}


@torch.inference_mode()
def measure(output, arm, checkpoint, reference, include_test, pilot):
    fixtures = json.loads(FIXTURES.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    factors = torch.load(FACTORS, map_location="cpu", weights_only=False, mmap=True)
    model = TensorizedLlama(payload) if arm == "original" else lossless.ExactResidualLlama(payload, factors)
    controls = {}
    for length in (128,512):
        ids = torch.tensor([fixtures["benchmark_ids"][0][:length]],device="cuda")
        logits, past = model(ids,use_cache=True)
        decoded, _ = model(ids[:,-1:],past=past,use_cache=False)
        controls[str(length)] = {"prefill":logits.cpu(), "cached_decode":decoded.cpu()}
        del logits,past,decoded
    torch.save(controls,output/"logit_controls.pt")
    result = {"status":"completed", "arm":arm,"parameter_bytes":model.storage_bytes(), "logit_controls_sha256":sha256(output/"logit_controls.pt")}
    if reference:
        expected = torch.load(reference/"logit_controls.pt",weights_only=True)
        result["logit_checks"] = {length:{kind:torch.equal(lossless.bits(value),lossless.bits(expected[length][kind])) for kind,value in values.items()} for length,values in controls.items()}
        result["all_control_logits_bitwise"] = all(v for row in result["logit_checks"].values() for v in row.values())
        if not result["all_control_logits_bitwise"]:
            raise RuntimeError("exact residual network changed control logits")
    result["benchmarks"] = benchmark(model,fixtures,1 if pilot else 5)
    if arm == "exact_residual" and not pilot:
        original_view = model.reference_view()
        result["paired_original_vs_residual"] = []
        for length, steps in ((128,0),(512,0),(128,32)):
            ids = torch.tensor([fixtures["benchmark_ids"][0][:length]],device="cuda")
            arms = ({"original":lambda:generate(original_view,ids,steps,False),
                     "exact_residual":lambda:generate(model,ids,steps,False)} if steps else
                    {"original":lambda:original_view(ids,use_cache=True),"exact_residual":lambda:model(ids,use_cache=True)})
            times = common.paired(arms)
            result["paired_original_vs_residual"].append({"length":length,"decode_steps":steps,"timings":times,
                                                        "speedup":times["original"]["wall_median_ms"]/times["exact_residual"]["wall_median_ms"]})
    result["systems_lifecycle_only_not_quality_evidence"] = pilot
    if not pilot:
        result["validation"] = quality(model,fixtures,"validation")
        summary = aggregate_quality(result["validation"])
        result["validation_summary"] = summary
        competent = summary["fragment_coverage"] >= .9 and summary["supported_line_rate"] >= .95
        result["validation_competent"] = competent
        if include_test and competent:
            result["test"] = quality(model,fixtures,"test")
            result["test_summary"] = aggregate_quality(result["test"])
        if reference:
            expected_result = json.loads((reference/"result.json").read_text())
            result["generation_checks"] = {split:{r["id"]:r["generated_ids"]==next(e["generated_ids"] for e in expected_result[split] if e["id"]==r["id"]) for r in result[split]}
                                           for split in ("validation","test") if split in result}
            result["all_generations_identical"] = all(v for rows in result["generation_checks"].values() for v in rows.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("audit_weights","projection","measure"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arm",choices=("original","exact_residual"),default="original")
    parser.add_argument("--checkpoint",type=Path,default=ORIGINAL)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--include-test",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    args = parser.parse_args()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(original_sha256=sha256(ORIGINAL), factor_sha256=sha256(FACTORS), checkpoint_sha256=sha256(args.checkpoint))
    producing = start(args.output,args.stage,config)
    if args.stage == "audit_weights":
        result = audit_weights(args.output)
    elif args.stage == "projection":
        result = projection(args.output)
    else:
        result = measure(args.output,args.arm,args.checkpoint,args.reference,args.include_test,args.pilot)
    result.update(producer_fingerprint=producing["fingerprint"],gpu_after=common.gpu_state())
    result["uncontended"] = common.uncontended(producing["gpu_before"],result["gpu_after"])
    write_json(args.output/"result.json",result)
    print(json.dumps({"completed":str(args.output),"status":result["status"],"validation":result.get("validation_summary"),"test":result.get("test_summary")}),flush=True)


if __name__ == "__main__":
    main()
