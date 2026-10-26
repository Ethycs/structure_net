"""Keep frozen VQ conditioning preparation per example while batching the head."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import types

import torch

from experiments.structure_net import ncm_example_batching_probe as original

ROOT = original.broad.runtime.ROOT/'20261026_example_batching_serial_codec_probe_v3'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-example-batching-codec-control.md')


@dataclass(frozen=True)
class Config(original.Config):
    codec_conditioning_preparation: str = 'separate_per_example_then_concatenate'
    attention_backend: str = 'unchanged_default_selection'
    original_default_and_math_gates_remain_failed: bool = True


def serial_codec_logits(head, features, source_mask, codes):
    if codes.shape != (len(features), sum(head.generator.levels)):
        raise ValueError('complete native hierarchy and source batch required')
    with torch.no_grad():
        prefix_ids = torch.full((len(features), 1), 101, device=features.device, dtype=torch.long)
        prefix = head.codec.encoder.wte(prefix_ids)
        root = head.generator.build_l0_query_input(len(features), dtype=prefix.dtype)
        earlier = torch.cat([head.codec.vq.idx_to_ncm_input(part) for part in codes.split(1)]).to(prefix.dtype)
        inputs = torch.cat((prefix, root, earlier), dim=1)
    with head.source_context(features, source_mask):
        return head.generator(inputs)


def run():
    import json
    failed_roots = [original.broad.runtime.ROOT/name for name in
                   ('20261026_example_batching_probe_v1', '20261026_example_batching_math_probe_v2')]
    for root in failed_roots:
        receipt = json.loads((root/'actual_tool_receipt.json').read_text())
        result = json.loads((root/'result.json').read_text())
        if receipt['actual_tool_exit_code'] != 0 or result['classification'] != 'FP32_example_batching_parity_failed':
            raise ValueError('both actual failed comparisons must remain recorded')
        for path, expected in receipt['artifacts'].items():
            if original.broad.file_hash(path) != expected:
                raise ValueError('failed comparison evidence changed')
    base = original.parity

    def parity(model, values, codes, *, noisy):
        # This target-aware diagnostic is not a source-only prediction.
        with torch.no_grad():
            ids = torch.tensor(codes, device='cuda', dtype=torch.long)
            single = torch.cat([model.codec.vq.idx_to_ncm_input(part) for part in ids.split(1)])
            batched = model.codec.vq.idx_to_ncm_input(ids)
            difference = batched-single
            metrics = {'noisy_history': noisy, 'condition': 'clean_codes_for_preparation_diagnostic',
                'shape': list(single.shape), 'max_absolute_difference': float(difference.abs().max()),
                'relative_L2_difference': float(torch.linalg.vector_norm(difference.double())/
                    torch.linalg.vector_norm(single.double()).clamp_min(1e-30)),
                'different_values': int((single != batched).sum()),
                'allclose_1e4': torch.allclose(single, batched, rtol=1e-4, atol=1e-4)}
            original.broad.write_json(ROOT/('codec_preparation_noisy.json' if noisy else 'codec_preparation_clean.json'), metrics)
        model.concepts.teacher_logits = types.MethodType(serial_codec_logits, model.concepts)
        return {**base(model, values, codes, noisy=noisy), 'codec_conditioning_preparation': 'per_example'}

    original.ROOT, original.PLAN, original.CONFIG, original.parity = ROOT, PLAN, Config(), parity
    original.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('run',))
    parser.parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if not existed and ROOT.exists():
            original.broad.write_json(ROOT/'failure.json', {'exception': type(error).__name__,
                'message': str(error), 'task_accuracy': None})
        raise
