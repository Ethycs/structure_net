"""Two fixed readouts of one source-predicted code sequence.

The caller owns source provenance, checkpoint admission, and semantic evaluation.
This module takes no references or target codes. Generate the parent outputs for
the complete population first, then install the new decoder once to decode those
same predicted codes. The original decoder is restored even on a failed readout.
"""
from contextlib import contextmanager

import torch


ARMS = ('new_codec_only', 'parent_with_source')
CODE_LENGTH, CODE_VOCAB = 641, 8192
WORD_LENGTH, WORD_VOCAB = 128, 30528


def require_unbound(model):
    if model._word_bound is not None or model.concepts._bound is not None:
        raise ValueError('source context must not survive across readouts')


def validate_codes(codes):
    if (codes.shape != (1, CODE_LENGTH) or codes.dtype != torch.long
            or bool((codes < 0).any()) or bool((codes >= CODE_VOCAB).any())):
        raise ValueError('one complete predicted native code hierarchy required')


def words_from_logits(logits):
    if logits.shape != (1, WORD_LENGTH, WORD_VOCAB) or not bool(torch.isfinite(logits).all()):
        raise ValueError('finite full-position full-vocabulary logits required')
    return logits.argmax(-1).detach().cpu()[0].tolist()


@torch.no_grad()
def predict_parent(model, features, mask, generate_codes):
    """Call the existing sequential predictor once; no codec encoder is used."""
    require_unbound(model)
    if (features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1
            or features.dtype != torch.float32 or mask.shape != features.shape[:2]
            or mask.dtype != torch.bool or mask.device != features.device
            or not bool(mask.all()) or not bool(torch.isfinite(features).all())):
        raise ValueError('one complete finite FP32 source with no padding or clipping required')
    codes = generate_codes(model, features, mask)
    validate_codes(codes)
    identity = codes.detach().clone()
    words = words_from_logits(model.word_logits(features, mask, codes))
    require_unbound(model)
    if not torch.equal(codes, identity):
        raise ValueError('parent readout changed the predicted codes')
    return {'codes': identity.cpu()[0].tolist(), 'raw_word_ids': words}


@contextmanager
def standalone_decoder(model, decoder_state):
    """Swap only complete decoder weights, keeping modes and flags unchanged.

    The model must already be frozen for inference. Prevalidate the entire state
    before loading, so even an invalid later entry cannot partially mutate it.
    """
    require_unbound(model)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('readout model must be frozen and in evaluation mode')
    current = model.codec.decoder.state_dict()
    if (set(current) != set(decoder_state) or any(
            not isinstance(decoder_state[k], torch.Tensor)
            or current[k].shape != decoder_state[k].shape
            or current[k].dtype != decoder_state[k].dtype
            or not bool(torch.isfinite(decoder_state[k]).all()) for k in current)):
        raise ValueError('exact finite decoder-only state required')
    original = {k: v.detach().cpu().clone() for k, v in current.items()}
    try:
        model.codec.decoder.load_state_dict(decoder_state, strict=True)
        yield
    finally:
        model.codec.decoder.load_state_dict(original, strict=True)


@torch.no_grad()
def decode_standalone(model, predicted_codes):
    """Decode saved source predictions; do not consult source adapters or encode."""
    require_unbound(model)
    if (len(predicted_codes) != CODE_LENGTH
            or any(type(v) is not int or not 0 <= v < CODE_VOCAB for v in predicted_codes)):
        raise ValueError('complete saved source prediction required')
    device = next(model.codec.decoder.parameters()).device
    codes = torch.tensor([predicted_codes], device=device, dtype=torch.long)
    words = words_from_logits(model.codec.decode(codes))
    require_unbound(model)
    if codes.cpu()[0].tolist() != predicted_codes:
        raise ValueError('standalone decoder changed the predicted codes')
    return {'codes': list(predicted_codes), 'raw_word_ids': words}


def paired_identity(parent, new):
    """Require all cases, their order, and their predicted codes to match."""
    if (not parent or len(parent) != len(new)
            or len({r['id'] for r in parent}) != len(parent)
            or [r['id'] for r in parent] != [r['id'] for r in new]
            or any(a['codes'] != b['codes'] for a, b in zip(parent, new, strict=True))):
        raise ValueError('both readouts must use the same complete ordered source predictions')
    for rows in (parent, new):
        for row in rows:
            if (len(row['codes']) != CODE_LENGTH or len(row['raw_word_ids']) != WORD_LENGTH
                    or any(type(v) is not int or not 0 <= v < CODE_VOCAB for v in row['codes'])
                    or any(type(v) is not int or not 0 <= v < WORD_VOCAB for v in row['raw_word_ids'])):
                raise ValueError('complete unmodified native predictions required')
    return len(parent)
