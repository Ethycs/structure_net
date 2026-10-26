"""Detached source-only histories and residual targets for their visited states."""
import torch

from experiments.structure_net import ncm_sequential_levels as sequential
from experiments.structure_net import ncm_packed_sequential_objective as packed

ARMS = ('fixed_targets', 'residual_targets')


@torch.no_grad()
def targets_for_history(vq, target_latent, history):
    """Oracle TRAIN labels conditional on the actual previous-code history.

    This differs from encoding a clean suffix: every state advances with the
    historical prediction, never the new oracle target. No gradient reaches
    the frozen target encoder or quantizer through this operation.
    """
    levels = vq.levels
    if (target_latent.ndim != 3 or target_latent.shape[1:] != (levels[-1], vq.embed_dim)
            or target_latent.dtype != torch.float32 or not torch.isfinite(target_latent).all()
            or history.dtype != torch.long or history.device != target_latent.device
            or history.shape != (len(target_latent), sum(levels))):
        raise ValueError('complete finite FP32 target latent and native history required')
    residual = target_latent.detach().clone()
    labels, offset = [], 0
    for level, length in enumerate(levels):
        emb = vq.embeddings[level]
        part = history[:, offset:offset+length]
        if part.min() < 0 or part.max() >= emb.num_embeddings:
            raise ValueError('history index outside its level codebook')
        value = vq._down_for_quant(residual, level, length) if level < len(levels)-1 else residual
        distances = vq._distances(value.reshape(-1, vq.embed_dim), emb.weight)
        if not torch.isfinite(distances).all():
            raise ValueError('nonfinite residual distances')
        labels.append(distances.argmin(-1).view(len(history), length))
        contribution = emb(part)
        if level < len(levels)-1:
            contribution = vq._up_to_full(contribution, level, levels[-1])
        residual -= vq._apply_quant_resi(contribution, level)
        offset += length
    return torch.cat(labels, dim=1)


def rollout_history(model, features, mask):
    """Run the unmodified source-only policy, then restore all module flags."""
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(features.device.type, enabled=False):
            return sequential.generate(model, features, mask).detach()
    finally:
        # Restore mixed modes (trainable generator, frozen eval codec) exactly;
        # model.train(True) alone would also enable dropout in frozen modules.
        for module, mode in modes.items():
            module.training = mode


def training_pair(model, features, mask, gold, words, arm, onpolicy):
    if arm not in ARMS or type(onpolicy) is not bool:
        raise ValueError('fixed study arm and Boolean history mode required')
    if (gold.dtype != torch.long or gold.shape != (len(features),sum(model.concepts.generator.levels))
            or words.dtype != torch.long or words.shape != (len(features),model.concepts.generator.levels[-1])):
        raise ValueError('complete native TRAIN words and codes required')
    # The rollout receives only model, source features and source mask.
    history = rollout_history(model,features,mask) if onpolicy else gold.detach().clone()
    if arm == 'residual_targets' and onpolicy:
        with torch.no_grad():
            latent = model.codec.encode_z(words)
            targets = targets_for_history(model.codec.vq,latent,history)
    else:
        targets = gold.detach().clone()
    return history, targets


def backward_example(model, features, mask, gold, words, arm, onpolicy, scaler, accumulation):
    if not model.training or type(accumulation) is not int or accumulation < 1:
        raise ValueError('training mode and positive integer accumulation required')
    history, targets = training_pair(model,features,mask,gold,words,arm,onpolicy)
    with torch.autocast(features.device.type,dtype=torch.float16,enabled=features.is_cuda):
        logits = model.concepts.teacher_logits(features,mask,history)
    terms = packed.losses(logits,targets,model.concepts.generator.levels)
    loss = terms.mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite on-policy objective')
    scaled = loss/accumulation
    (scaler.scale(scaled) if scaler is not None else scaled).backward()
    return {'onpolicy':onpolicy,'level_CE':terms.detach().cpu().tolist(),
        'mean_CE':float(loss.detach()),'history_errors':int((history != gold).sum()),
        'target_changes':int((targets != gold).sum()),
        'history_codes':history.cpu().tolist()[0] if onpolicy else None,
        'target_codes':targets.cpu().tolist()[0] if onpolicy else None}


def gradient_parity(model, features, mask, gold, words, arm, onpolicy):
    """Compare FP32 packed and explicit gradients for the identical visited states."""
    sequential.configure(model,False)
    history, targets = training_pair(model,features,mask,gold,words,arm,onpolicy)
    with torch.no_grad():
        clean = targets_for_history(model.codec.vq,model.codec.encode_z(words),gold)
    if not torch.equal(clean,gold):
        raise ValueError('clean-history targets differ from original native encoding')
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    previous, offset, outputs = None,0,[]
    levels = model.concepts.generator.levels
    cumulative = features.new_zeros(len(features),levels[-1],model.codec.embed_dim)
    for i,length in enumerate(levels):
        logits = sequential.level_logits(model,features,mask,previous,i)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),targets[:,offset:offset+length].flatten())/len(levels)
        loss.backward()
        outputs.append(logits.detach().cpu())
        cumulative, previous = sequential.advance(model.codec,cumulative,history[:,offset:offset+length],i)
        offset += length
    expected = {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in params.items()}
    model.zero_grad(set_to_none=True)
    actual = model.concepts.teacher_logits(features,mask,history)
    packed.losses(actual,targets,levels).mean().backward()
    error = norm = maximum = 0.
    for name,param in params.items():
        reference = expected[name]
        if reference is None:
            if param.grad is not None: raise ValueError('gradient presence differs:'+name)
            continue
        if param.grad is None: raise ValueError('missing packed gradient:'+name)
        difference = param.grad.detach().cpu()-reference
        error += float(difference.double().square().sum())
        norm += float(reference.double().square().sum())
        maximum = max(maximum,float(difference.abs().max()))
    reference_logits = torch.cat(outputs,dim=1)
    actual_logits = actual.detach().cpu()
    relative = (error/max(norm,1e-30))**.5
    result = {'arm':arm,'onpolicy':onpolicy,'relative_gradient_L2_error':relative,
        'max_abs_gradient_difference':maximum,
        'logits_allclose':torch.allclose(actual_logits,reference_logits,rtol=1e-4,atol=1e-4),
        'argmax_disagreements':int((actual_logits.argmax(-1) != reference_logits.argmax(-1)).sum()),
        'history_errors':int((history != gold).sum()),'target_changes':int((targets != gold).sum()),
        'clean_retarget_identical':True,'history_codes':history.cpu().tolist()[0],
        'target_codes':targets.cpu().tolist()[0]}
    result['passed'] = relative <= 5e-5 and maximum <= 1e-4 and result['logits_allclose'] and result['argmax_disagreements'] == 0
    model.zero_grad(set_to_none=True)
    if not result['passed']:
        raise ValueError('packed/explicit visited-state objective differs:'+str({k:v for k,v in result.items() if not k.endswith('_codes')}))
    return result
