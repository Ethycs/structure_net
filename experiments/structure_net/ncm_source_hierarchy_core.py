"""Concept-only training primitives; output words cannot shortcut this objective."""
import torch
import torch.nn.functional as F


def concept_scope(net):
    net.requires_grad_(False)
    net.source_head.requires_grad_(True)
    net.generator.lm_heads[-1].requires_grad_(False)
    return [p for p in net.parameters() if p.requires_grad]


def concept_logits(net,features,source_mask,history,source_ids):
    """One forward for earlier concept levels; no word labels or word decoder."""
    g=net.generator;codec=g.vqvae_local[0]
    source=net.source_head.source_latents(features,source_mask,source_ids)
    widths=g.levels[:-1];total=sum(widths)
    with torch.autocast(features.device.type,enabled=False):
        previous=codec.vq.idx_to_ncm_input(history).to(source.dtype)
    queries=torch.cat((g.build_l0_query_input(len(source),dtype=source.dtype),
        previous[:,:total-widths[0]]),dim=1)
    target=g.transformer.wte(queries)+g.transformer.wle(g.lvl_1L[:,:total])
    hidden=torch.cat((g.transformer.wte(source),target),dim=1);size=hidden.shape[1]
    mask=g.attn_mask_for_masking[...,:size,:size]+g._prefix_pad_bias(
        ~source_mask.bool(),size,g.attn_mask_for_masking.dtype,hidden.device)
    rope=(g.rope_cos[:size],g.rope_sin[:size])
    for block in g.transformer.blocks:hidden=block(hidden,attn_mask=mask,rope_cos_sin=rope)
    hidden=g.transformer.ln_f(hidden)[:,g.prefix_len:]
    outputs=[];offset=0
    for level,width in enumerate(widths):
        outputs.append(g.lm_heads[level](hidden[:,offset:offset+width]));offset+=width
    return outputs


def uniform_history(codes,levels,rng,maximum=.35,vocab=4096):
    if not 0<=maximum<=1 or vocab<2 or codes.ndim!=2 or codes.shape[1]!=sum(levels):
        raise ValueError("complete hierarchy and valid uniform corruption required")
    value=codes.detach().cpu().clone();offset=0
    rates=torch.linspace(maximum,0,len(levels)-1).tolist()
    for width,rate in zip(levels[:-1],rates):
        original=value[:,offset:offset+width]
        mask=torch.rand(original.shape,generator=rng)<rate
        replacements=torch.randint(vocab,original.shape,generator=rng)
        value[:,offset:offset+width]=torch.where(mask,replacements,original)
        offset+=width
    return value.to(codes.device)


def concept_loss(logits,targets,words,levels,blocks=4,pad=50257,root_weight=4.,pad_weight=.25):
    if (words.ndim!=2 or words.shape[1]!=levels[-1] or words.shape[1]%blocks
            or targets.shape!=(len(words),sum(levels)) or len(logits)!=len(levels)-1):
        raise ValueError("complete concept targets and word canvas required")
    active_blocks=words.reshape(len(words),blocks,-1).ne(pad).any(-1)
    if not active_blocks.any(-1).all():raise ValueError("every target needs a real word/EOS block")
    losses=[];details=[];offset=0;weights=[]
    for level,(output,width) in enumerate(zip(logits,levels[:-1])):
        if width%blocks or output.shape[:2]!=(len(words),width):raise ValueError("complete level shapes required")
        active=active_blocks.repeat_interleave(width//blocks,dim=1)
        labels=targets[:,offset:offset+width];offset+=width
        values=F.cross_entropy(output.float().flatten(0,1),labels.flatten(),reduction="none").reshape_as(labels)
        main=values[active].mean();padded=values[~active].mean() if (~active).any() else main*0
        loss=main+pad_weight*padded
        losses.append(loss);weights.append(root_weight if level==0 else 1.)
        details.append({"level":level,"active_CE":float(main.detach()),"padded_CE":float(padded.detach())})
    combined=sum(weight*loss for weight,loss in zip(weights,losses))/sum(weights)
    if not torch.isfinite(combined):raise ValueError("nonfinite source hierarchy loss")
    return combined,details
