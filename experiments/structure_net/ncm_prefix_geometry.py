"""Choose complete source-frame geometry without changing learned parameters."""
import torch


def set_source_frame(generator, frame, maximum=4096):
    if type(frame) is not int or not 1 <= frame <= maximum:
        raise ValueError("complete source frame must fit the declared maximum")
    if not generator.use_rope or generator.config.attn_pattern != "block_diagonal" or generator.use_flex_attention:
        raise ValueError("eager RoPE block-diagonal geometry required")
    from models.common import precompute_rope_cos_sin
    parameter = next(generator.parameters())
    key = (frame, str(parameter.device), parameter.dtype)
    cache = getattr(generator, "_ncm_source_geometry", {})
    if key not in cache:
        device = parameter.device
        levels = generator.levels
        labels = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i,n in enumerate(levels)])
        size = frame + len(labels)
        mask = torch.zeros(1,1,size,size,device=device,dtype=parameter.dtype)
        mask[:, :, :frame, frame:] = -torch.inf
        mask[:, :, frame:, frame:] = torch.where(labels[:,None] == labels[None,:], 0., -torch.inf)
        positions = torch.cat((torch.arange(frame,device=device), *[torch.arange(n,device=device) for n in levels]))
        dim = generator.config.n_embd // generator.config.n_head
        cos,sin = precompute_rope_cos_sin(positions,dim,generator.rope_base)
        geometry = {"attn_mask_for_masking":mask,"rope_cos":cos,"rope_sin":sin}
        for i,n in enumerate(levels):
            positions = torch.cat((torch.arange(frame,device=device),torch.arange(n,device=device)))
            cos,sin = precompute_rope_cos_sin(positions,dim,generator.rope_base)
            geometry[f"rope_cos_gen_{i}"] = cos
            geometry[f"rope_sin_gen_{i}"] = sin
        cache[key] = geometry
        generator._ncm_source_geometry = cache
    generator.prefix_len = frame
    generator.config.prefix_len = frame
    for name,value in cache[key].items():
        setattr(generator,name,value)


def source_frame(length, maximum=4096, quantum=256):
    if type(length) is not int or length < 1 or quantum < 1:
        raise ValueError("positive complete source length required")
    frame = ((length+quantum-1)//quantum)*quantum
    if frame > maximum:
        raise ValueError("source exceeds frame capacity; clipping prohibited")
    return frame
