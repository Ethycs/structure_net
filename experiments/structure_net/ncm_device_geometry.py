"""Device-safe geometry, retaining the released CPU RoPE calculation exactly."""
import torch


def set_source_frame(generator, frame, maximum=4096):
    if type(frame) is not int or not 1 <= frame <= maximum:
        raise ValueError("complete source frame must fit the declared maximum")
    if not generator.use_rope or generator.config.attn_pattern != "block_diagonal" or generator.use_flex_attention:
        raise ValueError("eager RoPE block-diagonal geometry required")
    from models.common import precompute_rope_cos_sin
    parameter = next(generator.parameters())
    key = (frame, str(parameter.device), parameter.dtype)
    cache = getattr(generator, "_ncm_device_source_geometry", {})
    if key not in cache:
        device = parameter.device
        levels = generator.levels
        labels = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i,n in enumerate(levels)])
        size = frame + len(labels)
        mask = torch.zeros(1,1,size,size,device=device,dtype=parameter.dtype)
        mask[:, :, :frame, frame:] = -torch.inf
        mask[:, :, frame:, frame:] = torch.where(labels[:,None] == labels[None,:], 0., -torch.inf)
        # The released helper creates frequencies on CPU. Preserve that arithmetic,
        # then transfer its results, instead of passing it CUDA positions.
        dim = generator.config.n_embd // generator.config.n_head
        positions = torch.cat((torch.arange(frame), *[torch.arange(n) for n in levels]))
        cos,sin = precompute_rope_cos_sin(positions,dim,generator.rope_base)
        geometry = {"attn_mask_for_masking":mask,"rope_cos":cos.to(device),"rope_sin":sin.to(device)}
        for i,n in enumerate(levels):
            positions = torch.cat((torch.arange(frame),torch.arange(n)))
            cos,sin = precompute_rope_cos_sin(positions,dim,generator.rope_base)
            geometry[f"rope_cos_gen_{i}"] = cos.to(device)
            geometry[f"rope_sin_gen_{i}"] = sin.to(device)
        cache[key] = geometry
        generator._ncm_device_source_geometry = cache
    generator.prefix_len = frame
    generator.config.prefix_len = frame
    for name,value in cache[key].items():
        setattr(generator,name,value)
