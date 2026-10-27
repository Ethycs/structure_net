"""Add source attention to every existing NCM block without changing initial outputs."""
import copy

import torch


def expand(model):
    head=model.concepts
    blocks=head.generator.transformer.blocks
    original=tuple(head.layers)
    if (len(blocks)!=12 or original!=(8,10,11) or head._bound is not None
            or set(head.adapters)!={str(i) for i in original} or len(head._handles)!=3):
        raise ValueError('unbound original twelve-block, three-source-adapter model required')
    if any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
           for adapter in head.adapters.values() for module in adapter.modules()):
        raise ValueError('source adapters themselves must have no copied hooks')
    added=[i for i in range(12) if i not in original]
    templates={i:min(original,key=lambda j:(abs(i-j),j)) for i in added}
    handles=[];created=[]
    try:
        for i in added:
            adapter=copy.deepcopy(head.adapters[str(templates[i])])
            with torch.no_grad():
                adapter.out.weight.zero_()
                if adapter.out.bias is not None:adapter.out.bias.zero_()
            head.adapters[str(i)]=adapter;created.append(i)
            def hook(module,args,output,layer=i):
                if head._bound is None:return output
                if output.ndim!=3 or output.shape[1]<1:
                    raise ValueError('native prefix plus target geometry required')
                prefix,targets=output[:,:1],output[:,1:]
                return torch.cat((prefix,targets+head.adapters[str(layer)](targets,head._bound)),dim=1)
            handles.append(blocks[i].register_forward_hook(hook))
        head._handles.extend(handles);head.layers=tuple(range(12))
    except Exception:
        for handle in handles:handle.remove()
        for i in created:del head.adapters[str(i)]
        raise
    return {'original_source_blocks':list(original),'source_blocks':list(head.layers),
        'added_source_blocks':added,'query_templates':{str(k):v for k,v in templates.items()},
        'new_parameters':sum(p.numel() for i in added for p in head.adapters[str(i)].parameters()),
        'initial_output_projection':'zero','generator_blocks':len(blocks)}
