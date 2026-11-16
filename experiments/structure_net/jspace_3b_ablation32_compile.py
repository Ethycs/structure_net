"""Compile all learned residual complements, then distill in causal layer order.

The initial control retains native head dimensions and every SwiGLU channel.
The searched projection is applied at entry and after both residual additions.
Native RMSNorm widths and nonuniform gains are preserved by the existing affine
compiler. Distillation uses the actual frozen student prefix and the exact
frozen native suffix, with original-native teacher targets.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_qwen_affine_stack import forward_reduced
from . import jspace_qwen_parallel_model as native
from . import jspace_qwen3_train as offload
from . import jspace_3b_response_train as response_training
from . import jspace_allblocks_artifact as artifact

SCHEMA='jspace.ablation32-compiled.v1'
PROJECTION_SITES=['entry','post_attention_residual','post_mlp_residual']


def _rows(means,positions):
    if means.ndim!=2 or len(means)<1:raise ValueError('Global or position-specific native means required')
    return means[0].expand(*positions.shape,-1) if len(means)==1 else means[positions]


def project_residual(hidden,discarded_basis,mean_by_position,position_ids):
    """Differentiable x-(x-mu)DD^T, retaining the caller's activation dtype."""
    if (hidden.ndim!=3 or discarded_basis.ndim!=2 or hidden.shape[-1]!=discarded_basis.shape[0]
            or position_ids.shape!=hidden.shape[:2] or mean_by_position.shape[-1]!=hidden.shape[-1]):
        raise ValueError('Aligned residual, discarded basis, means and positions required')
    dtype=torch.float64 if hidden.dtype==torch.float64 else torch.float32
    D=discarded_basis.to(device=hidden.device,dtype=dtype)
    means=_rows(mean_by_position.to(device=hidden.device,dtype=dtype),position_ids)
    value=hidden.to(dtype)
    return (value-((value-means)@D)@D.T).to(hidden.dtype)


def projected_native_block(original,hidden,discarded_basis,mean_by_position,*,position_ids,
                           position_embeddings,attention_mask=None,cache_position=None):
    """Exact frozen-block search semantics, differentiable in D and input.

    This explicitly projects residual variables; a normalization pre-hook alone
    would leave the subsequent residual addition on a different pathway.
    The caller owns parameter freezing, train/eval state and GPU reservation.
    """
    def project(x):return project_residual(x,discarded_basis,mean_by_position,position_ids)
    hidden=project(hidden)
    attention=original.self_attn(hidden_states=original.input_layernorm(hidden),
        attention_mask=attention_mask,position_ids=position_ids,position_embeddings=position_embeddings,
        cache_position=cache_position,use_cache=False)[0]
    hidden=project(hidden+attention)
    return project(hidden+original.mlp(original.post_attention_layernorm(hidden)))


def compile_block(original,retained_basis,mean_by_position,*,parameter_dtype=None):
    """Residual-only compilation; native heads/MLP and score scale unchanged."""
    block=AffineQwenBlock(original,retained_basis,mean_by_position,trainable=True,
        parameter_dtype=parameter_dtype or original.self_attn.q_proj.weight.dtype,
        learn_offsets=False,head_projection=None)
    if (block.has_head_projection or block.has_polynomial or block.head_dim!=block.native_head_dim
            or block.intermediate_size!=block.native_intermediate_size):
        raise RuntimeError('Residual-only control unexpectedly changed heads or MLP channels')
    return block


def compile_stack(native_blocks,geometries,*,parameter_dtype=None,expected_layers=36):
    """Compile EVERY block before any student optimization; no incremental fit."""
    if len(native_blocks)!=expected_layers or len(geometries)!=expected_layers:
        raise ValueError('All native blocks and complements must be supplied before compilation')
    blocks=[compile_block(original,g['retained'],g['anchor'],parameter_dtype=parameter_dtype)
            for original,g in zip(native_blocks,geometries)]
    if [b.layer_idx for b in blocks]!=list(range(expected_layers)):
        raise ValueError('Compiled blocks must span the full model in original order')
    return CompactQwenStack(blocks,geometry_mode='fp32',pad_to_multiple=128,
                            enabled=False,checkpoint_blocks=False)


def _write(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)


def _structure(stack):
    return dict(blocks=[dict(**{key:getattr(b,key)for key in artifact.BLOCK_FIELDS},
        inference_dtype=str(b.inference_dtype).removeprefix('torch.'),
        parameters=list(b._parameters),buffers=list(b._buffers))for b in stack.blocks],
        geometry_mode=stack.geometry_mode,pad_to_multiple=stack.pad_to_multiple)


def load_compiled(root):
    """Restore the fixed upfront bank on CPU without refolding native weights."""
    root=Path(root);receipt=json.loads((root/'manifest.json').read_text())
    if receipt.get('schema')!=SCHEMA or offload.sha(root/'compiled.pt')!=receipt['checkpoint_sha256']:
        raise ValueError('Compiled bank identity differs')
    payload=torch.load(root/'compiled.pt',map_location='cpu',weights_only=True,mmap=True)
    if (payload.get('schema')!=SCHEMA or payload.get('projection_sites')!=PROJECTION_SITES
            or not payload.get('all_blocks_compiled_before_distillation')
            or payload['geometry_manifest_sha256']!=receipt['geometry_manifest_sha256']):
        raise ValueError('Compiled bank provenance differs')
    state=payload['stack'];blocks=[]
    for index,item in enumerate(payload['structure']['blocks']):
        if item['packed']or item['has_head_projection']or item['has_polynomial']:
            raise ValueError('Expected the unpacked residual-only initialization')
        block=AffineQwenBlock.__new__(AffineQwenBlock);nn.Module.__init__(block)
        for name in artifact.BLOCK_FIELDS:setattr(block,name,item[name])
        block.inference_dtype=artifact.DTYPES[item['inference_dtype']]
        block.capture_mlp,block.last_mlp_hidden=False,None
        for kind in ('parameters','buffers'):
            for name in item[kind]:
                value=state.get(f'blocks.{index}.{name}')
                # Clone mutable learned weights off the read-only mmap; geometry
                # remains shared until the caller explicitly moves a block.
                if kind=='parameters':block.register_parameter(name,None if value is None else nn.Parameter(value.clone(),requires_grad=False))
                else:block.register_buffer(name,value)
        blocks.append(block)
    description=payload['structure']
    stack=CompactQwenStack(blocks,geometry_mode=description['geometry_mode'],
        pad_to_multiple=description['pad_to_multiple'],enabled=False)
    stack.load_state_dict(state,strict=True)
    if list(stack.layer_indices)!=list(range(receipt['layers'])):raise ValueError('Incomplete compiled bank')
    return stack.eval().requires_grad_(False),receipt


def restore_selected_block(stack,checkpoint,*,expected_sha256,norm=None):
    """Install only an authenticated selected stage, preserving parameter IDs."""
    if offload.sha(checkpoint)!=expected_sha256:raise ValueError('Selected block checkpoint identity differs')
    value=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if value.get('schema')!='jspace.ablation32-stage-checkpoint.v1':raise ValueError('Stage checkpoint required')
    stack.blocks[value['layer']].load_state_dict(value['block'],strict=True)
    if value.get('norm')is not None:
        if norm is None or value['layer']!=len(stack.blocks)-1:raise ValueError('Final norm belongs only to final stage')
        norm.load_state_dict(value['norm'],strict=True)
    return value


def pack_preserving_geometry(stack):
    """Deployment folding keeps all fixed geometry and shift tables in FP32."""
    from .jspace_qwen3b_progressive_model import cpu_module_clone
    packed=cpu_module_clone(stack).eval().requires_grad_(False)
    if packed.geometry_mode!='fp32':raise ValueError('Original arithmetic-precision geometry required')
    return packed.pack(dtype=torch.bfloat16,buffer_dtype=torch.float32,strip_unused=True)


def load_geometry_bank(root,*,expected_layers=36):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if (manifest.get('schema')!='ablation32.geometry-bank.v1'
            or manifest.get('projection_sites')!=PROJECTION_SITES
            or set(manifest['layers'])!={str(i)for i in range(expected_layers)}):
        raise ValueError('Complete geometry bank with matched three-site projection semantics required')
    values=[]
    for index in range(expected_layers):
        item=manifest['layers'][str(index)];path=(root/item['path']).resolve()
        if not path.is_relative_to(root)or offload.sha(path)!=item['sha256']:
            raise ValueError('Geometry payload identity differs')
        value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        B,D=value['retained'],value['discarded'];width=item['width'];rank=item['retained']
        if (B.shape!=(width,rank)or D.shape!=(width,width-rank)
                or value['anchor'].ndim!=2 or value['anchor'].shape[1]!=width
                or not torch.allclose(B.T@B,torch.eye(rank,dtype=B.dtype),atol=1e-9,rtol=1e-9)
                or not torch.allclose(D.T@D,torch.eye(width-rank,dtype=D.dtype),atol=1e-9,rtol=1e-9)
                or not torch.allclose(B.T@D,torch.zeros(rank,width-rank,dtype=B.dtype),atol=1e-9,rtol=1e-9)):
            raise ValueError('Exported retained/discarded complement is invalid')
        dc=B.new_full((width,),1/math.sqrt(width))
        if not torch.allclose(B[:,0],dc,atol=1e-12,rtol=1e-12):
            raise ValueError('The retained Qwen DC direction must be explicit and preserved')
        values.append(value)
    return manifest,values


def compile_all(root,model_path,geometry_root,*,arm,parameter_dtype=torch.bfloat16,guard=None,
                expected_layers=36):
    """CPU-stream native blocks into one upfront full-model compiled artifact."""
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if arm not in ('learned','random'):raise ValueError('Explicit learned or paired initial-random arm required')
    cfg=native._config(model_path)
    if cfg.num_hidden_layers!=expected_layers:raise ValueError('Complete native depth required')
    manifest,geometries=load_geometry_bank(geometry_root,expected_layers=expected_layers)
    blocks=[];started=time.monotonic()
    for layer,geometry in enumerate(geometries):
        if guard is not None:guard()
        original=native.load_native_block(model_path,layer,device='cpu',dtype=parameter_dtype,config=cfg)
        blocks.append(compile_block(original,geometry['retained'],geometry['anchor'],parameter_dtype=parameter_dtype))
        del original
    stack=CompactQwenStack(blocks,geometry_mode='fp32',pad_to_multiple=128,enabled=False)
    root.mkdir(parents=True)
    with (root/'compiled.pt').open('xb')as stream:
        torch.save(dict(schema=SCHEMA,arm=arm,stack=native.state_dict_cpu(stack),
            structure=_structure(stack),
            geometry_manifest_sha256=offload.sha(Path(geometry_root)/'manifest.json'),
            projection_sites=PROJECTION_SITES,all_blocks_compiled_before_distillation=True),stream)
    receipt=dict(schema=SCHEMA,status='completed',arm=arm,layers=expected_layers,
        geometry_root=str(Path(geometry_root).resolve()),geometry_manifest_sha256=offload.sha(Path(geometry_root)/'manifest.json'),
        model_path=str(Path(model_path).resolve()),configuration=cfg.to_dict(),parameter_dtype=str(parameter_dtype),
        ranks=list(stack.ranks),projection_sites=PROJECTION_SITES,
        native_head_dimensions=[b.head_dim for b in blocks],native_mlp_widths=[b.intermediate_size for b in blocks],
        additional_head_or_mlp_approximation=False,normalization_width=cfg.hidden_size,
        anchor_convention='Exported mu; compiler a=mu-(muB)B^T, z=xB; original-width RMS denominator',
        geometry_precision='FP32 stored affine geometry; native arithmetic/compiler rounding requires measured control',
        compiled_before_any_distillation=True,checkpoint=str(root/'compiled.pt'),checkpoint_sha256=offload.sha(root/'compiled.pt'),
        parameter_count=sum(v.numel()for v in stack.parameters()),storage=stack.storage_breakdown(),
        elapsed_seconds=time.monotonic()-started,confirmed=False)
    receipt['geometry_provenance']={k:v for k,v in manifest.items()if k!='layers'}
    _write(root/'manifest.json',receipt)
    return stack,receipt


def compile_pair(root,model_path,learned_root,random_root,*,parameter_dtype=torch.bfloat16,expected_layers=36):
    """Build both complete controls before either arm starts distillation."""
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    learned,lg=load_geometry_bank(learned_root,expected_layers=expected_layers)
    random,rg=load_geometry_bank(random_root,expected_layers=expected_layers)
    if any(a['retained'].shape!=b['retained'].shape or not torch.equal(a['anchor'],b['anchor'])for a,b in zip(lg,rg)):
        raise ValueError('Learned/random controls must share widths and exact TRAIN anchors')
    if (learned.get('arm')!='selected_learned' or random.get('arm')!='matched_initial_random'
            or random.get('epoch')!=0 or not learned.get('identities') or not learned.get('teacher_id')):
        raise ValueError('Selected search geometry and its exact initialized-random control required')
    for key in ('identities','teacher_id','model_configuration'):
        if learned.get(key)!=random.get(key):raise ValueError('Paired geometry provenance differs')
    for manifest in (learned,random):
        cp=manifest['checkpoint']
        if offload.sha(cp['path'])!=cp['sha256']:raise ValueError('Search checkpoint provenance differs')
        raw=torch.load(cp['path'],map_location='cpu',weights_only=True,mmap=True)
        if raw.get('identities')!=manifest['identities']or raw.get('teacher_id')!=manifest['teacher_id']:
            raise ValueError('Search checkpoint teacher/data binding differs')
    root.mkdir(parents=True);receipts={}
    for arm,path in (('learned',learned_root),('random',random_root)):
        stack,receipt=compile_all(root/arm,model_path,path,arm=arm,parameter_dtype=parameter_dtype,expected_layers=expected_layers)
        receipts[arm]=receipt;del stack
    result=dict(schema=SCHEMA,status='completed',both_arms_all_layers_compiled=True,arms=receipts,
        native_head_and_mlp_unchanged=True,residual_basis_source='learned discarded complement versus identical initialized-random parameterization',confirmed=False)
    _write(root/'pair.json',result);return result


@torch.no_grad()
def compiler_parity(original,compiled,discarded,means,hidden,*,position_ids,position_embeddings,
                    attention_mask=None,atol=1e-9,rtol=1e-9):
    """Compare the projected native function, never the unprojected teacher."""
    expected=projected_native_block(original,hidden,discarded,means,position_ids=position_ids,
        position_embeddings=position_embeddings,attention_mask=attention_mask,cache_position=position_ids[0])
    actual=compiled(hidden,position_ids=position_ids,position_embeddings=position_embeddings,
        attention_mask=attention_mask,cache_position=position_ids[0],use_cache=False)[0]
    error=(actual-expected).float()
    passed=bool(torch.all((actual-expected).abs()<=atol+rtol*expected.abs()))
    return dict(passed=passed,max_absolute_error=float(error.abs().max()),
        relative_rms_error=float(error.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-20)),
        atol=atol,rtol=rtol,projection_sites=PROJECTION_SITES,
        heads_unchanged=not compiled.has_head_projection,mlp_unchanged=not compiled.has_polynomial,
        reference='Same three-site projected native block',unprojected_teacher_equivalence_claim=False)


class DistillationDataset:
    """Add original boundary(layer+1) targets to the authenticated search data.

    Optional prefix_provider(index) supplies the separately authenticated actual
    student coordinate cache. This adapter never uses native boundary(layer) as
    a substitute for a compressed prefix.
    """
    def __init__(self,search_dataset,layer,*,prefix_provider=None):
        from . import jspace_3b_full_progressive_data as banks
        self.source=search_dataset;self.plan=search_dataset.plan;self.records=search_dataset.records
        self.rows=search_dataset.rows;self.layer=layer;self.prefix_provider=prefix_provider
        self.training=self.plan['distillation_indices'];self.development=self.plan['selection_indices']
        self.pad_token_id=search_dataset.pad_token_id
        if not 0<=layer<36:raise ValueError('Original next-boundary index must correspond to a model block')
        full_root=Path(self.plan['full_data_root'])
        full_plan,_=banks.verify(full_root,live=False)
        banks._manifest_files(full_root/'native',full_plan['fingerprint'])
        if full_plan['records_sha256']!=self.plan['records_sha256'] or full_plan['rows']!=self.rows:
            raise ValueError('Original next-boundary records differ from admitted response corpus')
        self.targets=banks._bank(full_root/'native'/f'boundary_{layer+1:02d}.npy',self.plan['total_tokens'],self.plan['width'])

    def batch(self,indices):
        if len(indices)!=1:raise ValueError('Streaming singleton microbatch required')
        batch=self.source.batch(indices);index=indices[0];row=self.rows[index]
        value=torch.from_numpy(np.array(self.targets[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
        batch['next_boundary_hidden']=value[None]
        if self.prefix_provider is not None:
            cached=self.prefix_provider(index)
            for key in ('student_input_coordinates','student_prefix_fingerprint','student_prefix_input_ids'):
                batch[key]=cached[key]
        return batch

    def close(self):
        if not self.targets._mmap.closed:self.targets._mmap.close()


class FrozenNativeLookahead(nn.Module):
    """Exact original remaining blocks: frozen weights, live input gradients."""
    def __init__(self,blocks,norm,embedding,rotary,*,start,expected_layers=36,checkpoint_blocks=True):
        super().__init__();blocks=list(blocks)
        if [b.self_attn.layer_idx for b in blocks]!=list(range(start,expected_layers)):
            raise ValueError('Complete exact native suffix through the final model layer required')
        self.blocks=nn.ModuleList(blocks);self.norm=deepcopy(norm);self.rotary=rotary
        self.register_buffer('embedding',embedding.detach())
        self.start,self.expected_layers,self.checkpoint_blocks=start,expected_layers,checkpoint_blocks
        self.eval().requires_grad_(False)

    def forward(self,hidden,positions,attention_mask):
        rope=self.rotary(hidden,positions)
        for block in self.blocks:
            def execute(value,current=block):
                return current(value,attention_mask=attention_mask,position_ids=positions,
                    position_embeddings=rope,cache_position=positions[0],use_cache=False)[0]
            hidden=checkpoint(execute,hidden,use_reentrant=False)if self.checkpoint_blocks and torch.is_grad_enabled()else execute(hidden)
        return self.norm(hidden)

    def readout(self,hidden):return F.linear(hidden.to(self.embedding.dtype),self.embedding)


@dataclass(frozen=True)
class DistillationConfig:
    epochs:int=12
    accumulation_examples:int=256
    learning_rate:float=4e-4
    minimum_lr_fraction:float=.15
    weight_decay:float=1e-4
    gradient_clip:float=1.
    temperature:float=2.
    feature_weight:float=1.
    kl_weight:float=1.
    token_chunk_size:int=8
    optimizer_chunk_bytes:int=4*1024**2
    max_cpu_state_bytes:int=4*1024**3

    def __post_init__(self):
        if (self.epochs<1 or self.accumulation_examples<1 or self.learning_rate<=0
                or not 0<self.minimum_lr_fraction<=1 or self.weight_decay<0 or self.gradient_clip<=0
                or self.temperature<=0 or self.feature_weight<0 or self.kl_weight<=0):
            raise ValueError('Valid explicit stage-distillation settings required')


class SequentialStage:
    """One current block; actual frozen student prefix and original suffix.

    Earlier prefix coordinates may be cached once per stage. A cache must bind
    exact token IDs and prefix state. Native boundary caches cannot substitute.
    All future compiled blocks already exist, but are not executed or trained.
    """
    def __init__(self,stack,embedding,student_rotary,lookahead,*,layer,guard=None,train_final_norm=False):
        if (stack.packed or stack.stripped or stack.layer_indices!=tuple(range(lookahead.expected_layers))
                or lookahead.start!=layer+1 or not 0<=layer<len(stack.blocks)):
            raise ValueError('Upfront whole-model stack and adjacent exact native suffix required')
        self.stack,self.embedding,self.rotary,self.lookahead=stack,embedding,student_rotary,lookahead
        self.layer,self.guard=layer,guard;self.device=embedding.device
        if train_final_norm and layer!=len(stack.blocks)-1:raise ValueError('Only the final stage may fit a private final norm')
        self.train_final_norm=bool(train_final_norm)
        if self.device.type=='cuda'and guard is None:raise ValueError('Caller reservation guard required')
        stack.eval().requires_grad_(False);embedding.requires_grad_(False);student_rotary.eval().requires_grad_(False)
        lookahead.eval().requires_grad_(False);self.block=stack.blocks[layer];self.block.requires_grad_(True)
        if self.train_final_norm:lookahead.norm.requires_grad_(True)
        self.prefix_fingerprint=offload.fingerprint(
            [(f'prefix.{i}.{n}',v)for i,b in enumerate(stack.blocks[:layer])for n,v in b.named_parameters()]
            +[(f'fixed.{n}',v)for n,v in stack.named_buffers()]+[('embedding',embedding)])['sha256']

    def trainable_parameters(self):
        return [*self.block.parameters(),*(self.lookahead.norm.parameters()if self.train_final_norm else [])]

    def _guard(self):
        if self.guard is not None:self.guard()

    def context(self,batch):
        self._guard();ids=batch['input_ids'].to(self.device);valid=batch['attention_mask'].to(self.device).bool()
        h=F.embedding(ids,self.embedding)
        positions=torch.arange(ids.shape[1],device=self.device)[None].expand_as(ids)
        allowed=torch.ones(ids.shape[1],ids.shape[1],device=self.device,dtype=torch.bool).tril()[None,None]&valid[:,None,None,:]
        mask=torch.zeros(allowed.shape,device=self.device,dtype=h.dtype).masked_fill(~allowed,torch.finfo(h.dtype).min)
        return h,positions,mask,self.rotary(h,positions)

    @torch.no_grad()
    def prefix_coordinates(self,batch):
        hidden,positions,mask,rope=self.context(batch)
        z=self.stack._geometry(hidden,self.stack.blocks[0].basis.T)
        for index in range(self.layer):
            block=self.stack.blocks[index]
            locations=block._positions(hidden,positions[0],positions,None)
            z,_=forward_reduced(block,z,locations,attention_mask=mask,cache_position=positions[0],position_embeddings=rope)
            bridge=self.stack.bridges[index]
            shift=block._rows(bridge.shift,positions,*z.shape[:2])
            z=self.stack._geometry(z,bridge.weight,shift)
        return z.detach()

    def forward(self,batch):
        hidden,positions,mask,rope=self.context(batch)
        if 'student_input_coordinates'in batch:
            if (batch.get('student_prefix_fingerprint')!=self.prefix_fingerprint
                    or not torch.equal(batch['student_prefix_input_ids'].cpu(),batch['input_ids'].cpu())):
                raise ValueError('Cached prefix is not bound to actual frozen student weights and exact tokens')
            z=batch['student_input_coordinates'].detach().to(self.embedding)
            if z.shape!=(*hidden.shape[:2],self.block.rank):raise ValueError('Cached student coordinate shape differs')
        else:z=self.prefix_coordinates(batch)
        locations=self.block._positions(hidden,positions[0],positions,None)
        coordinates,_=forward_reduced(self.block,z,locations,attention_mask=mask,cache_position=positions[0],position_embeddings=rope)
        anchor=self.block._rows(self.block.anchor,locations,*hidden.shape[:2])
        output=self.stack._geometry(coordinates,self.block.basis,anchor)
        final_hidden=self.lookahead(output,positions,mask)
        return coordinates,final_hidden


class StageTrainer:
    """Exact full-vocabulary T2 KL plus coordinate MSE, one block optimized."""
    def __init__(self,stage,dataset,*,teacher_id,train_indices,selection_indices,
                 config=DistillationConfig()):
        self.stage,self.dataset,self.config=stage,dataset,config
        self.training,self.selection=list(train_indices),list(selection_indices)
        if (not self.training or not self.selection or set(self.training)&set(self.selection)
                or any(dataset.records[i].split!='train'for i in [*self.training,*self.selection])):
            raise ValueError('Distinct admitted TRAIN fitting and held-out TRAIN selection indices required')
        self.parameters=stage.trainable_parameters()
        self.optimizer=offload.CPUGradientAdamW(self.parameters,lr=config.learning_rate,weight_decay=config.weight_decay,
            chunk_bytes=config.optimizer_chunk_bytes,max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.teacher=response_training.CachedHiddenTeacher(stage.embedding,teacher_id=teacher_id,guard=stage.guard)
        self.teacher.temperature=config.temperature
        self.loss_config=response_training.nal.DistillationConfig(objective='forward_kl',temperature=config.temperature,
            kd_weight=config.kl_weight,ce_weight=0.,token_chunk_size=config.token_chunk_size)
        self.total_updates=config.epochs*math.ceil(len(self.training)/config.accumulation_examples)
        self.updates=0;self.epoch=0;self.visited=set();self.parameter_ids=tuple(map(id,self.parameters))
        self.fixed=self.fixed_signature()

    def fixed_signature(self):
        s=self.stage
        return offload.fingerprint([(f'fixed.{n}',v)for n,v in s.stack.named_buffers()]
            +[(f'block.{i}.{n}',v)for i,b in enumerate(s.stack.blocks)if i!=s.layer for n,v in b.named_parameters()]
            +[(f'native.{n}',v)for n,v in [*s.lookahead.named_parameters(),*s.lookahead.named_buffers()]
                if not(s.train_final_norm and n.startswith('norm.'))]
            +[(f'rotary.{n}',v)for n,v in s.rotary.named_buffers()]+[('embedding',s.embedding)])

    def begin_epoch(self,epoch):
        if epoch!=self.epoch+1 or epoch>self.config.epochs or self.epoch and self.visited!=set(self.training):
            raise ValueError('Each declared stage epoch must cover all training examples exactly once')
        self.epoch=epoch;self.visited=set()

    def _features(self,coordinates,batch):
        target=batch['next_boundary_hidden'].detach().to(device=coordinates.device,dtype=torch.float32)
        target=F.linear(target,self.stage.block.basis.T.float())
        delta=coordinates.float()-target;mask=batch['attention_mask'].to(coordinates.device).bool()
        return delta,mask,delta.square().mean(-1)[mask].sum()

    def train_window(self,indices):
        indices=list(indices);cfg=self.config
        if (not self.epoch or not indices or len(indices)>cfg.accumulation_examples or len(set(indices))!=len(indices)
                or any(i not in self.training or i in self.visited for i in indices)):
            raise ValueError('Distinct unvisited stage TRAIN indices within a declared epoch required')
        if tuple(map(id,self.stage.trainable_parameters()))!=self.parameter_ids:raise RuntimeError('Live current-block parameter identities changed')
        count=sum(len(self.dataset.records[i].response_ids)for i in indices)
        feature_count=sum(len(self.dataset.records[i].input_ids)for i in indices)
        lr=cfg.learning_rate*(cfg.minimum_lr_fraction+(1-cfg.minimum_lr_fraction)*.5*(1+math.cos(math.pi*self.updates/max(1,self.total_updates-1))))
        self.optimizer.param_groups[0]['lr']=lr;self.optimizer.zero_grad();self.stage.block.train()
        feature_sum=kl_sum=0.;started=time.monotonic()
        try:
            for index in indices:
                batch=self.dataset.batch([index]);self.teacher.prepare(batch)
                coordinates,hidden=self.stage.forward(batch)
                delta,mask,feature=self._features(coordinates,batch)
                # Add the feature-loss adjoint at the reduced residual, so the
                # current block's shared graph is traversed exactly once.
                adjoint=(2*cfg.feature_weight/(feature_count*coordinates.shape[-1])*delta.detach()*mask[...,None]).to(coordinates.dtype)
                hook=coordinates.register_hook(lambda gradient,extra=adjoint:gradient+extra)
                try:
                    result=response_training.nal.chunked_backward(hidden,self.stage.lookahead.readout,
                        self.teacher,batch,self.loss_config,normalizer=count)
                finally:hook.remove();self.teacher.clear_cache()
                feature_sum+=float(feature.detach());kl_sum+=result['forward_kl']
            if any(self.optimizer.received.get(p,0)!=len(indices)for p in self.parameters):
                raise RuntimeError('Every current-block parameter must receive every microbatch gradient')
            receipt=self.optimizer.step(max_grad_norm=cfg.gradient_clip)
            self.updates+=1;self.visited.update(indices)
            return dict(epoch=self.epoch,update=self.updates,cases=len(indices),response_tokens=count,
                coordinate_tokens=feature_count,coordinate_mse=feature_sum/feature_count,teacher_kl=kl_sum/count,
                objective=cfg.feature_weight*feature_sum/feature_count+cfg.kl_weight*kl_sum/count,
                learning_rate=lr,optimizer=receipt,exact_frozen_native_lookahead=not self.stage.train_final_norm,
                native_suffix_blocks_frozen=True,original_teacher_targets_unchanged=True,
                only_current_block_trained=not self.stage.train_final_norm,
                private_final_norm_trained=self.stage.train_final_norm,seconds=time.monotonic()-started)
        finally:self.teacher.clear_cache();self.optimizer.zero_grad()

    @torch.inference_mode()
    def development(self):
        self.stage.block.eval();features=kl=0.;tokens=feature_tokens=0
        for index in self.selection:
            batch=self.dataset.batch([index]);self.teacher.prepare(batch)
            coordinates,hidden=self.stage.forward(batch);_,valid,feature=self._features(coordinates,batch)
            feature_tokens+=int(valid.sum())
            positions=batch['loss_mask'].nonzero().cpu();features+=float(feature);tokens+=len(positions)
            for start in range(0,len(positions),self.config.token_chunk_size):
                pos=positions[start:start+self.config.token_chunk_size]
                logits=self.stage.lookahead.readout(hidden[pos[:,0].to(hidden.device),pos[:,1].to(hidden.device)])
                target=batch['targets'][pos[:,0],pos[:,1]]
                values=response_training.nal.loss_vectors(logits,self.teacher(batch,pos),target,self.loss_config)
                kl+=float(values['forward_kl'].sum())
            self.teacher.clear_cache()
        return dict(cases=len(self.selection),response_tokens=tokens,coordinate_tokens=feature_tokens,coordinate_mse=features/feature_tokens,
            teacher_kl=kl/tokens,objective=self.config.feature_weight*features/feature_tokens+self.config.kl_weight*kl/tokens,
            scope='Exposed TRAIN selection objective, not task accuracy')

    def save_checkpoint(self,path,*,identities):
        if self.fixed_signature()!=self.fixed:raise RuntimeError('Frozen earlier/later blocks, native suffix or geometry changed')
        with Path(path).open('xb')as stream:
            torch.save(dict(schema='jspace.ablation32-stage-checkpoint.v1',layer=self.stage.layer,
                block=native.state_dict_cpu(self.stage.block),identities=identities,configuration=asdict(self.config),
                norm=native.state_dict_cpu(self.stage.lookahead.norm)if self.stage.train_final_norm else None,
                updates=self.updates,epoch=self.epoch,visited=sorted(self.visited),fixed_state=self.fixed),stream)
        return dict(path=str(Path(path).resolve()),sha256=offload.sha(path))

    def close(self):self.teacher.clear_cache();self.optimizer.close()
