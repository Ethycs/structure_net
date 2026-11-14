"""Prospective joint recovery using enriched bases and answer-ranking losses.

Native BF16-rounded initialization, full native attention heads, rank1024
compact updates, and a carried native residual. The original tied embedding /
full-vocabulary readout stays fixed. Training uses only the original512 train
and64 development cases; confirmation is neither opened nor selected on here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict,dataclass,field
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_joint_recovery as joint
from . import jspace_quadratic_recipe as recipe
from .jspace_semantic_loss import SemanticLossConfig,semantic_recovery_loss

SCHEMA="jspace.semantic-joint-recovery.v1"
BASIS_SCHEMA="jspace.all-layer-enriched-basis.v1"
METRICS=("loss","full_kl","choice_kl","choice_ce","weighted_full_kl","weighted_choice_kl","weighted_choice_ce",
         "accuracy","teacher_accuracy","teacher_agreement","choice_margin","valid_label_probability","answer_full_vocab_ce")


@dataclass(frozen=True)
class SemanticRecoveryConfig:
    epochs:int=3
    microbatch:int=1
    accumulation:int=16
    learning_rate:float=1e-5
    gradient_clip:float=1.
    seed:int=27863
    optimizer_chunk_bytes:int=32*1024**2
    max_cpu_state_bytes:int=32*1024**3
    objective:SemanticLossConfig=field(default_factory=SemanticLossConfig)

    def __post_init__(self):
        # Reuse validated optimizer/schedule settings without inheriting the
        # previous experiment's objective or its two formatting positions.
        joint.JointConfig(epochs=self.epochs,microbatch=self.microbatch,accumulation=self.accumulation,
            learning_rate=self.learning_rate,gradient_clip=self.gradient_clip,seed=self.seed,
            optimizer_chunk_bytes=self.optimizer_chunk_bytes,max_cpu_state_bytes=self.max_cpu_state_bytes)
        if not isinstance(self.objective,SemanticLossConfig):raise ValueError("Validated semantic loss configuration required")


def config_from_dict(value):
    values=dict(value);values["objective"]=SemanticLossConfig(**values["objective"])
    return SemanticRecoveryConfig(**values)


def ranking_summary(records):
    def one(rows):
        truth=Counter(chr(65+r["answer"]) for r in rows)
        predictions=Counter(chr(65+r["prediction"]) for r in rows)
        correct=sum(r["answer"]==r["prediction"] for r in rows)
        return dict(count=len(rows),correct=correct,accuracy=correct/len(rows),
            prediction_counts=dict(sorted(predictions.items())),true_answer_counts=dict(sorted(truth.items())),
            best_constant_label_correct=max(truth.values()),best_constant_label_accuracy=max(truth.values())/len(rows),
            gained_correct_over_best_constant_label=correct-max(truth.values()))
    if not records:raise ValueError("Nonempty ranking records required")
    return dict(**one(records),by_family={f:one([r for r in records if r["family"]==f])
        for f in sorted({r["family"] for r in records})})


class SemanticTrainer(joint.JointTrainer):
    """Reuse bounded CPU AdamW and append-only strict checkpoint lifecycle."""
    def __init__(self,model,dataset,*,config=SemanticRecoveryConfig(),guard=None):
        super().__init__(model,dataset,config=config,guard=guard)

    def losses(self,batch):
        # Compute only the two supervised full-vocabulary rows. Teacher cache
        # remains unchanged [B,4,V]; the pure helper selects its slots1/3.
        selected=batch["loss_positions"][:,[1,3]]
        logits=self.model(batch["input_ids"],batch["position_ids"],batch["attention_mask"],selected)
        cases=[self.dataset.items[i] for i in batch["indices"]]
        counts=torch.tensor([len(c["choices"]) for c in cases],device=self.device,dtype=torch.long)
        answers=torch.tensor([c["answer"] for c in cases],device=self.device,dtype=torch.long)
        values=semantic_recovery_loss(logits,batch["teacher"],label_ids=self.dataset.label_ids,
            choice_counts=counts,answers=answers,config=self.config.objective)
        return logits,values

    def _records(self,indices,logits,values):
        records=[]
        for row,index in enumerate(indices):
            case=self.dataset.items[index]
            label_logits=logits[row,-1,self.dataset.label_ids[:len(case["choices"])]]
            records.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                prediction=int(values["prediction"][row]),teacher_prediction=int(values["teacher_prediction"][row]),
                logits=label_logits.detach().float().cpu().tolist(),
                full_kl=float(values["per_case_full_kl"][row].detach()),
                choice_kl=float(values["per_case_choice_kl"][row].detach()),
                choice_ce=float(values["per_case_choice_ce"][row].detach()),
                valid_label_probability=float(values["per_case_valid_label_probability"][row]),
                choice_margin=float(values["per_case_choice_margin"][row]),
                answer_full_vocab_ce=float(values["per_case_answer_full_vocab_ce"][row]),
                # Compatibility with the read-only CE decomposition helper;
                # this alias is full-vocabulary CE, not the optimized choice CE.
                ce=float(values["per_case_answer_full_vocab_ce"][row])))
        return records

    def train_window(self,indices):
        if not indices or any(i not in self.dataset.training for i in indices):
            raise ValueError("Optimizer updates consume training identities only")
        self._guard();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        totals={key:0. for key in METRICS};records=[]
        for start in range(0,len(indices),self.config.microbatch):
            self._guard();selected=indices[start:start+self.config.microbatch]
            batch=self.dataset.batch(selected,self.device);logits,values=self.losses(batch)
            weight=len(selected)/len(indices)
            (values["loss"]*weight).backward()
            for key in totals:totals[key]+=float(values[key].detach())*weight
            records+=self._records(selected,logits,values)
            del logits,values,batch
        missing=[name for name,p in self.named_parameters if p.grad is None]
        if missing:raise RuntimeError("Joint parameters missing gradients: "+", ".join(missing))
        step=self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        self.optimizer.zero_grad(set_to_none=True);self.updates+=1
        return dict(**totals,cases=len(indices),update=self.updates,optimizer=step,ranking=ranking_summary(records),
            gradient_parameter_count=len(self.named_parameters),gradient_scalar_count=sum(p.numel() for _,p in self.named_parameters))

    @torch.no_grad()
    def development(self):
        self._guard();self.model.eval();self.optimizer.zero_grad(set_to_none=True)
        totals={key:0. for key in METRICS};records=[]
        for start in range(0,len(self.dataset.development),self.config.microbatch):
            self._guard();indices=self.dataset.development[start:start+self.config.microbatch]
            batch=self.dataset.batch(indices,self.device);logits,values=self.losses(batch)
            for key in totals:totals[key]+=float(values[key])*len(indices)
            records+=self._records(indices,logits,values)
        metrics={key:value/len(records) for key,value in totals.items()}
        metrics.update(ranking_summary(records))
        # Existing accuracy-first checkpoint lifecycle uses 'kl' solely as a
        # secondary tie-break. Its meaning is explicit in every report.
        metrics.update(kl=metrics["full_kl"],ce=metrics["answer_full_vocab_ce"],
            selection="highest_dev_accuracy_then_lowest_full_vocab_KL_midpoint_and_answer",
            kl_alias="full_kl_midpoint_and_answer",ce_alias="diagnostic_answer_full_vocab_ce_not_optimized_choice_ce")
        return metrics,records


def build_enriched_block(native,fit,*,parameter_dtype=torch.bfloat16):
    from .jspace_qwen_affine_block import AffineQwenBlock
    from .jspace_quadratic_qwen import QuadraticQwenBlock
    from .jspace_residual_carry import ResidualCarryBlock
    native=native.float().eval().requires_grad_(False)
    device=next(native.parameters()).device;width=native.self_attn.q_proj.in_features
    u,mean=fit["basis"].to(device=device,dtype=torch.float32),fit["means"].to(device=device,dtype=torch.float32)
    if u.shape!=(width,width//2) or mean.shape!=(1,width):raise ValueError("Half-width enriched basis/global mean required")
    # Stored head64 calibration arrays never enter this full-head architecture.
    affine=AffineQwenBlock(native,u,mean,parameter_dtype=torch.float32,trainable=True,head_projection=None)
    wrapped=QuadraticQwenBlock(affine,mode="softmax",learn_metric=False)
    for parameter in wrapped.parameters():
        parameter.data=parameter.data.to(parameter_dtype);parameter.requires_grad_(True)
    block=ResidualCarryBlock(wrapped)
    if block.basis.dtype!=torch.float32 or block.head_dim!=block.native_head_dim or block.has_head_projection:
        raise ValueError("Full native heads and FP32 geometry must survive construction")
    return block


def build_model(source,basis_root,device,*,guard=lambda:None):
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    parent=recipe.read(Path(source)/"plan.json");config=recipe.native_config(parent)
    if not config.tie_word_embeddings:raise ValueError("Original tied embedding/readout required")
    guard();embedding,norm,rotary=load_readout_norm(parent["model"]["path"],device=device,dtype=torch.bfloat16)
    blocks=[]
    for layer in range(parent["config"]["layers"]):
        guard();fit=recipe.load(Path(basis_root)/"basis"/f"block_{layer:02d}"/"fit.pt")
        native=load_native_block(parent["model"]["path"],layer,device=device,dtype=torch.bfloat16)
        block=build_enriched_block(native,fit)
        if (block.width,block.rank,block.head_dim,block.nq,block.nkv,block.intermediate_size)!=(2048,1024,128,16,2,11008):
            raise ValueError("Runtime architecture differs from the declared full-head Qwen recipe")
        blocks.append(block);del native,fit
    return joint.JointCarryModel(embedding,blocks,norm.weight,norm.variance_epsilon,rotary)


def validate_basis_root(basis_root,source,parent,items):
    basis_root,source=Path(basis_root).resolve(),Path(source).resolve()
    plan=recipe.read(basis_root/"plan.json");manifest=recipe.read(basis_root/"basis/manifest.json")
    if (plan.get("schema")!=BASIS_SCHEMA or plan.get("fingerprint")!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"})
            or Path(plan["source_root"]).resolve()!=source or plan["source_fingerprint"]!=parent["fingerprint"]
            or plan["layers"]!=list(range(36)) or plan["rank"]!=1024
            or manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]
            or manifest.get("source_fingerprint")!=parent["fingerprint"]):
        raise ValueError("Complete matching all36 enriched bases required")
    train_ids={c["identity"] for c in items if c["split"]=="train"}
    fitted,checked=set(plan["fit_training_identities"]),set(plan["internal_training_check_identities"])
    if not fitted or not checked or fitted&checked or not ((fitted|checked)<=train_ids):
        raise ValueError("Enrichment fitting/check identities must be disjoint training-only sets")
    expected={f"basis/block_{i:02d}/{name}" for i in range(36) for name in ("fit.pt","fit.json")}
    if set(manifest["artifacts"])!=expected:raise ValueError("Enriched manifest does not cover all36 blocks")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(basis_root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:
            raise ValueError("Enrichment producer changed")
    return plan,manifest


def prepare(root,source,basis_root,*,config=SemanticRecoveryConfig()):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU preparation")
    root,source,basis_root=map(lambda p:Path(p).resolve(),(root,source,basis_root))
    if root.exists():raise FileExistsError(root)
    parent=recipe.verify(source);items=recipe.read(source/"items.json")
    if Counter(c["split"] for c in items["items"])!={"train":512,"development":64}:
        raise ValueError("Original512train/64development population required")
    fitted,manifest=validate_basis_root(basis_root,source,parent,items["items"])
    teacher=Path(parent["teacher_root"])/"capture/teacher_logits.npy"
    inputs={str(teacher):parent["input_sha256"][str(teacher)]}
    for layer in range(36):
        path=basis_root/"basis"/f"block_{layer:02d}"/"fit.pt"
        inputs[str(path)]=manifest["artifacts"][str(path.relative_to(basis_root))]
    for path,expected in inputs.items():
        if recipe.sha(path)!=expected:raise ValueError("Changed training input: "+path)
    for path in (source/"plan.json",source/"items.json",basis_root/"plan.json",basis_root/"basis/manifest.json"):
        inputs[str(path)]=recipe.sha(path)
    producers=[Path(__file__),Path(__file__).with_name("jspace_semantic_loss.py"),
        Path(__file__).with_name("jspace_joint_recovery.py"),Path(__file__).with_name("jspace_streaming_adamw.py"),
        Path(__file__).with_name("jspace_residual_carry.py"),recipe.ROOT/"tests/structure_net/test_jspace_semantic_recovery.py",
        recipe.ROOT/"tests/structure_net/test_jspace_semantic_loss.py"]
    plan=dict(schema=SCHEMA,source=str(source),source_fingerprint=parent["fingerprint"],source_config=parent["config"],
        basis_root=str(basis_root),basis_fingerprint=fitted["fingerprint"],config=asdict(config),items_digest=recipe.digest(items),
        label_ids=parent["label_ids"],teacher_path=str(teacher),layers=36,rank=1024,executed_head_dim=128,
        stored_head_projection_used=False,attention="softmax",learn_metric=False,
        initialization="Original native BF16-rounded weights; fresh compile into enriched bases; no prior recovered weights",
        embedding_readout="Original frozen tied full-vocabulary BF16 E; FP32 final RMS gain trains",
        geometry_residual_dtype="float32",branch_parameter_dtype="bfloat16",intermediate_size=11008,
        objective="0.25*T^2 full-vocabulary KD at slots1/3 +0.25*T^2 answer conditional-choice KD +0.50 answer conditional-choice CE; T2",
        selection="Highest development accuracy, then lower full-vocabulary KL at midpoint/answer",
        pilot="Separate systems-only one accumulated update on16 longest training cases; pilot weights never enter train",
        input_sha256=inputs,input_identity={p:recipe.file_identity(p) for p in inputs},
        source_sha256={str(p):recipe.sha(p) for p in producers},
        optimizer_resume="Warm-start weights only; further training requires explicit optimizer reset in a new experiment")
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path in producers:
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan);recipe.write(root/"items.json",items)
    return dict(status="prepared",fingerprint=plan["fingerprint"],basis_fingerprint=fitted["fingerprint"])


def verify(root):
    root=Path(root);plan=recipe.read(root/"plan.json")
    if plan["schema"]!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("Changed semantic recovery plan")
    parent=recipe.verify(plan["source"]);items=recipe.read(root/"items.json")
    if parent["fingerprint"]!=plan["source_fingerprint"] or recipe.digest(items)!=plan["items_digest"]:
        raise ValueError("Changed original teacher/case identities")
    fitted,_=validate_basis_root(plan["basis_root"],plan["source"],parent,items["items"])
    if fitted["fingerprint"]!=plan["basis_fingerprint"]:raise ValueError("Changed enriched basis plan")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:
            raise ValueError("Changed semantic recovery producer")
    for path,identity in plan["input_identity"].items():
        if recipe.file_identity(path)!=identity:raise ValueError("Changed frozen recovery input metadata")
    return plan


def run(root,stage):
    if stage not in ("pilot","train"):raise ValueError("Only pilot/train may execute")
    root=Path(root).resolve();plan=verify(root);directory=root/stage
    if directory.exists():raise FileExistsError(directory)
    if stage=="train":joint.verify_pilot(root,plan["fingerprint"])
    from .jspace_full_lease import wait_ready
    started=time.monotonic();lease=None;dataset=None;cuda_started=False
    try:
        lease=wait_ready([0]);allocation=dict(lease["allocation"]);joint.check_reserved_peak(allocation,0)
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=allocation:raise RuntimeError("Semantic recovery lease changed")
            joint.check_reserved_peak(allocation,torch.cuda.max_memory_reserved())
        torch.set_num_threads(4);torch.manual_seed(plan["config"]["seed"])
        torch.cuda.init();cuda_started=True;torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
        config=config_from_dict(plan["config"])
        model=build_model(plan["source"],plan["basis_root"],"cuda:0",guard=guard)
        dataset=joint.JointDataset(recipe.read(root/"items.json")["items"],plan["teacher_path"],plan["label_ids"])
        trainer=SemanticTrainer(model,dataset,config=config,guard=guard)
        if stage=="pilot":
            directory.mkdir(exist_ok=False)
            order=sorted(dataset.training,key=lambda i:len(dataset.items[i]["input_ids"]),reverse=True)
            update=trainer.train_window(order[:config.microbatch*config.accumulation])
            result=dict(status="completed",systems_only=True,updates=1,update=update,
                longest_context=max(len(dataset.items[i]["input_ids"]) for i in order),executed_head_dim=128,
                trainable_scalars=sum(p.numel() for _,p in trainer.named_parameters))
            recipe.write(directory/"result.json",result)
        else:
            result=trainer.fit(directory,event=lambda kind,**values:recipe.event(root,kind,stage=stage,**values))
        guard()
        metrics=dict(status="completed",systems_only=stage=="pilot",fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"),elapsed_seconds=time.monotonic()-started,
            lease_id=lease["id"],allocation=allocation,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),reserved_budget_bytes=joint.check_reserved_peak(allocation,torch.cuda.max_memory_reserved()))
        recipe.write(directory/"manifest.json",metrics)
        return dict(**result,execution=metrics)
    except Exception as error:
        directory.mkdir(parents=True,exist_ok=True)
        recipe.write(directory/"failure.json",dict(status="failed",stage=stage,systems_only=stage=="pilot",fingerprint=plan["fingerprint"],
            error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,
            lease_id=None if lease is None else lease["id"],allocation=None if lease is None else lease["allocation"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_started else None))
        raise
    finally:
        if dataset is not None:dataset.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot","train"));parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--source",type=Path);parser.add_argument("--basis-root",type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":
        if args.source is None or args.basis_root is None:parser.error("prepare requires --source and --basis-root")
        result=prepare(args.root,args.source,args.basis_root)
    else:result=run(args.root,args.stage)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
