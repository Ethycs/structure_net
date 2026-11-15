"""Parameterize the verified Qwen2.5-3B one-block suffix expansion recipe.

Only orchestration/gates are generalized. The unchanged compact stack,
BF16 forward/backward, CPU FP32-master Adam, output objective and packed
scoring are reused. The caller authenticates the frozen native prefix,
accepted parent, freshly calibrated entrance block, pilot and reservations.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from . import jspace_incremental_suffix as prior
from . import jspace_qwen_parallel_train as core

CONFIG = core.RecoveryConfig(seed=20861, max_cpu_state_bytes=32 * 1024**3)
SELECTION = "highest_correct_then_lowest_full_vocab_ce_earlier_tie"
SCOPE = "One-block progressive Qwen2.5-3B suffix expansion; development only; no held-out or speed claim"
# The proven helper already supports any immediately preceding native layer.
preserve_inheritance = prior.preserve_inheritance
fixed_state = prior.fixed_state
master_fingerprint = prior.master_fingerprint
compare_rows = prior.compare_rows
selection_key = prior.selection_key


def gates(correct, *, native_correct, parent_correct, count, maximum_drop=.05):
    if type(count) is not int or count < 1 or any(type(x) is not int or not 0 <= x <= count
                                               for x in (correct,native_correct,parent_correct)):
        raise ValueError("Correct counts must be integers within the matched development count")
    if maximum_drop != .05:
        raise ValueError("This recipe fixes the accuracy budget to five percentage points")
    native_min=math.ceil(native_correct-maximum_drop*count)
    parent_min=math.ceil(parent_correct-maximum_drop*count)
    return dict(passed=correct>=max(native_min,parent_min),minimum_correct=max(native_min,parent_min),
        count=count,maximum_drop=maximum_drop,
        native=dict(correct=native_correct,minimum_correct=native_min,passed=correct>=native_min),
        parent=dict(correct=parent_correct,minimum_correct=parent_min,passed=correct>=parent_min))


class ProgressiveTrainer(prior.Trainer):
    """Train the entire accepted suffix plus exactly one new entrance block.

    ``parent_correct`` is explicit: the runner supplies its authenticated parent
    control count. The unchanged original-native count is supplied separately.
    Scientific gate failure is a completed outcome, never an implicit retry.
    """
    def __init__(self,stack,norm,readout_weight,rotary,dataset,label_ids,*,native_correct,parent_correct,
                 count=64,config=CONFIG,guard=None):
        layers=list(stack.layer_indices)
        if not layers or layers != list(range(layers[0],36)):
            raise ValueError("Expected a contiguous Qwen2.5-3B suffix ending at layer35")
        if config.dtype != "bfloat16":
            raise ValueError("The accepted 3B recovery recipe uses BF16 execution")
        if len(dataset.development) != count:
            raise ValueError("Control counts must match the complete development split")
        gates(0,native_correct=native_correct,parent_correct=parent_correct,count=count)
        self.native_correct,self.parent_correct,self.development_count=native_correct,parent_correct,count
        super().__init__(stack,norm,readout_weight,rotary,dataset,label_ids,config=config,guard=guard)

    def quality(self,correct):
        return gates(correct,native_correct=self.native_correct,parent_correct=self.parent_correct,
                     count=self.development_count)

    def master_update_probe(self,indices):
        """One real discarded update proves all blocks/norm receive effective gradients.

        CPU masters must change even when a single small update is below a BF16
        stored-weight ULP. StreamingCPUAdamW rejects all nonfinite gradients
        before modifying any state. This method makes the instance ineligible
        for scientific fit; construct a fresh trainer after the pilot.
        """
        if self.updates or self.finished:
            raise ValueError("Master probe requires an untouched discarded initializer")
        before=fixed_state(self.stack,self.readout,self.rotary)
        block_before=[master_fingerprint(block,self.optimizer) for block in self.stack.blocks]
        norm_before=master_fingerprint(self.norm,self.optimizer)
        stored_norm=prior.fingerprint(self.norm.named_parameters())
        update=self.train_window(indices,learning_rate=self.config.learning_rate)
        block_after=[master_fingerprint(block,self.optimizer) for block in self.stack.blocks]
        norm_after=master_fingerprint(self.norm,self.optimizer)
        changed=[a!=b for a,b in zip(block_before,block_after)]
        norm_changed=norm_before!=norm_after
        unchanged=fixed_state(self.stack,self.readout,self.rotary)==before
        gradient_norm=update["optimizer"]["unclipped_gradient_norm"]
        passed=(all(update["gradients_present_by_block"]) and all(changed) and norm_changed and unchanged
                and math.isfinite(gradient_norm) and gradient_norm>0)
        return dict(passed=passed,systems_only=True,discarded=True,update=update,
            layer_indices=list(self.stack.layer_indices),fp32_master_before=block_before,fp32_master_after=block_after,
            fp32_master_changed_by_block=changed,norm_fp32_master_before=norm_before,
            norm_fp32_master_after=norm_after,norm_fp32_master_changed=norm_changed,
            norm_bf16_storage_changed=stored_norm!=prior.fingerprint(self.norm.named_parameters()),
            fixed_state_unchanged=unchanged,fixed_state=before,
            gradient_evidence="Finite validated gradients and nonzero first FP32-master updates; BF16 storage changes are diagnostic")

    def fit(self,path,*,event=None,initial_reference=None):
        if self.updates or self.finished:
            raise ValueError("Scientific fit requires untouched initializer; discard pilot trainers")
        path=Path(path); path.mkdir(parents=True,exist_ok=False)
        event=event or (lambda *_a,**_kw:None)
        before=fixed_state(self.stack,self.readout,self.rotary); history=[]; best=None; best_epoch=None
        total=self.config.epochs*math.ceil(len(self.dataset.training)/(self.config.microbatch*self.config.accumulation))
        effective=self.config.microbatch*self.config.accumulation
        started=time.perf_counter();torch.manual_seed(self.config.seed)
        for epoch in range(self.config.epochs+1):
            training=[]
            if epoch:
                order=list(self.dataset.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),effective):
                    lr=self.config.learning_rate*.5*(1+math.cos(math.pi*self.updates/total))
                    update=self.train_window(order[start:start+effective],learning_rate=lr);training.append(update)
                    event("update",epoch=epoch,**update)
            metrics,records=self.packed_development();report=path/f"epoch_{epoch:02d}_development.json"
            if metrics["count"]!=self.development_count:
                raise ValueError("Incomplete development scoring")
            if epoch==0 and initial_reference is not None:
                check=compare_rows(records,initial_reference["initial_records"],
                    [self.dataset.items[i] for i in self.dataset.development],
                    expected_correct=initial_reference["initial"]["correct"])
                prior.base.write(path/"initializer_replay.json",check)
                if not check["passed"]:
                    raise RuntimeError("Scientific initializer differs from discarded pilot before training")
            prior.base.write(report,dict(metrics=metrics,records=records))
            changed=best is None or selection_key(metrics)>selection_key(best)
            row=dict(epoch=epoch,metrics=metrics,training=training,
                     development_sha256=prior.base.sha(report),checkpoint_updated=changed)
            if changed:
                pending=path/"selected.pt.pending";prior.save(pending,self.snapshot());os.replace(pending,path/"selected.pt")
                best,best_epoch=metrics,epoch;row["checkpoint_sha256"]=prior.base.sha(path/"selected.pt")
            history.append(row)
            prior.base.write(path/f"epoch_{epoch:02d}.json",row)
            event("development",epoch=epoch,selected_epoch=best_epoch,**metrics)
        if self.updates!=total:raise RuntimeError("Wrong optimizer update budget")
        state=torch.load(path/"selected.pt",map_location="cpu",weights_only=True,mmap=True)
        self.stack.load_state_dict(state["stack"],strict=True);self.norm.load_state_dict(state["norm"],strict=True)
        del state
        final,rows=self.packed_development()
        reference=json.loads((path/f"epoch_{best_epoch:02d}_development.json").read_text())
        if final!=best or rows!=reference["records"]:
            raise RuntimeError("Selected packed reload changed results")
        if fixed_state(self.stack,self.readout,self.rotary)!=before:
            raise RuntimeError("Recovery changed fixed geometry or readout")
        self.finished=True;quality=self.quality(best["correct"])
        result=dict(status="completed",systems_only=False,confirmed=False,
            layer_indices=list(self.stack.layer_indices),cut=self.stack.layer_indices[0],
            suffix_blocks=len(self.stack.blocks),completed_epochs=self.config.epochs,updates=self.updates,
            config=asdict(self.config),initial=history[0]["metrics"],selected=best,selected_epoch=best_epoch,
            initializer_selected=best_epoch==0,history=history,
            selected_checkpoint_sha256=prior.base.sha(path/"selected.pt"),quality=quality,quality_pass=quality["passed"],
            controls=dict(native_correct=self.native_correct,parent_correct=self.parent_correct,count=self.development_count),
            strict_reload_exact=True,fixed_state_unchanged=True,fixed_state=before,
            optimizer_memory=self.optimizer.memory_stats(),elapsed_seconds=time.perf_counter()-started,scope=SCOPE)
        prior.base.write(path/"result.json",result);return result
