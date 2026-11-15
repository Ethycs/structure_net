"""Cautious output KD after entrance prealignment, with a protected raw fallback.

Architecture, teacher loss, BF16 arithmetic and CPU Adam are inherited. Only
joint learning rate and prospective development eligibility/selection change.
"""
from collections.abc import Mapping
from dataclasses import asdict, replace
import gc
import math
import os
from pathlib import Path
import random
import time

import torch

from . import jspace_qwen3b_native_budget_train as inherited

prior = inherited.prior
old, core = prior.prior, prior.core
CONFIG = replace(inherited.CONFIG, learning_rate=2e-5)
TOLERANCE = 1e-8
SELECTION = "eligible_correct_then_ce_earliest; protected_raw_always_eligible"
selection_key = prior.selection_key


def candidate_eligibility(metrics, protected):
    """A new candidate cannot increase CE or original-teacher KL over raw."""
    for value in (metrics, protected):
        if (type(value.get("count")) is not int or value["count"] < 1 or
            type(value.get("correct")) is not int or not 0 <= value["correct"] <= value["count"]):
            raise ValueError("Valid integer development counts required")
    if metrics["count"] != protected["count"]:
        raise ValueError("Candidate and protected populations differ")
    if not all(math.isfinite(protected[k]) for k in ("ce", "original_kl")):
        raise ValueError("Protected CE and original-teacher KL must be finite")
    finite = all(math.isfinite(metrics[k]) for k in ("ce", "original_kl"))
    ce = finite and metrics["ce"] <= protected["ce"] + TOLERANCE
    kl = finite and metrics["original_kl"] <= protected["original_kl"] + TOLERANCE
    return dict(eligible=ce and kl, finite=finite, ce_pass=ce, original_kl_pass=kl,
                tolerance=TOLERANCE)


def best_eligible(history, protected):
    """Recompute prospective eligibility; preserve the earlier candidate on ties."""
    if not history or history[0].get("epoch") != -1 or history[0].get("metrics") != protected:
        raise ValueError("Protected raw candidate must lead history")
    if [r.get("epoch") for r in history] != list(range(-1, len(history)-1)):
        raise ValueError("History must have contiguous protected/warm/trained epochs")
    eligible = []
    for row in history:
        check = candidate_eligibility(row["metrics"], protected)
        if row.get("eligibility") != check:
            raise ValueError("Stored candidate eligibility disagrees with protected reference")
        if row["epoch"] == -1 or check["eligible"]:
            eligible.append(row)
    return max(eligible, key=lambda row: selection_key(row["metrics"]))


def _load(value):
    if isinstance(value, (str, Path)):
        value = torch.load(value, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, Mapping) or not all(isinstance(value.get(k), Mapping) for k in ("stack", "norm")):
        raise ValueError("Protected checkpoint must contain unpacked stack and norm states")
    if any(t.device.type != "cpu" for state in value.values() if isinstance(state, Mapping)
           for t in state.values() if isinstance(t, torch.Tensor)):
        raise ValueError("Protected checkpoint tensors must be on CPU")
    return value


def state_fingerprint(state):
    tensors = [(group+"."+name, value) for group in ("stack", "norm")
               for name, value in state[group].items() if isinstance(value, torch.Tensor)]
    metadata = {group: {name:value for name,value in state[group].items() if not isinstance(value,torch.Tensor)}
                for group in ("stack", "norm")}
    return dict(**old.fingerprint(tensors), metadata_sha256=old.base.digest(metadata))


class CautiousTrainer(inherited.NativeBudgetTrainer):
    def __init__(self, *args, config=CONFIG, **kwargs):
        super().__init__(*args, config=config, **kwargs)

    def fit(self, path, protected_state, protected_metrics, protected_records, *, event=None):
        if self.updates or self.finished:
            raise ValueError("Scientific fit requires untouched warm-start optimizer")
        candidate_eligibility(protected_metrics, protected_metrics)
        if protected_metrics["count"] != self.development_count:
            raise ValueError("Protected reference must cover complete development")
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        event=event or (lambda *_a,**_kw:None)
        self._guard();started=time.monotonic()
        before=prior.fixed_state(self.stack,self.readout,self.rotary)
        protected=_load(protected_state)
        warm=self.snapshot();warm_digest=state_fingerprint(warm)
        try:
            self.stack.load_state_dict(protected["stack"],strict=True)
            self.norm.load_state_dict(protected["norm"],strict=True)
            if prior.fixed_state(self.stack,self.readout,self.rotary)!=before:
                raise ValueError("Protected and warm-start fixed geometry/readout differ")
            measured,rows=self.packed_development()
            if measured!=protected_metrics or rows!=protected_records:
                raise RuntimeError("Protected initializer packed replay changed metrics or records")
        finally:
            self.stack.load_state_dict(warm["stack"],strict=True)
            self.norm.load_state_dict(warm["norm"],strict=True)
        restored=state_fingerprint(dict(stack=self.stack.state_dict(),norm=self.norm.state_dict()))
        if restored!=warm_digest:
            raise RuntimeError("Temporary protected validation changed warm-start model")
        del warm;gc.collect()
        proof=dict(protected_replay_exact=True,warm_restore_exact=True,
            warm_state_fingerprint_before=warm_digest,warm_state_fingerprint_after=restored,
            optimizer_untouched=True,scope="Model weights temporarily loaded; optimizer state never loaded or copied")
        old.base.write(path/"protected_replay.json",proof)
        history=[];best=None;best_epoch=None

        def record(epoch, metrics, records, updates, state=None):
            nonlocal best,best_epoch
            filename=f"epoch_{epoch:02d}_development.json"
            old.base.write(path/filename,dict(metrics=metrics,records=records))
            eligibility=candidate_eligibility(metrics,protected_metrics)
            changed=(epoch==-1 or (eligibility["eligible"] and selection_key(metrics)>selection_key(best)))
            row=dict(epoch=epoch,metrics=metrics,training=updates,eligibility=eligibility,
                development_sha256=old.base.sha(path/filename),checkpoint_updated=changed)
            if changed:
                pending=path/"selected.pt.pending"
                old.save(pending,state if state is not None else self.snapshot());os.replace(pending,path/"selected.pt")
                best,best_epoch=metrics,epoch;row["checkpoint_sha256"]=old.base.sha(path/"selected.pt")
            history.append(row);old.base.write(path/f"epoch_{epoch:02d}.json",row)
            event("development",epoch=epoch,selected_epoch=best_epoch,eligibility=eligibility,**metrics)

        record(-1,protected_metrics,protected_records,[],protected)
        del protected
        total=self.config.epochs*math.ceil(len(self.dataset.training)/(self.config.microbatch*self.config.accumulation))
        effective=self.config.microbatch*self.config.accumulation
        torch.manual_seed(self.config.seed)
        for epoch in range(self.config.epochs+1):
            updates=[]
            if epoch:
                order=list(self.dataset.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),effective):
                    lr=self.config.learning_rate*.5*(1+math.cos(math.pi*self.updates/total))
                    update=self.train_window(order[start:start+effective],learning_rate=lr);updates.append(update)
                    event("update",epoch=epoch,**update)
            metrics,rows=self.packed_development()
            record(epoch,metrics,rows,updates)
        if self.updates!=total:
            raise RuntimeError("Wrong optimizer update budget")
        winner=best_eligible(history,protected_metrics)
        if winner["epoch"]!=best_epoch:
            raise RuntimeError("Eligible checkpoint selection inconsistent")
        state=_load(path/"selected.pt")
        self.stack.load_state_dict(state["stack"],strict=True);self.norm.load_state_dict(state["norm"],strict=True)
        del state
        final,rows=self.packed_development()
        reference=old.read(path/f"epoch_{best_epoch:02d}_development.json")
        if final!=best or rows!=reference["records"]:
            raise RuntimeError("Selected packed reload changed results")
        if prior.fixed_state(self.stack,self.readout,self.rotary)!=before:
            raise RuntimeError("Recovery changed fixed geometry or readout")
        self.finished=True;quality=self.quality(best["correct"])
        result=dict(status="completed",systems_only=False,confirmed=False,
            layer_indices=list(self.stack.layer_indices),cut=self.stack.layer_indices[0],suffix_blocks=len(self.stack.blocks),
            completed_epochs=self.config.epochs,updates=self.updates,config=asdict(self.config),history=history,
            initial=protected_metrics,warm_initial=history[1]["metrics"],selected=best,selected_epoch=best_epoch,
            initializer_selected=best_epoch==-1,protected_epoch=-1,
            protected_reference=dict(metrics=protected_metrics,development_sha256=history[0]["development_sha256"],
                checkpoint_sha256=history[0]["checkpoint_sha256"]),protected_replay=proof,
            selection_policy=SELECTION,eligibility_tolerance=TOLERANCE,
            selected_checkpoint_sha256=old.base.sha(path/"selected.pt"),quality=quality,quality_pass=quality["passed"],
            controls=dict(native_correct=self.native_correct,parent_correct=self.parent_correct,count=self.development_count),
            strict_reload_exact=True,fixed_state_unchanged=True,fixed_state=before,
            optimizer_memory=self.optimizer.memory_stats(),elapsed_seconds=time.monotonic()-started,
            scope="Protected raw initializer and prealigned warm start; original-native budget; development only")
        old.base.write(path/"result.json",result)
        return result
