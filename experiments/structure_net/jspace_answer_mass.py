"""CPU-only read-only decomposition of saved development answer losses.

For the true label y and permitted answer-label set A,
 -log P(y) = -log P(y | A) - log P(A).
This distinguishes choosing the right letter from placing probability on the
answer-letter set at all. Inputs are existing development records; no models,
teacher tensors, confirmation cases, or GPU libraries are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def _sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def decompose_records(records):
    if not records or len({r["identity"] for r in records})!=len(records):
        raise ValueError("Nonempty unique development records required")
    output=[]
    for row in records:
        logits=row["logits"];answer=row["answer"];ce=row["ce"]
        if (not logits or type(answer) is not int or not 0<=answer<len(logits)
                or not all(math.isfinite(v) for v in [ce,*logits]) or ce<0):
            raise ValueError("Finite label logits, full-vocabulary CE and valid answer required")
        maximum=max(logits)
        log_label_sum=maximum+math.log(sum(math.exp(v-maximum) for v in logits))
        conditional_ce=log_label_sum-logits[answer]
        mass_ce=ce-conditional_ce
        if mass_ce < -1e-5:
            raise ValueError("Full-vocabulary CE cannot be lower than conditional label CE")
        prediction=max(range(len(logits)),key=logits.__getitem__)
        if prediction!=row["prediction"]:raise ValueError("Recorded prediction disagrees with label logits")
        others=[v for i,v in enumerate(logits) if i!=answer]
        output.append(dict(identity=row["identity"],family=row["family"],answer=answer,prediction=prediction,
            correct=prediction==answer,full_vocabulary_ce=ce,conditional_label_ce=conditional_ce,
            valid_label_mass_ce=mass_ce,valid_label_probability=min(1.,math.exp(-mass_ce)),
            conditional_correct_probability=math.exp(-conditional_ce),
            correct_label_margin=logits[answer]-max(others) if others else None,
            top_label_tied=sum(v==maximum for v in logits)>1))
    return output


def summarize(records):
    def summary(rows):
        n=len(rows);correct=sum(r["correct"] for r in rows)
        letter=lambda index:chr(65+index) if index<26 else str(index)
        truth=Counter(letter(r["answer"]) for r in rows)
        predictions=Counter(letter(r["prediction"]) for r in rows)
        means={"mean_"+name:sum(r[name] for r in rows)/n for name in
            ("full_vocabulary_ce","conditional_label_ce","valid_label_mass_ce", "valid_label_probability",
             "conditional_correct_probability")}
        return dict(count=n,correct=correct,accuracy=correct/n,**means,
            geometric_mean_valid_label_probability=math.exp(-means["mean_valid_label_mass_ce"]),
            top_label_ties=sum(r["top_label_tied"] for r in rows),
            true_answer_counts=dict(sorted(truth.items())),prediction_counts=dict(sorted(predictions.items())),
            best_constant_label_correct=max(truth.values()),best_constant_label_accuracy=max(truth.values())/n,
            gained_correct_over_best_constant_label=correct-max(truth.values()))
    return dict(overall=summary(records),by_family={f:summary([r for r in records if r["family"]==f])
        for f in sorted({r["family"] for r in records})})


def compare_records(baseline,current):
    before={r["identity"]:r for r in baseline};after={r["identity"]:r for r in current}
    if set(before)!=set(after):raise ValueError("Paired comparisons require identical development identities")
    changes=[]
    for identity,a in before.items():
        b=after[identity]
        if any(a[k]!=b[k] for k in ("family","answer")):raise ValueError("Paired truth/family differs")
        changes.append(dict(identity=identity,family=a["family"],baseline_correct=a["correct"],current_correct=b["correct"],
            baseline_prediction=a["prediction"],current_prediction=b["prediction"],
            full_ce_reduction=a["full_vocabulary_ce"]-b["full_vocabulary_ce"],
            conditional_ce_reduction=a["conditional_label_ce"]-b["conditional_label_ce"],
            label_mass_ce_reduction=a["valid_label_mass_ce"]-b["valid_label_mass_ce"]))
    def summary(rows):
        n=len(rows)
        means={"mean_"+k:sum(r[k] for r in rows)/n for k in
            ("full_ce_reduction","conditional_ce_reduction","label_mass_ce_reduction")}
        wins=sum(not r["baseline_correct"] and r["current_correct"] for r in rows)
        losses=sum(r["baseline_correct"] and not r["current_correct"] for r in rows)
        letter=lambda index:chr(65+index) if index<26 else str(index)
        transitions=Counter(letter(r["baseline_prediction"])+"->"+letter(r["current_prediction"]) for r in rows)
        total=means["mean_full_ce_reduction"]
        return dict(count=n,wins=wins,losses=losses,net_gained_correct=wins-losses,**means,
            same_prediction=sum(r["baseline_prediction"]==r["current_prediction"] for r in rows),
            both_correct=sum(r["baseline_correct"] and r["current_correct"] for r in rows),
            label_transitions=dict(sorted(transitions.items())),
            fraction_full_ce_reduction_from_label_mass=means["mean_label_mass_ce_reduction"]/total if total>0 else None,
            decomposition_residual=total-means["mean_conditional_ce_reduction"]-means["mean_label_mass_ce_reduction"])
    return dict(overall=summary(changes),by_family={f:summary([r for r in changes if r["family"]==f])
        for f in sorted({r["family"] for r in changes})},records=changes)


def report(current_path,*,baseline_path=None):
    current_path=Path(current_path)
    if not current_path.name.endswith("_development.json"):
        raise ValueError("Only saved epoch development reports are accepted")
    value=json.loads(current_path.read_text());rows=decompose_records(value["records"])
    result=dict(schema="jspace.answer-mass.v1",scope="Development-only diagnostic; full CE decomposed exactly into conditional-choice and label-mass losses",
        current=dict(path=str(current_path.resolve()),sha256=_sha(current_path),reported_metrics=value["metrics"]),
        summary=summarize(rows),records=rows,producer_sha256=_sha(__file__))
    if baseline_path is not None:
        baseline_path=Path(baseline_path)
        if not baseline_path.name.endswith("_development.json"):raise ValueError("Development baseline required")
        before=json.loads(baseline_path.read_text())
        result["baseline"]=dict(path=str(baseline_path.resolve()),sha256=_sha(baseline_path),reported_metrics=before["metrics"])
        result["comparison"]=compare_records(decompose_records(before["records"]),rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();value=report(args.current,baseline_path=args.baseline)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in value.items() if k not in ("records","comparison")},allow_nan=False))
    if "comparison" in value:
        print(json.dumps({k:v for k,v in value["comparison"].items() if k!="records"},allow_nan=False))


if __name__=="__main__":main()
