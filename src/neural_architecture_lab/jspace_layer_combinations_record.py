"""Typed exploratory compatibility evidence; no learned-repair conclusion."""
from dataclasses import asdict, dataclass
import math

from .jspace_single_layer_record import require, scores
from experiments.structure_net import jspace_layer_combination_policy as policy

STUDY_ID = "JSPACE-QWEN-LAYER-COMBINATIONS-01"
PRIMARY_METRIC_NAME = "most_compressed_tested_gate_passing_layer_count"
SCOPE = ("Conditional search among tested native/compact Qwen layer masks on the same64 development cases. "
         "No training, untouched confirmation, global-optimum, learned-repair, or inference-speed claim.")
CONTROLS = {"native":(policy.compressed(()),50), "accepted_two":(policy.compressed((0,1)),49),
            "all_compact":(policy.restored(()),27), "hybrid":(policy.restored((12,15)),24)}


def mask_layers(key):
    require(isinstance(key,str) and len(key) == 37 and key[0] == "m" and not set(key[1:])-{"0","1"}, "Malformed layer-mask ID")
    return [i for i,x in enumerate(key[1:]) if x == "1"]


def classify(compressed_count):
    require(type(compressed_count) is int and 2 <= compressed_count <= 36,"At least the accepted two-block control must pass")
    return ("additional_tested_compression_passes_development_gate" if compressed_count > 2
            else "no_tested_extension_beyond_accepted_two_blocks")


@dataclass(frozen=True)
class JSpaceLayerCombinationsRecord:
    study_id: str
    classification: str
    controls: dict
    unique_masks: int
    candidates: dict
    most_compressed: dict
    same_count: list
    selected_repair_target: dict
    pair_interactions: list
    selected_restoration_paired: dict
    model_parameters: int
    confirmed: bool = False
    independent_confirmations: int = 0
    learned_repair_evaluated: bool = False
    speed_measured: bool = False
    global_search_complete: bool = False
    scope: str = SCOPE

    @classmethod
    def from_evidence(cls,value):
        plan,result,pilot = (value[k] for k in ("plan","search","pilot"))
        require(value["status"] == result["status"] == "completed" and value["confirmed"] is False
            and plan["confirmed"] is False and result["confirmed"] is False,"Completed unconfirmed search evidence required")
        require(plan["schema"] == "jspace.layer-combinations.v1" and plan["study_id"] == STUDY_ID
            and plan["microbatch"] == 4 and plan["candidate_count"] == 6 and plan["repair_group_size"] == 3
            and plan["minimum_correct"] == 47 and plan["native_correct"] == 50 and plan["original_all_compact_correct"] == 27,
            "Frozen search population, policy and cumulative47/64 gate required")
        require(result["fingerprint"] == plan["fingerprint"] and result["passed"] is True and result["systems_only"] is False
            and result["fixed_state_unchanged"] is True and pilot["passed"] is True and pilot["systems_only"] is True
            and pilot["fixed_state_unchanged"] is True,"Successful unchanged-state pilot and search required")
        results = result["results"]; native = results[policy.mask_id(CONTROLS["native"][0])]["records"]
        verified = {}
        for key,row in results.items():
            layers = mask_layers(key); actual = scores(row["records"],native); metrics = row["metrics"]
            require(metrics["correct"] == actual["correct"] and metrics["count"] == 64 and metrics["accuracy"] == actual["accuracy"],
                "Mask accuracy differs from actual paired records")
            ce = margin = agreement = 0
            for record,teacher in zip(row["records"],native):
                logits = record["logits"]; peak = max(logits)
                expected_ce = math.log(sum(math.exp(x-peak) for x in logits))+peak-logits[record["answer"]]
                expected_margin = logits[record["answer"]]-max(x for i,x in enumerate(logits) if i != record["answer"])
                require(len(logits) == len(teacher["logits"]) and record["teacher_prediction"] == teacher["prediction"]
                    and math.isclose(record["choice_ce"],expected_ce,abs_tol=1e-4,rel_tol=1e-5)
                    and math.isclose(record["choice_margin"],expected_margin,abs_tol=1e-5,rel_tol=1e-5),
                    "Conditional metrics or original teacher decisions differ from saved logits")
                ce += record["choice_ce"]; margin += record["choice_margin"]
                agreement += record["prediction"] == teacher["prediction"]
            require(metrics["teacher_agreement_count"] == agreement and metrics["teacher_agreement"] == agreement/64
                and math.isclose(metrics["choice_ce"],ce/64,abs_tol=1e-10,rel_tol=1e-10)
                and math.isclose(metrics["choice_margin"],margin/64,abs_tol=1e-10,rel_tol=1e-10)
                and all(metrics["by_family"][f]["correct"] == v["correct"] and metrics["by_family"][f]["count"] == 16
                    for f,v in actual["by_family"].items()),"Aggregate selection metrics differ from actual rows")
            verified[key] = dict(mask_id=key,compressed_layers=layers,compressed_count=len(layers),**actual,
                choice_ce=ce/64,choice_margin=margin/64,teacher_agreement=agreement/64)
        candidates = policy.select_candidates(results)
        require(set(results) == set(policy.screen_masks()) | set(policy.refinement_masks(candidates["layers"])),
            "Complete681-mask screen plus both128-request refinements required, with exact deduplication")
        summary = policy.summarize(results,candidates)
        require(summary == result["summary"] and summary["confirmed"] is False and not summary["complete_global_search"],
            "Stored candidate selection, interactions or same-count summary changed")
        controls = {}
        for arm,(mask,count) in CONTROLS.items():
            key = policy.mask_id(mask); reference = value["references"][arm]["records"]
            require(verified[key]["correct"] == count,"Required control count changed")
            scores(reference,native)
            require(all(a["prediction"] == b["prediction"] and len(a["logits"]) == len(b["logits"])
                    and all(abs(x-y) <= 5e-4+1e-5*abs(y) for x,y in zip(a["logits"],b["logits"]))
                for a,b in zip(results[key]["records"],reference)),"Control predictions or tolerant logits failed")
            require(all(result["controls"][arm][k] is True for k in ("passed","predictions_exact","logits_pass")),
                "All four authenticated controls must pass")
            controls[arm] = verified[key]
        qualified = summary["most_compressed_qualified_mask"]
        require(qualified is not None and verified[qualified]["correct"] >= 47,"Most-compressed qualifying tested mask required")
        most = verified[qualified]
        rows = [dict(row,best=verified[row["best_mask"]],worst=verified[row["worst_mask"]]) for row in summary["same_count_rankings"]]
        selected = dict(summary["selected"],score=verified[summary["selected"]["restored_mask_id"]])
        require(selected["group_size"] == 3 and selected["score"]["compressed_count"] == 33
            and selected["rescue_correct"] == selected["score"]["correct"]-27,"Selected target must remain a conditional native-restoration triple")
        proof = value["selection_proof"]
        require(proof["selected_layers"] == selected["layers"] and proof["search_fingerprint"] == plan["fingerprint"],
            "Authenticated selection proof differs")
        inventory = value["parameter_inventory"]
        require(len(inventory["native_by_layer"]) == len(inventory["compact_by_layer"]) == 36
            and all(type(n) is int and n > 0 for n in [inventory["endpoints"],*inventory["native_by_layer"],*inventory["compact_by_layer"]]),
            "Actual original/compact parameter inventories required")
        parameters = inventory["endpoints"]+sum(inventory["compact_by_layer"][i] if i in most["compressed_layers"]
            else inventory["native_by_layer"][i] for i in range(36))
        return cls(STUDY_ID,classify(most["compressed_count"]),controls,len(results),candidates,most,rows,selected,
            summary["pair_interactions"],summary["paired_selected_restoration"],parameters)

    @property
    def primary_metric(self): return self.most_compressed["compressed_count"]

    def to_dict(self): return asdict(self)
