"""Descriptive figures and coefficient counts for the frozen country stack."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .jspace_country_stack import DEFAULT_ROOT, CONFIG, verify, checked
from .jspace_causal_components import cpu_only
from .jspace_relation_study import file_hash, write_json


def main(root=DEFAULT_ROOT):
    cpu_only();verify(root);checked(root,"fit");checked(root,"capture")
    result=json.loads((root/"result.json").read_text())
    fitted=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    predictions=torch.load(root/"predictions.pt",map_location="cpu",weights_only=True)
    selected=str(result["selected_degree"]);model=fitted["models"][selected]
    details={"selected_degree":int(selected),"unreduced_library_terms":len(model["unreduced_powers"]),
             "reduced_library_terms":len(model["powers"]),
             "unreduced_nonzero_coefficients":int(torch.count_nonzero(model["unreduced_numerators"])),
             "compressed_stored_nonzero_coefficients":int(torch.count_nonzero(model["numerators"][1:,:3])),
             "raw_score_scalars_per_world":36,"retained_score_scalars":3,"retained_attribute_bits":1,
             "continuous_tensor_payload_ratio":result["fit"]["uncompressed_numerical_bytes"]/result["fit"]["compressed_numerical_bytes"],
             "continuous_file_ratio":result["fit"]["uncompressed_file_bytes"]/result["fit"]["compressed_file_bytes"]}
    write_json(root/"presentation_metrics.json",details)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    scale=fitted["scale"].numpy();mean=fitted["mean"].numpy()
    for ax,cohort in zip(axes[:2],("source","dense")):
        actual=predictions[cohort]["actual"].numpy()
        denominator=np.sum(((actual-mean)/scale)**2,axis=(0,2))
        for name,color in (("carry_anchor","#999999"),("1","#007f8b"),("2","#c45a28")):
            value=predictions[cohort][name].numpy()
            error=np.sqrt(np.sum(((value-actual)/scale)**2,axis=(0,2))/denominator)
            ax.plot(CONFIG["layers"],error,marker=".",label={"1":"Linear","2":"Quadratic","carry_anchor":"Carry anchor"}[name],color=color)
        ax.set_title("Natural source" if cohort=="source" else "Full attention swap")
        ax.set_xlabel("Zero-based block output");ax.set_ylabel("Continuous country-score NRMSE")
        ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fit=result["fit"]
    values=[fit["uncompressed_file_bytes"]/1024,fit["compressed_file_bytes"]/1024]
    axes[2].bar([0,1],values,color=["#999999","#007f8b"])
    axes[2].set_xticks([0,1],["Nine separate\nreadouts","Compressed\nstack readout"])
    axes[2].set_ylabel("Serialized readout KiB");axes[2].set_title("Qwen backbone excluded from both")
    for i,value in enumerate(values):axes[2].text(i,value+5,f"{value:.1f}",ha="center")
    fig.suptitle("Country stack: exact surrogate reduction, measured neural reconstruction")
    for extension in ("png","svg"):fig.savefig(root/f"country_stack.{extension}",dpi=160)
    plt.close(fig)
    write_json(root/"presentation_manifest.json",{"source_sha256":file_hash(__file__),"scope":"post-evaluation descriptive presentation",
        "artifacts":{name:file_hash(root/name) for name in ("result.json","presentation_metrics.json","country_stack.png","country_stack.svg")}})
    return details


if __name__=="__main__":print(json.dumps(main(),allow_nan=False))
