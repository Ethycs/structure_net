"""Preserve the failed dictionary inverse without claiming irreversible code loss."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_codec_inverse_control import MEASUREMENT, MODES, PILOT
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261022_codec_inverse_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-codec-dictionary-inverse.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh completed evidence required")
    result = q.read(MEASUREMENT / "result.json")
    pilot = q.read(PILOT / "result.json")
    if result["status"] != "completed" or pilot["status"] != "passed" or not pilot["readout_order_exact"]:
        raise ValueError("completed controls and exact-order pilot required")
    for root in (PILOT, MEASUREMENT):
        verify(q.read(root / "producer.json"))
        receipt = q.read(root / "result.json")
        for path, expected in receipt["output_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("completed output changed")
    for model, cohorts in result["metrics"].items():
        for cohort, modes in cohorts.items():
            expected = 256 if cohort == "raw_validation" else 20
            if any(row["cases"] != expected for row in modes.values()):
                raise ValueError("incomplete cohort")
            oracle = modes["unquantized_oracle"]
            if oracle["all_correct"] != oracle["all_total"]:
                raise ValueError("inverse implementation control failed")
    qualified = any(result["codec_qualified"].values())
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (PILOT, MEASUREMENT):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-codec-inverse.v1", "classification": "dictionary_inverse_qualified" if qualified else "dictionary_inverse_failed",
        "measurement": result, "irreversible_information_loss_established": False,
        "task_accuracy": None, "trained": False, "confirmed": False, "underpowered": True,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = "\n".join("|"+model+"|"+cohort+"|"+"|".join(
        f"{100*modes[mode]['nonpad_reconstruction']:.4f}% ({modes[mode]['exact_texts']}/{modes[mode]['cases']})"
        for mode in MODES)+"|" for model,cohorts in result["metrics"].items() for cohort,modes in cohorts.items())
    REPORT.write_text(f'''# Token-dictionary inversion does not repair the NCM text codec

Date: 2026-10-22. `ncm-codec-inverse-control-v1` is complete, classified
**{aggregate['classification']}**. Neither released nor adapted codec meets
the target-blind replacement gate. No new summary accuracy or speed is measured;
the current attachment remains at its earlier0/64 Terra development result.

Entries are correct non-PAD target tokens, with exact target texts in parentheses.

| Codec | Cohort | Neural raw | Blind inverse,4 steps | Quantized, oracle statistics | Unquantized, oracle statistics |
| --- | --- | --- | --- | --- | --- |
{table}

The encoder is a lookup of token plus positional embeddings, followed by
8-group normalization. The candidate estimates normalization from neural token
predictions, reverses it, subtracts positional embeddings, then uses nearest
token-embedding distance. Four fixed alternating steps use no target IDs or
target statistics. They perform substantially worse than the original neural
decoder. No iteration/metric/decoder search or outcome-based rerun occurred.

Using true normalization statistics with the quantized latent also performs
worse than the neural decoder. That observation does **not** prove irreversible
information loss: the neural decoder demonstrably recovers more tokens than
this distance rule, and information can be distributed across positions. This
rejects the proposed geometric inverse as a decoder remedy. It cannot establish
an optimal decoding bound or excuse source-to-code prediction failure.

With the original unquantized normalized latent and true statistics, the same
inverse recovers every token ID exactly across all592 codec/case combinations,
including EOS and PAD. This supports implementation correctness for the stated
normalization inversion, not applicability of nearest-neighbor geometry after
learned residual quantization. Oracle variants know reference-derived
statistics and are not inference candidates. Actual generated-summary codes
were not tested here.

The20 seen/20 novel labels and raw256 source-disjoint validation windows are
unchanged. Every readout shares the same target codes. FP32, batch4, GPU1;
frozen states, finite values, code/prediction reload and the lifecycle pilot's
readout-order parity passed. Own peak allocation was
{result['peak_allocated_bytes']:,}bytes. The novel targets and raw validation
are development diagnostics; final task-test data remain sealed.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. Full readout IDs,
intermediate inverse iterations, target codes, original source snapshots and
input seals remain in immutable roots. Registry confirmed=false,
task_accuracy_measured=false, irreversible_information_loss_established=false.
''')
    print(json.dumps({"status": aggregate["classification"], "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-codec-inverse-control-v1": {"name": "NCM token-dictionary inverse information control",
            "description": "Four-step target-blind normalization and nearest-token inversion fails for released and adapted codecs on20 novel and256 raw texts. True-statistics quantized inverse is worse than the neural decoder; unquantized oracle inverse is exact. Irreversible information loss is not established; no summary quality or speed claim.",
            "confirmation_status": "dictionary_inverse_failed", "task_accuracy_measured": False,
            "irreversible_information_loss_established": False},
    })
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
