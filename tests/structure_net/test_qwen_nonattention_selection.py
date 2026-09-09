from experiments.structure_net.run_qwen_nonattention import select


def test_selection_requires_both_sources_jointly_and_uses_fixed_order():
    def row(source, variant, speed, gate=True):
        return {"source": source, "variant": variant, "length": 16384, "gate": gate,
                "speedup_vs_moba": speed, "metrics_vs_moba": {"change_pearson": 1.0}}
    rows = [row(source, name, speed) for source in ("wikipedia", "bnc_spoken")
            for name, speed in (("fused_fp16", 1.1), ("int8_mlp6", 1.4))]
    assert select(rows)["execution"]["name"] == "int8_mlp6"
    rows[-1]["gate"] = False
    assert select(rows)["execution"]["name"] == "fused_fp16"
    assert select(rows[:1]) is None
    assert select(rows[:1] * 2) is None
    assert select([]) is None


def test_eight_k_cannot_select_a_sixteen_k_candidate():
    rows = [{"source": s, "variant": "fused_fp16", "length": 8192, "gate": True,
             "speedup_vs_moba": 10, "metrics_vs_moba": {"change_pearson": 1.0}}
            for s in ("wikipedia", "bnc_spoken")]
    assert select(rows) is None
