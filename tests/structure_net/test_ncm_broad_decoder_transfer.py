import pytest

from experiments.structure_net.ncm_broad_decoder_transfer import qualify, source_windows, split_sources


class Tokenizer:
    eos_token_id, pad_token_id = 50256, 50257
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_windows_are_exact_spans_bounded_and_deterministic():
    text = "\n  ".join("word"+str(i) for i in range(128))
    source = {"id": "a", "source": text, "source_id": "s", "group_id": "g", "source_sha256": "h"}
    a, rejected = source_windows(source, Tokenizer())
    assert a == source_windows(source, Tokenizer())[0] and rejected == 0
    assert len(a) == 8 and len({r["target_sha256"] for r in a}) == 8
    for row in a:
        assert text[row["char_start"]:row["char_end"]] == row["target"]
        assert len(row["target"].split()) in (8, 16, 24, 32)
        assert len(row["native_target_ids"]) == 48


def test_source_split_is_stable_and_rejects_repeated_groups():
    records = [{k: str(i) for k in ("id", "source_id", "group_id", "source_sha256")} for i in range(8192)]
    a, b = split_sources(records)
    assert (a, b) == split_sources(records[::-1])
    assert len(a) == 7168 and len(b) == 1024
    assert not {r["group_id"] for r in a} & {r["group_id"] for r in b}
    bad = [dict(r) for r in records]; bad[-1]["group_id"] = bad[0]["group_id"]
    with pytest.raises(ValueError, match="repeated"):
        split_sources(bad)


def test_both_novel_and_raw_codec_metrics_must_pass_joint_gate():
    metrics = {k: {"nonpad_reconstruction": .995, "exact_texts": 18, "cases": 20}
               for k in ("raw_validation", "novel")}
    assert qualify(metrics)
    metrics["novel"]["exact_texts"] = 17
    assert not qualify(metrics)
    metrics["novel"]["exact_texts"] = 18
    metrics["raw_validation"]["nonpad_reconstruction"] = .99499
    assert not qualify(metrics)
