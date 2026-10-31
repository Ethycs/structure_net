from experiments.word_games.speculative_draft import convert_candidate
from pathlib import Path
import runpy
import pytest


def test_draft_ids_are_converted_through_bytes_and_cannot_rewrite_prefix():
    class Draft:
        def detokenize(self, ids, prev_tokens):
            assert ids == [200001, 200002] and prev_tokens == [18, 19]
            return b'"Alpha"'
    class Target:
        def tokenize(self, data, *, add_bos, special):
            assert data == b'"Alpha"' and not add_bos and not special
            return [1, 2, 3]
    assert convert_candidate(Target(), Draft(), [200001, 200002], [18, 19], 2) == [1, 2]


def test_byte_candidates_preserve_partial_unicode_without_decoding_loss():
    class Draft:
        def detokenize(self, ids, prev_tokens): return b'\x82\xac'
    class Target:
        def tokenize(self, data, **kwargs):
            assert data == b'\x82\xac'
            return [7]
    assert convert_candidate(Target(), Draft(), [3], [4], 8) == [7]


def test_faster_full_run_cannot_ignore_mismatched_output():
    evidence = runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/neural_architecture_lab/word_games_speculation_hypothesis.py"))["WordGamesSpeculationEvidence"]
    args = dict(stage="full", classification="observed_faster", requested=72, completed=72,
                pairs=36, exact_pairs=35, campaign="campaign.json", campaign_sha256="a" * 64,
                report="report.md", report_sha256="b" * 64)
    with pytest.raises(ValueError):
        evidence(**args)
    result = evidence(**{**args, "exact_pairs": 36}).record()
    assert result["tested"] and not result["confirmed"]
