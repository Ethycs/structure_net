from pathlib import Path
import pytest

from experiments.structure_net import ncm_reserved_code_probe_v2 as p


def test_only_exact_missing_schema_input_is_removed(tmp_path):
    result=tmp_path/'result.json';receipt=tmp_path/'actual_tool_receipt.json'
    for path in (result,receipt):path.write_text('{}')
    missing=tmp_path/'campaign_results.json'
    assert p.corrected_inputs([result,missing,receipt],missing)==[result,receipt]
    with pytest.raises(ValueError):p.corrected_inputs([result,receipt],missing)
    with pytest.raises(ValueError):p.corrected_inputs([result,missing,missing,receipt],missing)
    with pytest.raises(ValueError):p.corrected_inputs([result,missing,tmp_path/'unknown'],missing)
    missing.write_text('{}')
    with pytest.raises(ValueError):p.corrected_inputs([result,missing,receipt],missing)


def test_actual_recovery_manifest_has_complete_immutable_ancestry():
    paths=p.inputs()
    assert all(path.is_file() for path in paths)
    assert p.MISSING not in paths
    assert p.MISSING.parent/'result.json' in paths
    assert p.MISSING.parent/'actual_tool_receipt.json' in paths
    assert p.FAILED/'failure.json' in paths and p.FAILED/'actual_tool_receipt.json' in paths
    assert Path(p.__file__) in paths
