"""New decoder must qualify; pending parent lineage cannot be silently forked."""
import pytest
from experiments.structure_net import ncm_expanded_codec_semantics as run
from experiments.structure_net import ncm_sol64_grounded_oracle_semantics as parent


def test_only_expanded_arm_can_qualify_direction():
    assert run.classification([])=='fixed64_expanded_codec_quality_failed'
    assert run.classification(['domain'])=='fixed64_expanded_codec_quality_failed'
    assert run.classification(['expanded'])=='fixed64_expanded_codec_semantically_qualified'
    assert run.ARMS==('domain','expanded')


def test_pending_lineage_is_held_and_parent_globals_remain_original():
    assert run.SEMANTIC_LINEAGE_READY is False
    with pytest.raises(ValueError,match='matched-control descendant lineage'):
        run.workflow()
    assert parent.WORKFLOW==run.ROOT/'20261024_sol64_grounded_oracle_semantics_workflow_v1'
    assert parent.WORKFLOW!=run.WORKFLOW
