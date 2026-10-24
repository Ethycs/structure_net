"""Readiness and immutable first-packet recovery after the missing dependency."""
import importlib

from experiments.structure_net import ncm_sol64_bert_roundtrip_recovery as run


def test_actual_dependency_failure_has_no_semantic_firsts_to_retry():
    run.configure()
    assert importlib.import_module('tiktoken').get_encoding('cl100k_base')
    assert callable(importlib.import_module('requests').post)
    run.require_unstarted()
    assert run.base.expected_cases() == run.base.q.read(run.OLD_PREP / 'cases.json')
    assert run.base.MODULE == run.MODULE
    assert run.base.PILOT == run.PILOT
