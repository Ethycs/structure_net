"""Fail-closed phase reuse and scientific admission boundaries."""
import json
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_recovery_workflow as workflow


def put(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))
    return str(path)


def test_completed_receipt_rejects_changed_payload_and_lineage(tmp_path):
    payload = tmp_path/'result.json'
    put(payload, {'tokens':123})
    receipt = tmp_path/'receipt.json'
    put(receipt, {'schema':workflow.SCHEMA+'.phase', 'status':'completed',
                  'experiment_fingerprint':'original', 'output_sha256':{str(payload):workflow.sha(payload)}})
    assert workflow.authenticate_receipt(receipt, expected_fingerprint='original')['status'] == 'completed'
    with pytest.raises(ValueError, match='lineage'):
        workflow.authenticate_receipt(receipt, expected_fingerprint='different')
    put(payload, {'tokens':124})
    with pytest.raises(ValueError, match='output changed'):
        workflow.authenticate_receipt(receipt)


def test_failed_receipt_cannot_be_reused(tmp_path):
    path = tmp_path/'receipt.json'
    put(path, {'schema':workflow.SCHEMA+'.phase', 'status':'failed', 'output_sha256':{}})
    with pytest.raises(ValueError, match='no authenticated completion'):
        workflow.authenticate_receipt(path)


def admission(tmp_path):
    geometry = put(tmp_path/'context/manifest.json', {'qualified':True, 'max_length':2048})
    corpus = put(tmp_path/'data/manifest.json', {'spec_sha256':'spec'})
    spec = put(tmp_path/'spec.json', {'target_context':2048, 'geometry_manifest':geometry,
                                     'corpus_manifest':corpus, 'milestones':[100000]})
    pilot = tmp_path/'runtime/pilot/result.json'
    put(pilot, {'passed':True, 'discarded':True})
    return {'artifact_root':str(tmp_path), 'spec_path':spec, 'spec_sha256':'spec'}, Path(geometry), pilot


def test_training_rejects_unqualified_or_short_geometry(tmp_path):
    plan, geometry, _ = admission(tmp_path)
    phase = {'name':'train:reference:100000'}
    workflow.check_preconditions(plan, phase)
    for value in ({'qualified':False, 'max_length':2048}, {'qualified':True, 'max_length':416}):
        put(geometry, value)
        with pytest.raises(ValueError, match='not qualified'):
            workflow.check_preconditions(plan, phase)


def test_scientific_arm_rejects_failed_or_nondisposable_pilot(tmp_path):
    plan, _, pilot = admission(tmp_path)
    for value in ({'passed':False, 'discarded':True}, {'passed':True, 'discarded':False}):
        put(pilot, value)
        with pytest.raises(ValueError, match='pilot did not qualify'):
            workflow.check_preconditions(plan, {'name':'train:student_history:100000'})


def test_undeclared_arm_or_token_target_is_rejected(tmp_path):
    plan, _, _ = admission(tmp_path)
    for name in ('train:reference:999', 'train:unknown:100000', 'train:reference:0'):
        with pytest.raises(ValueError):
            workflow.phase_definition(plan, name)
