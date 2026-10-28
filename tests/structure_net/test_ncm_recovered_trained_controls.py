from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_recovered_trained_controls as run


def test_code_diagnostic_accounts_for_all_levels_without_accuracy_claim():
    old=[{'id':'one','codes':[0]*641}];new=[{'id':'one','codes':[1]*641}]
    gold={'one':[1]*641};value=run.compare_codes(new,old,gold)
    assert value['TRAIN_only'] and value['summary_semantic_accuracy'] is None
    assert value['target_codes_used_only_after_prediction'] and value['rows'][0]['changed_predicted_codes']==641
    for width,row in zip(run.LEVELS,value['rows'][0]['levels'],strict=True):
        assert row=={'width':width,'initializer_exact_codes':0,'trained_exact_codes':width}


def test_case_reordering_missing_codes_and_duplicates_are_rejected():
    old=[{'id':'one','codes':[0]*641},{'id':'two','codes':[1]*641}]
    targets={'one':[0]*641,'two':[1]*641}
    for new in (old[::-1],old[:1],[old[0],old[0]],
                [{'id':'one','codes':[0]*640},old[1]],
                [{'id':'one','codes':[8192]*641},old[1]]):
        with pytest.raises(ValueError):run.compare_codes(new,old,targets)


def test_loaded_model_must_match_exact_final_and_protected_hashes():
    actual={'model':'final','protected':{'codec':'frozen'}}
    training=SimpleNamespace(native=SimpleNamespace(vendor=SimpleNamespace(module_hash=lambda m:actual['model'])),
                             prior=SimpleNamespace(protected=lambda m:actual['protected']))
    trained={'final_model_sha256':'final','frozen_hashes':{'codec':'frozen'}}
    run.check_loaded(training,object(),trained)
    actual['model']='initializer'
    with pytest.raises(ValueError):run.check_loaded(training,object(),trained)
    actual['model']='final';actual['protected']={'codec':'changed'}
    with pytest.raises(ValueError):run.check_loaded(training,object(),trained)
