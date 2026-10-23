from experiments.neural_architecture_lab.finalize_ncm_native_components import capacity_gate


def test_native_capacity_gate_needs_free_exact_and_source_dependence():
    free = {'cases':32,'token_accuracy':.98,'exact':31}
    assert capacity_gate(free,{'token_accuracy':.92})
    assert not capacity_gate({**free,'exact':30},{'token_accuracy':.1})
    assert not capacity_gate({**free,'token_accuracy':.9799},{'token_accuracy':.1})
    assert not capacity_gate(free,{'token_accuracy':.98})
    assert not capacity_gate({**free,'cases':4},{'token_accuracy':.1})
