from experiments.structure_net.paws_frozen_campaign import completion_gates


def test_test_reveal_gate_reports_every_prior_experiment():
    gates=completion_gates()
    assert set(gates)=={f"{index:02d}" for index in range(1,10)}
    assert gates["01"] and gates["02"]
    assert all(isinstance(value,bool) for value in gates.values())
