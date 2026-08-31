import numpy as np
from experiments.structure_net.paws_embedding_router import lower_bound, route


def test_lower_bound_and_cheapest_route():
    success=np.ones((2,5,3)); weights=np.ones((2,5)); lcb=lower_bound(success,weights)
    assert np.allclose(lcb,1)
    assert route(lcb,.95).tolist()==[0,0]


def test_fallback_is_c():
    assert route(np.zeros((3,3)),.8).tolist()==[2,2,2]
