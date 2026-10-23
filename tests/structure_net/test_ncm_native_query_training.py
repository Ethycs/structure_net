from types import SimpleNamespace
import torch
from experiments.structure_net.ncm_native_query_training import actual_history,expert_gains
from experiments.structure_net.ncm_native_concepts import ConceptConfig,encode


def test_actual_history_has_no_target_and_expert_does_not_replace_its_codes():
    class Net:
        canvas = 4
        levels = (1,2)
        books = [torch.eye(4),torch.eye(4)]
        def level_logits(self,memory,prior,level):
            return torch.arange(4.).expand(len(prior),self.levels[level],4)
    net = Net(); memory = SimpleNamespace(source_mask=torch.ones(2,5))
    a,p = actual_history(net,memory,2,torch.Generator().manual_seed(42),(.5,.1))
    saved = [c.clone() for c in a]
    target = torch.randn(2,4,4)
    clean = encode(target,net.books,ConceptConfig(canvas=4,levels=(1,2),entries=4))[0]
    assert (expert_gains(target,net.books,net.levels,a,clean) >= -1e-6).all()
    b,q = actual_history(net,memory,2,torch.Generator().manual_seed(42),(.5,.1))
    assert torch.equal(p,q) and all(torch.equal(x,y) for x,y in zip(a,b))
    assert all(torch.equal(x,y) for x,y in zip(a,saved))
