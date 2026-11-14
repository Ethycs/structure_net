import torch
from transformers import RepetitionPenaltyLogitsProcessor
from experiments.structure_net.jspace_stacked_graph_decode_v2 import penalize


def test_gpu_membership_form_matches_hf_repetition_processor_with_duplicates_and_signs():
    ids=torch.tensor([[3,1,3,0,1]])
    scores=torch.tensor([[2.,-2.,4.,-5.,.0,1.]])
    seen=torch.zeros(6,dtype=torch.bool).index_fill_(0,ids.flatten(),True)
    reference=RepetitionPenaltyLogitsProcessor(1.05)(ids,scores.clone())
    assert torch.equal(reference,penalize(scores,seen,1.05))
