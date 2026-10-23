import pytest
import torch
from experiments.structure_net.ncm_native_concepts import ConceptConfig, check_repair, conditional_target, encode, expand, fit_book, pool


def test_native_concepts_pool_exact_cells_and_repair_actual_history():
    torch.manual_seed(42)
    cfg = ConceptConfig(canvas=8,levels=(2,4),entries=4,iterations=2,chunk=3)
    target = torch.randn(5,8,3)
    torch.testing.assert_close(pool(target,2),target.reshape(5,2,4,3).mean(2),rtol=0,atol=0)
    first = fit_book(pool(target,2).flatten(0,1),cfg,0,'cpu')
    ids = conditional_target(target,torch.zeros_like(target),first,2)
    residual = target-expand(first[ids],8)
    second = fit_book(pool(residual,4).flatten(0,1),cfg,1,'cpu')
    books = [first,second]
    assert all(torch.equal(book[0],torch.zeros(3)) for book in books)
    checks = check_repair(target,books,cfg)
    assert checks['conditional_expert_not_worse_at_same_history']
    assert checks['clean_history_label_parity']
    codes,reconstruction,errors = encode(target,books,cfg)
    assert reconstruction.shape == target.shape and errors.shape == (5,2)
    assert codes[0].shape == (5,2) and codes[1].shape == (5,4)
    repeated = encode(target,[book.clone() for book in books],cfg)
    assert all(torch.equal(a,b) for a,b in zip(codes,repeated[0]))


def test_native_concepts_reject_ambiguous_cell_boundaries():
    with pytest.raises(ValueError): ConceptConfig(canvas=8,levels=(3,4))
    with pytest.raises(ValueError): ConceptConfig(canvas=8,levels=(4,2))
    with pytest.raises(ValueError): pool(torch.zeros(2,8,3),3)
