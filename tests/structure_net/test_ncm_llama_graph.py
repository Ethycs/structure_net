import pytest
import ast
from pathlib import Path
import textwrap
import torch
from experiments.structure_net.ncm_llama_graph import validate_prefix,unpack_output
from experiments.structure_net.ncm_llama_graph import top1_categorical


def test_graph_input_contract_rejects_empty_outside_canvas_and_invalid_ids():
    validate_prefix([50256]+[50257]*511)
    for ids in ([50257]*512,[1]*511,[50258]+[50257]*511):
        with pytest.raises(ValueError):validate_prefix(ids)


def test_nonfinite_or_incomplete_graph_output_cannot_be_returned():
    with pytest.raises(ValueError):unpack_output([1]*48+[0],None)
    with pytest.raises(ValueError):unpack_output([1]*48,None)


def test_capture_shim_changes_only_two_scalar_allocation_expressions():
    from experiments.structure_net.ncm_llama import VENDOR
    source=(VENDOR/'models/next_concept.py').read_text();tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NextConceptTransformer')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='generate')
    expected=textwrap.dedent(ast.get_source_segment(source,method))
    for old,new in (("torch.tensor(float('-inf'), device=dev)","torch.full((), float('-inf'), device=dev)"),
                    ("torch.tensor(0.0, device=dev)","torch.full((), 0.0, device=dev)")):
        assert expected.count(old)==1;expected=expected.replace(old,new)
    shim=ast.parse(Path('experiments/structure_net/ncm_capture_generate.py').read_text())
    actual=next(x for x in shim.body if isinstance(x,ast.FunctionDef) and x.name=='generate')
    assert ast.get_source_segment(Path('experiments/structure_net/ncm_capture_generate.py').read_text(),actual)==expected


def test_categorical_keeps_unique_maxima_and_uniform_ties():
    torch.manual_seed(42)
    unique=torch.tensor([[0.,8.,2.,-1.]]).repeat(4096,1)
    assert torch.equal(top1_categorical(unique),torch.ones(4096,dtype=torch.long))
    tied=torch.tensor([[0.,8.,8.,-1.]]).repeat(10000,1)
    samples=top1_categorical(tied)
    assert ((samples==1)|(samples==2)).all()
    assert abs(float((samples==1).float().mean())-.5)<.02
