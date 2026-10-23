import pytest
import torch

from experiments.structure_net.ncm_native_parent_control import native_inputs


def test_native_prompt_padding_preserves_all_original_ids_and_rejects_overflow():
    class Tokenizer:
        eos_token_id = 9
        def apply_chat_template(self,messages,add_generation_prompt=True):
            n = int(messages[1]["content"].splitlines()[1])
            return list(range(n))
    inputs, ids = native_inputs([{"source":"3"},{"source":"7"}],Tokenizer(),"cpu")
    assert inputs["input_ids"].shape == (2,768)
    assert torch.equal(inputs["attention_mask"].sum(1),torch.tensor([3,7]))
    assert inputs["input_ids"][0,-3:].tolist() == ids[0] == [0,1,2]
    assert inputs["input_ids"][1,-7:].tolist() == ids[1] == list(range(7))
    with pytest.raises(ValueError,match="truncation prohibited"):
        native_inputs([{"source":"769"}],Tokenizer(),"cpu")
