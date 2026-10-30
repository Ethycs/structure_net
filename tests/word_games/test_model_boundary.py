from dataclasses import asdict
import json

import pytest

from experiments.word_games.agents import receiver_prompt
from experiments.word_games.backend import vendor
from experiments.word_games.campaign import evaluate_case, paired_checks
from experiments.word_games.common import canonical
from experiments.word_games.firewall import occupied
from experiments.word_games.metaphor import Metaphor, PUBLIC_TASK, select
from experiments.word_games.scenarios import cases, setup


M = Metaphor("spaceport", "ship", "station", "dock", "dock", "recall", "clearance", "departure")


def test_model_selects_the_actual_role_mapping(tmp_path):
    class Choice:
        def complete(self, prompt, path, *, max_tokens):
            assert prompt == PUBLIC_TASK and max_tokens == 256
            assert all(case["canary"] not in prompt for case in cases())
            assert "spaceport" not in prompt
            return canonical(asdict(M))
    assert select(Choice(), tmp_path) == M
    assert all(r["identical"] for r in paired_checks(M))


@pytest.mark.parametrize("change", [{"entity": "ship\nprivate"}, {"slot": "ship"}, {"deadline": "x" * 41}])
def test_invalid_metaphor_is_not_silently_repaired(change):
    with pytest.raises(ValueError):
        Metaphor(**{**asdict(M), **change})


def test_two_agent_path_discards_sender_prose_and_private_input(tmp_path):
    case = cases()[0]

    class TwoAgents:
        prompts = []

        def complete(self, prompt, path):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                assert case["canary"] in prompt
                context = json.loads(prompt.split("\nCURRENT INPUT\n", 1)[1].split("\nTHIS EVENT'S RETRIEVAL DIALOGUE\n", 1)[0])["context"]
                return canonical({"kind": "final", "target_text": case["canary"],
                    "proposal": vendor().proposal(context, case["expected_operations"], case["id"])})
            packet = json.loads(prompt.split("\nAUTHORIZED PACKET\n", 1)[1])
            assert prompt == receiver_prompt(packet)
            assert case["canary"] not in prompt
            assert "CURRENT INPUT" not in prompt and "private_workspace" not in prompt
            assert packet["metaphor"] == asdict(M)
            assert "ship" in packet["translation"]
            return canonical({"decision": packet["decision"], "occupied_slots": occupied(packet),
                              "summary": "The ship occupies its authorized dock."})

    client = TwoAgents()
    result = evaluate_case(client, case, tmp_path / "case", M)
    assert len(client.prompts) == 2
    assert result["sender_operation_match"] and result["receiver_state_correct"]
    assert not result["canary_in_receiver_input"]
