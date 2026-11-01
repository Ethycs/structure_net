"""The live workflow keeps actual state and never exposes its oracle."""
import json
from experiments.word_games.backend import vendor
from experiments.word_games.common import canonical
from experiments.word_games.firewall import occupied
from experiments.word_games.metaphor import Metaphor
from experiments.word_games_triton.normal_operation import (
    check_expected_workflow, workflow, context_for, evaluate)


def test_expected_workflow_is_permitted_and_returns_all_objects():
    assert check_expected_workflow() == {'steps': 7, 'final_revision': 7, 'all_expected_operations_valid': True}


def test_two_live_events_preserve_state_and_hide_oracle(tmp_path):
    engine = vendor().Engine.fixture()
    metaphor = Metaphor('library', 'book', 'branch', 'shelf', 'shelve', 'return', 'permit', 'due date')
    for case in workflow()[:2]:
        class Client:
            def complete(self, prompt, path):
                assert 'expected_operations' not in prompt and 'expected_state' not in prompt
                if '\nCURRENT INPUT\n' in prompt:
                    data = json.loads(prompt.split('\nCURRENT INPUT\n')[1].split("\nTHIS EVENT'S RETRIEVAL DIALOGUE\n")[0])
                    return canonical({'kind': 'final', 'target_text': case['canary'],
                        'proposal': vendor().proposal(data['context'], case['expected_operations'], case['id'])})
                packet = json.loads(prompt.split('\nAUTHORIZED PACKET\n')[1])
                assert case['canary'] not in prompt
                return canonical({'decision': packet['decision'], 'occupied_slots': occupied(packet),
                                  'summary': 'The books occupy their assigned shelves.'})
        context = context_for(engine, case)
        result = evaluate(Client(), engine, context, case, tmp_path / case['id'], metaphor)
        assert result['joint_structured_success']
    assert engine.revision == 2
    assert engine.loans['loan:bronze']['reservation']['slot'] == 'N1'
    assert engine.loans['loan:clock']['reservation']['slot'] == 'N2'


def test_correct_cancellation_does_not_mask_receiver_omission(tmp_path):
    engine = vendor().Engine.fixture()
    for case in workflow()[:2]:
        context = context_for(engine, case)
        engine.apply(vendor().proposal(context, case['expected_operations'], case['id']),
                     expected_context_id=context['context_id'])
    case = workflow()[4]  # Return Clock, while Heron remains in N1.
    context = context_for(engine, case)
    metaphor = Metaphor('library', 'book', 'branch', 'shelf', 'shelve', 'return', 'permit', 'due date')

    class Client:
        def complete(self, prompt, path):
            if '\nCURRENT INPUT\n' in prompt:
                return canonical({'kind': 'final', 'target_text': 'Return Clock.',
                    'proposal': vendor().proposal(context, case['expected_operations'], case['id'])})
            return canonical({'decision': 'applied', 'occupied_slots': [], 'summary': 'Clock returned.'})

    result = evaluate(Client(), engine, context, case, tmp_path / 'case', metaphor)
    assert result['sender_operation_match'] and result['backend_state_correct']
    assert not result['receiver_state_correct'] and not result['joint_structured_success']
    assert engine.loans['loan:bronze']['reservation']['slot'] == 'N1'


def test_name_only_lookup_is_resolvable_by_available_search():
    from experiments.word_games_triton.normal_followup import fixtures
    from experiments.word_games.agents import retrieve
    engine, case = next(fixtures())
    context = context_for(engine, case)
    found = retrieve(engine, context['context_id'], {'name': 'search', 'args': {'query': 'Copper Timekeeper'}})
    assert found['candidates'][0]['id'] == 'loan:clock'
    assert 'loan:clock' not in case['source']
