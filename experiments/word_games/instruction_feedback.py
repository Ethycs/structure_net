"""A small, typed pause request from Qwen to the upstream controller.

Only fixed reason codes cross this handoff. Actor prose and tool output stay in
the private run artifacts; they are not copied into the controller request.
"""
import json
import re


NEEDS = {
    'one_step': 'Give me one smaller action at a time.',
    'plain_language': 'Rewrite the instruction in short, direct sentences.',
    'inputs_outputs': 'Specify the exact inputs and expected output.',
    'success_check': 'Give a concrete check that tells me this step is done.',
    'resolve_conflict': 'Resolve the conflicting requirements before I continue.',
    'missing_information': 'Supply the missing task detail instead of asking me to guess.',
}

ASK_PROMPT = '''
The stronger upstream controller interprets the task. You must ask that controller
for simpler instructions when the next instruction is too complex, ambiguous,
contradictory, or missing a detail needed to execute it reliably. Ask before
editing when you can identify the issue up front. If it appears later, stop
further edits. Do not guess, silently weaken the task, or invent the missing fact.
To ask, end this turn with ONLY this JSON shape:
{"kind":"needs_simplification","needs":["one_step","success_check"]}
Use one to three distinct reason codes from:
one_step, plain_language, inputs_outputs, success_check, resolve_conflict,
missing_information. Choose only the codes that apply. Do not include free-form
questions, source names, file contents, or additional fields in this request.
The host will deliver the request to the stronger controller. Its revision must
pass through Qwen translation before execution. Do not claim the task is done
when you ask. A clear, executable instruction does not require clarification.
'''

REVIEW_PROMPT = '''You are Qwen checking whether an instruction is clear enough
for the Qwen executor. Read relevant task files if necessary. This is a read-only
review: do not implement, edit, or claim completion of the task. If the instruction
is clear and executable with the available information, end with ONLY
{"kind":"ready"}. Otherwise ask the stronger controller for simpler instructions.
''' + ASK_PROMPT


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate_feedback_key')
        obj[key] = value
    return obj


def _terminal_json(text):
    # OpenCode may emit an explanation before its final JSON verdict despite
    # the prompt. Accept only a complete terminal object; never relay the prose.
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, length = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if not text[index+length:].strip():
            return value
    raise ValueError('missing_terminal_json')


def extract(events, *, review=False):
    """Validate the last assistant message; never relay its arbitrary text.

    Ordinary execution can finish in prose. Read-only review requires an
    explicit valid ready/pause verdict. Tool output is never a verdict.
    """
    texts = [e['part'] for e in events if e.get('type') == 'text'
             and isinstance(e.get('part', {}).get('text'), str)]
    if not texts:
        if review:
            raise ValueError('missing_instruction_verdict')
        return None
    last_id = texts[-1].get('messageID')
    selected = [p for p in texts if p.get('messageID') == last_id] if last_id else texts[-1:]
    text = '\n'.join(p['text'] for p in selected).strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
    if fence:
        text = fence[1].strip()
    try:
        value = _terminal_json(text)
    except (ValueError, TypeError):
        if review or 'needs_simplification' in text:
            raise ValueError('invalid_instruction_verdict') from None
        return None
    if value == {'kind': 'ready'} and review:
        return {'kind': 'ready'}
    if not isinstance(value, dict) or value.get('kind') != 'needs_simplification':
        if review:
            raise ValueError('invalid_instruction_verdict')
        return None
    needs = value.get('needs')
    if (set(value) != {'kind', 'needs'} or not isinstance(needs, list)
            or not 1 <= len(needs) <= 3 or any(not isinstance(n, str) or n not in NEEDS for n in needs)
            or len(set(needs)) != len(needs)):
        raise ValueError('invalid_simplification_request')
    return {'kind': 'needs_simplification', 'needs': needs,
            'message': ' '.join(NEEDS[n] for n in needs)}
