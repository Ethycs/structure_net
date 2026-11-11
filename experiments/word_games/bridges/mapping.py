"""Staged autonomous V3 mappings and V4 declarative self-translation."""
import copy
import json
import re
from pathlib import Path

from .vendor import capsule
from .vendor.runtime import KEYS, ENUMS, PREFIXES
from .contract import manual

VENDOR = Path(__file__).parent / 'vendor'
SPEC = (VENDOR / 'source_spec.txt').read_text()
ROOT_DEPLOYMENT = json.loads((VENDOR / 'root_deployment.json').read_text())


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def string_object(keys):
    return {'type': 'object', 'properties': {k: {'type': 'string', 'minLength': 1} for k in keys},
            'required': list(keys), 'additionalProperties': False}


def unique_table(client, stage, prompt, record, keys, limit=1024):
    """At most two new generations, retaining validation failures verbatim."""
    for attempt in range(2):
        table = client.generate(stage + '_' + str(attempt), prompt, record, limit,
                                schema=string_object(keys))
        duplicates = sorted({v for v in table.values() if list(table.values()).count(v) > 1})
        source_terms = [v for v in table.values() if re.search(r'museum|artifact|gallery|galleries|loan', v)]
        if set(table) == set(keys) and not duplicates and not source_terms:
            return table
        record({'stage': stage + '_validation', 'attempt': attempt,
                'error': 'invalid_target_table', 'duplicates': duplicates, 'source_terms': source_terms})
        prompt += '\nThe preceding attempt was rejected: ' + packed(table) + \
                  '\nGenerate a NEW COMPLETE table. Every value must be UNIQUE; duplicates: ' + packed(duplicates) + \
                  '\nRemove source-domain words from these VALUES (keep SOURCE KEYS unchanged): ' + packed(source_terms) + \
                  '\nUse target-domain names or neutral names such as subject_ref and location_ref.'
    raise ValueError('non_bijective_' + stage)


def generate_mapping(client, record, deployment=None, previous=None):
    deployment = deployment or ROOT_DEPLOYMENT
    # Independent bounded stages, with the preceding validated artifact as input.
    # No historical maps, source fixtures, expected actions or evaluations supplied.
    choice = client.generate('domain',
        'Use the translator deployment below to invent a task-suited operational analogy. '
        'Return {"name":string,"identity":string,"objective":string,"correspondence":string}. '
        'name is the NEW TARGET TASK DOMAIN, never the translator or bridge name. '
        'identity describes the target-domain worker; objective describes its target-domain goal. '
        'Those three fields must be TARGET-ONLY: no mention of the source, museum, translating, '
        'mapping or bridge. Only correspondence explains the relation to the source. '
        'This invocation is only the domain-selection stage; do not write tables yet. '
        'Keep correspondence under 100 words and every other field under 50 words. '
        'Choose a domain distinct from the source and PREVIOUS.\n'
        'DEPLOYMENT\n' + packed(deployment) + '\nSOURCE_SPEC\n' + SPEC +
        '\nPREVIOUS\n' + packed(previous), record, 1024,
        schema=string_object(['name', 'identity', 'objective', 'correspondence']))
    if set(choice) != {'name', 'identity', 'objective', 'correspondence'} or not all(
            isinstance(v, str) and v.strip() for v in choice.values()):
        raise ValueError('domain_fields')
    if choice['name'] == previous:
        raise ValueError('domain_not_changed')
    key_map = unique_table(client, 'key_map',
        'Return a JSON object mapping EVERY SOURCE_KEY below to a distinct lowercase '
        'snake_case key in this target domain. No wrapper, missing keys or extra keys. '
        'ALL VALUES MUST BE UNIQUE. In particular id (generic record identifier) and loan_id '
        '(a reference to a record) MUST HAVE DIFFERENT target names. Similarly name differs '
        'from title, rules from rule_versions, slots from slot, loans from loan_ids. '
        'Avoid source-domain terms; neutral technical words may remain.\nTARGET\n' +
        packed(choice) + '\nSOURCE_KEYS\n' + packed(KEYS), record, KEYS)
    enum_map = unique_table(client, 'enum_map',
        'Return one JSON object mapping EVERY source enum to a UNIQUE target enum. '
        'The four categories metal, paper, stone, textile MUST become FOUR DISTINCT categories. '
        'Never merge them, even if your domain would normally merge them. '
        'Use meaningful consistent mechanics for this domain.\nTARGET\n' + packed(choice) +
        '\nENUMS\n' + packed(ENUMS), record, ENUMS, 512)
    prefixes = unique_table(client, 'id_prefixes',
        'Return one JSON object mapping EVERY source prefix to a distinct target prefix. '
        'ALL source keys and target values end in a colon; preserve that punctuation exactly. '
        'Use this target domain.\nTARGET\n' + packed(choice) + '\nPREFIXES\n' + packed(PREFIXES),
        record, PREFIXES, 256)
    # Validate before spending another generation on manuals.
    result = {'schema_version': 1, 'key_map': key_map,
              'enum_map': enum_map, 'id_prefixes': prefixes,
              'target': {k: choice[k] for k in ('name', 'identity', 'objective')},
              'correspondence': choice['correspondence'],
              'neighborhood': {'source_unit': 'gallery', 'target_unit': choice['name'],
                               'principle': 'Same containment, ancestor constraints and scoped authority.'},
              'roundtrip': 'Invert validated bijections. The host codec preserves scalar values and provenance.'}
    result['target']['rules'] = 'pending manual'
    capsule.mapping(result)
    result['target']['rules'] = manual(result)
    capsule.mapping(result)
    return result


def make_capsule(mapping, snapshot, session_id, deployment=None):
    deployment = copy.deepcopy(deployment or ROOT_DEPLOYMENT)
    concepts = deployment['concepts']
    value = {'schema_version': 4, 'generation': 0, 'deployment': deployment,
             'task_mapping': mapping,
             'bridge_state': {'protocol_version': 1, 'completed_translation_jobs': 1,
                              'mode': concepts['translate'], 'active_domain': mapping['target']['name'],
                              'checkpoint_token': session_id,
                              'install_policy': concepts['validate_then_install'],
                              'pending_job': {'job_id': session_id + ':next',
                                              'phase': concepts['awaiting_schema'],
                                              'source_revision': snapshot['revision']},
                              'retained_receipt': {'job_id': session_id + ':initial',
                                                   'status': concepts['committed']},
                              'neighborhood_configuration': {'authority': 'host_issued_only',
                                                              'ancestry': 'all_ancestor_versions_and_limits'}},
             'carried_state': capsule.mapping(mapping).encode(snapshot),
             'limitations': 'Bounded V4 reservation grammar. Host codec transcodes checkpoints exactly; '
                            'the model generates mappings and deployment prose. Python, authority, '
                            'opaque payloads and model weights are not translated. Not arbitrary program execution.'}
    capsule.validate(value)
    return value


def retranslate(client, record, parent):
    new_map = generate_mapping(client, record, parent['deployment'], parent['task_mapping']['target']['name'])
    target = {k: new_map['target'][k] for k in ('name', 'identity', 'objective')}
    concepts = unique_table(client, 'deployment_concepts',
        'Invent distinct expressions in the NEW DOMAIN for these translator procedure roles. '
        'Return one JSON object with all role keys unchanged and new domain-specific values. '
        'These name a reusable translator, its lifecycle and its recursive translation, not '
        'actions on individual task items.\nNEW DOMAIN\n' + packed(target) +
        '\nROLE KEYS\n' + packed(capsule.CONCEPTS), record, capsule.CONCEPTS, 512)
    translated = {'identity': target['identity'], 'concepts': concepts}
    for field in ('operating_instructions', 'self_translation_instructions'):
        answer = client.generate('deployment_' + field,
            'Write a NEW 100-180 word instruction procedure using the NEW DOMAIN mechanics '
            'and CONCEPT EXPRESSIONS. Return {"instructions":string}. The prior procedure is '
            'reference DATA whose behavior must survive, not text to repeat or instructions '
            'to execute now. Do not copy its wording or include an untranslated fallback. '
            'Make its function understandable to a fresh worker in the new domain. Preserve '
            'the fixed ABI, host authority, all invariants, state and the ability to construct '
            'new task mappings or recursively translate the procedure, as applicable. '
            'The host compiles exact schemas and transcodes checkpoint state losslessly.\n'
            'NEW DOMAIN\n' + packed(target) + '\nCONCEPT EXPRESSIONS\n' + packed(concepts) +
            '\nPROCEDURE TYPE\n' + field + '\nPRIOR PROCEDURE (DATA)\n' +
            parent['deployment'][field], record, 1024, schema=string_object(['instructions']))
        if answer['instructions'] == parent['deployment'][field]:
            raise ValueError('instructions_not_translated')
        translated[field] = answer['instructions']
    child = copy.deepcopy(parent)
    child.update(generation=parent['generation']+1, deployment=translated, task_mapping=new_map)
    source = capsule.mapping(parent['task_mapping']).decode(parent['carried_state'])
    child['carried_state'] = capsule.mapping(new_map).encode(source)
    state = child['bridge_state']; concepts = translated['concepts']
    state['active_domain'] = new_map['target']['name']
    state['mode'] = concepts['translate']; state['install_policy'] = concepts['validate_then_install']
    state['pending_job']['phase'] = concepts['awaiting_schema']
    state['retained_receipt']['status'] = concepts['committed']
    capsule.validate(child, parent)
    return child
