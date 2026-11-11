"""Durable, capability-separated reservation sessions over the V4 bridge.

The Engine is authoritative. The controller can propose actions but cannot issue
contexts, inspect source state, install mappings, or commit source mutations.
"""
import copy
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from .mapping import generate_mapping, make_capsule, packed, retranslate
from .contract import response_schema, translate_wire_goal
from .vendor import capsule
from .vendor.runtime import Bridge
from .vendor.source_runtime import Engine, Rejected


class Conflict(ValueError):
    pass


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def restore(data):
    engine = Engine.fixture()
    for key, value in data['snapshot'].items():
        setattr(engine, key, copy.deepcopy(value))
    engine.permissions = {k: set(v) for k, v in data['permissions'].items()}
    engine.contexts = copy.deepcopy(data['contexts'])
    engine.receipts = {tuple(row['key']): row['value'] for row in data['receipts']}
    engine.rebuild_indexes()
    return engine


def export(engine):
    return {'snapshot': engine.snapshot(),
            'permissions': {k: sorted(v) for k, v in engine.permissions.items()},
            'contexts': copy.deepcopy(engine.contexts),
            'receipts': [{'key': list(k), 'value': v} for k, v in engine.receipts.items()]}


class Privacy:
    """Stable, reversible opaque payload/identity handles outside the model.

    The V4 codec deliberately leaves prose untouched. This additional boundary
    prevents its protected title/note/name fields leaking the source domain.
    It preserves equality and exact quoted writes; arbitrary semantic searches
    through note contents are outside this first reservation workflow.
    """
    PROTECTED = {'title', 'text', 'notes', 'name', 'actor', 'event_id',
                 'question', 'context_id', 'tx_id'}
    PREFIX = re.compile(r'^(loan:|gal:|rule:|approval:|dept:)')

    def __init__(self, aliases):
        self.aliases = aliases

    def alias(self, value):
        if value not in self.aliases:
            match = self.PREFIX.match(value)
            self.aliases[value] = (match[0] if match else '') + 'opaque_' + secrets.token_hex(8)
        return self.aliases[value]

    def shield(self, value, field=None):
        if isinstance(value, dict):
            return {(self.alias(k) if self.PREFIX.match(k) else k): self.shield(v, k)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.shield(v, field) for v in value]
        if isinstance(value, str) and (field in self.PROTECTED or self.PREFIX.match(value)):
            return self.alias(value)
        return value

    def unshield(self, value):
        inverse = {v: k for k, v in self.aliases.items()}
        def walk(item):
            if isinstance(item, dict):
                return {inverse.get(k, k): walk(v) for k, v in item.items()}
            if isinstance(item, list):
                return [walk(v) for v in item]
            return inverse.get(item, item) if isinstance(item, str) else item
        return walk(value)

    def goal(self, goal):
        # Exact quoted payloads can be carried and later restored without sending
        # the original strings to the controller. This is not prose summarization.
        for quote in re.findall(r'"([^"\n]+)"', goal):
            self.alias(quote)
        if not self.aliases:
            return goal
        pattern = '|'.join(re.escape(k) for k in sorted(self.aliases, key=len, reverse=True))
        return re.sub(pattern, lambda m: self.aliases[m[0]], goal)


class Service:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls
                (id INTEGER PRIMARY KEY, session_id TEXT, record TEXT NOT NULL);
        ''')
        self.lock = threading.RLock()
        self.path.chmod(0o600)

    def record(self, sid):
        def save(value):
            self.db.execute('INSERT INTO calls(session_id,record) VALUES (?,?)', (sid, packed(value)))
        return save

    def get(self, sid):
        row = self.db.execute('SELECT data FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is None:
            raise KeyError('session_not_found')
        return json.loads(row[0])

    def save(self, session):
        # One SQLite statement is the durable commit point, including engine,
        # receipts, pending action, version and checkpoint. No filesystem journal gap.
        self.db.execute('INSERT OR REPLACE INTO sessions(id,data) VALUES (?,?)',
                        (session['id'], packed(session)))

    def create(self, client, caller, body):
        if set(body) - {'profile'} or body.get('profile', 'reservations-v4') != 'reservations-v4':
            raise ValueError('unsupported_profile')
        with self.lock:
            sid = secrets.token_hex(16)
            mapping = generate_mapping(client, self.record(sid))
            engine = Engine.fixture()
            # Stronger than table uniqueness: verify the actual source snapshot.
            mapper = capsule.mapping(mapping)
            expected = engine.snapshot()
            for group in ('loans', 'galleries'):
                for item in expected[group].values():
                    item.pop('target_id', None)  # V4's explicitly obsolete projection.
            if mapper.decode(mapper.encode(engine.snapshot())) != expected:
                raise ValueError('mapping_checkpoint_roundtrip')
            doer, controller = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            session = {'id': sid, 'version': 0, 'owner': caller, 'created_at': time.time(),
                       'tokens': {'doer': digest(doer), 'controller': digest(controller)},
                       'engine': export(engine), 'capsule': make_capsule(mapping, engine.snapshot(), sid),
                       'aliases': {}, 'pending': None, 'receipts': {}, 'last_result': None}
            self.save(session)
            return {'session_id': sid, 'version': 0, 'profile': 'reservations-v4',
                    'doer_capability': doer, 'controller_capability': controller,
                    'target': mapping['target']['name'], 'generation': 0}

    @staticmethod
    def authorize(session, capability, role):
        if not isinstance(capability, str) or not hmac.compare_digest(
                session['tokens'][role], digest(capability)):
            raise PermissionError('invalid_session_capability')

    @staticmethod
    def bridge(session, engine):
        # The upstream Bridge constructor only accepts a path. Use its untouched
        # methods with the validated in-memory mapping, avoiding shared temp paths.
        bridge = Bridge.__new__(Bridge)
        bridge.mapping = capsule.mapping(session['capsule']['task_mapping'])
        bridge.engine = engine
        return bridge

    def public_view(self, session):
        pending = session['pending']
        return {'session_id': session['id'], 'version': session['version'],
                'generation': session['capsule']['generation'],
                'manual': session['capsule']['task_mapping']['target'],
                'task': copy.deepcopy(pending['public']) if pending else None,
                'proposal': copy.deepcopy(pending.get('proposal')) if pending else None,
                'result': copy.deepcopy(session['last_result'])}

    def inspect(self, sid, capability, role):
        with self.lock:
            session = self.get(sid)
            self.authorize(session, capability, role)
            if role == 'controller':
                return self.public_view(session)
            return {'session_id': sid, 'version': session['version'],
                    'generation': session['capsule']['generation'],
                    'state': session['engine']['snapshot'],
                    'pending': session['pending'], 'last_result': session['last_result']}

    def mutate(self, sid, capability, role, action, body, client):
        if not isinstance(body.get('request_id'), str) or not 1 <= len(body['request_id']) <= 100:
            raise ValueError('request_id_required')
        if type(body.get('version')) is not int:
            raise ValueError('version_required')
        with self.lock:
            session = self.get(sid)
            self.authorize(session, capability, role)
            rid = body['request_id']; fingerprint = digest([role, action, body])
            receipt = session['receipts'].get(rid)
            if receipt:
                if receipt['fingerprint'] != fingerprint:
                    raise Conflict('request_id_conflict')
                return receipt['response']
            if session['version'] != body['version']:
                raise Conflict('stale_session_version')
            allowed = {('doer', 'prepare'), ('doer', 'apply'), ('doer', 'retranslate'),
                       ('controller', 'solve'), ('controller', 'submit')}
            if (role, action) not in allowed:
                raise PermissionError('role_operation_denied')
            engine = restore(session['engine'])
            bridge = self.bridge(session, engine)
            privacy = Privacy(session['aliases'])
            if action == 'prepare':
                if set(body) != {'request_id', 'version', 'goal', 'context'}:
                    raise ValueError('prepare_fields')
                if session['pending']:
                    raise Conflict('pending_task_exists')
                if not isinstance(body['goal'], str) or not 0 < len(body['goal']) <= 12000:
                    raise ValueError('goal_length')
                context = engine.issue_context(**body['context'])
                source_view = bridge.mapping.decode(bridge.view(context))
                view = bridge.mapping.encode(privacy.shield(source_view))
                goal = translate_wire_goal(privacy.goal(body['goal']), session['capsule']['task_mapping'])
                translated = client.generate('goal',
                    'Express the GOAL naturally in the TARGET domain. Its wire tokens and identifiers '
                    'are already translated correctly; NEVER change them or introduce new identifiers. '
                    'Return {"goal":string}. Do not solve it or choose additional actions. '
                    'Preserve every number, date, quoted opaque payload handle and ambiguity. '
                    'Do not invent permits, attributes or conditions.\nTARGET\n' +
                    packed(bridge.mapping.public_manual()) + '\nGOAL\n' + goal,
                    self.record(sid), 768)
                if set(translated) != {'goal'} or not isinstance(translated['goal'], str) or not translated['goal']:
                    raise ValueError('translated_goal_fields')
                handles = lambda text: set(re.findall(r'opaque_[0-9a-f]+', text))
                if handles(translated['goal']) != handles(goal):
                    raise ValueError('translated_goal_identity_changed')
                # Literal guard in addition to architectural view isolation.
                encoded = packed({'goal': translated['goal'], 'view': view})
                public_tokens = set(bridge.mapping.data['key_map'].values()) | set(bridge.mapping.data['enum_map'].values())
                for source, alias in session['aliases'].items():
                    # A provenance event named "transfer" is not disclosed by
                    # a legitimate target token such as "patient_transfer".
                    if (len(source) >= 8 and source not in public_tokens and source not in alias
                            and re.search(r'(?<![A-Za-z0-9_])' + re.escape(source) + r'(?![A-Za-z0-9_])', encoded)):
                        raise ValueError('source_literal_in_controller_view')
                session['pending'] = {'context': context, 'source_goal': body['goal'],
                                      'public': {'goal': translated['goal'], 'wire_goal': goal, 'view': view},
                                      'proposal': None, 'task_id': rid}
                response = {'status': 'prepared'}
            elif action in ('solve', 'submit'):
                expected_fields = {'request_id', 'version'} | ({'proposal'} if action == 'submit' else set())
                if set(body) != expected_fields:
                    raise ValueError('proposal_fields')
                pending = session['pending']
                if not pending or pending['proposal'] is not None:
                    raise Conflict('no_task_awaiting_proposal')
                prompt = ('Act as the target-only controller. Follow TARGET_MANUAL and the current host view. '
                    'wire_goal preserves the exact requested action tokens and identifiers; it is '
                    'authoritative if the fluent goal is ambiguous. '
                    'Return only target JSON operations or clarification. Do not invent context or '
                    'permissions. Perform ONLY the requested change, using the minimum number of '
                    'actions. Never add log/annotation actions unless a note was explicitly requested. '
                    'Inspect the current placement: null means unplaced. Never cancel or move an '
                    'unplaced item. Opaque handles are literal payloads; retain them exactly.\nTARGET_MANUAL\n' +
                    packed(bridge.mapping.public_manual()) + '\nTASK\n' + packed(pending['public']))
                for attempt in range(2 if action == 'solve' else 1):
                    proposal = body.get('proposal') if action == 'submit' else client.generate(
                        'solve_' + str(attempt), prompt, self.record(sid), 1024,
                        schema=response_schema(session['capsule']['task_mapping']))
                    # Validate on a throwaway Engine. Controller has no commit authority.
                    trial = self.bridge(session, restore(session['engine']))
                    source_proposal = privacy.unshield(bridge.mapping.decode(proposal))
                    result = trial.apply(bridge.mapping.encode(source_proposal), pending['context'], pending['task_id'])
                    if result['result']['status'] != 'rejected':
                        break
                    self.record(sid)({'stage': 'controller_validation', 'attempt': attempt,
                                      'result': result['result']})
                    code = result['result']['code']
                    feedback = {
                        'wrong_reservation_state': 'The chosen action has the wrong placement precondition. Null placement requires the action that establishes an initial placement.',
                        'not_reserved': 'This item has no placement to cancel. Inspect the current view.',
                        'operation_not_authorized': 'That action is outside the host-issued permitted actions.',
                        'inherited_due_limit': 'The date exceeds an ancestor deadline constraint.',
                        'approval_required': 'This item requires an existing matching permit.',
                        'slot_occupied': 'The proposed final placement collides with an occupied cell.',
                    }.get(code, 'Host validation rejected the proposed action. Check exact identifiers, current placement, authority, permits and constraints.')
                    prompt += '\nREJECTED_TARGET_RESPONSE\n' + packed(proposal) + '\nVALIDATION\n' + feedback + \
                              '\nReturn a NEW target response satisfying the original task and current view.'
                if result['result']['status'] == 'rejected':
                    raise ValueError('proposal_rejected')
                pending['proposal'] = proposal
                response = {'status': 'proposed', 'proposal': proposal}
            elif action == 'apply':
                if set(body) != {'request_id', 'version'}:
                    raise ValueError('apply_fields')
                pending = session['pending']
                if not pending or pending['proposal'] is None:
                    raise Conflict('no_proposal')
                decoded = privacy.unshield(bridge.mapping.decode(pending['proposal']))
                result = bridge.apply(bridge.mapping.encode(decoded), pending['context'], pending['task_id'])
                response = result
                # No decoded operations, source error codes or touched source IDs cross.
                session['last_result'] = {k: result['result'][k] for k in ('status', 'revision', 'mutated')}
                session['pending'] = None
            elif action == 'retranslate':
                if set(body) != {'request_id', 'version'}:
                    raise ValueError('retranslate_fields')
                if session['pending']:
                    raise Conflict('finish_pending_task_before_retranslation')
                if session['capsule']['generation'] >= 3:
                    raise ValueError('recursion_limit')
                parent = copy.deepcopy(session['capsule'])
                parent['carried_state'] = bridge.mapping.encode(engine.snapshot())
                parent['bridge_state']['pending_job']['source_revision'] = engine.revision
                child = retranslate(client, self.record(sid), parent)
                session['capsule'] = child
                response = {'status': 'translated', 'generation': child['generation'],
                            'checkpoint': capsule.validate(child, parent)}
            session['engine'] = export(engine)
            session['capsule']['carried_state'] = capsule.mapping(
                session['capsule']['task_mapping']).encode(engine.snapshot())
            session['version'] += 1
            response = {'version': session['version'], **response}
            session['receipts'][rid] = {'fingerprint': fingerprint, 'response': response}
            self.save(session)
            return response
