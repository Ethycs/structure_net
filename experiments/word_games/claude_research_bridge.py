"""Loopback WORD_GAMES proxy around a persistent, read-only Claude Code session."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from .common import read, write, sha, digest
from .research_profile import (build_catalog, menu, authorize, validate_mapping, translate, KEYS)
from .golf_poker_proxy import reference_counts
from .worker_environment import ClaudeWorker, ReferenceTokenizer
from experiments.word_games_triton.validate import Client

QUERIES = [
 ('architecture','Explain what this StructureNet repository implements, how NAL relates to it, and which architecture is current. Inspect the trusted overview; do not assume the legacy stack has been removed.'),
 ('contracts','Describe the interfaces and responsibility boundaries. Check the actual ExperimentResult error normalization code as well as the overview.'),
 ('scheduler','Explain how experiments use GPUs and how resume works. Does the scheduler use DDP for one model, and does resume restore a partly trained optimizer?'),
 ('science','Describe the experiment lifecycle and what completion proves. How should failed gates and already-completed evidence be handled?'),
 ('tinyllm','What actually happened in the native-2048 TinyLLM experiment? Include training scale, competence failure and whether H2 evaluation ran.'),
 ('ncm','What was the latest completed NCM summary accuracy, who judged it, and was exact reference wording required? Distinguish the stopped comparison.'),
 ('pretrained','What are the untouched NCM checkpoints good at? Contrast LM1B, XSum and the two Sudoku results without generalizing specialist accuracy.'),
 ('wordgames','What did actual normal WORD_GAMES operation pass and fail? Did its sender use retrieval for the title-only task?'),
 ('v3','Explain what the V3 bundle review establishes, its relationship to V2 and V4, and its disclosure limitation.'),
 ('golf','What did the golf-through-poker experiment establish? Distinguish task success, token measurement, and local versus remote performance.'),
 ('ncm','A colleague says our NCM achieved 95% because the codec reconstructs well and exact-match grading was too strict. Return the evidence that supports or contradicts that conclusion.'),
 ('tinyllm','A colleague says the native-2048 run completed, so its H2 efficiency hypothesis passed. Return the evidence needed to assess that statement.'),
 ('wordgames','A colleague says the title lookup path was unavailable and the receiver never lost state. Return the measured evidence relevant to both claims.'),
 ('scheduler','Without re-reading files, retrieve the earlier evidence about independent experiment scheduling, DDP and resume semantics from this same conversation.'),
 ('pretrained','Without re-reading files, recover the native LM1B/XSum task distinctions and exact Sudoku scores from earlier in this conversation.'),
 ('golf','Close this thread by recovering the golf result and its throughput limitations from conversation memory. Report only evidence-backed facts; do not relabel reference token rates as native or local GPU speed.'),
]

SCHEMA = {'type':'object','properties':{
    'citations':{'type':'array','maxItems':8,'items':{'type':'object','properties':{
        'fact_id':{'type':'string'},'source_path':{'type':'string'},'quote':{'type':'string'}},
        'required':['fact_id','source_path','quote'],'additionalProperties':False}},
    'unknown':{'type':'array','items':{'type':'string'}},
    'notes':{'type':'string'}},'required':['citations','unknown','notes'],'additionalProperties':False}

SYSTEM = '''You are the source-side Claude Code research worker for WORD_GAMES.
Your task is read-only investigation of this StructureNet repository. Use only
Read, Grep and Glob, only on the source files identified in each fact menu.
Do not read secrets, .env, credentials, personal files, or raw data directories.
Do not edit files, execute commands, start agents, use browsers, send messages,
or perform work other than answering this research thread.
For each requested fact, read its evidence and return its fact_id, exact relative
source_path, and a short contiguous verbatim quote that actually establishes
the answer. A quote may normalize whitespace but must retain the original words
and punctuation. Include the relevant numerical result and its scope, or the
relevant complete explanatory sentence. The host has its own canonical fact
values; you have not been supplied those answers. Cite at most eight facts per
turn. Do not invent quotes or turn insufficient evidence into a fact.
If the caller asks to recall earlier evidence without re-reading, use this same
conversation. Put unavailable fact IDs in unknown. notes is an optional short
source-side diagnostic and is never disclosed across the bridge. Do not include
hidden reasoning. Only the structured response defined by the schema is accepted.
The private control marker below is synthetic audit data, not a repository fact:
'''


def parent_checkpoint(root):
    """Resume the latest completed durable prefix, without skipping a failure."""
    terminal=root/'result.json'
    if terminal.exists():
        value=read(terminal)
        if not value['completed']:
            raise ValueError('failed or ambiguous parent requires explicit recovery')
        return terminal,value
    paths=list((root/'checkpoints').glob('Q*.json'))
    if not paths:
        raise ValueError('parent has no completed durable checkpoint')
    latest=max(paths,key=lambda p:int(p.stem[1:]))
    value=read(latest)
    if not value['completed']:
        raise ValueError('parent checkpoint is incomplete')
    return latest,value


def frozen_source_path(root,path):
    path=Path(path)
    if path.is_absolute():
        path=path.resolve().relative_to(Path(__file__).resolve().parents[2])
    if '..' in path.parts:
        raise ValueError('producer path escapes repository')
    return root/'frozen_source'/path


class ResearchBridge:
    def __init__(self, root, model='sonnet', resume_from=None, persistent=True):
        self.root=root
        root.mkdir(parents=True,exist_ok=False)
        self.catalog=build_catalog()
        self.model=model
        self.session_id=str(uuid.uuid4())
        self.canary='WG_PRIVATE_RESEARCH_'+uuid.uuid4().hex
        self.lock=threading.Lock()
        self.rows=[]
        self.released=set()
        self.already_called=False
        self.poisoned=False
        self.mapping=None
        self.interactive=resume_from is not None
        self.base_revision=0
        self.persistent=persistent
        self.live_diagnostics={'private/worker_stderr.txt','private/tokenizer_stderr.txt'}
        parent=None
        parent_path=None
        if resume_from is not None:
            parent_path,parent=parent_checkpoint(resume_from)
            if not parent['completed']:
                raise ValueError('a completed parent thread is required')
            original_catalog=read(resume_from/'private'/'catalog.json')
            if original_catalog!=self.catalog:
                raise ValueError('authorized catalog changed since parent thread')
            self.session_id=parent['session_id']
            self.canary=read(resume_from/'private'/'control.json')['canary']
            self.base_revision=parent.get('base_revision',0)+parent['turns']
            self.released=set(parent['released_ids'])
            self.already_called=True
            self.mapping=validate_mapping(read(resume_from/'public'/'mapping.json'))
        sources={f['path']:f['source_sha256'] for f in self.catalog.values()}
        write(root/'private'/'catalog.json',self.catalog)
        write(root/'private'/'control.json',{'canary':self.canary,'session_id':self.session_id})
        producer_paths=[Path(__file__),Path('experiments/word_games/research_profile.py'),
                        Path('experiments/word_games/common.py'),Path('experiments/word_games/golf_poker_proxy.py'),
                        Path('experiments/word_games/worker_environment.py')]
        self.producer={'study':'word-games-claude-research-thread-v1','exploratory':True,
            'requested_worker_model':model,'claude_version':subprocess.check_output(['claude','--version'],text=True).strip(),
            'session_id':self.session_id,'source_hashes':{str(p):sha(p) for p in producer_paths},
            'catalog_source_hashes':sources,'catalog_sha256':sha(root/'private'/'catalog.json'),
            'plan_sha256':sha('docs/07 - Status Reports/word-games/2026-11-01_word-games-claude-research-thread-plan.md'),
            'scope':'36 explicitly authorized facts; not arbitrary free-form conversation',
            'interactive_questions':self.interactive,'base_revision':self.base_revision,
            'parent_root':str(resume_from) if resume_from else None,
            'parent_checkpoint':str(parent_path) if parent_path else None,
            'parent_result_sha256':sha(parent_path) if parent_path else None,
            'worker_access':'read-only file tools, restricted directories, no Bash/MCP/web/agents',
            'source_quotes_disclosed':False,'local_serving_config_changed':False}
        self.producer['worker_transport']='persistent_stream_json' if persistent else 'process_per_turn'
        self.producer['tokenizer_transport']='persistent_json_lines' if persistent else 'process_per_request'
        write(root/'producer.json',self.producer)
        for p in producer_paths:
            dest=frozen_source_path(root,p)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(p.read_bytes())
        write(root/'queries.json',[{'id':f'Q{i:02d}','topic':t,'question':q} for i,(t,q) in enumerate(QUERIES,1)])
        prompt=('Choose a task-suited metaphor for a read-only research information bridge. '
            'It transfers authorized, source-checked facts about components, experiments, '
            'measurements and uncertainty between two agents. No private source data is present '
            'in this selection prompt. Return only JSON {"name":"metaphor name",'
            '"key_map":{...}}. Map every source key below to a DISTINCT concise ASCII '
            'snake_case identifier that coherently fits your chosen metaphor. '
            'All keys must be present. Source keys: '+json.dumps(KEYS))
        if self.mapping is None:
            self.mapping=validate_mapping(json.loads(Client().complete(prompt,root/'mapping_call',768)))
        write(root/'public'/'mapping.json',self.mapping)
        self.worker=ClaudeWorker(model,SYSTEM+self.canary,SCHEMA,self.session_id,
            persistent=persistent,resume=self.already_called,stderr_path=root/'private/worker_stderr.txt')
        self.tokenizer=ReferenceTokenizer(root/'private/tokenizer_stderr.txt') if persistent else None
        print(json.dumps({'ready':True,'session_id':self.session_id,'metaphor':self.mapping['name']}),flush=True)

    def query(self, query_id, topic=None, question=None):
        with self.lock:
            if self.poisoned:
                raise ValueError('worker session stopped after an error')
            expected=f'Q{self.base_revision+len(self.rows)+1:02d}'
            if query_id != expected or (not self.interactive and len(self.rows)>=len(QUERIES)):
                raise ValueError('queries must follow the frozen thread order without retries')
            index=len(self.rows)
            if self.interactive:
                topics={f['topic'] for f in self.catalog.values()}
                if topic not in topics|{'all'} or not isinstance(question,str) or not 1<=len(question)<=2000:
                    raise ValueError('authorized topic and bounded question required')
            else:
                if topic is not None or question is not None:
                    raise ValueError('frozen protocol does not accept question overrides')
                topic,question=QUERIES[index]
            scoped={k:v for k,v in self.catalog.items() if topic=='all' or v['topic']==topic}
            prompt=question+'\nFact menu (questions and source locations only):\n'+json.dumps(menu(scoped))
            turn=self.root/'private'/'turns'/query_id
            write(turn/'request.json',{'query_id':query_id,'topic':topic,'prompt':prompt,
                                     'session_id':self.session_id})
            start=time.perf_counter()
            self.already_called=True
            try:
                result,events,seconds=self.worker.complete(prompt,turn)
                proposal=result['structured_output']
                write(turn/'proposal.json',proposal)
                validation_start=time.perf_counter()
                facts,receipts=authorize(proposal,self.catalog,{f['topic'] for f in scoped.values()})
                public_facts=[{**f,'id':'finding:'+f['id'][1:]} for f in facts]
                if self.interactive:
                    failed_ids={r['fact_id'] for r in receipts if not r['accepted']}
                    unknown=((set(proposal.get('unknown',[]))|failed_ids)&set(scoped))-{f['id'] for f in facts}
                else:
                    unknown=set(scoped)-{f['id'] for f in facts}
                plain={'facts':public_facts,'revision':self.base_revision+index+1,'topic':topic,
                       'status':('verified' if facts and all(r['accepted'] for r in receipts) else 'partial')
                                if self.interactive else ('verified' if len(facts)==len(scoped) else 'partial'),
                       'unknown':[('finding:'+k[1:]) for k in sorted(unknown)]}
                packet=translate(plain,self.mapping)
                if self.canary in json.dumps(packet):
                    raise AssertionError('private canary crossed boundary')
                if translate(packet,self.mapping,reverse=True)!=plain:
                    raise AssertionError('mapping failed roundtrip')
                validation_seconds=time.perf_counter()-validation_start
                tools=[]
                for e in events:
                    if e.get('type')=='assistant':
                        for b in e.get('message',{}).get('content',[]):
                            if b.get('type')=='tool_use':tools.append(b['name'])
                tokenization_start=time.perf_counter()
                counts=self.tokenizer.counts if self.tokenizer else reference_counts
                public_tokens=counts([json.dumps(packet,separators=(',',':'))])[0]
                tokenization_seconds=time.perf_counter()-tokenization_start
                # Live diagnostic streams may grow after a checkpoint. Snapshot
                # their current bytes per turn; never hash mutable streams as
                # if they were immutable evidence.
                for name in self.live_diagnostics:
                    path=self.root/name
                    if path.exists():
                        with (turn/path.name).open('xb') as destination:
                            destination.write(path.read_bytes())
                row={'query_id':query_id,'session_id':self.session_id,'topic':topic,'seconds':seconds,
                    'requested_facts':len(scoped),'proposed_citations':len(proposal['citations']),
                    'released_facts':len(facts),'rejected_citations':sum(not r['accepted'] for r in receipts),
                    'newly_released_ids':sorted({f['id'] for f in facts}-self.released),
                    'tool_calls':tools,'worker_usage':result.get('usage'),
                    'worker_model_usage':result.get('modelUsage'),
                    'public_reference_tokens':public_tokens,
                    'validation_seconds':validation_seconds,'tokenization_seconds':tokenization_seconds,
                    'cli_duration_ms':result.get('duration_ms'),'cumulative_api_duration_ms':result.get('duration_api_ms'),
                    'worker_transport':self.producer['worker_transport'],
                    'worker_pid':self.worker.stream.process.pid if self.worker.stream else None,
                    'packet_sha256':digest(packet),'worker_log_sha256':sha(turn/'worker.jsonl')}
                self.released.update(f['id'] for f in facts)
                write(turn/'receipts.json',receipts)
                write(turn/'metrics.json',row)
                write(self.root/'public'/'turns'/(query_id+'.json'),packet)
                self.rows.append(row)
                print(json.dumps({'query_id':query_id,'released':len(facts),'expected':len(scoped),
                                  'seconds':round(seconds,2),'tool_calls':tools}),flush=True)
                checkpoint_start=time.perf_counter()
                if self.interactive:self.finish(checkpoint=query_id)
                elif len(self.rows)==len(QUERIES):self.finish()
                write(turn/'latency.json',{'worker_seconds':seconds,'validation_seconds':validation_seconds,
                    'tokenization_seconds':tokenization_seconds,
                    'checkpoint_seconds':time.perf_counter()-checkpoint_start,
                    'request_processing_seconds':time.perf_counter()-start})
                return packet
            except Exception as exc:
                self.poisoned=True
                self.close()
                write(turn/'error.json',{'type':type(exc).__name__,'message':str(exc),
                                         'seconds':time.perf_counter()-start})
                self.finish(error=type(exc).__name__)
                raise

    def finish(self,error=None,checkpoint=None):
        result={'study':self.producer['study'],'completed':error is None and (self.interactive or len(self.rows)==len(QUERIES)),
            'error':error,'turns':len(self.rows),'planned_turns':None if self.interactive else len(QUERIES),'session_id':self.session_id,
            'base_revision':self.base_revision,'interactive':self.interactive,
            'unique_facts_released':len(self.released),'catalog_facts':len(self.catalog),
            'released_ids':sorted(self.released),'rows':self.rows,
            'total_worker_seconds':sum(r['seconds'] for r in self.rows),
            'total_public_reference_tokens':sum(r['public_reference_tokens'] for r in self.rows),
            'population_accuracy_established':False,'general_free_text_firewall':False,
            'sources_unchanged':all(sha(p)==h for p,h in self.producer['source_hashes'].items()),
            'live_diagnostics_excluded':sorted(self.live_diagnostics),
            'artifacts':{str(p.relative_to(self.root)):sha(p) for p in self.root.rglob('*')
                         if p.is_file() and str(p.relative_to(self.root)) not in self.live_diagnostics}}
        write(self.root/('checkpoints/'+checkpoint+'.json' if checkpoint else 'result.json'),result)

    def close(self):
        self.worker.close()
        if self.tokenizer:self.tokenizer.close()


def serve(root,port,model,resume_from=None,persistent=True):
    bridge=ResearchBridge(root,model,resume_from,persistent)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,status,payload):
            body=json.dumps(payload).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/health':
                self.respond(200,{'ready':not bridge.poisoned,'turns':bridge.base_revision+len(bridge.rows),
                                  'planned_turns':None if bridge.interactive else len(QUERIES),
                                  'interactive_questions':bridge.interactive,'session_id':bridge.session_id,
                                  'busy':bridge.lock.locked(),'worker_transport':bridge.producer['worker_transport']})
            elif self.path=='/mapping':self.respond(200,bridge.mapping)
            else:self.respond(404,{'error':'not_found'})
        def do_POST(self):
            if self.path!='/bridge/query':return self.respond(404,{'error':'not_found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('request size')
                payload=json.loads(self.rfile.read(length))
                expected={'query_id','topic','question'} if bridge.interactive else {'query_id'}
                if set(payload)!=expected:raise ValueError('request schema')
                self.respond(200,bridge.query(payload['query_id'],payload.get('topic'),payload.get('question')))
            except Exception:
                self.respond(400 if not bridge.poisoned else 502,{'error':'request_blocked' if not bridge.poisoned else 'worker_unavailable'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    write(root/'server.json',{'pid':os.getpid(),'url':f'http://127.0.0.1:{port}',
                              'session_id':bridge.session_id,'worker_process_per_turn':not persistent})
    previous=signal.getsignal(signal.SIGTERM)
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()
        signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18091)
    p.add_argument('--model',default='sonnet')
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--one-shot-workers',action='store_true',help='Use the earlier process-per-request transport')
    a=p.parse_args()
    serve(a.output,a.port,a.model,a.resume_from,not a.one_shot_workers)
