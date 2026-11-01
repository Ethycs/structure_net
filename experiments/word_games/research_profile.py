"""Explicitly authorized repository facts for the first read-only research bridge."""
from pathlib import Path

from .common import digest, sha

OVERVIEW = 'docs/03 - Architecture/structure-net-overview.md'
GUIDE = 'docs/02 - Implementation/experimental-system-master-guide.md'
TINY = 'docs/08 - Analysis/tinyllm/2026-09-07_tinyllm-native-2048-h2-crossover.md'
NCM = 'docs/08 - Analysis/ncm/2026-10-29_ncm-latest-recovered-semantic-grade.md'
NATIVE = 'docs/08 - Analysis/ncm/2026-10-30_ncm-pretrained-capabilities.md'
WG = 'docs/08 - Analysis/word-games/2026-11-01_word-games-normal-operation.md'
V3 = 'docs/08 - Analysis/word-games/2026-11-01_word-games-v3-bundle-review.md'
GOLF = 'docs/08 - Analysis/word-games/2026-11-01_word-games-golf-poker-proxy.md'

# Source worker sees the ID, topic, question and path, not these canonical values
# or the required evidence anchor. Only this host-owned projection can leave.
SPECS = [
 ('architecture','What is Structure Net for?',OVERVIEW,'sparse neural-network construction, measurement, and structural evolution','Structure Net','purpose','Sparse neural-network construction, measurement, and structural evolution.'),
 ('architecture','How does NAL relate to Structure Net?',OVERVIEW,'packaged as a sibling Python package','Neural Architecture Lab','relationship','A sibling Python package that provides the hypothesis-and-experiment harness.'),
 ('architecture','Has the legacy evolution stack been removed?',OVERVIEW,'The legacy directory cannot yet be described as removed','legacy evolution','status','Still present beside the newer components; compatibility users remain.'),
 ('architecture','What is the preferred coordination surface?',OVERVIEW,'New coordination work targets `components/orchestrators`','coordination','preferred surface','components/orchestrators'),
 ('architecture','Can unmodified llama.cpp execute TinyLLM recurrent feedback models?',OVERVIEW,'cannot represent their recurrent execution','TinyLLM feedback','runtime compatibility','Unmodified GPT-2, GGUF and llama.cpp runtimes cannot represent the recurrent execution.'),
 ('contracts','Should a metric select a mutation?',OVERVIEW,'A metric should measure, not choose a mutation','metrics','responsibility','Measure; analysis, planning and mutation are separate responsibilities.'),
 ('contracts','Which explicit interface contracts are named?',OVERVIEW,'`ComponentContract`, `EvolutionContext`, `AnalysisReport`, `EvolutionPlan`','component interfaces','contracts',['ComponentContract','EvolutionContext','AnalysisReport','EvolutionPlan']),
 ('contracts','What happens to an error-bearing completed ExperimentResult?', 'src/neural_architecture_lab/core.py','if self.error is not None and self.status == ExperimentStatus.COMPLETED:','ExperimentResult','error normalization','An error-bearing COMPLETED result is changed to FAILED.'),
 ('scheduler','How are independent experiments scheduled?',OVERVIEW,'logical CUDA\ndevice slots using CUDA-safe spawned processes','NAL local runner','scheduling','Independent experiments occupy logical CUDA slots in spawned processes.'),
 ('scheduler','Does the canonical local scheduler distribute one model with DDP?',GUIDE,'It\ndoes not distribute one model with DDP','NAL local scheduler','DDP','It parallelizes independent experiments rather than distributing one model with DDP.'),
 ('scheduler','Does NAL resume automatically restore optimizer state?',GUIDE,'It does not generally restore a model halfway through training','NAL resume','scope','Reuses successful fingerprint-matched results; does not generally restore a partially trained model or optimizer.'),
 ('science','How should completed artifacts and changed runs be handled?',GUIDE,'Completed raw evidence is\nappend-only','experiment evidence','immutability','Completed raw evidence is append-only; changed code or configuration uses a new artifact root.'),
 ('science','Does completing all seeds imply that a hypothesis passed?',GUIDE,'the hypothesis passed;','campaign completion','interpretation','Successful execution of all seeds does not establish that scientific gates passed.'),
 ('science','May a failed prerequisite be ignored to run a conditional stage?',GUIDE,'Respect decision order.','scientific gates','decision order','A failed prerequisite stops its conditional downstream stage; secondary results cannot rescue the primary gate.'),
 ('tinyllm','How many updates did native-2048 training complete?',TINY,'completed all 12,000 optimizer updates','native-2048 TinyLLM','optimizer updates',12000),
 ('tinyllm','How many target tokens did native-2048 training use?',TINY,'73,728,000 target tokens','native-2048 TinyLLM','training target tokens',73728000),
 ('tinyllm','What happened at the competence gate?',TINY,'8.70%, exceeding the preregistered maximum of 5%','native-2048 TinyLLM','competence gate','Short-context loss degraded 8.70%, exceeding the 5% margin.'),
 ('tinyllm','Was H2 evaluation run after the competence failure?',TINY,'no H2 cell was executed','native-2048 TinyLLM','H2 evaluation','No H2 cell ran; the registered competence gate failed.'),
 ('ncm','What is the latest complete-summary DEV result?',NCM,'0/128 complete-summary semantic','latest recovered NCM','DEV complete-summary accuracy','0/128 (0.0%); the final 320 cases remain sealed.'),
 ('ncm','Which semantic judge graded the completed NCM arm?',NCM,'The judge was `codex_sdk/gpt-5.6-terra`','latest recovered NCM','semantic judge','codex_sdk/gpt-5.6-terra'),
 ('ncm','Were summaries graded by exact reference wording?',NCM,'Paraphrases\nare allowed; exact reference wording is not required','NCM summary grading','paraphrases','Allowed; complete summaries must be faithful, salient, readable and within the 128-token cap.'),
 ('ncm','Was the original-readout comparison fully completed?',NCM,'0/113 valid judged cases','original NCM readout','partial grade','0/113 judged; 15 cases ungraded when the user stopped the remaining comparison.'),
 ('pretrained','Was LM1B pretrained for conditional conversation summaries?',NATIVE,'unconditional language model, with a single\nCLS prefix and a 128-position output canvas','released LM1B','pretraining','Unconditional language generation with a single CLS prefix and 128-position output canvas; not conversation summarization.'),
 ('pretrained','What source/output limits does released XSum use?',NATIVE,'512-position source and a 48-position output canvas','released XSum','native task','Conditional news summarization with a 512-position source and 48-position output canvas.'),
 ('pretrained','What was the released NCM easy-Sudoku result?',NATIVE,'1,988/2,000 = 99.4%','released NCM Sudoku easy','exact correctness','1988/2000 (99.4%)'),
 ('pretrained','What was the released NCM hard-Sudoku result?',NATIVE,'1,726/2,000 = 86.3%','released NCM Sudoku hard','exact correctness','1726/2000 (86.3%)'),
 ('wordgames','How many independent ordinary WORD_GAMES tasks passed?',WG,'The four independent ordinary tasks passed','WORD_GAMES ordinary tasks','joint success','4/4 independent ordinary cases passed.'),
 ('wordgames','What failed in the persistent WORD_GAMES workflow?',WG,'| Persistent workflow, executed prefix | 4/5','WORD_GAMES persistent workflow','joint success','4/5 executed events passed; the receiver omitted the remaining occupied slot at event five.'),
 ('wordgames','Did the WORD_GAMES sender actually use retrieval?',WG,'zero\nmodel-selected retrieval calls','WORD_GAMES sender','retrieval use','Zero model-selected retrieval calls; title-only lookup invented an ID and failed.'),
 ('v3','Does the V3 replay establish fresh model accuracy?',V3,'These are saved-answer replays','V3 mapping review','evidence scope','16 saved sequential states matched; this was replay, not fresh Qwen accuracy.'),
 ('v3','How do the V3 engine and codec relate to V2 and V4?',V3,'source Engine is byte-identical to the active V2 vendor','V3 implementation','identity','The source Engine matches active V2; the generic mapping runtime matches V4.'),
 ('v3','Is the V3 translated view already compliant with our private-note policy?',V3,'A synthetic\nprivate note survives V3','V3 disclosure','integration requirement','V3 preserves private note payloads; apply the stricter WORD_GAMES disclosure policy before integration.'),
 ('golf','What did poker-only golf accomplish?',GOLF,'56/56 optimal actions','golf through poker','task result','15/15 holes, 56/56 optimal actions, and 58 strokes, matching direct golf.'),
 ('golf','What was poker-only golf throughput?',GOLF,'**12.92**','golf through poker','throughput','12.92 visible Qwen-token-equivalents/s over full proxy requests.'),
 ('golf','Did the proxy supply native token counts?',GOLF,'returned zero prompt, completion and total tokens','golf proxy','token measurement','The SDK proxy returned zero usage; visible text was counted with the local Qwen tokenizer.'),
 ('golf','Did that proxy experiment measure local GPU generation speed?',GOLF,"not a measurement of\nthe local three-GPU model's generation speed",'golf proxy','speed scope','Remote proxy measurements do not measure local three-GPU generation speed.'),
]


def build_catalog():
    catalog = {}
    for index,(topic,question,path,anchor,subject,property_,value) in enumerate(SPECS,1):
        text = Path(path).read_text()
        if anchor not in text:
            raise ValueError(f'evidence anchor missing: {index} {path}')
        key = f'F{index:02d}'
        catalog[key] = {'id':key,'topic':topic,'question':question,'path':path,'anchor':anchor,
                        'source_sha256':sha(path),'subject':subject,'property':property_,
                        'value':value,'evidence_line':text[:text.index(anchor)].count('\n')+1}
    return catalog


def menu(catalog):
    return [{k:fact[k] for k in ('id','topic','question','path')} for fact in catalog.values()]


def authorize(proposal, catalog, allowed_topics):
    """Never forward model-authored prose, quotes, paths or unknown strings."""
    if not isinstance(proposal,dict) or set(proposal)-{'citations','unknown','notes'}:
        raise ValueError('proposal schema')
    citations = proposal.get('citations')
    if not isinstance(citations,list) or len(citations)>8:
        raise ValueError('citation count')
    released, receipts, seen = [], [], set()
    for item in citations:
        if not isinstance(item,dict) or set(item)!={'fact_id','source_path','quote'}:
            raise ValueError('citation schema')
        key = item['fact_id']
        fact = catalog.get(key)
        reason = None
        if fact is None or fact['topic'] not in allowed_topics or key in seen:
            reason = 'identity_or_scope'
        elif item['source_path'] != fact['path'] or sha(fact['path']) != fact['source_sha256']:
            reason = 'source_identity'
        else:
            quote = item['quote']
            # Whitespace-normalized literal evidence; no invented paraphrase or
            # source mixing can satisfy the fixed per-fact witness.
            norm = lambda s:' '.join(s.split())
            if (not isinstance(quote,str) or not quote.strip() or len(quote)>1800 or
                norm(quote) not in norm(Path(fact['path']).read_text()) or
                norm(fact['anchor']) not in norm(quote)):
                reason = 'evidence_witness'
        receipts.append({'fact_id':key,'accepted':reason is None,'reason':reason})
        if reason is None:
            seen.add(key)
            released.append({'id':key,'subject':fact['subject'],'property':fact['property'],
                             'value':fact['value'],'evidence':'record:'+fact['source_sha256'][:12],
                             'status':'verified'})
    return released, receipts


KEYS = ('facts','id','subject','property','value','evidence','status','revision','topic','unknown')


def validate_mapping(mapping):
    if not isinstance(mapping,dict) or set(mapping)!={'name','key_map'}:
        raise ValueError('mapping schema')
    if not isinstance(mapping['name'],str) or not 1<=len(mapping['name'])<=60:
        raise ValueError('mapping name')
    keys=mapping['key_map']
    if set(keys)!=set(KEYS) or len(set(keys.values()))!=len(KEYS):
        raise ValueError('mapping bijection')
    if any(not isinstance(v,str) or not v.isidentifier() or not v.isascii() or len(v)>40 for v in keys.values()):
        raise ValueError('mapping keys')
    return mapping


def translate(value,mapping,reverse=False):
    keys=mapping['key_map']
    if reverse:keys={v:k for k,v in keys.items()}
    if isinstance(value,dict):return {keys.get(k,k):translate(v,mapping,reverse) for k,v in value.items()}
    if isinstance(value,list):return [translate(v,mapping,reverse) for v in value]
    return value
