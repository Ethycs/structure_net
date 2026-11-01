"""Ask the ongoing Claude Code thread through its authorized WORD_GAMES channel."""
import argparse
from pathlib import Path
import json
import time

import requests

from .common import write,sha
from .research_profile import translate,validate_mapping


def ask(url,question,topic,out):
    out.mkdir(parents=True,exist_ok=False)
    http=requests.Session()
    health=http.get(url+'/health',timeout=5);health.raise_for_status()
    state=health.json()
    if not state['ready'] or not state.get('interactive_questions'):
        raise ValueError('an interactive continuation bridge is required')
    response=http.get(url+'/mapping',timeout=5);response.raise_for_status()
    mapping=validate_mapping(response.json())
    body={'query_id':f"Q{state['turns']+1:02d}",'topic':topic,'question':question}
    write(out/'request.json',body)
    write(out/'producer.json',{'source_sha256':sha(__file__),'initial_health':state,'url':url})
    start=time.perf_counter()
    response=http.post(url+'/bridge/query',json=body,timeout=(5,330))
    elapsed=time.perf_counter()-start
    write(out/'response.json',{'status':response.status_code,'seconds':elapsed,'body':response.json()})
    response.raise_for_status()
    packet=translate(response.json(),mapping,reverse=True)
    write(out/'authorized_answer.json',packet)
    print(json.dumps(packet,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18091')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--question');group.add_argument('--question-file',type=Path)
    p.add_argument('--topic',default='all')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ask(a.url,a.question if a.question is not None else a.question_file.read_text(),a.topic,a.output)
