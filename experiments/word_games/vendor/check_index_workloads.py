"""Run both preserved seeded workloads with independent index checks after EVERY apply.
Checks do not rebuild or repair indexes. Includes rejected and idempotent calls.
"""
import json,re
from bridge import Engine
from test_bridge import stress
from stress_valid import run
original=Engine.apply
checks=0
def checked(self,*args,**kwargs):
    global checks
    try:return original(self,*args,**kwargs)
    finally:
        locations={g:set() for g in self.galleries};occupancy={};tokens={}
        for lid,loan in self.loans.items():
            loc=loan['reservation']['gallery_id'] if loan['reservation'] else loan['home'];locations[loc].add(lid)
            if loan['reservation']:occupancy[(loc,loan['reservation']['slot'])]=lid
            tokens[lid]=re.findall(r'\w+',(lid+' '+loan['title']).lower())
        assert locations==self._locations
        assert occupancy==self._occupancy
        assert tokens==self._tokens
        checks+=1
Engine.apply=checked
r={'randomized':stress(),'valid_heavy':run(),'index_checks':checks,'assertion_failures':0,'method':'independent recomputation after each apply; no repair'}
with open('index_workloads_report.json','w') as f:json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
