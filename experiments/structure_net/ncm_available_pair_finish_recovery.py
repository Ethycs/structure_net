"""Resume the not-yet-started join stages with an absolute Pixi executable."""
import json
from pathlib import Path
import subprocess
import time

from experiments.structure_net import ncm_available_pair_finish as original

ROOT,archive=original.ROOT,original.archive
PIXI=Path('/home/rabbit/.pixi/bin/pixi')


def main():
    state=subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-finish-v1.service',
        '--property=ExecMainPID,ExecMainStatus,InvocationID,ActiveState'],text=True)
    if ('ExecMainPID=1424405' not in state or 'ExecMainStatus=1' not in state
            or 'InvocationID=62938333b5164c478e6c518d5fbeb041' not in state
            or (ROOT/'stores_producer.json').exists() or not PIXI.is_file()):
        raise ValueError('exact pre-stage launch failure and untouched store stage required')
    archive.write(ROOT/'finish_recovery_producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'previous_failure_state':state,'previous_log_sha256':archive.sha(ROOT/'finish.log'),
        'change':'absolute Pixi path; no counting/data/split/model change','pixi':str(PIXI)})
    stages=[('stores','ncm_available_pair_census_fast','stores'),('sources','ncm_available_pair_receipts','sources'),
            ('receipt_index','ncm_available_pair_receipts','index'),('receipt_join','ncm_available_pair_receipts','join')]
    for name,module,action in stages:
        if name=='receipt_index':
            original.wait_result(ROOT/'payload_pack_result.json','all_indexed_JSON_payloads_packed',
                                 'structure-net-ncm-available-pair-pack-v1.service')
        command=[str(PIXI),'run','--locked','-e','ncm','python','-m','experiments.structure_net.'+module,action]
        print(json.dumps({'starting_stage':name,'command':command}),flush=True)
        started=time.monotonic();result=subprocess.run(command)
        archive.write(ROOT/(name+'_process_receipt.json'),{'command':command,'actual_child_exit_code':result.returncode,
            'elapsed_seconds':time.monotonic()-started,'observer':'ncm_available_pair_finish_recovery subprocess.run'})
        if result.returncode:raise ValueError('census stage failed: '+name)
    archive.write(ROOT/'finish_result.json',{'status':'known_artifact_census_stages_completed',
        'extra_unindexed_payload_resolution_complete':False,'training_admitted':False})


if __name__=='__main__':main()
