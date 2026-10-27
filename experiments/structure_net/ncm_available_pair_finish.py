"""Finish the known-artifact census in order after the live restorers complete."""
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiments.structure_net import ncm_available_pair_archive as archive

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1')


def wait_result(path,status,unit):
    while not path.exists():
        raw=subprocess.check_output(['systemctl','--user','show',unit,
            '--property=ActiveState,SubState,MainPID,Result,ExecMainStatus'],text=True)
        if 'SubState=running' not in raw and 'SubState=start' not in raw:
            raise ValueError('required producer stopped without completed result: '+unit)
        time.sleep(5)
    value=archive.read(path)
    if value['status']!=status:raise ValueError('required producer has unsuccessful result')
    return archive.sha(path)


def run_stage(name,module,stage):
    command=[shutil.which('pixi'),'run','--locked','-e','ncm','python','-m',module,stage]
    receipt=ROOT/(name+'_process_receipt.json')
    if receipt.exists():raise ValueError('stage already attempted; no blind relaunch')
    print(json.dumps({'starting_stage':name,'command':command}),flush=True)
    started=time.monotonic()
    result=subprocess.run(command)
    archive.write(receipt,{'command':command,'actual_child_exit_code':result.returncode,
        'elapsed_seconds':time.monotonic()-started,'observer':'ncm_available_pair_finish subprocess.run'})
    if result.returncode:raise ValueError('census stage failed: '+name)


def main():
    archive.write(ROOT/'finish_producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'stages':['stores','sources','receipt_index','receipt_join'],
        'new_model_calls':0,'training_admitted':False})
    if archive.read(ROOT/'core_result.json')['status']!='all_core_stores_counted':
        raise ValueError('all500 core stores must be counted first')
    wait_result(archive.ROOT/'restore_result.json','all_indexed_summary_artifacts_restored',
                'structure-net-ncm-available-pair-archive-v1.service')
    run_stage('stores','experiments.structure_net.ncm_available_pair_census_fast','stores')
    run_stage('sources','experiments.structure_net.ncm_available_pair_receipts','sources')
    wait_result(ROOT/'payload_pack_result.json','all_indexed_JSON_payloads_packed',
                'structure-net-ncm-available-pair-pack-v1.service')
    run_stage('receipt_index','experiments.structure_net.ncm_available_pair_receipts','index')
    run_stage('receipt_join','experiments.structure_net.ncm_available_pair_receipts','join')
    archive.write(ROOT/'finish_result.json',{'status':'known_artifact_census_stages_completed',
        'extra_unindexed_payload_resolution_complete':False,'training_admitted':False})


if __name__=='__main__':main()
