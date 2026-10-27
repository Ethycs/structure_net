"""Run the unchanged pair census with its transactional index on SSD."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_available_pair_census as census

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('core','stores','sources','caches','report'))
    args=parser.parse_args()
    census.ROOT=ROOT;census.STORAGE=ROOT
    path=ROOT/'runtime_producer.json'
    hashes={str(p):census.archive.sha(p) for p in (Path(__file__),Path(census.__file__),ROOT/'migration.json')}
    if path.exists():
        if census.archive.read(path)['source_bindings']!=hashes:
            raise ValueError('SSD census producer or migration binding changed')
    elif args.stage!='report':
        census.archive.write(path,{'source_bindings':hashes,'change':'database storage placement only',
            'source_payload_storage':str(census.archive.STORAGE),'counting_logic_changed':False})
    getattr(census,args.stage)()


if __name__=='__main__':main()
