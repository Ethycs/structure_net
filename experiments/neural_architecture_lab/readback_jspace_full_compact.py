"""Fresh-process registry readback with legacy Chroma/NumPy compatibility."""
import json
from pathlib import Path

import numpy as np

from experiments.structure_net import jspace_full_capture as base
from experiments.structure_net.jspace_full_affine_v2 import DEFAULT_ROOT
from experiments.structure_net.jspace_bench_data import sha,write
from .store_jspace_full_compact import REGISTRY,readback


def main():
    base.cpu_only()
    # The locked default environment has NumPy 2 and a Chroma consumer that still
    # refers to np.NaN. This alias is process-local; no package or DB is patched.
    compatibility="NaN" not in vars(np)
    if compatibility:np.NaN=np.nan
    result=readback(DEFAULT_ROOT,REGISTRY,base.ROOT/"data/chroma_db")
    result.update(numpy_nan_compatibility_alias=compatibility,
                  reader_source_sha256=sha(Path(__file__)))
    write(DEFAULT_ROOT/"registry_readback.json",result)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":main()
