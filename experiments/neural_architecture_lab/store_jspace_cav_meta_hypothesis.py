"""Register an exploratory J-space pilot after its final report is available."""
import argparse
import json
from pathlib import Path

from neural_architecture_lab.jspace_cav_meta_hypothesis import store_jspace_cav_meta_hypothesis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    args = parser.parse_args()
    record = store_jspace_cav_meta_hypothesis(args.results, args.report, args.output,
                                           chromadb_path=args.chromadb_path)
    print(json.dumps(record["storage"]["readback"], indent=2))


if __name__ == "__main__":
    main()
