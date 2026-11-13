"""Register a completed exploratory relation study after its final report exists."""
import argparse
import json
from pathlib import Path

from neural_architecture_lab.jspace_relation_meta_hypothesis import (
    build_jspace_relation_meta_hypothesis, store_jspace_relation_meta_hypothesis,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--validate-only", action="store_true", help="Validate archived evidence without opening Chroma or writing an output")
    args = parser.parse_args(argv)
    if args.validate_only:
        record = build_jspace_relation_meta_hypothesis(args.results, args.report)
        response = {"status": "validated_without_storage", "hypothesis_id": record["hypothesis"]["id"],
                    "classification": record["result"]["classification"], "confirmed": False}
    else:
        if args.output is None:
            parser.error("--output is required unless --validate-only is set")
        record = store_jspace_relation_meta_hypothesis(args.results, args.report, args.output,
                                                     chromadb_path=args.chromadb_path)
        response = record["storage"]["readback"]
    print(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
