import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="WORD_GAMES two-agent information firewall experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import-bundle")
    p = sub.add_parser("runtime")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("regression")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "import-bundle":
        from .import_bundle import run
        result = run()
    elif args.command == "runtime":
        from .runtime import prepare
        result = prepare(args.output)
    elif args.command == "regression":
        from .regression import run
        result = run(args.output)
    else:
        from .campaign import run
        result = run(args.output, args.identity, pilot=args.pilot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
