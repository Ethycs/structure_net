"""StructureNet's reservation-controller CLI, runnable without ML dependencies."""

import argparse
import json
import sys

from .client import ControllerClient, ReservationError, allocation_map
from .launcher import run_reserved


def parse_allocation(value):
    result = {}
    try:
        for item in value.split(","):
            gpu, mib = item.split("=")
            if gpu in result:
                raise ValueError("duplicate GPU")
            result[gpu] = int(mib)
        return allocation_map(result)
    except (ValueError, ReservationError) as exc:
        raise argparse.ArgumentTypeError("Use GPU-<full UUID>=<MiB>,... with unique cards") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("health", "status", "registry"):
        commands.add_parser(action)
    for action in ("validate", "register", "remove", "load", "unload"):
        cmd = commands.add_parser(action)
        cmd.add_argument("model")
        if action in ("register", "remove"):
            cmd.add_argument("--reason", required=True)
    release = commands.add_parser("release")
    release.add_argument("lease_id")
    run = commands.add_parser("run")
    run.add_argument("--allocation", type=parse_allocation, required=True)
    run.add_argument("--duration", type=int, required=True, help="Lease window in seconds, including startup")
    run.add_argument("--reason", required=True)
    run.add_argument("--contact", required=True)
    run.add_argument("--receipt", required=True, help="New JSONL file; records lease lifecycle, never credentials")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    client = ControllerClient()
    try:
        if args.action == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            return run_reserved(command, allocation=args.allocation, duration=args.duration,
                                reason=args.reason, contact=args.contact, receipt=args.receipt, client=client)
        if args.action in ("health", "status", "registry"):
            result = getattr(client, args.action)()
        elif args.action == "validate":
            result = client.validate(args.model)
        elif args.action in ("register", "remove"):
            result = getattr(client, args.action)(args.model, args.reason)
        elif args.action == "release":
            result = client.release(args.lease_id)
        else:
            result = client.model_action(args.model, args.action)
        print(json.dumps(result, indent=2))
        if args.action == "validate" and result.get("result") != "pass":
            return 1
        return 0
    except (ReservationError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
