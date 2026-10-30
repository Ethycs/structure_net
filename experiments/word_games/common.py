import hashlib
import json
from pathlib import Path

PROJECT = "WORD_GAMES"
SCHEMA = "word-games.information-firewall.v1"
HYPOTHESIS = "word-games-phi4-authorized-view-v1"
ROOT = Path("data/experiments/word_games")
PACKAGE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())
