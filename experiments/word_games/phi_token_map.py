"""Byte-exact vocabulary bridges, not interchangeable BPE tokenizers."""
import base64
from dataclasses import dataclass
import gzip
import json
import operator
from pathlib import Path

from .common import canonical, digest, read, sha, write

SCHEMA = "word-games.phi-token-map.v1"
BLOCKED_ATTRIBUTES = 1 | 2 | 8 | 16  # unknown, unused, control, user-defined


@dataclass
class Vocabulary:
    name: str
    pieces: list
    attributes: list
    metadata: dict

    def ordinary(self, i):
        return bool(self.pieces[i]) and bool(self.attributes[i] & (4 | 32)) and not self.attributes[i] & BLOCKED_ATTRIBUTES

    def record(self):
        return {"name": self.name, "pieces": [base64.b64encode(p).decode("ascii") for p in self.pieces],
                "attributes": self.attributes, "metadata": self.metadata}

    @property
    def fingerprint(self):
        r = self.record()
        del r["name"]
        return digest(r)

    @classmethod
    def restore(cls, r):
        pieces = [base64.b64decode(x, validate=True) for x in r["pieces"]]
        if len(pieces) != len(r["attributes"]):
            raise ValueError("vocabulary array length mismatch")
        return cls(r["name"], pieces, r["attributes"], r["metadata"])

    def decode(self, ids):
        values = []
        for token in ids:
            i = operator.index(token)
            if i < 0 or i >= len(self.pieces) or not self.ordinary(i):
                raise ValueError("control, empty, or invalid token in ordinary text")
            values.append(self.pieces[i])
        return b"".join(values)


@dataclass
class TokenMap:
    source: Vocabulary
    destination: Vocabulary
    direct: list
    sequences: list
    kinds: list

    def validate(self):
        n = len(self.source.pieces)
        if any(len(x) != n for x in (self.direct, self.sequences, self.kinds)):
            raise ValueError("map array length mismatch")
        for i, sequence in enumerate(self.sequences):
            if not self.source.ordinary(i):
                if sequence is not None or self.direct[i] != -1 or self.kinds[i] != "blocked":
                    raise ValueError("special tokens must be blocked")
                continue
            if sequence is None or not sequence or self.destination.decode(sequence) != self.source.pieces[i]:
                raise ValueError("token decomposition changes source bytes")
            direct = self.direct[i]
            if direct != -1 and self.destination.decode([direct]) != self.source.pieces[i]:
                raise ValueError("invalid direct token mapping")

    def translate(self, ids, limit=None):
        """Return proposal IDs. Full target verification is REQUIRED at generation.

        Canonical tokenization of adjacent source pieces can merge across their
        boundary. These byte-exact IDs are not a drop-in tokenizer or logit map.
        """
        if limit is not None and limit < 0:
            raise ValueError("negative token limit")
        result = []
        for token in ids:
            i = operator.index(token)
            if i < 0 or i >= len(self.sequences) or self.sequences[i] is None:
                raise ValueError("unmapped/control token; preserve model-specific chat protocol")
            result.extend(self.sequences[i])
            if limit is not None and len(result) >= limit:
                return result[:limit]
        return result

    def record(self):
        return {"schema": SCHEMA, "source_sha256": self.source.fingerprint,
                "destination_sha256": self.destination.fingerprint,
                "direct": self.direct, "sequences": self.sequences, "kinds": self.kinds}

    def summary(self):
        ordinary = sum(self.source.ordinary(i) for i in range(len(self.source.pieces)))
        direct = sum(i >= 0 for i in self.direct)
        return {"source_vocab_size": len(self.source.pieces), "ordinary_tokens": ordinary,
                "blocked_tokens": len(self.source.pieces) - ordinary,
                "exact_byte_counterparts": direct, "exact_byte_counterpart_fraction": direct / ordinary,
                "same_id_counterparts": sum(i == t for i, t in enumerate(self.direct)),
                "single_token_decompositions": sum(x is not None and len(x) == 1 for x in self.sequences),
                "multi_token_decompositions": sum(x is not None and len(x) > 1 for x in self.sequences),
                "max_decomposition_tokens": max(len(x) for x in self.sequences if x is not None),
                "kinds": {k: self.kinds.count(k) for k in sorted(set(self.kinds))}}


def build_map(source, destination, encode):
    by_piece = {}
    for i, piece in enumerate(destination.pieces):
        if destination.ordinary(i):
            by_piece.setdefault(piece, []).append(i)
    byte_ids = {b[0]: ids[0] for b, ids in by_piece.items() if len(b) == 1}
    direct, sequences, kinds = [], [], []
    for i, piece in enumerate(source.pieces):
        if not source.ordinary(i):
            direct.append(-1); sequences.append(None); kinds.append("blocked")
            continue
        matches = by_piece.get(piece, [])
        counterpart = matches[0] if matches else -1
        try:
            piece.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            sequence = [counterpart] if matches else [byte_ids[b] for b in piece]
            kind = "raw_bytes_direct" if matches else "raw_byte_fallback"
        else:
            sequence = list(encode(piece))
            if not sequence or destination.decode(sequence) != piece:
                raise ValueError("native standalone tokenization changed bytes")
            if len(sequence) == 1:
                counterpart = sequence[0]
            kind = "canonical_standalone"
        direct.append(counterpart); sequences.append(sequence); kinds.append(kind)
    result = TokenMap(source, destination, direct, sequences, kinds)
    result.validate()
    return result


def save_gzip(path, record):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(canonical(record).encode("utf-8"))


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class TokenMapBundle:
    def __init__(self, small_to_large, large_to_small):
        self.small_to_large, self.large_to_small = small_to_large, large_to_small

    def save(self, directory, provenance):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        files = {"small_vocab.json.gz": self.small_to_large.source.record(),
                 "large_vocab.json.gz": self.small_to_large.destination.record(),
                 "small_to_large.json.gz": self.small_to_large.record(),
                 "large_to_small.json.gz": self.large_to_small.record()}
        for name, value in files.items():
            save_gzip(directory / name, value)
        write(directory / "manifest.json", {"schema": SCHEMA, "provenance": provenance,
            "files": {name: sha(directory / name) for name in files},
            "small_to_large": self.small_to_large.summary(), "large_to_small": self.large_to_small.summary()})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        m = read(directory / "manifest.json")
        names = {"small_vocab.json.gz", "large_vocab.json.gz", "small_to_large.json.gz", "large_to_small.json.gz"}
        if m["schema"] != SCHEMA or set(m["files"]) != names:
            raise ValueError("unexpected token-map schema/files")
        for name, expected in m["files"].items():
            if sha(directory / name) != expected:
                raise ValueError("token map artifact hash mismatch")
        small = Vocabulary.restore(read_gzip(directory / "small_vocab.json.gz"))
        large = Vocabulary.restore(read_gzip(directory / "large_vocab.json.gz"))
        result = []
        for name, source, target in [("small_to_large", small, large), ("large_to_small", large, small)]:
            r = read_gzip(directory / (name + ".json.gz"))
            if r["schema"] != SCHEMA or r["source_sha256"] != source.fingerprint or r["destination_sha256"] != target.fingerprint:
                raise ValueError("tokenizer fingerprint mismatch")
            mapping = TokenMap(source, target, r["direct"], r["sequences"], r["kinds"])
            mapping.validate()
            result.append(mapping)
        return cls(*result)

    def validate_tokenizers(self, small, large):
        if (small.fingerprint != self.small_to_large.source.fingerprint or
                large.fingerprint != self.small_to_large.destination.fingerprint):
            raise ValueError("runtime tokenizers differ from saved token map")
