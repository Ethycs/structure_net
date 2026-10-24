"""Deterministic components and quota matching for public NCM source splits."""
from collections import Counter, defaultdict
import hashlib

from experiments.structure_net import ncm_source_split_integrity as guard


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def bucket(tokens):
    return next((i for i, (lo, hi) in enumerate(((192, 511), (512, 767),
        (768, 1023), (1024, 1535), (1536, 2048))) if lo <= tokens <= hi), None)


class SourceGraph:
    """Exact inverted 5-gram comparisons; aliases and overlap join components."""
    def __init__(self):
        self.parent = []
        self.rows = []
        self.signatures = []
        self.aliases = {}
        self.postings = defaultdict(list)
        self.edges = Counter()

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def add(self, row):
        i = len(self.rows)
        self.parent.append(i)
        self.rows.append(row)
        sig = guard.signature(row)
        self.signatures.append(sig)
        aliases = [("group", x) for x in row.get("groups", [row["group_id"]])]
        aliases += [("family", x) for x in (set(row.get("families", ())) | sig["families"])]
        aliases += [("turn", x) for x in sig["turns"]]
        aliases += [("source_hash", row["source_sha256"])]
        aliases += [("words", sig["words"])] if sig["words"] else []
        for key in aliases:
            if key in self.aliases:
                self.union(i, self.aliases[key])
                self.edges[key[0]] += 1
            else:
                self.aliases[key] = i
        counts = Counter()
        for gram in sig["grams"]:
            counts.update(self.postings.get(gram, ()))
        for j, common in counts.items():
            minimum = min(len(sig["grams"]), len(self.signatures[j]["grams"]))
            if common >= 32 and common * 5 >= minimum * 4:
                self.union(i, j)
                self.edges["near_text_overlap"] += 1
        for gram in sig["grams"]:
            self.postings[gram].append(i)
        return i

    def components(self):
        groups = defaultdict(list)
        for i, row in enumerate(self.rows):
            groups[self.find(i)].append(row)
        result = {}
        for values in groups.values():
            identifier = min(r["source_sha256"] for r in values)
            result[identifier] = {"rows": values, "blocked": any(r["blocked"] for r in values)}
        return result


def assign_components(components, quotas, namespace):
    """Augmenting paths handle components that can fill several length bands."""
    choices = {b: sorted((key for key, value in components.items() if not value["blocked"]
            and any(r.get("candidate") and r["bin"] == b for r in value["rows"])),
            key=lambda key: (digest(namespace + ":component:" + key), key)) for b in quotas}
    owner = {}

    def take(b, seen_bands, seen_groups):
        if b in seen_bands:
            return False
        seen_bands.add(b)
        for key in choices[b]:
            if key in seen_groups:
                continue
            seen_groups.add(key)
            previous = owner.get(key)
            if previous is None or take(previous, seen_bands, seen_groups):
                owner[key] = b
                return True
        return False

    for b in sorted(quotas, key=lambda b: (len(choices[b]) / quotas[b], b)):
        for _ in range(quotas[b]):
            if not take(b, set(), set()):
                raise ValueError("insufficient disjoint source components for fixed quotas: "
                    + str({k: {"available": len(v), "required": quotas[k]} for k, v in choices.items()}))
    if Counter(owner.values()) != Counter(quotas):
        raise ValueError("complete component quotas required")
    selected = {}
    for key, b in owner.items():
        candidates = [r for r in components[key]["rows"] if r.get("candidate") and r["bin"] == b]
        selected[key] = min(candidates, key=lambda r: (digest(namespace + ":row:" + r["id"]), r["id"]))
    return selected, {b: len(v) for b, v in choices.items()}


def split_assignment(selected, final_quotas, development_quotas, namespace):
    result = {s: [] for s in ("final_short", "final_long", "development_short", "development_long")}
    for b in final_quotas:
        values = sorted(((key, row) for key, row in selected.items() if row["bin"] == b),
            key=lambda x: (digest(namespace + ":cohort:" + x[0]), x[0]))
        if len(values) != final_quotas[b] + development_quotas[b]:
            raise ValueError("full final and development quotas required before splitting")
        for i, (key, row) in enumerate(values):
            split = ("final" if i < final_quotas[b] else "development") + ("_short" if b == 0 else "_long")
            result[split].append({**row, "source_component": key, "split": split})
    return result
