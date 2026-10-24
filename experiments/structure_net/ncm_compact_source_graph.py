"""Exact source components with interned words/5-grams and sparse postings.

IDs use dictionary equality, not probabilistic fingerprints. Prior gram sets
are unnecessary: postings supply intersections and only their sizes are kept.
Pairs already joined by aliases need no additional overlap edge.
"""
from collections import Counter

from experiments.structure_net import ncm_clean_source_graph as reference


class CompactSourceGraph(reference.SourceGraph):
    def __init__(self):
        self.parent, self.rows = [], []
        self.aliases, self.words, self.grams = {}, {}, {}
        self.postings, self.gram_sizes = [], []
        self.edges = Counter()

    def add(self, row):
        i = len(self.rows)
        self.parent.append(i)
        self.rows.append(row)
        words = tuple(self.words.setdefault(w, len(self.words)) for w in reference.guard.normalized_words(row["source"]))
        gram_ids = set()
        for j in range(max(0, len(words) - 4)):
            gram = words[j:j + 5]
            identifier = self.grams.get(gram)
            if identifier is None:
                identifier = len(self.grams)
                self.grams[gram] = identifier
                self.postings.append(None)
            gram_ids.add(identifier)
        self.gram_sizes.append(len(gram_ids))
        families = set(row.get("families", ())) | {reference.guard.source_family(v) for v in
            [row["source_id"], *[s["source_id"] for s in row.get("spans", [])]]}
        aliases = [("group", v) for v in row.get("groups", [row["group_id"]])]
        aliases += [("family", v) for v in families]
        aliases += [("turn", s["turn_text_sha256"]) for s in row.get("spans", [])]
        aliases += [("source_hash", row["source_sha256"])]
        if words:
            aliases.append(("words", words))
        for key in aliases:
            if key in self.aliases:
                self.union(i, self.aliases[key])
                self.edges[key[0]] += 1
            else:
                self.aliases[key] = i
        root = self.find(i)
        external, counts = {}, Counter()
        for identifier in gram_ids:
            posting = self.postings[identifier]
            for j in (() if posting is None else (posting,) if isinstance(posting, int) else posting):
                outside = external.get(j)
                if outside is None:
                    outside = self.find(j) != root
                    external[j] = outside
                if outside:
                    counts[j] += 1
        for j, common in counts.items():
            minimum = min(len(gram_ids), self.gram_sizes[j])
            if common >= 32 and common * 5 >= minimum * 4:
                self.union(i, j)
                self.edges["cross_component_near_text_edge"] += 1
        for identifier in gram_ids:
            posting = self.postings[identifier]
            if posting is None:
                self.postings[identifier] = i
            elif isinstance(posting, int):
                self.postings[identifier] = [posting, i]
            else:
                posting.append(i)
        return i
