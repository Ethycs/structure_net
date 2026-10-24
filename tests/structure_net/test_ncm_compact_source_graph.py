from copy import deepcopy
import random

from experiments.structure_net import ncm_compact_source_graph as run
from experiments.structure_net import ncm_clean_source_graph as old


def row(identity, words, sid=None, blocked=False):
    source = " ".join(words)
    return {"id": identity, "source_id": sid or identity, "group_id": sid or identity,
        "source": source, "source_sha256": old.digest(source), "spans": [], "blocked": blocked}


def canonical(graph):
    return sorted((sorted(r["id"] for r in v["rows"]), v["blocked"]) for v in graph.components().values())


def test_interned_graph_preserves_exact_components_and_transitive_blocking():
    words = [f"word{i}" for i in range(120)]
    rows = [row("known", words, "sharegpt_x_0", True),
        row("contained", words[20:100]), row("alias", ["unrelated", "words"], "sharegpt_x_2"),
        row("normalized", ["UNRELATED", "WORDS"]), row("separate", ["totally", "different"])]
    a, b = old.SourceGraph(), run.CompactSourceGraph()
    for r in rows:
        a.add(deepcopy(r)); b.add(deepcopy(r))
    assert canonical(a) == canonical(b)
    assert not hasattr(b, "signatures")


def test_reference_and_compact_graphs_agree_on_varied_order_and_overlap():
    rng = random.Random(882)
    for _ in range(15):
        base = [f"w{i}" for i in range(150)]
        rows = []
        for i in range(20):
            start = rng.randrange(30)
            words = base[start:start + rng.randrange(40, 120)]
            if rng.randrange(4) == 0:
                words = [f"fresh{i}_{j}" for j in range(60)]
            rows.append(row(str(i), words, "sharegpt_x_" + str(i) if i % 7 == 0 else None, i == 0))
        rng.shuffle(rows)
        a, b = old.SourceGraph(), run.CompactSourceGraph()
        for r in rows:
            a.add(r); b.add(r)
        assert canonical(a) == canonical(b)
