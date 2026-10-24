"""Define bounded input fragments from complete contiguous public exchanges.

Boundaries precede model use. Every constituent turn stays literal and complete;
this module does not truncate model inputs or repair generated outputs.
"""
from experiments.structure_net.ncm_memory_data import text_hash
from experiments.structure_net.ncm_source_split_integrity import source_family

BANDS = ((512, 767), (768, 1023), (1024, 1535), (1536, 2048))


def exchange_boundaries(session):
    if not session:
        return []
    return [0, *[i for i, turn in enumerate(session) if i and turn["role"] == "user"], len(session)]


def windows(source_id, session, encoding, namespace="ncm-public-exchange-input-v2"):
    """First complete window per band, with fixed start/end traversal order."""
    boundaries = exchange_boundaries(session)
    result = {}
    for start_index, start in enumerate(boundaries[:-1]):
        for end in boundaries[start_index + 1:]:
            turns = session[start:end]
            if any(not turn["content"] for turn in turns):
                continue
            text = "\n".join(turn["content"] for turn in turns)
            count = len(encoding.encode(text, disallowed_special=()))
            if count > BANDS[-1][1]:
                break
            band = next((i + 1 for i, (low, high) in enumerate(BANDS) if low <= count <= high), None)
            if band is None or band in result:
                continue
            spans = []
            for turn in turns:
                sha = text_hash(turn["content"])
                spans.append({"source_id": source_id, "role": turn["role"],
                    "turn_id": "public:" + text_hash(source_id + "|" + turn["role"] + "|" + sha),
                    "turn_text_sha256": sha, "span_text_sha256": sha,
                    "start_char": 0, "end_char": len(turn["content"])})
            sha = text_hash(text)
            result[band] = {"id": text_hash(namespace + "|" + source_id + "|" + sha),
                "source_id": source_id, "group_id": source_family(source_id), "source_sha256": sha,
                "source": text, "spans": spans, "source_cl100k_tokens": count, "bin": band,
                "exchange_start_turn": start, "exchange_end_turn": end, "source_truncated": False}
            if len(result) == len(BANDS):
                return list(result.values())
    return list(result.values())


def comparison_view(row, full_turn_tokens):
    """Generic bypass turns cannot join unrelated families through greetings.

The full source text and source family remain in the guard. Only shared-turn
hash aliases of turns below the application's192-token threshold are omitted.
Literal provenance always uses the original complete span list.
"""
    if any(s["turn_text_sha256"] not in full_turn_tokens for s in row["spans"]):
        raise ValueError("complete public full-turn token inventory required")
    return {**row, "spans": [s for s in row["spans"] if full_turn_tokens[s["turn_text_sha256"]] >= 192]}
