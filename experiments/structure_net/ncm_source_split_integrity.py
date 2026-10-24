"""Conservative source-family and text-overlap guards for new NCM splits.

Family IDs join observed session-name variants. They supplement existing group
IDs, exact source hashes and public-turn provenance; they are not a claim that
arbitrary identifier formats expose every related conversation.
"""
import re
import unicodedata


def source_family(source_id):
    for pattern in (r"(sharegpt_.+)_\d+", r"(answer_[0-9a-f]{8})(?:_abs)?(?:_\d+)?", r"([0-9a-f]{8})(?:_\d+)?"):
        match = re.fullmatch(pattern, source_id)
        if match:
            return match.group(1)
    return source_id


def normalized_words(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def signature(row):
    words = normalized_words(row["source"])
    return {"words": words, "grams": {words[i:i + 5] for i in range(max(0, len(words) - 4))},
        "families": {source_family(v) for v in [row["source_id"], *[s["source_id"] for s in row.get("spans", [])]]},
        "turns": {s["turn_text_sha256"] for s in row.get("spans", [])}}


def overlap(a, b, left=None, right=None):
    left, right = signature(a) if left is None else left, signature(b) if right is None else right
    common = len(left["grams"] & right["grams"])
    minimum = min(len(left["grams"]), len(right["grams"]))
    flags = {"same_id": a["id"] == b["id"], "same_source_id": a["source_id"] == b["source_id"],
        "same_group_id": a["group_id"] == b["group_id"], "same_source_hash": a["source_sha256"] == b["source_sha256"],
        "same_source_family": bool(left["families"] & right["families"]),
        "shared_public_turn": bool(left["turns"] & right["turns"]),
        "normalized_text_equal": bool(left["words"]) and left["words"] == right["words"],
        "near_text_overlap": common >= 32 and common * 5 >= minimum * 4}
    if not any(flags.values()):
        return None
    return {**flags, "common_word_5grams": common, "smaller_word_5grams": minimum,
        "shorter_text_containment": common / minimum if minimum else None}


def cross_overlaps(training, heldout):
    left = [(r, signature(r)) for r in training]
    right = [(r, signature(r)) for r in heldout]
    result = []
    for b, sb in right:
        for a, sa in left:
            value = overlap(a, b, sa, sb)
            if value is not None:
                result.append({"train_id": a["id"], "heldout_id": b["id"], **value})
    return result


def require_disjoint(training, heldout):
    conflicts = cross_overlaps(training, heldout)
    if conflicts:
        raise ValueError(f"source family/text/turn overlap in {len(conflicts)} TRAIN/heldout pairs")
