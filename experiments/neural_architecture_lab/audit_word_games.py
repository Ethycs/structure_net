"""Read back raw two-agent traffic, replay transactions and verify artifact bindings.

This is a structural audit, not a semantic judge of metaphor suitability.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.word_games.agents import receiver_prompt
from experiments.word_games.common import canonical, digest, read, sha, write
from experiments.word_games.firewall import exchange
from experiments.word_games.metaphor import Metaphor, PUBLIC_TASK
from experiments.word_games.scenarios import setup


def audit(root):
    root = Path(root)
    campaign_path = root / "campaign_results.json"
    c, p = read(campaign_path), read(root / "producer.json")
    if p.pop("fingerprint") != digest(p) or digest(p) != c["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    if Counter(sha(f) for f in (root / "producer_source").iterdir()) != Counter(p["source_hashes"].values()):
        raise ValueError("source snapshot mismatch")
    for path, expected in p["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError("audit must use the producing implementation: " + path)
    for path, expected in c["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("case artifact mismatch")
    selection_prompt = read(root / "metaphor/request.json")["body"]["inputs"][0]["data"][0]
    if selection_prompt != PUBLIC_TASK:
        raise ValueError("selector saw more than the public task")
    metaphor = Metaphor(**c["metaphor"])
    rows = []
    for case in p["cases"]:
        path = root / "cases" / case["id"]
        packet, result = read(path / "receiver_packet.json"), read(path / "result.json")
        prompt = read(path / "receiver/request.json")["body"]["inputs"][0]["data"][0]
        if prompt != receiver_prompt(packet):
            raise ValueError("receiver input contains material outside the approved packet")
        e, _ = setup(case)
        context = read(path / "context.json")
        cid = context["context_id"]
        e.contexts = {cid: {k: v for k, v in context.items() if k != "context_id"}}
        e.rebuild_indexes()
        if e.snapshot() != read(path / "before.json"):
            raise ValueError("initial state mismatch")
        replay, receipt = exchange(e, read(path / "proposal.json"), cid, metaphor)
        if replay != packet or receipt != read(path / "private_receipt.json") or e.snapshot() != read(path / "after.json"):
            raise ValueError("transaction replay mismatch")
        wire = json.loads(read(path / "receiver/response.json")["body"])
        text = next(o["data"][0] for o in wire["outputs"] if o["name"] == "text_output")
        try: response = json.loads(text)
        except ValueError: response = None
        if response != result["receiver_response"]:
            raise ValueError("recorded receiver response differs from raw model output")
        leak = case["canary"] in prompt or case["canary"] in text or any(x in canonical(packet) for x in ("loan:", "gal:", "ctx:"))
        rows.append({"id": case["id"], "exact_replay": True, "receiver_only_authorized_packet": True,
                     "disclosure_failure": leak})
    metrics = {k: sum(read(root / "cases" / x["id"] / "result.json")[k] for x in rows) for k in c["metrics"]}
    if metrics != c["metrics"] or len(rows) != c["completed"]:
        raise ValueError("aggregate mismatch")
    usage = Counter()
    calls = 0
    for path in root.rglob("response.json"):
        response = read(path)
        wire = json.loads(response["body"])
        if response["http_status"] != 200 or wire["model_name"] != c["model_identity"]["model"]:
            raise ValueError("model or transport mismatch")
        info = json.loads(next(o["data"][0] for o in wire["outputs"] if o["name"] == "generation_info"))
        usage.update(info["usage"])
        calls += 1
    bindings = {str(f): sha(f) for f in root.rglob("*") if f.is_file()}
    result = {"status": "readback_verified", "campaign_sha256": sha(campaign_path),
              "cases": rows, "cases_with_disclosure_failure": sum(r["disclosure_failure"] for r in rows),
              "model_calls": calls, "token_usage_including_selection": dict(usage),
              "semantic_judging": False, "artifacts": bindings}
    write(root / "readback_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "artifacts")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
