"""New compatibility receipts for the supplied historical backend and trials."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import re
import unittest

from .backend import vendor
from .common import PACKAGE, read, sha, write
from .import_bundle import INPUTS, V1, V2


def replay_v1():
    b = vendor()
    root = INPUTS / V1 / "stress_test"
    scenario, prior, result = None, {}, []
    for case in read(root / "cases.json"):
        if scenario != case["scenario"]:
            engine, scenario, prior = b.Engine.fixture(), case["scenario"], {}
        saved = root / "run_v1" / case["id"]
        ctx = read(saved / "session.json")["context"]
        engine.contexts[ctx["context_id"]] = {k: v for k, v in ctx.items() if k != "context_id"}
        engine.rebuild_indexes()
        admin = case.get("admin") or {}
        if "after_context_rule_patch" in admin:
            engine.update_rule(*admin["after_context_rule_patch"])
        assert engine.snapshot() == read(saved / "before.json")
        p = read(saved / "response.json")["proposal"]
        bound = prior[admin["replay"]]["context_id"] if "replay" in admin else ctx["context_id"]
        try:
            receipt = engine.apply(p, expected_context_id=bound)
        except b.Rejected as exc:
            receipt = {"status": "rejected", "reason": str(exc)}
        assert receipt == read(saved / "result.json")["result"]
        assert engine.snapshot() == read(saved / "after.json")
        if receipt["status"] == "committed":
            prior[case["id"]] = p
        result.append({"id": case["id"], "exact_replay": True})
    return result


def replay_v2():
    b = vendor()
    root = INPUTS / V2 / "long_workflow"
    for name, expected in read(root / "FREEZE.json").items():
        assert sha(INPUTS / V2 / name) == expected, name
    engine = b.Engine.fixture()
    for name, value in read(root / "fixture.json").items():
        setattr(engine, name, value)
    engine.rebuild_indexes()
    result = []
    for case in read(root / "cases.json"):
        saved = root / "run" / case["id"]
        admin = case.get("admin") or {}
        if "revoke" in admin:
            engine.approvals.pop(admin["revoke"])
            engine.rebuild_indexes()
        ctx = read(saved / "session.json")["context"]
        engine.contexts[ctx["context_id"]] = {k: v for k, v in ctx.items() if k != "context_id"}
        engine.rebuild_indexes()
        if "after_context_rule_patch" in admin:
            engine.update_rule(*admin["after_context_rule_patch"])
        assert engine.snapshot() == read(saved / "before.json")
        try:
            receipt = engine.apply(read(saved / "response.json")["proposal"], expected_context_id=ctx["context_id"])
        except b.Rejected as exc:
            receipt = {"status": "rejected", "reason": str(exc)}
        assert receipt == read(saved / "result.json")["result"]
        assert engine.snapshot() == read(saved / "after.json")
        result.append({"id": case["id"], "exact_replay": True})
    return result


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    b = vendor()
    logs = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module("test_v2"))
    with contextlib.redirect_stdout(logs):
        units = unittest.TextTestRunner(stream=logs, verbosity=2).run(suite)
        if not units.wasSuccessful():
            (out / "tests.log").write_text(logs.getvalue())
            raise RuntimeError("upstream backend unit failure")
        original, checks = b.Engine.apply, [0]

        def checked(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            finally:
                locations = {g: set() for g in self.galleries}
                occupancy, tokens = {}, {}
                for lid, item in self.loans.items():
                    locations[self.location(item)].add(lid)
                    if item["reservation"]:
                        r = item["reservation"]
                        occupancy[(r["gallery_id"], r["slot"])] = lid
                    tokens[lid] = re.findall(r"\w+", (lid + " " + item["title"]).lower())
                assert (locations, occupancy, tokens) == (self._locations, self._occupancy, self._tokens)
                checks[0] += 1

        b.Engine.apply = checked
        try:
            mixed = importlib.import_module("test_bridge").stress()
            valid = importlib.import_module("stress_valid").run()
        finally:
            b.Engine.apply = original
        v1, v2 = replay_v1(), replay_v2()
    (out / "tests.log").write_text(logs.getvalue())
    result = {"status": "compatibility_passed", "fresh_model_calls": 0, "unit_tests": units.testsRun,
              "mixed": mixed, "valid_heavy": valid, "independent_index_checks": checks[0],
              "v1_archived_replays": v1, "v2_archived_replays": v2,
              "upstream_provenance": read(PACKAGE / "vendor/provenance.json")}
    write(out / "result.json", result)
    return {"status": result["status"], "unit_tests": units.testsRun,
            "index_checks": checks[0], "archived_replays": len(v1) + len(v2)}
