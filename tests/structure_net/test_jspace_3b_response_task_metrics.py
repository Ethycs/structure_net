import json
from pathlib import Path
import sys

import pytest

from experiments.structure_net import jspace_3b_response_task_metrics as m


def test_command_isolation_and_readonly_runtime():
    command = m.sandbox_command()
    assert "--unshare-all" in command and "--clearenv" in command and "--cap-drop" in command
    assert "--ro-bind" in command and "/runtime" in command
    assert "/home/rabbit" not in command and "/data" not in command
    assert "-I" in command and "-S" in command


def test_no_plain_execution_when_bubblewrap_missing(tmp_path):
    result = m.run_mbpp("raise RuntimeError('must not execute')",dict(test_list=["assert True"]), bwrap=str(tmp_path/"missing-bwrap"))
    assert result["status"] == "unavailable" and result["passed"] is None


@pytest.mark.parametrize("text,expected", [("The answer is 1,200.","1.2E+3"),
    ("Work: 3+4=7. #### 7","7"), ("Final answer: 3/4", "0.75"), ("No answer",None)])
def test_numeric_answer_extraction(text, expected):
    assert m.final_number(text) == expected


def test_reference_requires_gsm_final_delimiter():
    assert m.grade_answer("3",dict(family="gsm8k",response="Reasoning mentions 3"))["available"] is False


def test_summaries_never_get_exact_match_grade():
    for family in ("xsum","dolly"):
        result=m.grade_answer("identical text",dict(family=family,response="identical text"))
        assert result["passed"] is None and result["status"]=="semantic_judge_required"


def test_complete_answer_exact_check_is_diagnostic():
    result=m.grade_answer("The CAT.",dict(family="arc",response="the cat"))
    assert result["passed"] and result["diagnostic_only"]


def test_code_fence_extraction_rejects_ambiguous_blocks():
    assert m.extract_code("Here:\n```python\ndef f(): return 1\n```\n") == "def f(): return 1\n"
    with pytest.raises(ValueError):m.extract_code("```python\nx=1\n")


def fixture_records():
    contexts={"math":dict(family="gsm8k",response="2+2=4\n#### 4"),
              "code":dict(family="mbpp",response="def f(): return 1",test_list=["assert f()==1"])}
    records=[dict(example_id=k,input_ids=[1,2],response="4" if k=="math" else "def f(): return 1") for k in contexts]
    return records,contexts


def test_missing_sandbox_fails_gate_but_preserves_numeric_result(tmp_path):
    rows,contexts=fixture_records()
    result=m.grade_generated_tasks(rows,rows,contexts,bwrap=str(tmp_path/"missing"))
    assert not result["passed"] and not result["gates"]["mbpp"]["available"]
    assert result["gates"]["gsm8k"]["passed"]


def test_prompt_mismatch_rejected():
    rows,contexts=fixture_records()
    other=[dict(r,input_ids=[3]) for r in rows]
    with pytest.raises(ValueError,match="prompts differ"):
        m.grade_generated_tasks(rows,other,contexts)


def test_live_sandbox_limits_and_test_verdicts():
    probe=m.sandbox_probe(str(Path(sys.prefix).resolve()))
    if not probe["available"]:
        pytest.skip("Kernel/sandbox does not permit nested bubblewrap; unavailable is fail-closed")
    context=dict(test_list=["assert f(2)==3"])
    assert m.run_mbpp("def f(x): return x+1",context)["passed"]
    assert not m.run_mbpp("def f(x): return x",context)["passed"]
    assert not m.run_mbpp("while True: pass",context,limits=m.SandboxLimits(cpu_seconds=1,wall_seconds=3))["passed"]
    assert not m.run_mbpp("open('/home/rabbit/.ssh/id_rsa').read()",context)["passed"]
