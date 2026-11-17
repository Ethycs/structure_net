
import pytest

from experiments.structure_net import jspace_3b_recovery_study_eval as m


def plan():
    return dict(grader_identity="pinned-grader", spec_snapshot=dict(native_model_revision="a"*40),
        decoding=dict(max_new_tokens=4, temperature=0., do_sample=False, eos_token_ids=[9]),
        family_weights={family:.25 for family in m.corpus.FAMILIES}, alpha=.05, margin=.05,
        required_pairs=m.corpus.sample_size(), confirmation=dict(sample_size_ready=True))


def record():
    return m.data.ResponseRecord(example_id="id", group_id="group", split="development", task="benchmark:gsm8k",
        prompt_ids=(1, 2), response_ids=(3, 9), finish_reason="eos", eos_token_id=9,
        provenance=dict(source_uri="https://public.example", source_revision="a"*40,
            tokenizer_sha256="tokenizer", chat_template_sha256="template", prompt_provenance="source",
            response_provenance="reference", assistant_boundary_verified=True))


def response(**changes):
    value = dict(example_id="id", group_id="group", task="benchmark:gsm8k", checkpoint="a"*40,
        role="native", response="Answer", output_ids=[3, 9], input_ids=[1, 2], stop_reason="eos",
        decoding=plan()["decoding"], grader_identity="pinned-grader")
    value.update(changes)
    return value


def evidence(family="math", group="g", native=True, student=True, status=None):
    def grade(passed):
        return dict(status=status or ("correct" if passed else "incorrect"), available=status is None,
                    passed=passed if status is None else None, useful_completed=passed if status is None else False,
                    natural_eos=True)
    return dict(category=family, group_id=group, native=grade(native), student=grade(student))


def test_actual_math_success_not_natural_termination_proxy():
    n, s = m.grade_pair(dict(response="4"), dict(response="5"), dict(family="gsm8k", response="2+2=4\n#### 4"))
    assert n["status"] == "correct" and n["passed"]
    assert s["status"] == "incorrect" and not s["passed"]


def test_unsupported_missing_reference_and_wrong_answer_are_distinct():
    unknown, _ = m.grade_pair(dict(response="x"), dict(response="x"), dict(family="unknown"))
    missing, _ = m.grade_pair(dict(response="4"), dict(response="4"), dict(family="gsm8k", response="no final delimiter"))
    incorrect, _ = m.grade_pair(dict(response="5"), dict(response="5"), dict(family="gsm8k", response="#### 4"))
    assert (unknown["status"], missing["status"], incorrect["status"]) == ("unsupported", "unavailable", "incorrect")
    assert unknown["passed"] is missing["passed"] is None and incorrect["passed"] is False


def test_apps_test_payload_failure_not_candidate_incorrect():
    value = m.grade_code("print(1)", dict(family="apps", apps_tests=dict(inputs=[""], outputs=[])))
    assert value["status"] == "unavailable" and value["passed"] is None
    unsupported = m.grade_code("print(1)", dict(family="apps", apps_tests=dict(inputs=[12], outputs=[12])))
    assert unsupported["status"] == "unsupported" and unsupported["passed"] is None


def test_generated_code_never_executes_in_host_process(tmp_path):
    marker = tmp_path/"unsafe"
    malicious = f"open({str(marker)!r}, 'w').write('escaped')"
    context = dict(family="apps", apps_tests=dict(inputs=[""], outputs=[""]))
    harness, assertions = m.apps_program(malicious, context)
    assert malicious in harness and assertions
    assert not marker.exists()
    value = m.grade_code(malicious, context, bwrap=str(tmp_path/"missing-bubblewrap"))
    assert value["status"] == "unavailable" and value["passed"] is None
    assert not marker.exists()


def test_pair_identity_requires_exact_template_prompt_and_decoder():
    m.validate_response(response(), record(), plan())
    for changed in (dict(input_ids=[1, 8]), dict(group_id="wrong"), dict(task="wrong"),
                    dict(grader_identity="different"), dict(checkpoint="unversioned-model")):
        with pytest.raises(ValueError):
            m.validate_response(response(**changed), record(), plan())
    changed = response()
    changed["decoding"] = dict(changed["decoding"], temperature=.8)
    with pytest.raises(ValueError, match="Unpinned"):
        m.validate_response(changed, record(), plan())


def test_length_stop_never_implies_invented_eos():
    m.validate_response(response(output_ids=[3, 4, 5, 6], stop_reason="length"), record(), plan())
    with pytest.raises(ValueError, match="Length termination"):
        m.validate_response(response(output_ids=[3, 4], stop_reason="length"), record(), plan())
    with pytest.raises(ValueError, match="Length termination"):
        m.validate_response(response(output_ids=[3, 4, 5, 9], stop_reason="length"), record(), plan())
    with pytest.raises(ValueError, match="EOS"):
        m.validate_response(response(output_ids=[3, 4], stop_reason="eos"), record(), plan())
    with pytest.raises(ValueError, match="continues after EOS"):
        m.validate_response(response(output_ids=[9, 3]), record(), plan())


def test_confirmation_outputs_require_prospective_selection_binding():
    with pytest.raises(ValueError, match="prospective selection"):
        m.validate_response(response(), record(), plan(), selection_sha256="sealed")
    m.validate_response(response(selection_sha256="sealed"), record(), plan(), selection_sha256="sealed")


def test_weighted_intervals_use_source_groups_and_not_sixteen_case_certificate():
    values = {family:[0.]*4 for family in m.corpus.FAMILIES}
    interval = m.paired_interval(values, plan()["family_weights"])
    assert interval["low"] < -.05 < interval["high"]
    rows = [evidence(family=family, group=f"{family}-{i}") for family in m.corpus.FAMILIES for i in range(4)]
    summary = m.summarize(rows, plan(), "confirmation")
    assert summary["verdict"] == "inconclusive"
    assert not summary["planned_sample_size_met"]
    duplicated_group_rows = [evidence(family=family, group=family) for family in m.corpus.FAMILIES for _ in range(100)]
    assert m.summarize(duplicated_group_rows, plan(), "confirmation")["task_accuracy_difference"]["groups"] == 4


def test_powered_complete_paired_endpoint_and_dev_remain_distinct():
    rows = [evidence(family=family, group=f"{family}-{i}") for family in m.corpus.FAMILIES for i in range(800)]
    summary = m.summarize(rows, plan(), "confirmation")
    assert summary["noninferiority_eligible"] and summary["verdict"] == "noninferior"
    assert m.summarize(rows, plan(), "development")["verdict"] == "inconclusive"
    rows[0] = evidence(status="unavailable")
    incomplete = m.summarize(rows, plan(), "confirmation")
    assert not incomplete["complete_endpoint"] and incomplete["verdict"] == "inconclusive"
    assert incomplete["statuses"]["native"]["unavailable"] == 1


def test_teacher_binding_includes_context_and_both_outputs():
    native, student = response(), response(role="student", checkpoint="b"*64)
    for value in (native, student):
        value.update(max_new_tokens=4, eos_token_ids=[9])
    context = dict(prompt="Do the task", source="Fact 7", reference="7")
    packet, binding = m.semantic.pair_packet(native, student, context=context)
    _, changed = m.semantic.pair_packet(native, student, context=dict(context, source="Fact 8"))
    assert binding["context"] != changed["context"]
    assert binding["native"] == m.data.digest(native)
    assert binding["student"] == m.data.digest(student)
    assert {candidate["candidate_id"] for candidate in packet["candidates"]} == {"A", "B"}
    assert all("role" not in candidate for candidate in packet["candidates"])


def test_live_existing_sandbox_apps_stdin_and_callable():
    probe = m.tasks.sandbox_probe()
    if not probe["available"]:
        pytest.skip("Existing fail-closed sandbox unavailable in this kernel")
    context = dict(family="apps", apps_tests=dict(inputs=["2 3\n"], outputs=["5\n"]))
    assert m.grade_code("a,b=map(int,input().split()); print(a+b)", context)["passed"]
    assert m.grade_code("import sys\na,b=map(int,sys.stdin.buffer.read().split()); sys.stdout.buffer.write(str(a+b).encode())", context)["passed"]
    assert not m.grade_code("print(9)", context)["passed"]
    context = dict(family="apps", apps_tests=dict(fn_name="add", inputs=[[2, 3]], outputs=[5]))
    assert m.grade_code("class Solution:\n def add(self,a,b): return a+b", context)["passed"]
    assert not m.grade_code("class Solution:\n def add(self,a,b): return a-b", context)["passed"]


def test_check_prepared_only_then_detects_consumed_payload_change(tmp_path):
    payload = tmp_path/"records.jsonl"
    payload.write_text("prepared payload\n")
    manifest_path = tmp_path/"manifest.json"
    m.corpus.save(manifest_path, dict(sha256={str(payload):m.corpus.sha(payload)}))
    spec = tmp_path/"spec.json"
    spec.write_text("{}")
    root = tmp_path/"evaluation"
    root.mkdir()
    frozen = dict(spec_path=str(spec), spec_sha256=m.corpus.sha(spec), implementation_sha256={},
                  corpus_manifest=str(manifest_path), corpus_manifest_sha256=m.corpus.sha(manifest_path))
    frozen["fingerprint"] = m.data.digest(frozen)
    m.corpus.save(root/"plan.json", frozen)
    assert m.check(root)["completed"] == {}
    payload.write_text("altered")
    with pytest.raises(ValueError, match="Artifact changed"):
        m.check(root)
