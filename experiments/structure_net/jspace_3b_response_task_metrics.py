"""Small paired generated-answer checks, separate from semantic judging.

MBPP code runs only inside bubblewrap with fresh namespaces, no network, no
home/data mounts, a cleared environment and bounded CPU/memory/files/output.
An unavailable sandbox never falls back to ordinary execution or passes a gate.
These tiny response-task screens are not official benchmark accuracy estimates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 2
    wall_seconds: float = 5.
    memory_bytes: int = 256 * 1024**2
    output_bytes: int = 64 * 1024
    source_bytes: int = 64 * 1024


# Executed by a fresh isolated Python interpreter. No caller globals/credentials
# are available. Test setup, generated code and dataset assertions share their
# own namespace, as required by ordinary MBPP function tests.
_RUNNER = r'''
import json,os,resource,sys,traceback
p=json.load(sys.stdin)
l=p['limits']
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_CPU,(l['cpu_seconds'],l['cpu_seconds']))
resource.setrlimit(resource.RLIMIT_AS,(l['memory_bytes'],l['memory_bytes']))
resource.setrlimit(resource.RLIMIT_FSIZE,(l['output_bytes'],l['output_bytes']))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
isolation={'hidden_paths':{x:not os.path.exists(x) for x in ('/home/rabbit','/data','/root','/opt')},
           'environment_keys':sorted(os.environ),'network_namespace':os.readlink('/proc/self/ns/net')}
ns={'__name__':'__mbpp__'}
completed=0
result={'status':'failed','completed_tests':0,'test_count':len(p['tests']),'isolation':isolation}
try:
    if not p['tests']: raise ValueError('Nonempty dataset tests required')
    exec(compile(p['setup'],'<dataset setup>','exec'),ns)
    exec(compile(p['code'],'<model response>','exec'),ns)
    for t in p['tests']:
        exec(compile(t,'<dataset test>','exec'),ns)
        completed+=1
    result.update(status='passed',completed_tests=completed)
except BaseException as e:
    result.update(error_type=type(e).__name__,error=str(e)[:1000],completed_tests=completed)
print('\n'+p['marker']+json.dumps(result,sort_keys=True),flush=True)
'''


def sandbox_command(*, runtime=None, bwrap="/usr/bin/bwrap"):
    runtime = Path(runtime or sys.prefix).resolve()
    if not (runtime/"bin/python").exists():
        raise ValueError("Explicit Python runtime with bin/python required")
    return [str(bwrap), "--unshare-all", "--new-session", "--die-with-parent",
            "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--ro-bind", str(runtime), "/runtime",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/work",
            "--chdir", "/work", "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PATH", "/runtime/bin:/usr/bin", "--setenv", "OMP_NUM_THREADS", "1",
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "/runtime/bin/python", "-I", "-S", "-c", _RUNNER]


def _sandbox_run(code, tests, setup="", *, limits=SandboxLimits(), runtime=None, bwrap="/usr/bin/bwrap"):
    if (not isinstance(code, str) or not isinstance(setup, str) or not isinstance(tests, list)
            or not tests or any(not isinstance(t, str) or not t.strip() for t in tests)):
        return dict(status="unavailable", available=False, reason="Missing or invalid dataset code/tests")
    if (limits.cpu_seconds < 1 or limits.wall_seconds <= 0 or limits.memory_bytes < 64*1024**2
            or limits.output_bytes < 1024 or limits.source_bytes < 1024):
        raise ValueError("Invalid sandbox limits")
    if sum(len(x.encode()) for x in [code, setup, *tests]) > limits.source_bytes:
        return dict(status="failed", available=True, reason="Source/test byte limit exceeded")
    marker = "MBPP_RESULT_"+secrets.token_hex(16)+":"
    payload = json.dumps(dict(code=code, tests=tests, setup=setup, marker=marker, limits=asdict(limits))).encode()
    try:
        command = sandbox_command(runtime=runtime, bwrap=bwrap)
    except (ValueError, OSError) as error:
        return dict(status="unavailable", available=False, reason=str(error))
    # Regular output files are bounded by RLIMIT_FSIZE; pipes would permit
    # unbounded captured output before the wall-time timeout.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(command, input=payload, stdout=stdout, stderr=stderr,
                env={"PATH":"/usr/bin:/bin", "LANG":"C.UTF-8"}, timeout=limits.wall_seconds, check=False)
        except subprocess.TimeoutExpired:
            return dict(status="timeout", available=True, passed=False, reason="Sandbox wall-time limit", limits=asdict(limits))
        except OSError as error:
            return dict(status="unavailable", available=False, reason=str(error))
        stdout.seek(0); out = stdout.read(limits.output_bytes).decode(errors="replace")
        stderr.seek(0); err = stderr.read(limits.output_bytes).decode(errors="replace")
    lines = [line[len(marker):] for line in out.splitlines() if line.startswith(marker)]
    if process.returncode != 0 or len(lines) != 1:
        # Setup failure is only called unavailable by the trusted probe. An
        # admitted sandbox's candidate may crash, print too much or terminate.
        return dict(status="execution_failed", available=True, passed=False, returncode=process.returncode,
                    reason="Sandbox did not produce exactly one completed result", stderr=err[:2000], limits=asdict(limits))
    try:
        result = json.loads(lines[0])
        passed = result["status"] == "passed" and result["completed_tests"] == result["test_count"] == len(tests)
    except (ValueError, KeyError, TypeError):
        return dict(status="execution_failed", available=True, passed=False, reason="Invalid sandbox result")
    return dict(result, available=True, passed=passed, returncode=process.returncode, limits=asdict(limits))


@lru_cache(maxsize=4)
def sandbox_probe(runtime=None, bwrap="/usr/bin/bwrap"):
    result = _sandbox_run("def add(a,b): return a+b", ["assert add(2,3)==5"], runtime=runtime, bwrap=bwrap)
    isolation = result.get("isolation", {})
    hidden = isolation.get("hidden_paths", {})
    allowed_env = {"HOME", "LANG", "PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PWD"}
    try:
        host_net = os.readlink("/proc/self/ns/net")
    except OSError:
        host_net = None
    passed = (result.get("passed") is True and len(hidden) == 4 and all(hidden.values())
        and set(isolation.get("environment_keys", [])) <= allowed_env
        and host_net is not None and isolation.get("network_namespace") != host_net)
    return dict(available=passed, status="available" if passed else "unavailable", probe=result,
                reason=None if passed else "Bubblewrap execution or isolation checks failed; no unsandboxed fallback")


def extract_code(response):
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty code response")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.S|re.I)
    if len(blocks) == 1:
        return blocks[0].strip()+"\n"
    if blocks or "```" in response:
        raise ValueError("Ambiguous or incomplete code fence")
    return response.strip()+"\n"


def run_mbpp(response, context, *, limits=SandboxLimits(), runtime=None, bwrap="/usr/bin/bwrap"):
    probe = sandbox_probe(str(Path(runtime or sys.prefix).resolve()), bwrap)
    if not probe["available"]:
        return dict(status="unavailable", available=False, passed=None, reason=probe["reason"], sandbox_probe=probe)
    try:
        code = extract_code(response)
    except ValueError as error:
        return dict(status="failed", available=True, passed=False, reason=str(error))
    return _sandbox_run(code, context.get("test_list"), context.get("test_setup_code", ""),
                        limits=limits, runtime=runtime, bwrap=bwrap)


_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"


def final_number(text, *, reference=False):
    text = str(text)
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    elif reference:
        return None
    matches = list(re.finditer(_NUMBER, text))
    if not matches:
        return None
    token = matches[-1].group(0).replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/")
            value = Decimal(numerator)/Decimal(denominator)
        else:
            value = Decimal(token)
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def normalized_answer(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def grade_answer(response, context, **sandbox_options):
    family = context.get("family")
    reference = context.get("response")
    if family in {"dolly", "xsum"}:
        return dict(status="semantic_judge_required", available=False, passed=None,
                    reason="No exact-match semantic grading for summaries or general instructions")
    if family == "mbpp":
        return run_mbpp(response, context, **sandbox_options)
    if family == "gsm8k":
        expected, actual = final_number(reference, reference=True), final_number(response)
        if expected is None:
            return dict(status="unavailable", available=False, passed=None, reason="Invalid GSM8K reference final number")
        passed = actual is not None and Decimal(actual) == Decimal(expected)
        return dict(status="passed" if passed else "failed", available=True, passed=passed,
                    expected=expected, extracted=actual, extraction="Last numeric value after final #### when present; otherwise last numeric value")
    if family in {"arc", "boolq", "hellaswag", "winogrande"}:
        passed = bool(normalized_answer(response)) and normalized_answer(response) == normalized_answer(reference)
        return dict(status="passed" if passed else "failed", available=True, passed=passed,
                    metric="normalized_complete_answer_exact_match", diagnostic_only=True)
    return dict(status="unavailable", available=False, passed=None, reason="Unsupported task family")


def grade_generated_tasks(native_records, student_records, contexts, *, maximum_drop_pp=5.,
                          minimum_student_rate=.5, **sandbox_options):
    def index(records):
        values = {r["example_id"]: r for r in records}
        if len(values) != len(records) or not values:
            raise ValueError("Unique nonempty generation records required")
        return values
    native, student = index(native_records), index(student_records)
    if native.keys() != student.keys() or any(key not in contexts for key in native):
        raise ValueError("Paired example IDs and exact source contexts required")
    rows, families = [], defaultdict(list)
    for key, n in native.items():
        s, context = student[key], contexts[key]
        if not n.get("input_ids") or n["input_ids"] != s.get("input_ids"):
            raise ValueError("Native/student generation prompts differ or lack token IDs")
        family = context["family"]
        reference_check = None
        if family == "mbpp":
            reference_check = run_mbpp(context["response"], context, **sandbox_options)
        if reference_check is not None and reference_check.get("passed") is not True:
            a = b = dict(status="unavailable", available=False, passed=None,
                         reason="MBPP reference tests could not be qualified", reference_check=reference_check)
        else:
            a = grade_answer(n.get("response", ""), context, **sandbox_options)
            b = grade_answer(s.get("response", ""), context, **sandbox_options)
        row = dict(example_id=key, family=family, native=a, student=b,
                   native_stop_reason=n.get("stop_reason"), student_stop_reason=s.get("stop_reason"),
                   reference_check=reference_check)
        rows.append(row); families[family].append(row)
    summaries = {}
    for family, entries in families.items():
        available = all(r[arm].get("available") is True for r in entries for arm in ("native", "student"))
        n, s = (sum(r[arm].get("passed") is True for r in entries) for arm in ("native", "student"))
        summaries[family] = dict(count=len(entries), available=available, native_correct=n if available else None,
            student_correct=s if available else None, native_rate=n/len(entries) if available else None,
            student_rate=s/len(entries) if available else None, delta_pp=100*(s-n)/len(entries) if available else None,
            unavailable_pairs=sum(not all(r[a].get("available") is True for a in ("native", "student")) for r in entries),
            diagnostic_only=family not in {"mbpp", "gsm8k"})
    gates = {}
    for family in ("mbpp", "gsm8k"):
        summary = summaries.get(family)
        available = bool(summary and summary["available"])
        passed = bool(available and summary["student_rate"] >= minimum_student_rate
                      and summary["delta_pp"] >= -maximum_drop_pp)
        gates[family] = dict(passed=passed, available=available, minimum_student_rate=minimum_student_rate,
            maximum_drop_pp=maximum_drop_pp, reason="Missing or unavailable paired task check" if not available else None)
    return dict(schema="jspace.generated-task-metrics.v1", passed=all(g["passed"] for g in gates.values()),
        available=all(g["available"] for g in gates.values()), gates=gates, families=summaries, records=rows,
        scope="Tiny paired generated-response correctness screen; not official benchmark accuracy or statistical noninferiority",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
