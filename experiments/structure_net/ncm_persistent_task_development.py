"""Admit the persistent SQLite-isolated route through the original task gates."""
from pathlib import Path

from experiments.structure_net import ncm_persistent_judge_qualification_v2 as qualification
from experiments.structure_net import ncm_isolated_task_development as consumer

CPU=qualification.original.BASE/'20261028_persistent_task_evaluation_CPU_v1.json'
PLAN=Path('docs/07 - Status Reports/2026-10-28_ncm-persistent-task-evaluation.md')


def install():
    qualification.install()
    consumer.PLAN=PLAN
    consumer.CPU=CPU


if __name__=='__main__':
    install()
    # The existing consumer requires both full audits, preserves historical
    # first judgments, serializes requests and rejects incomplete admission.
    consumer.main()
