"""Shared dataclasses used across all agents."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    name: str
    task_type: str          # "hardware_probe" | "operator_profiling"
    run: str | None         # path to executable (None for pure probes)
    extra: dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    task: Task
    strategies: list[str]   # e.g. ["probe", "ncu"]
    probe_name: str | None  # e.g. "pointer_chasing"
    ncu_metrics: list[str]
    ncu_sections: list[str]


@dataclass
class RawResult:
    task: Task
    plan: TaskPlan
    ncu_csv_path: str | None = None
    probe_stdout: str | None = None
    probe_log_path: str | None = None
    error: str | None = None


@dataclass
class AnalyzedResult:
    task: Task
    value: Any              # numeric or string classification
    unit: str = ""
    confidence: str = "medium"
    evidence: dict = field(default_factory=dict)
    reasoning: str = ""
    error: str | None = None
