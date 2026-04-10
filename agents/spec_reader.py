"""SpecReader: parse target_spec.json into List[Task]."""
import json
import logging
from pathlib import Path
from .models import Task

log = logging.getLogger(__name__)

# Keywords that indicate a hardware probe target (vs operator profiling)
HARDWARE_PROBE_KEYWORDS = {
    "latency", "bandwidth", "clock", "bank_conflict", "cache_capacity",
    "l1_cache", "l2_cache", "dram", "sm_count", "warp_scheduler",
    "register_file", "throughput_peak", "boost_clock", "actual_clock",
    "memory_latency", "cache_size",
}


class SpecReader:
    def read(self, spec_path: str) -> list[Task]:
        with open(spec_path) as f:
            spec = json.load(f)

        run = spec.get("run")  # path to executable, may be None
        targets = spec.get("targets", [])

        tasks = []
        for target in targets:
            if isinstance(target, str):
                name = target
                extra = {}
            elif isinstance(target, dict):
                name = target.get("name", str(target))
                extra = {k: v for k, v in target.items() if k != "name"}
            else:
                name = str(target)
                extra = {}

            task_type = self._classify(name, extra)
            # Allow spec to override type
            if "type" in extra:
                task_type = extra["type"]

            tasks.append(Task(name=name, task_type=task_type, run=run, extra=extra))
            log.info(f"Task: {name} → {task_type}")

        return tasks

    def _classify(self, name: str, extra: dict) -> str:
        name_lower = name.lower()
        for kw in HARDWARE_PROBE_KEYWORDS:
            if kw in name_lower:
                return "hardware_probe"
        return "operator_profiling"
