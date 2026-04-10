"""Planner: route each Task to a profiling strategy."""
import logging
from .models import Task, TaskPlan

log = logging.getLogger(__name__)

# Core ncu metrics for operator profiling
BASE_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes.sum",
    "l2__t_bytes.sum",
    "dram__bytes.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "sm__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "sm__cycles_elapsed.sum",
]

# Routing table: (keyword_in_target_name) → (probe_name, ncu_sections, extra_metrics)
ROUTING_TABLE = [
    # Hardware probes
    (["dram_latency", "memory_latency", "global_latency"],
     "pointer_chasing", ["MemoryWorkloadAnalysis"], []),
    (["l2_cache", "cache_capacity", "cache_size"],
     "bandwidth_sweep", ["MemoryWorkloadAnalysis"], []),
    (["bank_conflict"],
     "bank_conflict", [], []),
    (["boost_clock", "actual_clock", "clock_mhz", "sm_clock"],
     "clock_probe", [], []),
    (["bandwidth", "memory_bandwidth", "peak_bandwidth"],
     "bandwidth_sweep", ["MemoryWorkloadAnalysis"], []),
    # Operator profiling
    (["bottleneck", "bound"],
     None, ["SpeedOfLight", "MemoryWorkloadAnalysis"], []),
    (["tensor_core", "tc_util"],
     None, ["SpeedOfLight_RooflineChart"], [
         "sm__inst_executed_pipe_tensor_op_hmma.sum",
     ]),
    (["memory_hierarchy", "cache_hit", "l1", "l2"],
     None, ["MemoryWorkloadAnalysis", "CacheHitRate"], []),
    (["occupancy"],
     None, ["OccupancyAnalysis"], [
         "sm__warps_active.avg.pct_of_peak_sustained_active",
         "sm__maximum_warps_per_active_cycle_pct",
     ]),
]


class Planner:
    def plan(self, task: Task) -> TaskPlan:
        name_lower = task.name.lower()

        probe_name = None
        ncu_sections = []
        extra_metrics = []

        for keywords, probe, sections, metrics in ROUTING_TABLE:
            if any(kw in name_lower for kw in keywords):
                probe_name = probe
                ncu_sections = sections
                extra_metrics = metrics
                break

        # Default fallback for operator profiling
        if task.task_type == "operator_profiling" and not ncu_sections:
            ncu_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis"]

        ncu_metrics = list(dict.fromkeys(BASE_NCU_METRICS + extra_metrics))

        strategies = []
        if probe_name:
            strategies.append("probe")
        if task.run and (task.task_type == "operator_profiling" or ncu_sections):
            strategies.append("ncu")
        if not strategies:
            strategies = ["probe"]

        plan = TaskPlan(
            task=task,
            strategies=strategies,
            probe_name=probe_name,
            ncu_metrics=ncu_metrics,
            ncu_sections=ncu_sections,
        )
        log.info(f"Plan [{task.name}]: strategies={strategies}, probe={probe_name}, sections={ncu_sections}")
        return plan
