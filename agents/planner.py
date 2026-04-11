"""Planner: route each Task to a profiling strategy."""
import logging
from .models import Task, TaskPlan

log = logging.getLogger(__name__)

# Core ncu metrics for operator profiling
BASE_NCU_METRICS = [
    # Compute/memory throughput % — primary bottleneck classification signals (spec §3.2)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    # Raw byte traffic — memory hierarchy breakdown
    "l1tex__t_bytes.sum",
    "l2__t_bytes.sum",
    "dram__bytes.sum",
    # Compute instruction mix
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    # Tensor core activity — spec mandates pct_of_peak_sustained_active (spec §4.1)
    "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",  # kept for cross-check
    # FMA pipe activity
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    # Instruction throughput
    "sm__inst_executed.sum",
    "sm__sass_thread_inst_executed_per_inst_executed",  # warp divergence proxy
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    # Memory access pattern
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu.sum",
    # Timing
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
    # Shared memory capacity / warp topology — shmem_probe + nvml_fallback
    (["max_shmem", "shmem", "shared_mem", "warp_size"],
     "shmem_probe", [], []),
    # Operator profiling
    (["bottleneck", "bound"],
     None, ["SpeedOfLight", "MemoryWorkloadAnalysis"], []),
    (["tensor_core", "tc_util"],
     None, ["SpeedOfLight_RooflineChart"], [
         "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
         "sm__inst_executed_pipe_tensor_op_hmma.sum",
     ]),
    (["memory_hierarchy", "cache_hit", "l1", "l2"],
     None, ["MemoryWorkloadAnalysis", "CacheHitRate"], []),
    (["occupancy"],
     None, ["OccupancyAnalysis"], [
         "sm__warps_active.avg.pct_of_peak_sustained_active",
         "sm__maximum_warps_per_active_cycle_pct",
     ]),
    # Warp divergence — needs InstructionStats for per-inst-executed ratio
    (["warp_divergence", "divergence"],
     None, ["InstructionStatistics"], [
         "sm__sass_thread_inst_executed_per_inst_executed",
         "sm__inst_executed.sum",
     ]),
    # Uncoalesced access detection — needs sector/request counts
    (["uncoalesced_access", "uncoalesced", "access_pattern", "coalescing"],
     None, ["MemoryWorkloadAnalysis", "GlobalAccess"], [
         "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
         "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
         "l1tex__data_bank_conflicts_pipe_lsu.sum",
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
