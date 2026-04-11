"""Analyzer: parse raw results into structured AnalyzedResult."""
import logging
import json
from pathlib import Path
from .models import RawResult, AnalyzedResult
from parsers.ncu_parser import parse_ncu_csv
from parsers.probe_parser import parse_probe_output

log = logging.getLogger(__name__)


class Analyzer:
    def analyze(self, raw: RawResult) -> AnalyzedResult:
        if raw.error:
            return AnalyzedResult(
                task=raw.task, value=None, error=raw.error,
                reasoning=f"Execution failed: {raw.error}"
            )

        if raw.task.task_type == "hardware_probe":
            return self._analyze_probe(raw)
        else:
            return self._analyze_operator(raw)

    # ------------------------------------------------------------------
    def _analyze_probe(self, raw: RawResult) -> AnalyzedResult:
        name = raw.task.name.lower()
        stdout = raw.probe_stdout or ""
        evidence = {}

        parsed = parse_probe_output(raw.plan.probe_name, stdout)
        evidence["probe_parsed"] = parsed

        value = None
        reasoning = ""

        if "clock" in name or raw.plan.probe_name == "clock_probe":
            value = parsed.get("clock_mhz")
            reasoning = f"Measured SM clock via clock64() correlation: {value} MHz"

        elif "bank_conflict" in name or raw.plan.probe_name == "bank_conflict":
            value = parsed.get("penalty_ratio")
            reasoning = (
                f"Bank conflict penalty = cycles(stride=32)/cycles(stride=1) = {value:.2f}x"
                if value else "Could not compute penalty ratio"
            )

        elif "latency" in name or raw.plan.probe_name == "pointer_chasing":
            value = parsed.get("dram_latency_cycles")
            evidence.update({
                "l1_latency_cycles": parsed.get("l1_latency_cycles"),
                "l2_latency_cycles": parsed.get("l2_latency_cycles"),
                "dram_latency_cycles": parsed.get("dram_latency_cycles"),
            })
            reasoning = (
                f"Pointer-chasing latency hierarchy: "
                f"L1={parsed.get('l1_latency_cycles')} cycles, "
                f"L2={parsed.get('l2_latency_cycles')} cycles, "
                f"DRAM={parsed.get('dram_latency_cycles')} cycles"
            )

        elif "bandwidth" in name or "l2_cache" in name or raw.plan.probe_name == "bandwidth_sweep":
            if "l2_cache" in name or "cache_capacity" in name or "cache_size" in name:
                value = parsed.get("l2_cache_bytes")
                reasoning = f"L2 cache capacity estimated from bandwidth knee: {value} bytes"
            else:
                value = parsed.get("peak_read_GBs")
                reasoning = f"Peak memory bandwidth: {value} GB/s"

        elif raw.plan.probe_name == "shmem_probe" or "shmem" in name or "shared_mem" in name or "max_shmem" in name:
            if "warp_size" in name or ("warp" in name and "size" in name):
                value = 32
                reasoning = "Warp size is architecturally fixed at 32 on all NVIDIA GPUs"
            elif "bandwidth" in name:
                value = parsed.get("shmem_bandwidth_GBs")
                reasoning = f"Shared memory bandwidth: {value} GB/s"
            elif "optin" in name:
                value = parsed.get("max_shmem_optin_kb")
                reasoning = f"Max shared memory per block (opt-in): {value} KB"
            else:
                value = parsed.get("max_shmem_per_block_kb")
                reasoning = f"Max shared memory per block (default): {value} KB"

        elif "warp_size" in name or ("warp" in name and "size" in name):
            value = 32
            reasoning = "Warp size is architecturally fixed at 32 on all NVIDIA GPUs"

        if value is None:
            value = parsed.get("value") or parsed or None
            reasoning = reasoning or f"Raw probe output: {parsed}"

        # Last resort: unknown hardware probe with no matching probe file
        # Try pynvml for device-level info
        if value is None or value == {}:
            value, reasoning = self._nvml_fallback(name)

        return AnalyzedResult(
            task=raw.task,
            value=value,
            evidence=evidence,
            reasoning=reasoning,
            confidence="high" if value is not None else "low",
        )

    # ------------------------------------------------------------------
    def _analyze_operator(self, raw: RawResult) -> AnalyzedResult:
        name = raw.task.name.lower()
        evidence = {}
        reasoning_parts = []

        # Check for torch profiler data (fallback when ncu unavailable)
        torch_data = None
        if raw.probe_stdout and raw.probe_stdout.startswith("TORCH_PROFILE:"):
            import json
            torch_data = json.loads(raw.probe_stdout[len("TORCH_PROFILE:"):])
            evidence["torch_profiler"] = torch_data

        metrics = {}
        if raw.ncu_csv_path and Path(raw.ncu_csv_path).exists():
            csv_text = Path(raw.ncu_csv_path).read_text()
            if csv_text.strip():
                metrics = parse_ncu_csv(csv_text)
                evidence["ncu_metrics"] = metrics

        if not metrics and not torch_data:
            return AnalyzedResult(
                task=raw.task, value=None,
                reasoning="No profiling data available (ncu permission denied, torch profiler also failed)",
                confidence="low",
            )

        # Use torch profiler data if ncu unavailable
        if not metrics and torch_data:
            return self._analyze_from_torch(raw, name, torch_data, evidence)

        # Aggregate across kernels (use the kernel with most activity)
        agg = self._aggregate_metrics(metrics)
        evidence["aggregated"] = agg

        value = None

        if "bottleneck" in name or "bound" in name:
            value, r = self._classify_bound(agg)
            reasoning_parts.append(r)

        elif "tensor_core" in name or "tc_util" in name:
            value, r = self._tensor_core_util(agg)
            reasoning_parts.append(r)

        elif "occupancy" in name:
            value = agg.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0) / 100.0
            reasoning_parts.append(f"Achieved occupancy: {value:.2%}")

        elif "memory_hierarchy" in name or "cache_hit" in name:
            value, r = self._memory_hierarchy(agg)
            reasoning_parts.append(r)

        elif "warp_divergence" in name or "divergence" in name:
            value, r = self._classify_warp_divergence(agg)
            reasoning_parts.append(r)

        elif "uncoalesced_access" in name or "uncoalesced" in name or "coalescing" in name or "access_pattern" in name:
            value, r = self._classify_uncoalesced_access(agg)
            reasoning_parts.append(r)

        else:
            # Generic: return compute utilization
            value = agg.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
            reasoning_parts.append(f"SM throughput utilization: {value:.1f}%")

        return AnalyzedResult(
            task=raw.task,
            value=value,
            evidence=evidence,
            reasoning="\n".join(reasoning_parts),
            confidence="high" if value is not None else "low",
        )

    def _analyze_from_torch(self, raw: RawResult, name: str, torch_data: dict, evidence: dict) -> AnalyzedResult:
        """Analyze using nvidia-smi dmon data when ncu is unavailable."""
        peak_sm = torch_data.get("peak_sm_util_pct")
        avg_sm = torch_data.get("avg_sm_util_pct")
        peak_mem = torch_data.get("peak_mem_util_pct")
        method = torch_data.get("profiling_method", "dmon")

        reasoning_parts = [
            f"Analysis via {method}.",
            f"Peak SM utilization: {peak_sm}%, Avg SM utilization: {avg_sm}%",
            f"Peak memory utilization: {peak_mem}%",
        ]

        value = None

        if "bottleneck" in name or "bound" in name:
            if peak_sm is not None and peak_mem is not None:
                if peak_sm > peak_mem * 1.2:
                    value = "compute_bound"
                    reasoning_parts.append(f"SM util ({peak_sm}%) dominates mem util ({peak_mem}%) → compute_bound")
                else:
                    value = "memory_bound"
                    reasoning_parts.append(f"Mem util ({peak_mem}%) comparable to SM util ({peak_sm}%) → memory_bound")
            else:
                value = "unknown"
                reasoning_parts.append("Insufficient dmon samples to classify")

        elif "tensor_core" in name:
            # Can't determine tensor core usage from dmon; report SM util as proxy
            value = (peak_sm or 0) / 100.0
            reasoning_parts.append("Tensor core usage not measurable via dmon; SM util used as proxy")

        elif "memory_hierarchy" in name:
            value = {
                "peak_sm_util_pct": peak_sm,
                "peak_mem_util_pct": peak_mem,
                "note": "ncu unavailable; hierarchy detail requires hardware counters",
            }

        else:
            value = peak_sm

        return AnalyzedResult(
            task=raw.task,
            value=value,
            evidence=evidence,
            reasoning="\n".join(reasoning_parts),
            confidence="medium",
        )

    # ------------------------------------------------------------------
    def _aggregate_metrics(self, metrics: dict) -> dict:
        """Sum/average metrics across all kernels."""
        if not metrics:
            return {}
        agg = {}
        for kernel_metrics in metrics.values():
            for k, v in kernel_metrics.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0) + v
        # For percentage metrics, take max instead of sum
        pct_keys = [k for k in agg if "pct" in k or "throughput" in k]
        for k in pct_keys:
            vals = [m.get(k, 0) for m in metrics.values() if isinstance(m.get(k), (int, float))]
            if vals:
                agg[k] = max(vals)
        return agg

    def _classify_bound(self, agg: dict) -> tuple:
        # --- Primary classification: spec §3.2 mandates throughput-% comparison ---
        # Use gpu__compute_memory_throughput (combined L1/L2/DRAM pressure) vs
        # sm__throughput (compute pipe saturation).
        compute_pct = agg.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        mem_pct = agg.get(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 0
        )
        dram_pct = agg.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        l2_pct = agg.get("l2__throughput.avg.pct_of_peak_sustained_elapsed", 0)

        # Prefer the combined memory throughput metric; fall back to max(dram, l2) if missing
        effective_mem_pct = mem_pct if mem_pct > 0 else max(dram_pct, l2_pct)

        # Classify: whichever throughput % is higher wins; ties go to memory_bound
        if compute_pct > effective_mem_pct:
            bound = "compute_bound"
        elif effective_mem_pct > 0:
            bound = "memory_bound"
        else:
            # No throughput-% data available — fall back to arithmetic intensity
            dram_bytes = agg.get("dram__bytes.sum", 0)
            flops = agg.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0) * 2
            ai = flops / max(dram_bytes, 1)
            ridge_point = 82.0  # RTX 4090: ~82.6 TFLOPS / 1008 GB/s
            bound = "compute_bound" if ai > ridge_point else "memory_bound"
            r = (
                f"Throughput-% metrics unavailable; using arithmetic intensity fallback. "
                f"AI={ai:.2f} FLOP/byte (ridge={ridge_point}). "
                f"Classification: {bound}"
            )
            return bound, r

        r = (
            f"Compute throughput: {compute_pct:.1f}% of peak. "
            f"Memory throughput (gpu__compute_memory): {mem_pct:.1f}% of peak "
            f"(DRAM: {dram_pct:.1f}%, L2: {l2_pct:.1f}%). "
            f"Classification: {bound}"
        )
        return bound, r

    def _tensor_core_util(self, agg: dict) -> tuple:
        # Spec §4.1: use sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active
        # as the primary tensor core utilization metric (directly comparable to peak).
        tc_pct = agg.get(
            "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active", None
        )
        if tc_pct is not None:
            util = tc_pct / 100.0
            r = (
                f"Tensor core pipe active: {tc_pct:.1f}% of peak sustained active "
                f"(sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active)"
            )
            return util, r

        # Fallback: instruction-count ratio (less accurate — counts instructions not cycles)
        tc = agg.get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0)
        total = agg.get("sm__inst_executed.sum", 1)
        util = tc / max(total, 1)
        r = (
            f"Tensor core utilization (instruction ratio fallback): "
            f"{tc} TC insts / {total} total insts = {util:.2%}. "
            f"Note: sm__pipe_tensor_op_hmma_cycle_active metric unavailable."
        )
        return util, r

    def _memory_hierarchy(self, agg: dict) -> tuple:
        l1 = agg.get("l1tex__t_bytes.sum", 0)
        l2 = agg.get("l2__t_bytes.sum", 0)
        dram = agg.get("dram__bytes.sum", 0)
        total = max(l1, 1)
        l2_hit = 1 - (l2 / total) if l2 < total else 0
        dram_hit = 1 - (dram / max(l2, 1)) if dram < l2 else 0
        r = (
            f"L1 bytes: {l1:.0f}, L2 bytes: {l2:.0f}, DRAM bytes: {dram:.0f}. "
            f"L1 hit rate: {l2_hit:.1%}, L2 hit rate: {dram_hit:.1%}"
        )
        return {"l1_bytes": l1, "l2_bytes": l2, "dram_bytes": dram,
                "l1_hit_rate": l2_hit, "l2_hit_rate": dram_hit}, r

    def _classify_warp_divergence(self, agg: dict) -> tuple:
        """Classify warp divergence using sm__sass_thread_inst_executed_per_inst_executed.

        Reports average threads that executed each instruction (ideally 32 = no divergence).
        Spec §5.1: use this metric as the primary divergence indicator.
        """
        threads_per_inst = agg.get(
            "sm__sass_thread_inst_executed_per_inst_executed", None
        )
        total_insts = agg.get("sm__inst_executed.sum", 0)

        if threads_per_inst is None:
            return None, (
                "sm__sass_thread_inst_executed_per_inst_executed not available; "
                "divergence analysis requires InstructionStatistics ncu section."
            )

        # threads_per_inst is reported in [0, 32]; normalise to [0, 1]
        warp_size = 32.0
        efficiency = min(threads_per_inst / warp_size, 1.0)

        if efficiency >= 0.95:
            classification = "low_divergence"
        elif efficiency >= 0.75:
            classification = "moderate_divergence"
        else:
            classification = "high_divergence"

        r = (
            f"Warp execution efficiency: {efficiency:.1%} "
            f"(threads_per_inst={threads_per_inst:.2f} / warp_size=32). "
            f"Total instructions executed: {total_insts:.0f}. "
            f"Divergence classification: {classification}"
        )
        return {"classification": classification, "warp_efficiency": efficiency,
                "threads_per_inst": threads_per_inst}, r

    def _classify_uncoalesced_access(self, agg: dict) -> tuple:
        """Detect uncoalesced global memory access via sector-to-request ratio.

        Ideal: 1 request = 1 sector (all 32 threads access same 128-byte cache line).
        >1 sector/request means uncoalesced accesses (multiple cache lines touched).
        Spec §5.2: use l1tex__t_sectors / l1tex__t_requests for the global load path.
        """
        sectors = agg.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0)
        requests = agg.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 0)
        bank_conflicts = agg.get("l1tex__data_bank_conflicts_pipe_lsu.sum", 0)

        if requests == 0:
            return None, (
                "No global load requests recorded; "
                "uncoalesced access analysis requires MemoryWorkloadAnalysis ncu section."
            )

        sectors_per_request = sectors / requests
        if sectors_per_request <= 1.5:
            classification = "well_coalesced"
        elif sectors_per_request <= 3.0:
            classification = "partially_uncoalesced"
        else:
            classification = "highly_uncoalesced"

        r = (
            f"Global load coalescing: {sectors_per_request:.2f} sectors/request "
            f"(sectors={sectors:.0f}, requests={requests:.0f}). "
            f"L1 bank conflicts: {bank_conflicts:.0f}. "
            f"Classification: {classification}"
        )
        return {
            "classification": classification,
            "sectors_per_request": sectors_per_request,
            "sectors": sectors,
            "requests": requests,
            "bank_conflicts": bank_conflicts,
        }, r

    def _nvml_fallback(self, name: str) -> tuple:
        """Use pynvml to answer common device-level queries when no probe covers them."""
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name_lower = name.lower()

            if "sm_count" in name_lower or "sm_number" in name_lower:
                # Prefer torch.cuda.get_device_properties which directly reports SM count
                try:
                    import torch
                    props = torch.cuda.get_device_properties(0)
                    val = props.multi_processor_count
                    return val, f"SM count from torch.cuda.get_device_properties: {val} SMs"
                except Exception:
                    # Fallback: total CUDA cores ÷ 128 (Ada Lovelace / Ampere: 128 CUDA cores per SM)
                    cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(h)
                    val = cuda_cores // 128
                    return val, f"SM count estimated: {cuda_cores} CUDA cores ÷ 128 = {val} SMs"

            elif "max_shmem" in name_lower or "shared_mem" in name_lower or "shmem" in name_lower:
                # torch.cuda.get_device_properties exposes max_shared_memory_per_block in bytes
                try:
                    import torch
                    props = torch.cuda.get_device_properties(0)
                    val_bytes = props.max_shared_memory_per_block
                    val_kb = val_bytes / 1024
                    return val_kb, (
                        f"Max shared memory per block from torch.cuda.get_device_properties: "
                        f"{val_kb:.0f} KB ({val_bytes} bytes)"
                    )
                except Exception:
                    return None, "max_shared_memory_per_block not exposed by installed pynvml version"

            elif "memory" in name_lower and "total" in name_lower:
                val = pynvml.nvmlDeviceGetMemoryInfo(h).total
                return val, f"Total VRAM from nvml: {val} bytes"

            elif "warp_size" in name_lower or ("warp" in name_lower and "size" in name_lower):
                return 32, "Warp size is architecturally fixed at 32 on all NVIDIA GPUs"

            elif "warp" in name_lower:
                # Warp size is architecturally fixed at 32
                return 32, "Warp size is architecturally fixed at 32 on all NVIDIA GPUs"
            else:
                # Generic: return SM clock as a best-effort value
                val = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                return val, f"No dedicated probe for '{name}'; nvml SM clock={val} MHz as proxy"
        except Exception as e:
            return None, f"nvml fallback failed: {e}"
