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

        if value is None:
            value = parsed.get("value") or parsed
            reasoning = reasoning or f"Raw probe output: {json.dumps(parsed)}"

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

        elif "memory_hierarchy" in name:
            value, r = self._memory_hierarchy(agg)
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
        compute_util = agg.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        dram_bytes = agg.get("dram__bytes.sum", 0)
        flops = agg.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0) * 2
        cycles = agg.get("sm__cycles_elapsed.sum", 1)

        # Roofline: arithmetic intensity
        ai = flops / max(dram_bytes, 1)
        # RTX 4090 ridge point ~= 82.6 TFLOPS / 1008 GB/s ≈ 82 FLOP/byte
        ridge_point = 82.0
        bound = "compute_bound" if ai > ridge_point else "memory_bound"

        r = (
            f"Arithmetic intensity: {ai:.2f} FLOP/byte (ridge={ridge_point}). "
            f"SM compute util: {compute_util:.1f}%. "
            f"Classification: {bound}"
        )
        return bound, r

    def _tensor_core_util(self, agg: dict) -> tuple:
        tc = agg.get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0)
        total = agg.get("sm__inst_executed.sum", 1)
        util = tc / max(total, 1)
        r = f"Tensor core instructions: {tc} / {total} total = {util:.2%}"
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
