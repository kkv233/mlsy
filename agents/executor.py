"""Executor: run ncu profiling and/or CUDA probes, save raw outputs."""
import subprocess
import logging
import os
import json
import shutil
from pathlib import Path
from .models import TaskPlan, RawResult

log = logging.getLogger(__name__)

PROBES_DIR = Path(__file__).parent.parent / "probes"
LOGS_DIR = Path("logs")
TMP_DIR = Path("/tmp/mlsy_probes")

NCU_BIN = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
NVCC_BIN = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
GPU_ARCH = "sm_89"  # RTX 4090 D, CC 8.9


class Executor:
    def __init__(self):
        LOGS_DIR.mkdir(exist_ok=True)
        TMP_DIR.mkdir(exist_ok=True)

    def execute(self, plan: TaskPlan) -> RawResult:
        result = RawResult(task=plan.task, plan=plan)

        if "probe" in plan.strategies and plan.probe_name:
            self._run_probe(plan, result)

        if "ncu" in plan.strategies and plan.task.run:
            self._run_ncu(plan, result)
            # If ncu produced no useful data, fall back to torch profiler
            if not self._ncu_has_data(result):
                self._run_torch_profiler(plan, result)

        return result

    def _ncu_has_data(self, result: RawResult) -> bool:
        if not result.ncu_csv_path:
            return False
        content = Path(result.ncu_csv_path).read_text()
        # Real data has more than just the header/error lines
        data_lines = [l for l in content.splitlines()
                      if l.strip() and not l.startswith("==")]
        return len(data_lines) > 2

    # ------------------------------------------------------------------
    def _run_probe(self, plan: TaskPlan, result: RawResult):
        probe_src = PROBES_DIR / f"{plan.probe_name}.cu"
        if not probe_src.exists():
            result.error = f"Probe source not found: {probe_src}"
            log.error(result.error)
            return

        bin_path = TMP_DIR / plan.probe_name
        log.info(f"Compiling probe: {probe_src} → {bin_path}")
        ret = subprocess.run(
            [NVCC_BIN, "-O3", f"-arch={GPU_ARCH}", str(probe_src), "-o", str(bin_path)],
            capture_output=True, text=True, timeout=60,
        )
        if ret.returncode != 0:
            result.error = f"nvcc failed:\n{ret.stderr}"
            log.error(result.error)
            return

        log.info(f"Running probe: {bin_path}")
        ret = subprocess.run(
            [str(bin_path)],
            capture_output=True, text=True, timeout=120,
        )
        stdout = ret.stdout
        if ret.returncode != 0:
            log.warning(f"Probe exited {ret.returncode}: {ret.stderr}")
            stdout = ret.stdout + "\n" + ret.stderr

        log_path = LOGS_DIR / f"probe_{plan.task.name}.txt"
        log_path.write_text(stdout)
        result.probe_stdout = stdout
        result.probe_log_path = str(log_path)
        log.info(f"Probe output saved: {log_path}")

    # ------------------------------------------------------------------
    def _run_ncu(self, plan: TaskPlan, result: RawResult):
        csv_path = LOGS_DIR / f"ncu_{plan.task.name}.csv"
        exe = plan.task.run

        if not Path(exe).exists():
            log.warning(f"Executable not found: {exe}, skipping ncu")
            return

        # Build ncu command
        cmd = [NCU_BIN, "--csv", "--target-processes", "all"]

        if plan.ncu_sections:
            for sec in plan.ncu_sections:
                cmd += ["--section", sec]
        else:
            cmd += ["--metrics", ",".join(plan.ncu_metrics)]

        cmd += ["-o", str(csv_path).replace(".csv", ""), exe]

        log.info(f"Running ncu: {' '.join(cmd)}")
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # ncu writes to .ncu-rep file; also try --csv export
        # Try direct CSV output approach
        cmd2 = [NCU_BIN, "--csv", "--target-processes", "all"]
        if plan.ncu_sections:
            for sec in plan.ncu_sections:
                cmd2 += ["--section", sec]
        else:
            cmd2 += ["--metrics", ",".join(plan.ncu_metrics)]
        cmd2.append(exe)

        log.info(f"Running ncu (stdout CSV): {' '.join(cmd2)}")
        ret2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)

        output = ret2.stdout
        if ret2.returncode != 0 and not output.strip():
            log.warning(f"ncu failed (rc={ret2.returncode}): {ret2.stderr[:500]}")
            # Try with --set basic as fallback
            cmd3 = [NCU_BIN, "--csv", "--set", "basic", "--target-processes", "all", exe]
            log.info(f"ncu fallback: {' '.join(cmd3)}")
            ret3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=300)
            output = ret3.stdout

        csv_path.write_text(output)
        result.ncu_csv_path = str(csv_path)
        log.info(f"ncu CSV saved: {csv_path} ({len(output)} bytes)")

    # ------------------------------------------------------------------
    def _run_torch_profiler(self, plan: TaskPlan, result: RawResult):
        """Fallback when ncu lacks permissions: use nvidia-smi dmon to sample
        GPU utilization while the target executable runs."""
        if plan.task.task_type == "hardware_probe" and result.probe_stdout:
            log.info(f"Skipping fallback profiler for hardware probe {plan.task.name}")
            return

        exe = plan.task.run
        log.info(f"Falling back to nvidia-smi dmon profiling for {plan.task.name}")

        import tempfile, threading

        dmon_output = []
        stop_event = threading.Event()

        def run_dmon():
            # Sample every 100ms: SM util, memory util, memory clock, SM clock
            ret = subprocess.run(
                ["nvidia-smi", "dmon", "-s", "ucm", "-d", "1"],
                capture_output=True, text=True, timeout=60,
            )
            dmon_output.append(ret.stdout)

        # Start dmon in background thread
        dmon_thread = threading.Thread(target=run_dmon, daemon=True)
        dmon_thread.start()

        # Run the target executable
        try:
            exe_ret = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            exe_stdout = exe_ret.stdout
        except Exception as e:
            exe_stdout = str(e)
        finally:
            # Stop dmon
            subprocess.run(["pkill", "-f", "nvidia-smi dmon"], capture_output=True)
            dmon_thread.join(timeout=3)

        dmon_text = dmon_output[0] if dmon_output else ""

        # Parse dmon output: extract peak SM util and memory util
        sm_utils, mem_utils = [], []
        for line in dmon_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    sm_utils.append(int(parts[1]))
                    mem_utils.append(int(parts[2]))
                except ValueError:
                    continue

        data = {
            "exe_stdout": exe_stdout[:500],
            "peak_sm_util_pct": max(sm_utils) if sm_utils else None,
            "avg_sm_util_pct": round(sum(sm_utils) / len(sm_utils), 1) if sm_utils else None,
            "peak_mem_util_pct": max(mem_utils) if mem_utils else None,
            "dmon_samples": len(sm_utils),
            "profiling_method": "nvidia-smi dmon (ncu unavailable)",
        }

        log_path = LOGS_DIR / f"dmon_{plan.task.name}.json"
        log_path.write_text(json.dumps(data, indent=2))
        result.probe_stdout = "TORCH_PROFILE:" + json.dumps(data)
        result.probe_log_path = str(log_path)
        log.info(f"dmon profile saved: {log_path} (peak SM={data['peak_sm_util_pct']}%)")
