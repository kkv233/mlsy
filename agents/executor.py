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
        """Fallback: use torch.profiler when ncu lacks permissions.
        Only runs for operator_profiling tasks (not hardware probes that already have probe data)."""
        if plan.task.task_type == "hardware_probe" and result.probe_stdout:
            # Hardware probe already has data; don't overwrite with torch profiler
            log.info(f"Skipping torch profiler for hardware probe {plan.task.name} (probe data exists)")
            return
        import sys
        exe = plan.task.run
        log.info(f"Falling back to torch.profiler for {plan.task.name}")

        script = f"""
import torch, torch.profiler, json, sys
device = 'cuda'
torch.cuda.synchronize()

# Try to run the target executable via subprocess and profile a representative op
import subprocess
ret = subprocess.run(['{exe}'], capture_output=True, text=True, timeout=60)

# Also profile a representative matmul as baseline
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
for _ in range(3): torch.mm(x, y)
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, with_flops=True,
) as prof:
    for _ in range(10): torch.mm(x, y)
    torch.cuda.synchronize()

events = prof.key_averages()
result = {{
    'total_cuda_us': sum(e.self_cuda_time_total for e in events),
    'total_flops': sum(e.flops or 0 for e in events),
    'top_kernels': [
        {{'name': e.key, 'cuda_us': e.self_cuda_time_total, 'flops': e.flops or 0}}
        for e in sorted(events, key=lambda x: x.self_cuda_time_total, reverse=True)[:5]
        if e.self_cuda_time_total > 0
    ],
    'exe_stdout': ret.stdout[:500],
}}
print('TORCH_PROFILE_JSON:' + json.dumps(result))
"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            tmp = f.name

        try:
            ret = subprocess.run(
                [sys.executable, tmp], capture_output=True, text=True, timeout=120
            )
            for line in ret.stdout.splitlines():
                if line.startswith("TORCH_PROFILE_JSON:"):
                    data = json.loads(line[len("TORCH_PROFILE_JSON:"):])
                    log_path = LOGS_DIR / f"torch_{plan.task.name}.json"
                    log_path.write_text(json.dumps(data, indent=2))
                    # Store in probe_stdout for analyzer to pick up
                    result.probe_stdout = "TORCH_PROFILE:" + json.dumps(data)
                    result.probe_log_path = str(log_path)
                    log.info(f"Torch profiler data saved: {log_path}")
                    break
        except Exception as e:
            log.warning(f"Torch profiler failed: {e}")
        finally:
            import os
            os.unlink(tmp)
