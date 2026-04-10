"""
torch_profiler_runner.py
Profiles a PyTorch operation using torch.profiler and outputs structured metrics.
Can be used as a standalone script or imported.
"""
import sys
import json
import torch
import torch.profiler


def profile_matmul(size: int = 1024, dtype=torch.float32) -> dict:
    """Profile a matrix multiply and return metrics."""
    device = "cuda"
    x = torch.randn(size, size, device=device, dtype=dtype)
    y = torch.randn(size, size, device=device, dtype=dtype)

    # Warm up
    for _ in range(3):
        z = torch.mm(x, y)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            z = torch.mm(x, y)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total_cuda_us = sum(e.self_cuda_time_total for e in events)
    total_flops = sum(e.flops for e in events if e.flops)

    # Find the main kernel
    kernels = [e for e in events if e.self_cuda_time_total > 0 and e.key != "cudaDeviceSynchronize"]
    kernels.sort(key=lambda e: e.self_cuda_time_total, reverse=True)

    result = {
        "total_cuda_us": total_cuda_us,
        "total_flops": total_flops,
        "top_kernels": [
            {
                "name": e.key,
                "cuda_us": e.self_cuda_time_total,
                "flops": e.flops,
                "calls": e.count,
            }
            for e in kernels[:5]
        ],
    }

    # Compute TFLOPS
    if total_cuda_us > 0 and total_flops > 0:
        result["tflops"] = total_flops / (total_cuda_us * 1e6)

    return result


def profile_executable_python(exe_path: str) -> dict:
    """
    Profile a Python script using torch.profiler by importing and running it.
    Falls back to subprocess profiling if not a Python file.
    """
    import subprocess
    import tempfile
    import os

    if not exe_path.endswith(".py"):
        return {}

    # Run the script with profiling wrapper
    wrapper = f"""
import torch
import torch.profiler
import sys
sys.argv = ['{exe_path}']

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    exec(open('{exe_path}').read())

events = prof.key_averages()
import json
result = {{
    'total_cuda_us': sum(e.self_cuda_time_total for e in events),
    'total_flops': sum(e.flops for e in events if e.flops),
    'top_kernels': [
        {{'name': e.key, 'cuda_us': e.self_cuda_time_total, 'flops': e.flops}}
        for e in sorted(events, key=lambda x: x.self_cuda_time_total, reverse=True)[:5]
        if e.self_cuda_time_total > 0
    ],
}}
print('TORCH_PROFILE_JSON:' + json.dumps(result))
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper)
        tmp = f.name

    try:
        ret = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=120)
        for line in ret.stdout.splitlines():
            if line.startswith("TORCH_PROFILE_JSON:"):
                return json.loads(line[len("TORCH_PROFILE_JSON:"):])
    finally:
        os.unlink(tmp)

    return {}


if __name__ == "__main__":
    result = profile_matmul()
    print(json.dumps(result, indent=2))
