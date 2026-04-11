"""Parse stdout from CUDA probe binaries into structured dicts."""
import re
import logging
import numpy as np

log = logging.getLogger(__name__)


def parse_probe_output(probe_name: str | None, stdout: str) -> dict:
    if not probe_name or not stdout.strip():
        return {}

    parsers = {
        "clock_probe": _parse_clock,
        "pointer_chasing": _parse_pointer_chasing,
        "bandwidth_sweep": _parse_bandwidth_sweep,
        "bank_conflict": _parse_bank_conflict,
        "shmem_probe": _parse_shmem_probe,
    }
    fn = parsers.get(probe_name, _parse_generic)
    try:
        return fn(stdout)
    except Exception as e:
        log.warning(f"Probe parse error ({probe_name}): {e}")
        return {"raw": stdout[:500]}


def _parse_clock(stdout: str) -> dict:
    """Expected output: 'clock_mhz: 2520.5' or 'CLOCK_MHZ 2520'"""
    for line in stdout.splitlines():
        m = re.search(r"clock[_\s]*mhz[:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            return {"clock_mhz": float(m.group(1))}
    # Fallback: find any float on a line with "mhz"
    for line in stdout.splitlines():
        if "mhz" in line.lower():
            nums = re.findall(r"[0-9]+\.?[0-9]*", line)
            if nums:
                return {"clock_mhz": float(nums[-1])}
    return {}


def _parse_pointer_chasing(stdout: str) -> dict:
    """Expected output lines: '<size_bytes> <latency_cycles>'"""
    data = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                size = int(parts[0])
                latency = float(parts[1])
                data.append((size, latency))
            except ValueError:
                continue

    if not data:
        return {}

    data.sort(key=lambda x: x[0])
    sizes = [d[0] for d in data]
    latencies = [d[1] for d in data]

    # Find plateaus using gradient analysis
    result = {"data": data}

    # L1: smallest sizes (< 256KB)
    l1_data = [(s, l) for s, l in data if s < 256 * 1024]
    if l1_data:
        result["l1_latency_cycles"] = min(l for _, l in l1_data)

    # L2: medium sizes (256KB - 64MB)
    l2_data = [(s, l) for s, l in data if 256 * 1024 <= s < 64 * 1024 * 1024]
    if l2_data:
        result["l2_latency_cycles"] = float(np.median([l for _, l in l2_data]))

    # DRAM: large sizes (> 64MB)
    dram_data = [(s, l) for s, l in data if s >= 64 * 1024 * 1024]
    if dram_data:
        result["dram_latency_cycles"] = float(np.median([l for _, l in dram_data]))
    elif latencies:
        result["dram_latency_cycles"] = max(latencies)

    # L2 cache capacity: find knee point (where latency jumps significantly)
    if len(data) >= 3:
        result["l2_cache_bytes"] = _find_knee(sizes, latencies)

    return result


def _parse_bandwidth_sweep(stdout: str) -> dict:
    """Expected output lines: '<bytes> <read_GBs> <write_GBs>'"""
    data = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                size = int(parts[0])
                read_bw = float(parts[1])
                write_bw = float(parts[2]) if len(parts) >= 3 else read_bw
                data.append((size, read_bw, write_bw))
            except ValueError:
                continue

    if not data:
        return {}

    data.sort(key=lambda x: x[0])
    result = {"data": data}

    # Peak bandwidth: max at large sizes
    large = [(s, r, w) for s, r, w in data if s >= 64 * 1024 * 1024]
    if large:
        result["peak_read_GBs"] = max(r for _, r, _ in large)
        result["peak_write_GBs"] = max(w for _, _, w in large)
    else:
        result["peak_read_GBs"] = max(r for _, r, _ in data)
        result["peak_write_GBs"] = max(w for _, _, w in data)

    # L2 bandwidth: small sizes
    small = [(s, r, w) for s, r, w in data if s <= 4 * 1024 * 1024]
    if small:
        result["l2_bandwidth_GBs"] = max(r for _, r, _ in small)

    # L2 cache capacity: knee in bandwidth curve
    sizes = [d[0] for d in data]
    bws = [d[1] for d in data]
    if len(data) >= 3:
        result["l2_cache_bytes"] = _find_knee_bw(sizes, bws)

    return result


def _parse_bank_conflict(stdout: str) -> dict:
    """Expected output lines: '<stride> <cycles>'"""
    data = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                stride = int(parts[0])
                cycles = float(parts[1])
                data.append((stride, cycles))
            except ValueError:
                continue

    if not data:
        return {}

    result = {"data": data}
    stride_map = {s: c for s, c in data}

    base = stride_map.get(1, None)
    worst = stride_map.get(32, None)
    if base and worst and base > 0:
        result["penalty_ratio"] = worst / base
        result["no_conflict_cycles"] = base
        result["max_conflict_cycles"] = worst

    return result


def _parse_shmem_probe(stdout: str) -> dict:
    """Parse shmem_probe.cu output.
    Expected lines:
      max_shmem_per_block_kb: <val>
      max_shmem_optin_kb: <val>
      shmem_bandwidth_GBs: <val>
    """
    result = {}
    for line in stdout.splitlines():
        m = re.match(r"([\w_]+):\s*([0-9.]+)", line.strip())
        if m:
            result[m.group(1)] = float(m.group(2))
    return result


def _parse_generic(stdout: str) -> dict:
    """Try to extract any key: value pairs."""
    result = {}
    for line in stdout.splitlines():
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)[\s:=]+([0-9.]+)", line.strip())
        if m:
            result[m.group(1)] = float(m.group(2))
    return result


def _find_knee(sizes: list, latencies: list) -> int:
    """Find the size where latency jumps (cache capacity boundary)."""
    if len(sizes) < 3:
        return sizes[-1] if sizes else 0
    # Find largest jump in latency
    max_jump = 0
    knee_idx = len(sizes) // 2
    for i in range(1, len(latencies)):
        jump = latencies[i] - latencies[i - 1]
        if jump > max_jump:
            max_jump = jump
            knee_idx = i
    return sizes[knee_idx - 1]


def _find_knee_bw(sizes: list, bws: list) -> int:
    """Find size where bandwidth drops significantly (cache → DRAM transition)."""
    if len(sizes) < 3:
        return sizes[-1] if sizes else 0
    max_drop = 0
    knee_idx = len(sizes) // 2
    for i in range(1, len(bws)):
        drop = bws[i - 1] - bws[i]
        if drop > max_drop:
            max_drop = drop
            knee_idx = i
    return sizes[knee_idx - 1]
