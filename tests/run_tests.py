#!/usr/bin/env python3
"""
tests/run_tests.py — MLSYS Phase 1 全场景测试
运行方式: python3 tests/run_tests.py
"""
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"

results = []


def run_spec(spec_path: str, desc: str, expect_keys: list[str] = None,
             expect_no_crash: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"TEST: {desc}")
    print(f"SPEC: {spec_path}")
    print(f"{'='*60}")

    ret = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), spec_path],
        capture_output=False,
        cwd=str(ROOT),
    )

    passed = True
    notes = []

    # 1. 不崩溃
    if expect_no_crash and ret.returncode != 0:
        print(f"  {FAIL} 进程异常退出 (rc={ret.returncode})")
        passed = False
    else:
        print(f"  {PASS} 进程正常退出 (rc={ret.returncode})")

    # 2. results.json 存在
    results_path = ROOT / "results.json"
    if not results_path.exists():
        print(f"  {FAIL} results.json 不存在")
        passed = False
    else:
        data = json.loads(results_path.read_text())
        print(f"  {PASS} results.json 存在，包含 {len(data)} 个 key: {list(data.keys())}")

        # 3. 期望的 key 都在
        if expect_keys:
            for k in expect_keys:
                if k not in data:
                    print(f"  {FAIL} 缺少 key: {k}")
                    passed = False
                elif data[k] is None:
                    print(f"  {WARN} key={k} 值为 None")
                    notes.append(f"{k}=None")
                else:
                    print(f"  {PASS} {k} = {data[k]}")

    # 4. summary.txt 存在且有 Summary 段
    summary_path = ROOT / "logs" / "summary.txt"
    if not summary_path.exists():
        print(f"  {FAIL} logs/summary.txt 不存在")
        passed = False
    else:
        summary = summary_path.read_text()
        if "[ Summary ]" in summary:
            print(f"  {PASS} summary.txt 含 LLM Summary 段")
        else:
            print(f"  {WARN} summary.txt 无 LLM Summary（可能 API key 未配置）")

    status = "PASS" if passed else "FAIL"
    print(f"\n  → {status}" + (f" ({', '.join(notes)})" if notes else ""))
    results.append((desc, passed, notes))
    return {"passed": passed}


def main():
    # 确保 sample_matmul 已编译
    if not Path("/tmp/sample_matmul").exists():
        print("编译 sample_matmul...")
        subprocess.run(
            ["nvcc", "-O3", "-arch=sm_89",
             str(ROOT / "probes/sample_matmul.cu"), "-o", "/tmp/sample_matmul"],
            check=True
        )

    # ----------------------------------------------------------------
    # Case 1: Hardware probe + Operator profiling 混合（核心场景）
    # ----------------------------------------------------------------
    run_spec(
        str(ROOT / "tests/spec_full.json"),
        "混合场景：Hardware probe + Operator profiling",
        expect_keys=[
            "actual_boost_clock_mhz",
            "dram_latency_cycles",
            "bank_conflict_penalty",
            "l2_cache_capacity",
            "bottleneck_diagnosis",
            "tensor_core_utilization",
            "memory_hierarchy_analysis",
        ],
    )

    # ----------------------------------------------------------------
    # Case 2: run 路径不存在（边界）
    # ----------------------------------------------------------------
    run_spec(
        str(ROOT / "tests/spec_bad_run.json"),
        "边界：run 路径不存在",
        expect_keys=["bottleneck_diagnosis", "actual_boost_clock_mhz"],
        expect_no_crash=True,  # 不应崩溃，应降级处理
    )

    # ----------------------------------------------------------------
    # Case 3: spec 里没有 run 字段（纯 hardware probe）
    # ----------------------------------------------------------------
    run_spec(
        str(ROOT / "tests/spec_no_run.json"),
        "边界：无 run 字段（纯 hardware probe）",
        expect_keys=["actual_boost_clock_mhz", "dram_latency_cycles"],
    )

    # ----------------------------------------------------------------
    # Case 4: targets 全是未知名称
    # ----------------------------------------------------------------
    run_spec(
        str(ROOT / "tests/spec_unknown_targets.json"),
        "边界：targets 含未知名称",
        expect_keys=["warp_efficiency", "sm_count", "unknown_metric_xyz"],
        expect_no_crash=True,
    )

    # ----------------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    for desc, p, notes in results:
        icon = PASS if p else FAIL
        print(f"  {icon} {desc}" + (f"  [{', '.join(notes)}]" if notes else ""))
    print(f"\n  {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
