#!/usr/bin/env python3
"""
MLSYS Phase 1 — GPU Profiling Multi-Agent Framework
Entrypoint: reads target_spec.json, runs profiling pipeline, writes results.json
"""
import json
import sys
import os
import logging
from pathlib import Path

from agents.spec_reader import SpecReader
from agents.planner import Planner
from agents.executor import Executor
from agents.analyzer import Analyzer
from agents.reporter import Reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run.log", mode="w"),
    ],
)
log = logging.getLogger("main")


def main():
    spec_path = sys.argv[1] if len(sys.argv) > 1 else "target_spec.json"
    if not Path(spec_path).exists():
        log.error(f"Spec file not found: {spec_path}")
        sys.exit(1)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    log.info(f"=== MLSYS Phase 1 Pipeline ===")
    log.info(f"Spec: {spec_path}")

    # Stage 1: Parse spec
    reader = SpecReader()
    tasks = reader.read(spec_path)
    log.info(f"Tasks: {[t.name for t in tasks]}")

    # Stage 2: Plan
    planner = Planner()
    plans = [planner.plan(t) for t in tasks]

    # Stage 3: Execute
    executor = Executor()
    raw_results = [executor.execute(p) for p in plans]

    # Stage 4: Analyze
    analyzer = Analyzer()
    analyzed = [analyzer.analyze(r) for r in raw_results]

    # Stage 5: Report
    reporter = Reporter()
    reporter.report(analyzed, output_path="results.json")

    log.info("=== Done. results.json written. ===")


if __name__ == "__main__":
    main()
