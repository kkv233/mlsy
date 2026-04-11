"""Reporter: write results.json and per-target reasoning logs."""
import json
import logging
from pathlib import Path
from .models import AnalyzedResult
from llm_client import LLMClient

log = logging.getLogger(__name__)
LOGS_DIR = Path("logs")


class Reporter:
    def __init__(self):
        self.llm = LLMClient()

    def report(self, results: list[AnalyzedResult], output_path: str = "results.json"):
        LOGS_DIR.mkdir(exist_ok=True)
        final = {}
        results_with_llm = []

        for r in results:
            llm_reasoning = ""
            try:
                llm_reasoning = self.llm.interpret(r)
            except Exception as e:
                log.warning(f"LLM interpretation failed for {r.task.name}: {e}")

            results_with_llm.append((r, llm_reasoning))

            log_path = LOGS_DIR / f"reasoning_{r.task.name}.txt"
            log_path.write_text(self._format_reasoning(r, llm_reasoning))

            value = r.value
            if llm_reasoning:
                try:
                    parsed = json.loads(llm_reasoning)
                    if "value" in parsed:
                        value = parsed["value"]
                except Exception:
                    pass

            final[r.task.name] = value
            log.info(f"Result [{r.task.name}]: {value}")

        with open(output_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        log.info(f"results.json written: {output_path}")

        # Generate concise summary via dedicated summarizer call
        summary_text = ""
        try:
            summary_text = self.llm.summarize(results, final)
        except Exception as e:
            log.warning(f"Summary generation failed: {e}")

        self._write_summary(results_with_llm, final, summary_text)

    def _write_summary(self, results_with_llm: list[tuple], final: dict, summary_text: str):
        lines = [
            "=" * 60,
            "MLSYS Phase 1 — GPU Profiling Analysis Report",
            "=" * 60,
            "",
            "[ Final Results ]",
            "",
        ]
        for k, v in final.items():
            lines.append(f"  {k}: {v}")

        if summary_text:
            lines += ["", "[ Summary ]", "", summary_text]

        lines += ["", "=" * 60, "", "[ Per-Target Details ]", ""]

        for r, llm_text in results_with_llm:
            lines.append(f">> {r.task.name}")
            lines.append(f"   value      : {final.get(r.task.name, r.value)}")
            lines.append(f"   confidence : {r.confidence}")
            lines.append(f"   method     : {r.reasoning}")
            if r.error:
                lines.append(f"   error      : {r.error}")
            lines.append("")

        (LOGS_DIR / "summary.txt").write_text("\n".join(lines))

    def _format_reasoning(self, r: AnalyzedResult, llm_text: str) -> str:
        lines = [
            f"=== Target: {r.task.name} ===",
            f"Task type: {r.task.task_type}",
            f"",
            f"--- Analyzer Result ---",
            f"Value: {r.value}",
            f"Confidence: {r.confidence}",
            f"Reasoning: {r.reasoning}",
            f"",
            f"--- Evidence ---",
            json.dumps(r.evidence, indent=2, default=str),
            f"",
        ]
        if llm_text:
            lines += [f"--- LLM Interpretation ---", llm_text, ""]
        if r.error:
            lines += [f"--- Error ---", r.error, ""]
        return "\n".join(lines)
