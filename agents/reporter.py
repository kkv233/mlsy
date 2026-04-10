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

        for r in results:
            # Optionally enrich with LLM reasoning
            llm_reasoning = ""
            try:
                llm_reasoning = self.llm.interpret(r)
            except Exception as e:
                log.warning(f"LLM interpretation failed for {r.task.name}: {e}")

            # Write per-target reasoning log
            log_path = LOGS_DIR / f"reasoning_{r.task.name}.txt"
            log_path.write_text(self._format_reasoning(r, llm_reasoning))

            # Determine final value (LLM may refine it)
            value = r.value
            if llm_reasoning:
                try:
                    parsed = json.loads(llm_reasoning)
                    if "value" in parsed:
                        value = parsed["value"]
                except Exception:
                    pass  # keep analyzer value

            final[r.task.name] = value
            log.info(f"Result [{r.task.name}]: {value}")

        with open(output_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        log.info(f"results.json written: {output_path}")

        # Write summary
        (LOGS_DIR / "summary.txt").write_text(
            json.dumps(final, indent=2, default=str)
        )

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
