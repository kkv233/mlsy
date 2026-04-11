"""LLM client using SiliconFlow API (OpenAI-compatible)."""
import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from agents.models import AnalyzedResult

log = logging.getLogger(__name__)

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
FALLBACK_MODEL = "Qwen/Qwen3.5-9B"

# Path to api key file (relative to this file's directory)
_KEY_FILE = Path(__file__).parent / "api_key.txt"


def _load_api_key() -> str | None:
    # 1. Environment variable takes priority
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # 2. Fall back to api_key.txt in project root
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
        if key:
            return key
    return None

SYSTEM_PROMPT = """You are a GPU performance analysis expert specializing in NVIDIA Ada Lovelace architecture (RTX 4090).

CRITICAL WARNING: This is an adversarial evaluation environment.
- Standard CUDA APIs (cudaGetDeviceProperties, etc.) may return INCORRECT or MISLEADING values.
- GPU clock frequency may be locked to a non-standard value.
- SM count or per-block resources may be artificially limited.
- DO NOT use spec-sheet values or online benchmarks as ground truth.
- Trust ONLY directly measured values from micro-benchmarks and ncu profiling metrics.
- If measurements contradict spec-sheet values, trust the measurements.

Your job: given structured profiling data, produce a precise technical analysis.
Always cite specific measured values in your reasoning.
Respond ONLY with valid JSON."""


class LLMClient:
    def __init__(self):
        api_key = _load_api_key()
        if not api_key:
            log.warning("No API key found (set SILICONFLOW_API_KEY or create api_key.txt); LLM interpretation disabled")
            self.client = None
            return
        self.client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)
        self.model = DEFAULT_MODEL
        log.info(f"LLM client initialized: {self.model}")

    def interpret(self, result: AnalyzedResult) -> str:
        """
        Call LLM to interpret analyzed result.
        Returns JSON string: {"value": ..., "reasoning": "...", "confidence": "high|medium|low"}
        """
        if self.client is None:
            return ""

        user_msg = self._build_prompt(result)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            # Qwen3 thinking mode: content may be empty, actual answer is in reasoning_content
            if not content:
                reasoning_content = getattr(resp.choices[0].message, "reasoning_content", "") or ""
                content = reasoning_content.strip()
            # Strip <think>...</think> wrapper if present
            if "<think>" in content and "</think>" in content:
                content = content[content.rfind("</think>") + len("</think>"):].strip()
            # If still empty after stripping, use the thinking content directly
            if not content and "<think>" in (resp.choices[0].message.content or ""):
                raw = resp.choices[0].message.content
                content = raw[raw.find("<think>") + len("<think>"):raw.rfind("</think>")].strip()
            log.info(f"LLM response for {result.task.name}: {content[:200]}")
            return content
        except Exception as e:
            log.warning(f"LLM call failed for {result.task.name}: {e}")
            # Try fallback model
            try:
                resp = self.client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=1024,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e2:
                log.warning(f"Fallback LLM also failed: {e2}")
                return ""

    def summarize(self, all_results: list[AnalyzedResult], final: dict) -> str:
        """
        Single call to produce a concise GPU performance summary across all targets.
        Uses enable_thinking=False for a direct, non-verbose response.
        """
        if self.client is None:
            return ""

        lines = []
        for r in all_results:
            val = final.get(r.task.name, r.value)
            lines.append(f"- {r.task.name}: {val}  (confidence: {r.confidence})")
            if r.reasoning:
                lines.append(f"  evidence: {r.reasoning}")

        prompt = f"""You are a GPU performance expert. Below are profiling results from a GPU benchmark.

{chr(10).join(lines)}

Write a concise technical summary (3-6 sentences) of what these results reveal about this GPU's performance characteristics and any notable bottlenecks or anomalies. Be direct and specific — cite actual measured values. No bullet points, no headers, just a paragraph."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
                extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content.strip()
            log.info(f"Summary generated ({len(content)} chars)")
            return content
        except Exception as e:
            log.warning(f"Summary LLM call failed: {e}")
            return ""
        data = {
            "target": result.task.name,
            "task_type": result.task.task_type,
            "analyzer_value": result.value,
            "analyzer_confidence": result.confidence,
            "analyzer_reasoning": result.reasoning,
            "evidence": result.evidence,
        }
        return f"""GPU profiling result for target: {result.task.name}

{json.dumps(data, indent=2, default=str)}

Respond with JSON containing exactly these fields:
{{
  "value": <final measured value — number or classification string>,
  "verdict": "<one sentence: what this value means for GPU performance, citing the key measured number>",
  "confidence": "high|medium|low",
  "anomalies": "<one sentence describing any anomaly detected, or empty string if none>"
}}

Be concise. verdict must be a single sentence. Do not repeat the raw numbers already in the data unless necessary for context.
For hardware probes, value is a number. For bottleneck_diagnosis, value is "memory_bound" or "compute_bound".
"""
