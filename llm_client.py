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
            # Strip thinking tags if present (Qwen3 thinking mode)
            if "<think>" in content:
                content = content[content.rfind("</think>") + len("</think>"):].strip()
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

    def _build_prompt(self, result: AnalyzedResult) -> str:
        data = {
            "target": result.task.name,
            "task_type": result.task.task_type,
            "analyzer_value": result.value,
            "analyzer_confidence": result.confidence,
            "analyzer_reasoning": result.reasoning,
            "evidence": result.evidence,
        }
        return f"""Analyze this GPU profiling result and provide your assessment.

Target: {result.task.name}
Task type: {result.task.task_type}

Measured data:
{json.dumps(data, indent=2, default=str)}

Provide a JSON response with exactly these fields:
{{
  "value": <the final measured value — number or classification string>,
  "reasoning": "<step-by-step explanation citing specific measured values>",
  "confidence": "high|medium|low",
  "anomalies": "<any detected anomalies or empty string>"
}}

For hardware probe targets, "value" should be a number.
For operator profiling targets like bottleneck_diagnosis, "value" should be a string like "memory_bound" or "compute_bound".
"""
