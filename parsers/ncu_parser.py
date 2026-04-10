"""Parse ncu --csv output into {kernel_name: {metric_name: value}}."""
import csv
import io
import re
import logging

log = logging.getLogger(__name__)


def parse_ncu_csv(text: str) -> dict[str, dict[str, float]]:
    """
    ncu --csv output has comment lines starting with '==' and then CSV data.
    Returns {kernel_name: {metric_name: numeric_value}}.
    """
    # Strip comment lines
    lines = [l for l in text.splitlines() if not l.startswith("==")]
    clean = "\n".join(lines)

    if not clean.strip():
        return {}

    try:
        reader = csv.DictReader(io.StringIO(clean))
        rows = list(reader)
    except Exception as e:
        log.warning(f"CSV parse error: {e}")
        return {}

    result: dict[str, dict[str, float]] = {}

    for row in rows:
        # Column names vary by ncu version; try common patterns
        kernel = (
            row.get("Kernel Name") or row.get("kernel_name") or
            row.get("ID") or "unknown"
        )
        metric = (
            row.get("Metric Name") or row.get("metric_name") or
            row.get("Metric") or ""
        )
        raw_val = (
            row.get("Metric Value") or row.get("metric_value") or
            row.get("Value") or ""
        )

        if not metric:
            continue

        value = _parse_value(raw_val)
        if kernel not in result:
            result[kernel] = {}
        # Accumulate (sum) repeated metrics across invocations
        if metric in result[kernel]:
            result[kernel][metric] += value
        else:
            result[kernel][metric] = value

    return result


def _parse_value(s: str) -> float:
    """Parse ncu metric value strings like '1,234,567', '98.5%', '1.2 K', '3.4 G'."""
    if not s:
        return 0.0
    s = s.strip().replace(",", "")
    # Remove % sign
    s = s.rstrip("%")
    # Handle suffixes
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0
