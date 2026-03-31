"""
Production-aligned recall evaluation dataclass and utilities.

Provides RecallResult, page-type configuration matching production exactly,
and formatting helpers for recall pipeline results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RecallResult:
    """Result from a single recall evaluation query."""
    soft_recall: float
    hard_recall: float
    winsorized_soft_recall: float = 0.0
    proportion: float = 0.0
    num_requests: int = 0

    def to_dict(self) -> dict:
        return {
            "soft_recall": self.soft_recall,
            "hard_recall": self.hard_recall,
            "winsorized_soft_recall": self.winsorized_soft_recall,
            "proportion": self.proportion,
            "num_requests": self.num_requests,
        }


# PageTypeConfig matching production exactly.
# Keys are page_type IDs; values are the pipeline stage cardinalities.
PAGE_TYPE_CONFIG_MAP = {
    19: {
        "num_ads_pm_scored": 190905,
        "num_ads_fm_scanned": 15371,
        "num_ads_fm_matched": 11119,
        "num_ads_ai_ranked": 4189,
        "num_ads_ai_returned": 709,
        "num_ads_af_ranked": 534,
        "num_ads_af_returned": 84,
    },
    35: {
        "num_ads_pm_scored": 190905,
        "num_ads_fm_scanned": 15371,
        "num_ads_fm_matched": 11119,
        "num_ads_ai_ranked": 4189,
        "num_ads_ai_returned": 709,
        "num_ads_af_ranked": 534,
        "num_ads_af_returned": 84,
    },
    49: {
        "num_ads_pm_scored": 190905,
        "num_ads_fm_scanned": 15371,
        "num_ads_fm_matched": 11119,
        "num_ads_ai_ranked": 4189,
        "num_ads_ai_returned": 709,
        "num_ads_af_ranked": 534,
        "num_ads_af_returned": 84,
    },
}


def parse_recall_result(df_row: dict, proportion: float = 0.0) -> RecallResult:
    """Parse recall from a Presto query result row.

    Args:
        df_row: A dict from the Presto JSON result with keys like
                'soft_recall', 'hard_recall', 'winsorized_soft_recall'.
        proportion: The traffic proportion used for this evaluation.

    Returns:
        A RecallResult populated from the row.
    """
    return RecallResult(
        soft_recall=float(df_row.get("soft_recall", 0.0) or 0.0),
        hard_recall=float(df_row.get("hard_recall", 0.0) or 0.0),
        winsorized_soft_recall=float(df_row.get("winsorized_soft_recall", 0.0) or 0.0),
        proportion=proportion,
    )


def format_results_table(results: Dict[float, RecallResult]) -> str:
    """Format recall results as a human-readable table string.

    Args:
        results: Mapping from proportion -> RecallResult.

    Returns:
        A formatted table string.
    """
    header = (
        f"{'Proportion':>12s}  "
        f"{'Soft Recall':>12s}  "
        f"{'Hard Recall':>12s}  "
        f"{'Winsorized':>12s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for proportion in sorted(results.keys()):
        r = results[proportion]
        lines.append(
            f"{proportion:>12.2f}  "
            f"{r.soft_recall:>12.6f}  "
            f"{r.hard_recall:>12.6f}  "
            f"{r.winsorized_soft_recall:>12.6f}"
        )

    return "\n".join(lines)


def format_results_markdown(results: Dict[float, RecallResult]) -> str:
    """Format recall results as a Markdown table.

    Args:
        results: Mapping from proportion -> RecallResult.

    Returns:
        A Markdown table string.
    """
    lines = [
        "| Proportion | Soft Recall | Hard Recall | Winsorized Soft Recall |",
        "|------------|-------------|-------------|------------------------|",
    ]
    for proportion in sorted(results.keys()):
        r = results[proportion]
        lines.append(
            f"| {proportion:.2f} | {r.soft_recall:.6f} | {r.hard_recall:.6f} | {r.winsorized_soft_recall:.6f} |"
        )
    return "\n".join(lines)
