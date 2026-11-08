"""Post-process PromoPilot ranking output with a minimum shortlist score.

This module does not alter scores, re-rank candidates, or call an LLM.
It preserves the full ranking and only removes low-scoring selections from
 the recommended shortlist. Editorial flags remain visible and do not by
themselves disqualify a candidate.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def apply_shortlist_policy(ranking: dict[str, Any], min_score: float = 40.0) -> dict[str, Any]:
    if not isinstance(ranking, dict):
        raise TypeError("Ranking input must be a JSON object")
    if min_score < 0 or min_score > 100:
        raise ValueError("min_score must be between 0 and 100")
    full_ranking = ranking.get("full_ranking")
    shortlist = ranking.get("shortlist")
    if not isinstance(full_ranking, list) or not isinstance(shortlist, list):
        raise ValueError("Input must contain list fields 'full_ranking' and 'shortlist'")

    out = deepcopy(ranking)
    by_id = {c.get("candidate_id"): c for c in out["full_ranking"] if isinstance(c, dict)}
    kept: list[dict[str, Any]] = []
    below_threshold: list[dict[str, Any]] = []

    for selected in out["shortlist"]:
        candidate_id = selected.get("candidate_id")
        score = float(selected.get("ranking_score", selected.get("adjusted_score", 0)))
        flags = list(selected.get("quality_flags") or [])
        selected["editorial_review_required"] = bool(flags)
        selected["selection_status"] = "provisional_editorial_review" if flags else "recommended"
        if score < min_score:
            selected["shortlist_selected"] = False
            selected["shortlist_eligible"] = False
            selected["shortlist_exclusion_reason"] = (
                f"Below minimum shortlist score ({score:g} < {min_score:g}); retained in full ranking for audit."
            )
            below_threshold.append(selected)
            full_item = by_id.get(candidate_id)
            if full_item is not None:
                full_item["shortlist_selected"] = False
                full_item["shortlist_eligible"] = False
                full_item["shortlist_exclusion_reason"] = selected["shortlist_exclusion_reason"]
                full_item["editorial_review_required"] = bool(flags)
                full_item["selection_status"] = "below_shortlist_threshold"
        else:
            selected["shortlist_selected"] = True
            kept.append(selected)
            full_item = by_id.get(candidate_id)
            if full_item is not None:
                full_item["shortlist_selected"] = True
                full_item["editorial_review_required"] = bool(flags)
                full_item["selection_status"] = selected["selection_status"]

    # Keep original full_ranking scores/order/flags; annotate review state for every entry.
    for item in out["full_ranking"]:
        flags = list(item.get("quality_flags") or [])
        item.setdefault("editorial_review_required", bool(flags))
        if item.get("shortlist_selected") and item.get("selection_status") is None:
            item["selection_status"] = "provisional_editorial_review" if flags else "recommended"

    out["shortlist"] = kept
    out["editorial_review_excluded"] = below_threshold
    out["shortlist_policy"] = {
        "name": "minimum_score_with_editorial_flags_preserved",
        "minimum_score": min_score,
        "score_field": "ranking_score",
        "scores_modified": False,
        "editorial_flags_auto_disqualify": False,
        "below_threshold_candidates_retained_in_full_ranking": True,
    }
    summary = out.setdefault("summary", {})
    summary["shortlist_count"] = len(kept)
    summary["shortlist_target"] = out.get("ranking_policy", {}).get("top_n")
    target = summary.get("shortlist_target")
    summary["shortlist_slots_unfilled"] = max(0, int(target) - len(kept)) if target is not None else None
    summary["shortlist_below_threshold_count"] = len(below_threshold)
    summary["shortlist_editorial_review_count"] = sum(bool(x.get("editorial_review_required")) for x in kept)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a minimum-score policy to PromoPilot ranking output.")
    parser.add_argument("--ranking", required=True, help="Path to epXX_ranking.json")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--min-score", type=float, default=40.0, help="Minimum score for recommended shortlist (default: 40)")
    args = parser.parse_args()
    source = Path(args.ranking)
    destination = Path(args.out)
    try:
        ranking = json.loads(source.read_text(encoding="utf-8"))
        result = apply_shortlist_policy(ranking, args.min_score)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved shortlist policy output: {destination}")
    print(f"Recommended shortlist: {result['summary']['shortlist_count']}")
    print(f"Below threshold: {result['summary']['shortlist_below_threshold_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
