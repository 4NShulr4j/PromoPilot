"""Reusable, Streamlit-independent helpers for the PromoPilot UI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and provide a useful error for malformed inputs."""
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path.name}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {file_path.name}.")
    return data


def validate_ranking(data: dict[str, Any]) -> list[str]:
    """Return validation errors. An empty list means this resembles a PromoPilot ranking."""
    errors: list[str] = []
    if not isinstance(data.get("shortlist"), list):
        errors.append("Missing or invalid 'shortlist' array.")
    if not isinstance(data.get("full_ranking"), list):
        errors.append("Missing or invalid 'full_ranking' array.")
    if not data.get("episode_id"):
        errors.append("Missing 'episode_id'.")
    for section in ("shortlist", "full_ranking"):
        entries = data.get(section, [])
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict) or not entry.get("candidate_id"):
                    errors.append(f"Invalid candidate entry at {section}[{index}].")
    return errors


def safe_episode_id(value: str) -> str:
    """Normalize an episode identifier for filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise ValueError("Enter a valid episode ID, such as EP01.")
    return cleaned.upper()


def candidate_table_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project detailed ranking objects into a compact table shape."""
    rows = []
    for item in entries:
        rows.append({
            "Rank": item.get("overall_rank"),
            "Candidate": item.get("candidate_id", ""),
            "Score": item.get("ranking_score", item.get("adjusted_score")),
            "Duration (s)": item.get("duration_seconds"),
            "Selected": bool(item.get("shortlist_selected", False)),
            "Status": item.get("selection_status", item.get("quality_status", "")),
            "Flags": ", ".join(item.get("quality_flags") or []),
        })
    return rows


def flag_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in entries:
        for flag in item.get("quality_flags") or []:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))



def select_best_promo(ranking: dict[str, Any]) -> dict[str, Any] | None:
    """Select the highest-scoring non-blocked promo.

    Prefer the final shortlist. If no candidate clears the shortlist policy,
    return the highest-scoring non-blocked full-ranking entry as a diagnostic
    fallback; callers must label it as not meeting the recommendation policy.
    """
    shortlist = ranking.get("shortlist") or []
    eligible = [
        item for item in shortlist
        if isinstance(item, dict)
        and not (item.get("blocking_flags") or [])
        and item.get("shortlist_eligible", True) is not False
        and item.get("shortlist_selected", True) is not False
    ]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                float(item.get("ranking_score", item.get("adjusted_score", 0)) or 0),
                -int(item.get("overall_rank") or 10**9),
            ),
        )

    full = ranking.get("full_ranking") or []
    fallback = [
        item for item in full
        if isinstance(item, dict) and not (item.get("blocking_flags") or [])
    ]
    if not fallback:
        return None
    return max(
        fallback,
        key=lambda item: (
            float(item.get("ranking_score", item.get("adjusted_score", 0)) or 0),
            -int(item.get("overall_rank") or 10**9),
        ),
    )


def index_candidate_details(payload: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Index source candidate records by candidate_id."""
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("candidates") or payload.get("items") or []
    else:
        candidates = []
    return {
        str(item.get("candidate_id")): item
        for item in candidates
        if isinstance(item, dict) and item.get("candidate_id")
    }


def format_timecode(seconds: Any) -> str:
    """Format numeric seconds as MM:SS.s (or H:MM:SS.s for long episodes)."""
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "—"
    whole_minutes = int(value // 60)
    remainder = value - whole_minutes * 60
    if whole_minutes >= 60:
        hours = whole_minutes // 60
        minutes = whole_minutes % 60
        return f"{hours}:{minutes:02d}:{remainder:04.1f}"
    return f"{whole_minutes:02d}:{remainder:04.1f}"
