"""PromoPilot deterministic ranking and shortlist selection.

Uses the Evaluation Quality Gate's adjusted score as the ranking score. It does
not call an LLM and does not apply the quality-gate penalty a second time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

MID_THOUGHT_FLAGS = {"CUT_MID_THOUGHT"}
START_KEYS = {
    "start_seconds", "start_sec", "start_time_seconds", "start_time_sec",
    "clip_start_seconds", "clip_start_sec", "start", "start_time",
    "start_timestamp", "start_timecode", "in_point", "clip_start",
}
END_KEYS = {
    "end_seconds", "end_sec", "end_time_seconds", "end_time_sec",
    "clip_end_seconds", "clip_end_sec", "end", "end_time",
    "end_timestamp", "end_timecode", "out_point", "clip_end",
}
DURATION_KEYS = {"duration_seconds", "duration_sec", "clip_duration_seconds", "duration"}
EVENT_CONTAINER_KEYS = {
    "event_ids", "source_event_ids", "timeline_events", "source_events",
    "event_sequence", "included_events", "events",
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def parse_time(value: Any) -> float | None:
    """Parse seconds, MM:SS(.ms), or HH:MM:SS(.ms)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def walk_dicts(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_dicts(value)


def first_candidate_id(node: dict[str, Any]) -> str | None:
    for d in walk_dicts(node):
        for key, value in d.items():
            if norm_key(key) in {"candidate_id", "promo_candidate_id", "id"} and isinstance(value, str):
                if re.search(r"PROMO|CANDIDATE", value, re.I):
                    return value
    return None


def collect_candidate_records(root: Any) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for d in walk_dicts(root):
        cid = None
        for key, value in d.items():
            if norm_key(key) in {"candidate_id", "promo_candidate_id"} and isinstance(value, str):
                cid = value
                break
        if cid and ("PROMO" in cid.upper() or "CANDIDATE" in cid.upper()):
            # Keep the largest matching object; it is most likely the full record.
            if cid not in found or len(d) > len(found[cid]):
                found[cid] = d
    # Some schemas use a plain `id` field inside a candidate collection.
    for d in walk_dicts(root):
        if "id" in d and isinstance(d["id"], str) and re.search(r"PROMO|CANDIDATE", d["id"], re.I):
            cid = d["id"]
            if cid not in found or len(d) > len(found[cid]):
                found[cid] = d
    return found


def extract_span(record: dict[str, Any]) -> tuple[float | None, float | None]:
    start = end = duration = None
    for d in walk_dicts(record):
        for key, value in d.items():
            nk = norm_key(key)
            if start is None and nk in START_KEYS:
                start = parse_time(value)
            elif end is None and nk in END_KEYS:
                end = parse_time(value)
            elif duration is None and nk in DURATION_KEYS:
                duration = parse_time(value)
        if start is not None and end is not None:
            break
    if start is not None and end is None and duration is not None:
        end = start + duration
    if end is not None and start is None and duration is not None:
        start = end - duration
    if start is not None and end is not None and end < start:
        # Handle malformed/reversed ranges defensively instead of silently swapping.
        return None, None
    return start, end


def extract_event_ids(record: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for d in walk_dicts(record):
        for key, value in d.items():
            nk = norm_key(key)
            if nk in EVENT_CONTAINER_KEYS:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and re.search(r"EV\d+", item):
                            ids.add(item)
                        elif isinstance(item, dict):
                            for k2, v2 in item.items():
                                if norm_key(k2) in {"event_id", "id"} and isinstance(v2, str) and re.search(r"EV\d+", v2):
                                    ids.add(v2)
                elif isinstance(value, str) and re.search(r"EV\d+", value):
                    ids.add(value)
    return ids


def containment_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return overlap / shorter if shorter > 0 else 0.0


def event_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def build_overlap_clusters(items: list[dict[str, Any]], threshold: float = 0.80) -> None:
    """Annotate near-duplicate groups using time containment or event overlap."""
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            time_sim = 0.0
            if a.get("start_seconds") is not None and b.get("start_seconds") is not None:
                time_sim = containment_overlap(
                    (a["start_seconds"], a["end_seconds"]),
                    (b["start_seconds"], b["end_seconds"]),
                )
            event_sim = event_overlap(a.get("source_event_ids", set()), b.get("source_event_ids", set()))
            sim = max(time_sim, event_sim)
            if sim >= threshold:
                union(i, j)
                a.setdefault("overlap_with", []).append(b["candidate_id"])
                b.setdefault("overlap_with", []).append(a["candidate_id"])
                a["overlap_similarity_max"] = max(a.get("overlap_similarity_max", 0.0), round(sim, 3))
                b["overlap_similarity_max"] = max(b.get("overlap_similarity_max", 0.0), round(sim, 3))

    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    group_num = 0
    for idxs in groups.values():
        group_num += 1
        cluster_id = f"OVERLAP_{group_num:02d}"
        for idx in idxs:
            items[idx]["overlap_cluster_id"] = cluster_id
            items[idx]["overlap_cluster_size"] = len(idxs)


def run_ranking(quality_gate: dict[str, Any], candidates_doc: Any | None = None, top_n: int = 5,
                overlap_threshold: float = 0.80) -> dict[str, Any]:
    gate_results = quality_gate.get("results", [])
    candidate_records = collect_candidate_records(candidates_doc) if candidates_doc is not None else {}
    items: list[dict[str, Any]] = []
    missing_metadata: list[str] = []

    for result in gate_results:
        cid = result.get("candidate_id")
        if not cid:
            continue
        candidate = candidate_records.get(cid, {})
        start, end = extract_span(candidate) if candidate else (None, None)
        event_ids = extract_event_ids(candidate) if candidate else set()
        flags = result.get("flags") or []
        codes = [f.get("code", "UNKNOWN") for f in flags]
        blocking = sorted(set(codes) & MID_THOUGHT_FLAGS)
        score = float(result.get("adjusted_score", result.get("original_model_score", 0.0)))
        if start is None or end is None:
            missing_metadata.append(cid)
        items.append({
            "candidate_id": cid,
            "original_model_score": float(result.get("original_model_score", score)),
            "adjusted_score": score,
            "ranking_score": score,
            "quality_status": result.get("quality_status", "unknown"),
            "quality_flags": codes,
            "blocking_flags": blocking,
            "shortlist_eligible": not bool(blocking),
            "shortlist_exclusion_reason": "Confirmed mid-thought cut; keep out of recommended shortlist pending editorial repair." if blocking else None,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": round(end - start, 3) if start is not None and end is not None else None,
            "source_event_ids": event_ids,
            "overlap_with": [],
            "overlap_similarity_max": 0.0,
        })

    build_overlap_clusters(items, threshold=overlap_threshold)
    # Deterministic total order: score, fewer blockers, fewer editorial flags, ID.
    items.sort(key=lambda x: (-x["ranking_score"], len(x["blocking_flags"]), len(x["quality_flags"]), x["candidate_id"]))
    for rank, item in enumerate(items, 1):
        item["overall_rank"] = rank

    # Select one eligible representative from each near-duplicate cluster.
    best_by_cluster: dict[str, dict[str, Any]] = {}
    for item in items:
        if not item["shortlist_eligible"]:
            continue
        cluster = item["overlap_cluster_id"]
        best_by_cluster.setdefault(cluster, item)
    representatives = sorted(best_by_cluster.values(), key=lambda x: (-x["ranking_score"], len(x["quality_flags"]), x["candidate_id"]))
    shortlist = representatives[:max(1, top_n)]
    selected_ids = {x["candidate_id"] for x in shortlist}
    for item in items:
        item["shortlist_selected"] = item["candidate_id"] in selected_ids
        if item["shortlist_selected"]:
            item["ranking_reason"] = "Highest adjusted score among eligible candidates in its overlap cluster."
        elif item["blocking_flags"]:
            item["ranking_reason"] = item["shortlist_exclusion_reason"]
        elif item["overlap_cluster_size"] > 1:
            item["ranking_reason"] = "Another eligible candidate in this overlapping cluster ranked higher."
        else:
            item["ranking_reason"] = "Outside the requested shortlist size."

    def public(item: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in item.items() if k != "source_event_ids"}

    return {
        "schema_version": "1.0",
        "engine": "PromoPilot Ranking Engine",
        "mode": "deterministic_adjusted_score_overlap_diversity_no_llm_calls",
        "episode_id": quality_gate.get("episode_id"),
        "input_candidate_count": len(gate_results),
        "candidate_metadata_matched_count": len(gate_results) - len(missing_metadata),
        "candidate_metadata_missing_count": len(missing_metadata),
        "candidate_metadata_missing": missing_metadata,
        "ranking_policy": {
            "score_source": "quality_gate.adjusted_score",
            "quality_gate_penalty_applied_again": False,
            "blocking_flags": sorted(MID_THOUGHT_FLAGS),
            "editorial_review_flags_are_not_automatic_exclusions": True,
            "overlap_threshold": overlap_threshold,
            "overlap_measure": "intersection_over_shorter_clip_duration_or_source_event_containment",
            "one_representative_per_overlap_cluster": True,
            "top_n": top_n,
        },
        "summary": {
            "ranked_count": len(items),
            "shortlist_count": len(shortlist),
            "blocked_mid_thought_count": sum(bool(x["blocking_flags"]) for x in items),
            "overlap_cluster_count": len({x["overlap_cluster_id"] for x in items}),
        },
        "shortlist": [public(x) for x in shortlist],
        "full_ranking": [public(x) for x in items],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank PromoPilot promo candidates using Evaluation Quality Gate output.")
    parser.add_argument("--quality-gate", required=True, help="Path to epXX_quality_gate.json")
    parser.add_argument("--candidates", help="Path to epXX_candidates.json (used for overlap detection)")
    parser.add_argument("--out", required=True, help="Output ranked JSON path")
    parser.add_argument("--top-n", type=int, default=5, help="Maximum diverse shortlist size (default: 5)")
    parser.add_argument("--overlap-threshold", type=float, default=0.80, help="Near-duplicate containment threshold (default: 0.80)")
    args = parser.parse_args(argv)
    try:
        gate = load_json(args.quality_gate)
        candidates = load_json(args.candidates) if args.candidates else None
        if not isinstance(gate, dict) or not isinstance(gate.get("results"), list):
            raise ValueError("Quality-gate JSON must contain a 'results' array.")
        if args.top_n < 1:
            raise ValueError("--top-n must be at least 1.")
        if not 0.0 < args.overlap_threshold <= 1.0:
            raise ValueError("--overlap-threshold must be in the interval (0, 1].")
        report = run_ranking(gate, candidates, args.top_n, args.overlap_threshold)
        write_json(args.out, report)
        print(f"Ranking complete: {report['summary']['ranked_count']} ranked, {report['summary']['shortlist_count']} shortlisted.")
        if report["candidate_metadata_missing_count"]:
            print(f"Warning: timing/event metadata not extracted for {report['candidate_metadata_missing_count']} candidate(s); overlap detection may be incomplete.")
        print(f"Saved: {args.out}")
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Ranking failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
