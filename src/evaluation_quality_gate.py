"""Deterministic source-based QA layer for PromoPilot candidate evaluations.

This module does not call an LLM and never overwrites evaluator scores. It joins
candidate/evaluation/source-event records by IDs, emits transparent flags, and
writes an additive quality-gate report.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any

ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…|--|—)\s*[\"'”’\)\]\}]*$")
QUESTION_RE = re.compile(r"\?\s*[\"'”’)]*$")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first(d: Any, *keys: str, default=None):
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def records(obj: Any, preferred: tuple[str, ...]) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in preferred:
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # Some artifacts wrap the actual payload one level deeper.
        for value in obj.values():
            if isinstance(value, dict):
                found = records(value, preferred)
                if found:
                    return found
    return []


def event_id(event: dict) -> str | None:
    return first(event, "event_id", "id", "source_event_id", "eventId")


def event_start(event: dict) -> float | None:
    value = first(event, "start_time", "start_seconds", "start", "timestamp_start", "startTime")
    if value is None and isinstance(event.get("timing"), dict):
        value = first(event["timing"], "start", "start_time", "start_seconds")
    return as_float(value)


def event_end(event: dict) -> float | None:
    value = first(event, "end_time", "end_seconds", "end", "timestamp_end", "endTime")
    if value is None and isinstance(event.get("timing"), dict):
        value = first(event["timing"], "end", "end_time", "end_seconds")
    return as_float(value)


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            # Accept HH:MM:SS(.ms) and MM:SS(.ms).
            parts = s.split(":")
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 3:
                    return nums[0] * 3600 + nums[1] * 60 + nums[2]
                if len(nums) == 2:
                    return nums[0] * 60 + nums[1]
            except ValueError:
                return None
    return None


def event_text(event: dict) -> str:
    value = first(event, "content", "text", "dialogue", "description", "raw_text", default="")
    return str(value or "").strip()


def event_speaker(event: dict) -> str:
    value = first(event, "speaker", "character", "character_name", "entity", default="")
    if isinstance(value, dict):
        value = first(value, "name", "label", "canonical_name", default="")
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def candidate_id(item: dict) -> str | None:
    return first(item, "candidate_id", "id", "promo_id")


def get_candidate_window(c: dict) -> tuple[float | None, float | None]:
    start = as_float(first(c, "start_time", "start_seconds", "start"))
    end = as_float(first(c, "end_time", "end_seconds", "end"))
    if start is None or end is None:
        segs = first(c, "segments", "timeline_segments", default=[])
        if isinstance(segs, list) and segs:
            start = start if start is not None else as_float(first(segs[0], "start_time", "start_seconds", "start"))
            end = end if end is not None else as_float(first(segs[-1], "end_time", "end_seconds", "end"))
    return start, end


def candidate_event_ids(c: dict) -> tuple[str | None, str | None]:
    start_id = first(c, "start_event_id", "first_event_id")
    end_id = first(c, "end_event_id", "last_event_id")
    segments = first(c, "segments", "timeline_segments", default=[])
    if isinstance(segments, list) and segments:
        start_id = start_id or first(segments[0], "start_event_id", "event_id")
        end_id = end_id or first(segments[-1], "end_event_id", "event_id")
    timeline = first(c, "timeline_events", "events", default=[])
    if isinstance(timeline, list) and timeline:
        start_id = start_id or event_id(timeline[0])
        end_id = end_id or event_id(timeline[-1])
    hook = first(c, "hook", "hook_anchor", default={})
    ending = first(c, "ending", "ending_anchor", default={})
    start_id = start_id or (first(hook, "event_id", "source_event_id") if isinstance(hook, dict) else None)
    end_id = end_id or (first(ending, "event_id", "source_event_id") if isinstance(ending, dict) else None)
    return start_id, end_id


def score_total(evaluation: dict) -> float | None:
    for key in ("total_score", "overall_score", "weighted_score", "score"):
        value = as_float(evaluation.get(key))
        if value is not None:
            return value
    scores = evaluation.get("scores")
    if isinstance(scores, dict):
        for key in ("total", "overall", "weighted_total"):
            value = as_float(scores.get(key))
            if value is not None:
                return value
    return None


def evaluate_one(c: dict, ev: dict, events: list[dict], hook_threshold: float, response_window: float) -> dict:
    by_id = {event_id(x): x for x in events if event_id(x)}
    ordered = sorted((x for x in events if event_start(x) is not None), key=event_start)
    start, end = get_candidate_window(c)
    first_id, last_id = candidate_event_ids(c)
    hook = first(c, "hook", "hook_anchor", default={})
    hook_id = first(hook, "event_id", "source_event_id") if isinstance(hook, dict) else None
    hook_id = hook_id or first(c, "hook_event_id")
    hook_event = by_id.get(hook_id)
    if hook_event is None and isinstance(hook, dict):
        hook_time = as_float(first(hook, "start_time", "start_seconds", "time"))
    else:
        hook_time = event_start(hook_event) if hook_event else None
    hook_offset = round(hook_time - start, 3) if hook_time is not None and start is not None else None

    end_event = by_id.get(last_id) if last_id else None
    if end_event is None and end is not None:
        before = [x for x in ordered if event_end(x) is not None and event_end(x) <= end + 0.05]
        end_event = before[-1] if before else None
    end_text = event_text(end_event) if end_event else ""
    end_speaker = event_speaker(end_event) if end_event else ""
    end_event_end = event_end(end_event) if end_event else end

    flags: list[dict] = []
    if hook_offset is not None:
        hook_pass = 0 <= hook_offset <= hook_threshold
        if not hook_pass:
            flags.append({"code": "HOOK_START_LATE", "severity": "warning", "detail": f"Hook source event begins {hook_offset:.2f}s after clip start; target is <= {hook_threshold:.1f}s."})
        # A zero model rating is a contradiction only when the model rationale says timing caused it.
        scores = ev.get("scores", {}) if isinstance(ev.get("scores"), dict) else {}
        hook_score = as_float(first(scores, "hook_strength", "hook", "hook_score"))
        if hook_pass and hook_score == 0:
            flags.append({"code": "HOOK_SCORE_REVIEW", "severity": "review", "detail": "Source timing confirms the hook begins within the timing threshold, but the evaluator assigned hook strength 0. Review the rationale; timing alone does not establish creative strength."})
    else:
        hook_pass = None
        flags.append({"code": "HOOK_SOURCE_UNRESOLVED", "severity": "review", "detail": "Could not resolve hook event/time from candidate and source-event records."})

    # Look for a same-speaker continuation immediately after the clip boundary.
    next_events = []
    for x in ordered:
        xs = event_start(x)
        if xs is None or end_event_end is None or xs < end_event_end - 0.05:
            continue
        if end_event and event_id(x) == event_id(end_event):
            continue
        next_events.append(x)
    continuation = None
    if end_text and (ELLIPSIS_RE.search(end_text) or end_text.endswith((",", ":"))):
        for x in next_events[:5]:
            gap = (event_start(x) - end_event_end) if event_start(x) is not None else None
            sp = event_speaker(x)
            txt = event_text(x)
            if gap is not None and gap <= 5.0 and sp and sp == end_speaker and txt:
                continuation = {"event_id": event_id(x), "gap_seconds": round(gap, 3), "speaker": sp, "text_preview": txt[:180]}
                break
    if continuation:
        flags.append({"code": "CUT_MID_THOUGHT", "severity": "high", "detail": "The ending event appears unfinished and the same speaker continues shortly after the candidate cut.", "evidence": continuation})

    # Post-cut context is reported as context, not treated as semantic proof of an answer.
    post_cut = []
    if end_event_end is not None:
        for x in next_events[:3]:
            xs = event_start(x)
            gap = round(xs - end_event_end, 3) if xs is not None else None
            post_cut.append({"event_id": event_id(x), "gap_seconds": gap, "speaker": event_speaker(x), "text_preview": event_text(x)[:180]})
    immediate_response = None
    if end_text and QUESTION_RE.search(end_text):
        for item in post_cut:
            if item["gap_seconds"] is not None and 0 <= item["gap_seconds"] <= response_window:
                immediate_response = item
                break
    if immediate_response:
        flags.append({"code": "QUESTION_FOLLOWED_QUICKLY", "severity": "review", "detail": "A dialogue question at the cut is followed by another source event shortly afterward. This may weaken the cut as a cliffhanger; semantic answer detection is not inferred.", "evidence": immediate_response})

    # Distinguish an emotional turn at episode end from a conventional unresolved cliffhanger.
    ending = first(c, "ending", "ending_anchor", default={})
    ending_label = first(ending, "type", "ending_type", "label") if isinstance(ending, dict) else None
    if end_event and any(term in (end_text + " " + str(ending_label or "")).casefold() for term in ("defiant", "defiance", "turns", "hardens", "cut to black", "music swell")):
        flags.append({"code": "EMOTIONAL_TURN_NOT_VERIFIED_CLIFFHANGER", "severity": "review", "detail": "The source evidence may support an emotional character turn. Do not automatically label it a cliffhanger without an unresolved story question."})

    high = sum(1 for f in flags if f["severity"] == "high")
    review = sum(1 for f in flags if f["severity"] == "review")
    # Transparent conservative adjustment: only a demonstrable split thought is penalized.
    original = score_total(ev)
    penalty = 10 if continuation else 0
    adjusted = round(max(0.0, original - penalty), 2) if original is not None else None
    return {
        "candidate_id": candidate_id(c) or candidate_id(ev),
        "original_model_score": original,
        "adjusted_score": adjusted,
        "adjustment_points": -penalty,
        "hook_timing": {"hook_event_id": hook_id, "offset_seconds": hook_offset, "within_threshold": hook_pass, "threshold_seconds": hook_threshold},
        "ending_event": {"event_id": last_id or (event_id(end_event) if end_event else None), "text": end_text[:240]},
        "post_cut_context": post_cut,
        "flags": flags,
        "quality_status": "needs_review" if (high or review) else "pass",
    }


def run_gate(episode_obj: Any, candidates_obj: Any, evaluations_obj: Any, hook_threshold: float = 3.0, response_window: float = 2.0) -> dict:
    events = records(episode_obj, ("events", "timeline_events", "source_events"))
    candidates = records(candidates_obj, ("candidates", "promo_candidates", "items"))
    evaluations = records(evaluations_obj, ("evaluations", "results", "candidate_evaluations", "items"))
    cand_map = {candidate_id(c): c for c in candidates if candidate_id(c)}
    output = []
    unmatched = []
    for ev in evaluations:
        cid = candidate_id(ev)
        c = cand_map.get(cid)
        if not c:
            # Evaluation artifacts sometimes embed the candidate record.
            c = first(ev, "candidate", "candidate_data")
        if not isinstance(c, dict):
            unmatched.append(cid or "unknown")
            continue
        output.append(evaluate_one(c, ev, events, hook_threshold, response_window))
    return {
        "schema_version": "1.0",
        "gate_name": "PromoPilot Evaluation Quality Gate",
        "mode": "deterministic_source_checks_no_llm_calls",
        "policy": {"original_evaluation_preserved": True, "hook_start_threshold_seconds": hook_threshold, "immediate_response_window_seconds": response_window, "mid_thought_penalty_points": 10, "note": "Only an evidenced same-speaker continuation is automatically penalized. Rapid post-cut events are flags for editorial review, not semantic answer claims."},
        "episode_id": first(episode_obj, "episode_id", "id", "episode"),
        "candidate_count": len(candidates),
        "evaluation_count": len(evaluations),
        "quality_checked_count": len(output),
        "unmatched_evaluations": unmatched,
        "summary": {
            "pass": sum(x["quality_status"] == "pass" for x in output),
            "needs_review": sum(x["quality_status"] == "needs_review" for x in output),
            "mid_thought_cuts": sum(any(f["code"] == "CUT_MID_THOUGHT" for f in x["flags"]) for x in output),
            "rapid_question_followups": sum(any(f["code"] == "QUESTION_FOLLOWED_QUICKLY" for f in x["flags"]) for x in output),
            "hook_score_reviews": sum(any(f["code"] == "HOOK_SCORE_REVIEW" for f in x["flags"]) for x in output),
        },
        "results": sorted(output, key=lambda x: (x["adjusted_score"] is None, -(x["adjusted_score"] or 0))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", required=True, help="Parsed episode JSON")
    p.add_argument("--candidates", required=True, help="Candidate generator JSON")
    p.add_argument("--evaluations", required=True, help="Promo evaluator JSON")
    p.add_argument("--out", default="outputs/evaluations/ep01_quality_gate.json")
    p.add_argument("--hook-threshold", type=float, default=3.0)
    p.add_argument("--response-window", type=float, default=2.0)
    args = p.parse_args()
    report = run_gate(load_json(args.episode), load_json(args.candidates), load_json(args.evaluations), args.hook_threshold, args.response_window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Quality gate complete: {report['quality_checked_count']} candidates checked")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {out}")

if __name__ == "__main__":
    main()
