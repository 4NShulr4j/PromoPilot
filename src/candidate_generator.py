"""Evidence-grounded continuous promo candidate generation for PromoPilot.

This module does not call an LLM. It combines normalized dramatic-moment
anchors with the complete parsed screenplay timeline. All clip times are
resolved from parser events, and Python validates duration and provenance.

The first milestone supports continuous, single-episode promo candidates.
Multi-segment and cross-episode promos are intentionally deferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MIN_DURATION = 20.0
DEFAULT_MAX_DURATION = 60.0
DEFAULT_MIN_HOOK_SCORE = 4
DEFAULT_MIN_ENDING_SCORE = 3


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Invalid non-finite {field}: {value!r}")
    return result


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _moment_event_ids(moment: dict[str, Any]) -> list[str]:
    ids = moment.get("source_event_ids") or moment.get("event_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    return _unique([str(x) for x in ids])


def _moment_flags(moment: dict[str, Any]) -> list[dict[str, Any]]:
    flags = moment.get("editorial_review_flags") or []
    return [x for x in flags if isinstance(x, dict)]


def _event_text(event: dict[str, Any]) -> str:
    return str(event.get("content") or "").strip()


def _is_dialogue(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "").casefold() == "dialogue"


def _ends_with_question(event: dict[str, Any]) -> bool:
    # Screenplay dialogue is usually quoted; inspect the text rather than rely
    # on the final character being a question mark.
    return _is_dialogue(event) and "?" in _event_text(event)


def _stable_candidate_id(episode_id: str, start_id: str, end_id: str, basis: str) -> str:
    seed = f"{episode_id}|{start_id}|{end_id}|{basis}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10].upper()
    return f"{episode_id}_PROMO_{digest}"


def _format_timestamp(seconds: float) -> str:
    total_hundredths = round(seconds * 100)
    minutes, remainder = divmod(total_hundredths, 6000)
    whole_seconds, hundredths = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{hundredths:02d}"


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float, str]:
    return (
        _as_float(event.get("start_seconds"), "event.start_seconds"),
        _as_float(event.get("end_seconds"), "event.end_seconds"),
        str(event.get("event_id") or ""),
    )


def _build_lookup(episode: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    events = episode.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Parsed episode must contain a non-empty events list")

    lookup: dict[str, dict[str, Any]] = {}
    clean_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Every parsed event must be an object")
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Parsed event is missing event_id")
        if event_id in lookup:
            raise ValueError(f"Duplicate event_id: {event_id}")
        start = _as_float(event.get("start_seconds"), f"{event_id}.start_seconds")
        end = _as_float(event.get("end_seconds"), f"{event_id}.end_seconds")
        if end <= start:
            raise ValueError(f"Event {event_id} has non-positive duration")
        lookup[event_id] = event
        clean_events.append(event)

    clean_events.sort(key=_event_sort_key)
    return lookup, clean_events


def _collect_hook_anchors(
    analysis: dict[str, Any],
    event_lookup: dict[str, dict[str, Any]],
    min_hook_score: int,
) -> list[dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for moment_index, moment in enumerate(analysis.get("dramatic_moments") or []):
        if not isinstance(moment, dict):
            continue
        score = int(moment.get("hook_potential") or 0)
        moment_type = str(moment.get("moment_type") or "").casefold()
        if score < min_hook_score and moment_type != "hook":
            continue

        valid_events = [event_lookup[eid] for eid in _moment_event_ids(moment) if eid in event_lookup]
        if not valid_events:
            continue
        valid_events.sort(key=_event_sort_key)
        # Prefer dialogue if the moment contains both dialogue and visual setup;
        # otherwise use the earliest evidenced event. This keeps the hook tied
        # to the moment while avoiding a camera-only boundary when possible.
        dialogue_events = [event for event in valid_events if _is_dialogue(event)]
        anchor_event = dialogue_events[0] if dialogue_events else valid_events[0]
        event_id = str(anchor_event["event_id"])
        flags = _moment_flags(moment)
        visual_only = any(flag.get("code") == "visual_only_hook_review" for flag in flags)
        current = anchors.get(event_id)
        candidate = {
            "event_id": event_id,
            "start_seconds": _as_float(anchor_event["start_seconds"], "hook.start_seconds"),
            "hook_potential": score,
            "moment_type": moment_type,
            "description": str(moment.get("description") or "").strip(),
            "why_significant": str(moment.get("why_significant") or "").strip(),
            "visual_only": visual_only or not _is_dialogue(anchor_event),
            "context_dependent": bool(moment.get("context_dependent", False)),
            "context_needed": str(moment.get("context_needed") or "").strip(),
            "moment_index": moment_index,
        }
        if current is None or candidate["hook_potential"] > current["hook_potential"]:
            anchors[event_id] = candidate

    return sorted(anchors.values(), key=lambda item: (item["start_seconds"], item["event_id"]))


def _collect_ending_anchors(
    episode: dict[str, Any],
    analysis: dict[str, Any],
    event_lookup: dict[str, dict[str, Any]],
    timeline: list[dict[str, Any]],
    min_ending_score: int,
) -> list[dict[str, Any]]:
    anchors: dict[tuple[str, str], dict[str, Any]] = {}

    # Model-proposed dramatic endpoints. They remain candidates, not confirmed
    # cliffhangers; the downstream evaluator and editor must assess them.
    for moment_index, moment in enumerate(analysis.get("dramatic_moments") or []):
        if not isinstance(moment, dict):
            continue
        score = int(moment.get("cliffhanger_potential") or 0)
        if score < min_ending_score:
            continue
        valid_events = [event_lookup[eid] for eid in _moment_event_ids(moment) if eid in event_lookup]
        if not valid_events:
            continue
        end_event = max(valid_events, key=_event_sort_key)
        event_id = str(end_event["event_id"])
        key = (event_id, "model_cliffhanger_anchor")
        anchors[key] = {
            "event_id": event_id,
            "end_seconds": _as_float(end_event["end_seconds"], "ending.end_seconds"),
            "ending_potential": score,
            "ending_basis": "model_cliffhanger_anchor",
            "description": str(moment.get("description") or "").strip(),
            "why_significant": str(moment.get("why_significant") or "").strip(),
            "review_required": True,
            "review_reason": "Model-scored ending anchor; confirm that the cut leaves a meaningful outcome unresolved.",
            "moment_index": moment_index,
        }

    # An unanswered question can be a useful cut point. Keep this as a
    # heuristic review candidate, never as a confirmed cliffhanger.
    for index, event in enumerate(timeline):
        if not _ends_with_question(event):
            continue
        end_time = _as_float(event["end_seconds"], "question.end_seconds")
        next_events = [
            later for later in timeline[index + 1:]
            if _as_float(later["start_seconds"], "next.start_seconds") >= end_time - 0.01
        ]
        if not next_events:
            continue
        next_event = next_events[0]
        gap = _as_float(next_event["start_seconds"], "next.start_seconds") - end_time
        if gap > 4.0:
            continue
        if _is_dialogue(next_event) and event.get("speaker") and next_event.get("speaker") == event.get("speaker"):
            continue
        event_id = str(event["event_id"])
        key = (event_id, "question_cut_requires_review")
        anchors[key] = {
            "event_id": event_id,
            "end_seconds": end_time,
            "ending_potential": 3,
            "ending_basis": "question_cut_requires_review",
            "description": "Candidate cut immediately after a question and before the next story beat.",
            "why_significant": "The question may create an unanswered beat when used as the promo ending.",
            "review_required": True,
            "review_reason": "Question-cut heuristic; inspect the following response and rendered pacing before approval.",
            "next_event_id": str(next_event.get("event_id") or ""),
            "next_event_start_seconds": _as_float(next_event["start_seconds"], "next.start_seconds"),
        }

    # Use the real final event as an episode-ending candidate. It is not
    # automatically described as a cliffhanger.
    ending_context = analysis.get("episode_ending_context") or {}
    context_events = ending_context.get("events") or []
    context_event_ids = [str(item.get("event_id")) for item in context_events if isinstance(item, dict)]
    final_event = None
    for event_id in reversed(context_event_ids):
        if event_id in event_lookup:
            final_event = event_lookup[event_id]
            break
    if final_event is None:
        final_event = timeline[-1]
    final_id = str(final_event["event_id"])
    final_duration = _as_float(episode.get("duration_seconds"), "episode.duration_seconds")
    key = (final_id, "episode_ending_candidate")
    anchors[key] = {
        "event_id": final_id,
        "end_seconds": final_duration,
        "ending_potential": 3,
        "ending_basis": "episode_ending_candidate",
        "description": "Candidate ending at the actual episode end.",
        "why_significant": "Uses the supplied episode's final timeline beat; whether it is a cliffhanger requires editorial judgment.",
        "review_required": True,
        "review_reason": "Episode-end candidate, not a verified cliffhanger. Review whether the final beat creates a strong unresolved question.",
    }

    return sorted(anchors.values(), key=lambda item: (item["end_seconds"], item["event_id"], item["ending_basis"]))


def _collect_timeline_events(
    timeline: list[dict[str, Any]], start_seconds: float, end_seconds: float
) -> list[dict[str, Any]]:
    # Include every event overlapping the clip. This preserves simultaneous
    # camera, action, dialogue, and sound directions from the source timeline.
    return [
        {
            "event_id": event["event_id"],
            "label": event.get("label"),
            "speaker": event.get("speaker"),
            "event_type": event.get("event_type"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "start_seconds": _as_float(event["start_seconds"], "event.start_seconds"),
            "end_seconds": _as_float(event["end_seconds"], "event.end_seconds"),
            "clip_overlap_start_seconds": max(
                start_seconds, _as_float(event["start_seconds"], "event.start_seconds")
            ),
            "clip_overlap_end_seconds": min(
                end_seconds, _as_float(event["end_seconds"], "event.end_seconds")
            ),
            "starts_before_clip": _as_float(event["start_seconds"], "event.start_seconds") < start_seconds,
            "continues_after_clip": _as_float(event["end_seconds"], "event.end_seconds") > end_seconds,
            "content": event.get("content", ""),
        }
        for event in timeline
        if _as_float(event["start_seconds"], "event.start_seconds") < end_seconds
        and _as_float(event["end_seconds"], "event.end_seconds") > start_seconds
    ]


def validate_candidate(
    candidate: dict[str, Any],
    episode: dict[str, Any],
    event_lookup: dict[str, dict[str, Any]] | None = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[str]:
    """Return validation errors; an empty list means structurally valid."""
    errors: list[str] = []
    if event_lookup is None:
        event_lookup, _ = _build_lookup(episode)
    segments = candidate.get("segments") or []
    if len(segments) != 1:
        errors.append("MVP candidates must contain exactly one continuous segment")
        return errors

    segment = segments[0]
    start_id = str(segment.get("start_event_id") or "")
    end_id = str(segment.get("end_event_id") or "")
    if start_id not in event_lookup:
        errors.append("Unknown start_event_id")
    if end_id not in event_lookup:
        errors.append("Unknown end_event_id")
    if start_id not in event_lookup or end_id not in event_lookup:
        return errors

    start = _as_float(segment.get("start_seconds"), "segment.start_seconds")
    end = _as_float(segment.get("end_seconds"), "segment.end_seconds")
    duration = end - start
    episode_duration = _as_float(episode.get("duration_seconds"), "episode.duration_seconds")
    if start < 0 or end > episode_duration + 0.001:
        errors.append("Segment timestamps fall outside the episode duration")
    if end <= start:
        errors.append("Segment end must be after segment start")
    if duration < min_duration - 0.001 or duration > max_duration + 0.001:
        errors.append(f"Duration must be between {min_duration:g} and {max_duration:g} seconds")

    start_event = event_lookup[start_id]
    end_event = event_lookup[end_id]
    if abs(start - _as_float(start_event["start_seconds"], "start event time")) > 0.011:
        errors.append("Segment start does not match the source hook event start")
    if abs(end - _as_float(end_event["end_seconds"], "end event time")) > 0.011 and segment.get("ending_basis") != "episode_ending_candidate":
        errors.append("Segment end does not match the source ending event end")

    hook_id = str((candidate.get("hook") or {}).get("event_id") or "")
    if hook_id != start_id:
        errors.append("Hook event must be the segment's start event")
    hook = event_lookup.get(hook_id)
    if hook:
        hook_offset = _as_float(hook["start_seconds"], "hook start") - start
        if not 0 <= hook_offset <= 3:
            errors.append("Hook must occur within the first 3 seconds")

    ending_id = str((candidate.get("ending") or {}).get("event_id") or "")
    if ending_id != end_id:
        errors.append("Ending event must match the segment's end event")
    if candidate.get("episode_id") != episode.get("episode_id"):
        errors.append("Candidate episode_id does not match parsed episode")

    return errors


def generate_candidates(
    episode: dict[str, Any],
    normalized_analysis: dict[str, Any],
    *,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    min_hook_score: int = DEFAULT_MIN_HOOK_SCORE,
    min_ending_score: int = DEFAULT_MIN_ENDING_SCORE,
    max_candidates: int = 40,
) -> dict[str, Any]:
    """Generate reproducible single-episode promo candidates from source events."""
    episode_id = str(episode.get("episode_id") or "").strip()
    if not episode_id:
        raise ValueError("Parsed episode is missing episode_id")
    analysis_episode_id = str(normalized_analysis.get("episode_id") or "").strip()
    if analysis_episode_id and analysis_episode_id != episode_id:
        raise ValueError(
            f"Parsed episode {episode_id!r} does not match normalized analysis {analysis_episode_id!r}"
        )
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("Invalid minimum/maximum duration configuration")
    if not 1 <= min_hook_score <= 5 or not 1 <= min_ending_score <= 5:
        raise ValueError("Hook and ending score thresholds must be between 1 and 5")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    event_lookup, timeline = _build_lookup(episode)
    hooks = _collect_hook_anchors(normalized_analysis, event_lookup, min_hook_score)
    endings = _collect_ending_anchors(
        episode, normalized_analysis, event_lookup, timeline, min_ending_score
    )

    candidates: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for hook in hooks:
        for ending in endings:
            start = hook["start_seconds"]
            end = ending["end_seconds"]
            duration = end - start
            if duration < min_duration - 0.001 or duration > max_duration + 0.001:
                continue
            if end <= start:
                continue
            # Avoid clipping before the selected hook ends.
            hook_event = event_lookup[hook["event_id"]]
            if end < _as_float(hook_event["end_seconds"], "hook event end"):
                continue
            timeline_events = _collect_timeline_events(timeline, start, end)
            candidate_id = _stable_candidate_id(
                episode_id, hook["event_id"], ending["event_id"], ending["ending_basis"]
            )
            visual_flag = hook["visual_only"]
            review_flags = []
            if visual_flag:
                review_flags.append({
                    "code": "visual_hook_requires_video_review",
                    "severity": "advisory",
                    "message": "The opening hook is visual/non-dialogue; screenplay text alone cannot confirm scroll-stopping impact.",
                })
            if hook["context_dependent"]:
                review_flags.append({
                    "code": "standalone_context_review",
                    "severity": "warning",
                    "message": hook["context_needed"] or "Hook moment may depend on prior story context.",
                })
            review_flags.append({
                "code": ending["ending_basis"],
                "severity": "warning" if ending["review_required"] else "advisory",
                "message": ending["review_reason"],
            })

            # Candidate-generation priority is a sorting heuristic, not the
            # final promo quality score. The evaluator will score candidates.
            hook_component = (hook["hook_potential"] / 5.0) * 60.0
            ending_component = (ending["ending_potential"] / 5.0) * 40.0
            priority = round(hook_component + ending_component, 2)
            if ending["ending_basis"] == "episode_ending_candidate":
                ending_display_time = _format_timestamp(end)
            else:
                ending_display_time = str(event_lookup[ending["event_id"]].get("end_time") or _format_timestamp(end))

            candidate = {
                "candidate_id": candidate_id,
                "show_id": str(normalized_analysis.get("show_id") or ""),
                "episode_id": episode_id,
                "promo_type": "continuous_single_episode",
                "segments": [{
                    "episode_id": episode_id,
                    "start_seconds": round(start, 2),
                    "end_seconds": round(end, 2),
                    "start_time": str(hook_event.get("start_time") or ""),
                    "end_time": ending_display_time,
                    "start_event_id": hook["event_id"],
                    "end_event_id": ending["event_id"],
                    "ending_basis": ending["ending_basis"],
                }],
                "duration_seconds": round(duration, 2),
                "hook": {
                    "event_id": hook["event_id"],
                    "start_time": hook_event.get("start_time"),
                    "text": _event_text(hook_event),
                    "description": hook["description"],
                    "hook_potential": hook["hook_potential"],
                    "visual_only": visual_flag,
                    "context_dependent": hook["context_dependent"],
                },
                "ending": {
                    "event_id": ending["event_id"],
                    "end_time": ending_display_time,
                    "text": _event_text(event_lookup[ending["event_id"]]),
                    "description": ending["description"],
                    "ending_potential": ending["ending_potential"],
                    "ending_basis": ending["ending_basis"],
                    "confirmed_cliffhanger": False,
                    "review_required": ending["review_required"],
                },
                "timeline_events": timeline_events,
                "source_event_ids": [item["event_id"] for item in timeline_events],
                "candidate_generation_priority": priority,
                "selection_reason": (
                    f"Starts at a source-grounded hook anchor (potential {hook['hook_potential']}/5) "
                    f"and ends at {ending['ending_basis']} (potential {ending['ending_potential']}/5)."
                ),
                "editorial_review_flags": review_flags,
                "validation_errors": [],
                "validation_passed": False,
            }
            errors = validate_candidate(
                candidate, episode, event_lookup, min_duration=min_duration, max_duration=max_duration
            )
            candidate["validation_errors"] = errors
            candidate["validation_passed"] = not errors
            if errors:
                rejected_pairs.append({
                    "hook_event_id": hook["event_id"],
                    "ending_event_id": ending["event_id"],
                    "errors": errors,
                })
            else:
                candidates.append(candidate)

    # Stable ordering, then avoid exact duplicates by source hook/end IDs.
    candidates.sort(
        key=lambda item: (
            -item["candidate_generation_priority"],
            item["duration_seconds"],
            item["candidate_id"],
        )
    )
    unique: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        segment = candidate["segments"][0]
        pair = (segment["start_event_id"], segment["end_event_id"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        unique.append(candidate)

    selected = unique[:max_candidates]
    return {
        "schema_version": 1,
        "show_id": str(normalized_analysis.get("show_id") or ""),
        "episode_id": episode_id,
        "generation_method": "source_event_anchors_and_rule_based_windowing",
        "model_calls_made": 0,
        "constraints": {
            "min_duration_seconds": min_duration,
            "max_duration_seconds": max_duration,
            "min_hook_score": min_hook_score,
            "min_ending_score": min_ending_score,
            "continuous_single_episode_only": True,
        },
        "source_event_count": len(timeline),
        "hook_anchor_count": len(hooks),
        "ending_anchor_count": len(endings),
        "candidate_count": len(selected),
        "rejected_pair_count": len(rejected_pairs),
        "candidates": selected,
        "rejected_pairs": rejected_pairs,
        "notes": [
            "Generation priority is a heuristic for ordering candidates, not the final promo quality score.",
            "No ending is marked as a confirmed cliffhanger by this module; endings require downstream evaluation and editorial review.",
            "The complete overlapping source timeline is retained for each candidate.",
            "This milestone generates continuous single-episode candidates only.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate timestamp-validated continuous promo candidates from PromoPilot story analysis."
    )
    parser.add_argument("--episode", required=True, help="Parsed episode JSON from src.parser")
    parser.add_argument("--analysis", required=True, help="Normalized story analysis JSON from src.story_normalizer")
    parser.add_argument("--output", help="Output JSON path; defaults to outputs/candidates/<episode>_candidates.json")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--min-hook-score", type=int, default=DEFAULT_MIN_HOOK_SCORE)
    parser.add_argument("--min-ending-score", type=int, default=DEFAULT_MIN_ENDING_SCORE)
    parser.add_argument("--max-candidates", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not 0 <= args.min_hook_score <= 5 or not 0 <= args.min_ending_score <= 5:
        raise SystemExit("Hook and ending thresholds must be between 0 and 5")
    episode = load_json(args.episode)
    analysis = load_json(args.analysis)
    result = generate_candidates(
        episode,
        analysis,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_hook_score=args.min_hook_score,
        min_ending_score=args.min_ending_score,
        max_candidates=args.max_candidates,
    )
    output_path = Path(args.output) if args.output else Path("outputs/candidates") / f"{episode['episode_id'].lower()}_candidates.json"
    save_json(output_path, result)
    print("Promo candidate generation completed.")
    print(f"Episode:          {result['episode_id']}")
    print(f"Source events:    {result['source_event_count']}")
    print(f"Hook anchors:     {result['hook_anchor_count']}")
    print(f"Ending anchors:   {result['ending_anchor_count']}")
    print(f"Candidates:       {result['candidate_count']}")
    print(f"Rejected pairs:   {result['rejected_pair_count']}")
    print(f"Model calls:      {result['model_calls_made']}")
    print(f"Saved:            {output_path}")


if __name__ == "__main__":
    main()
