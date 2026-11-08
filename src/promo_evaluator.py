"""LLM-assisted promo quality evaluation for PromoPilot.

The evaluator scores source-grounded candidates; it does not edit clips or
claim screenplay-only evidence proves visual/audio quality. Python performs
all structural validation and weighted-score arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

PROMPT_VERSION = "promo-evaluator-v1"
WEIGHTS = {
    "hook_strength": 25,
    "cliffhanger_strength": 25,
    "standalone_clarity": 20,
    "emotional_engagement": 15,
    "pacing": 10,
    "clip_integrity": 5,
}
DEFAULT_MAX_COMPLETION_TOKENS = 900
DEFAULT_MIN_DURATION = 20.0
DEFAULT_MAX_DURATION = 60.0

SYSTEM_INSTRUCTIONS = """You are a rigorous senior editor evaluating promotional clips for vertical microdrama.
Evaluate only the proposed clip, using the supplied screenplay events as evidence.
A promo must attract a new viewer, have a strong hook within its first 3 seconds,
end on an unresolved and meaningful beat, and make sense without prior episodes.
Be critical: a dramatic scene is not automatically a good advertisement. A question
is not automatically a cliffhanger. A final episode beat is not automatically unresolved.
The post-cut context is supplied only to help judge whether the ending is meaningful;
it is NOT part of the proposed promo. Never invent dialogue, events, timestamps,
character relationships or visual/audio details not supported by the screenplay.
Return one compact JSON object matching the requested fields. Scores must be integers 0-5.
"""


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "candidate"))[:120]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def _event_lookup(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    events = episode.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Parsed episode must contain a non-empty events list")
    lookup: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Every parsed event must be an object")
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Parsed source event is missing event_id")
        if event_id in lookup:
            raise ValueError(f"Duplicate source event_id: {event_id}")
        start = _finite_float(event.get("start_seconds"), f"{event_id}.start_seconds")
        end = _finite_float(event.get("end_seconds"), f"{event_id}.end_seconds")
        if end <= start:
            raise ValueError(f"Source event {event_id} has invalid duration")
        lookup[event_id] = event
    return lookup


def validate_candidate_source(
    candidate: dict[str, Any],
    episode: dict[str, Any],
    lookup: dict[str, dict[str, Any]] | None = None,
    *,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[str]:
    """Revalidate candidate structure and source provenance independently."""
    errors: list[str] = []
    lookup = lookup or _event_lookup(episode)
    candidate_id = str(candidate.get("candidate_id") or "<unknown>")
    segments = candidate.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        return ["MVP evaluator currently supports exactly one continuous segment"]

    segment = segments[0]
    start_id = str(segment.get("start_event_id") or "")
    end_id = str(segment.get("end_event_id") or "")
    if start_id not in lookup:
        errors.append(f"Unknown start_event_id: {start_id}")
    if end_id not in lookup:
        errors.append(f"Unknown end_event_id: {end_id}")
    if start_id not in lookup or end_id not in lookup:
        return errors

    try:
        start = _finite_float(segment.get("start_seconds"), "segment.start_seconds")
        end = _finite_float(segment.get("end_seconds"), "segment.end_seconds")
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    duration = end - start
    episode_duration = _finite_float(episode.get("duration_seconds"), "episode.duration_seconds")
    if start < 0 or end > episode_duration + 0.011:
        errors.append("Segment timestamps fall outside the episode duration")
    if end <= start:
        errors.append("Segment end must be after segment start")
    if duration < min_duration - 0.011 or duration > max_duration + 0.011:
        errors.append(f"Duration must be between {min_duration:g} and {max_duration:g} seconds")

    start_event = lookup[start_id]
    end_event = lookup[end_id]
    source_start = _finite_float(start_event.get("start_seconds"), "source start")
    source_end = _finite_float(end_event.get("end_seconds"), "source end")
    if abs(start - source_start) > 0.011:
        errors.append("Segment start does not match the original hook event start")
    if segment.get("ending_basis") != "episode_ending_candidate" and abs(end - source_end) > 0.011:
        errors.append("Segment end does not match the original ending event end")
    if segment.get("ending_basis") == "episode_ending_candidate" and abs(end - episode_duration) > 0.011:
        errors.append("Episode-ending candidate does not end at actual episode duration")

    hook = candidate.get("hook") or {}
    ending = candidate.get("ending") or {}
    if str(hook.get("event_id") or "") != start_id:
        errors.append("Hook event must match segment start_event_id")
    if str(ending.get("event_id") or "") != end_id:
        errors.append("Ending event must match segment end_event_id")
    hook_offset = source_start - start
    if hook_offset < -0.011 or hook_offset > 3.011:
        errors.append("Hook must occur within the first 3 seconds")

    source_ids = candidate.get("source_event_ids") or []
    if not isinstance(source_ids, list):
        errors.append("source_event_ids must be a list")
    else:
        unknown = [str(item) for item in source_ids if str(item) not in lookup]
        if unknown:
            errors.append("Unknown source_event_ids: " + ", ".join(unknown[:5]))

    candidate_episode_id = str(candidate.get("episode_id") or "")
    episode_id = str(episode.get("episode_id") or "")
    if candidate_episode_id != episode_id:
        errors.append(f"Candidate episode_id {candidate_episode_id!r} does not match parsed episode {episode_id!r}")
    if candidate.get("validation_passed") is False:
        errors.append("Candidate generator marked this candidate as invalid")
    return errors


def _short_event(event: dict[str, Any], *, max_chars: int = 300) -> dict[str, Any]:
    content = str(event.get("content") or "").strip()
    if len(content) > max_chars:
        content = content[: max_chars - 1].rstrip() + "…"
    return {
        "event_id": event.get("event_id"),
        "start": event.get("start_time"),
        "end": event.get("end_time"),
        "type": event.get("event_type"),
        "speaker": event.get("speaker"),
        "label": event.get("label"),
        "content": content,
    }


def build_candidate_context(
    candidate: dict[str, Any],
    episode: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, source-grounded context for one candidate."""
    lookup = _event_lookup(episode)
    segment = candidate["segments"][0]
    clip_ids = candidate.get("source_event_ids") or []
    clip_events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event_id in clip_ids:
        event_id = str(event_id)
        if event_id in lookup and event_id not in seen:
            clip_events.append(_short_event(lookup[event_id]))
            seen.add(event_id)

    # If source_event_ids is absent or incomplete, reconstruct overlapping
    # source events from the parsed screenplay timeline.
    start = _finite_float(segment["start_seconds"], "segment.start_seconds")
    end = _finite_float(segment["end_seconds"], "segment.end_seconds")
    for event_id, event in lookup.items():
        ev_start = _finite_float(event["start_seconds"], f"{event_id}.start_seconds")
        ev_end = _finite_float(event["end_seconds"], f"{event_id}.end_seconds")
        if ev_start < end and ev_end > start and event_id not in seen:
            clip_events.append(_short_event(event))
            seen.add(event_id)
    clip_events.sort(key=lambda item: (lookup[str(item["event_id"])] ["start_seconds"], str(item["event_id"])))

    # Supply up to three later source events as context ONLY. They are not part
    # of the promo and allow the evaluator to judge whether an apparent
    # unanswered question is immediately resolved in the episode.
    later = [
        event for event in lookup.values()
        if _finite_float(event["start_seconds"], "later.start") >= end - 0.001
    ]
    later.sort(key=lambda item: (_finite_float(item["start_seconds"], "later.start"), str(item["event_id"])))
    post_cut = [_short_event(event, max_chars=220) for event in later[:3]]

    logline = str((analysis or {}).get("logline") or episode.get("logline") or "")
    if len(logline) > 700:
        logline = logline[:699].rstrip() + "…"
    return {
        "candidate_id": candidate.get("candidate_id"),
        "episode_id": candidate.get("episode_id"),
        "promo_type": candidate.get("promo_type"),
        "duration_seconds": round(end - start, 2),
        "clip_start": segment.get("start_time"),
        "clip_end": segment.get("end_time"),
        "hook": candidate.get("hook") or {},
        "ending": candidate.get("ending") or {},
        "editorial_review_flags": candidate.get("editorial_review_flags") or [],
        "show_logline": logline,
        "clip_events": clip_events,
        "post_cut_context_not_in_promo": post_cut,
    }


def build_evaluation_prompts(context: dict[str, Any]) -> tuple[str, str]:
    """Return compact prompts; no full show or unrelated episodes are sent."""
    user_prompt = (
        "Evaluate this one candidate promo. Score each criterion from 0 to 5, "
        "using integer values. A 0 means missing/ineffective; 3 acceptable; "
        "5 excellent. Be critical and evidence-based.\n\n"
        "WEIGHTS (Python calculates the weighted total):\n"
        "hook_strength 25; cliffhanger_strength 25; standalone_clarity 20; "
        "emotional_engagement 15; pacing 10; clip_integrity 5.\n\n"
        "Important evaluation rules:\n"
        "- Hook must happen within the first 3 seconds of the clip.\n"
        "- A dramatic statement is not automatically a strong hook.\n"
        "- A question or episode-end cut is not automatically a cliffhanger.\n"
        "- Judge whether a new viewer can understand the basic situation.\n"
        "- Judge pacing from the supplied source events and clip duration.\n"
        "- Clip integrity includes chronology, understandable transitions, and a cut that does not truncate a key source event.\n"
        "- Visual-only hooks require video review; do not claim to have seen the video.\n"
        "- post_cut_context_not_in_promo is context only; it is not included in the clip.\n"
        "- Cite only event IDs included in clip_events or post_cut_context_not_in_promo.\n\n"
        "Return exactly this JSON shape:\n"
        '{"scores":{"hook_strength":0,"cliffhanger_strength":0,"standalone_clarity":0,'
        '"emotional_engagement":0,"pacing":0,"clip_integrity":0},'
        '"hook_assessment":"...", "ending_assessment":"...", '
        '"cliffhanger_type":"strong_unresolved_outcome|open_question|weak_cut|resolved_beat|ambiguous", '
        '"standalone_clarity_explanation":"...", "why_it_works":"...", '
        '"risks":["..."], "strengths":["..."], '
        '"evidence_event_ids":["..."], "confidence":0.0}\n\n'
        "CANDIDATE CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    return SYSTEM_INSTRUCTIONS, user_prompt


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate short leading/trailing text from plain-text fallback, but
        # require a complete first JSON object.
        start = text.find("{")
        if start < 0:
            raise ValueError("LLM response did not contain a JSON object")
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(result, dict):
        raise ValueError("LLM evaluation must be a JSON object")
    return result


def normalize_evaluation_response(
    model_result: dict[str, Any],
    *,
    allowed_event_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    raw_scores = model_result.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("LLM response is missing the 'scores' object")
    scores: dict[str, int] = {}
    for criterion in WEIGHTS:
        value = raw_scores.get(criterion)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Score {criterion!r} must be numeric")
        if not 0 <= value <= 5:
            raise ValueError(f"Score {criterion!r} must be between 0 and 5")
        if float(value) != int(value):
            warnings.append(f"Score {criterion} was non-integer and rounded")
        scores[criterion] = int(round(value))

    evidence = model_result.get("evidence_event_ids") or []
    if not isinstance(evidence, list):
        raise ValueError("evidence_event_ids must be a list")
    valid_evidence: list[str] = []
    for item in evidence:
        event_id = str(item)
        if event_id in allowed_event_ids and event_id not in valid_evidence:
            valid_evidence.append(event_id)
        else:
            warnings.append(f"Ignored invalid or duplicate evidence event ID: {event_id}")

    confidence = model_result.get("confidence", 0.5)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        warnings.append("Invalid confidence value; defaulted to 0.5")
        confidence = 0.5

    cliffhanger_type = str(model_result.get("cliffhanger_type") or "ambiguous").strip()
    allowed_types = {
        "strong_unresolved_outcome", "open_question", "weak_cut", "resolved_beat", "ambiguous"
    }
    if cliffhanger_type not in allowed_types:
        warnings.append(f"Unknown cliffhanger_type {cliffhanger_type!r}; defaulted to ambiguous")
        cliffhanger_type = "ambiguous"

    def clean_string_list(field: str) -> list[str]:
        value = model_result.get(field) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            warnings.append(f"Invalid {field}; defaulted to an empty list")
            return []
        return [str(item).strip() for item in value if str(item).strip()][:8]

    normalized = {
        "scores": scores,
        "hook_assessment": str(model_result.get("hook_assessment") or "").strip(),
        "ending_assessment": str(model_result.get("ending_assessment") or "").strip(),
        "cliffhanger_type": cliffhanger_type,
        "standalone_clarity_explanation": str(model_result.get("standalone_clarity_explanation") or "").strip(),
        "why_it_works": str(model_result.get("why_it_works") or "").strip(),
        "risks": clean_string_list("risks"),
        "strengths": clean_string_list("strengths"),
        "evidence_event_ids": valid_evidence,
        "confidence": float(confidence),
    }
    if not normalized["evidence_event_ids"]:
        warnings.append("No valid evidence_event_ids returned by evaluator")
    return normalized, warnings


def calculate_weighted_score(scores: dict[str, int]) -> float:
    total = 0.0
    for criterion, weight in WEIGHTS.items():
        if criterion not in scores:
            raise ValueError(f"Missing score: {criterion}")
        value = scores[criterion]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 5:
            raise ValueError(f"Invalid score for {criterion}: {value!r}")
        total += (float(value) / 5.0) * weight
    return round(total, 2)


def classify_candidate(
    weighted_score: float,
    scores: dict[str, int],
    structural_warnings: list[str],
    evaluator_warnings: list[str],
) -> str:
    """Editorial triage label; never an automatic publishing approval."""
    if structural_warnings:
        return "rejected_or_fix_source"
    if scores["hook_strength"] < 3 or scores["cliffhanger_strength"] < 3:
        return "needs_editor_review"
    if weighted_score >= 80 and scores["hook_strength"] >= 4 and scores["cliffhanger_strength"] >= 4:
        return "strong_candidate_for_editor_review"
    if weighted_score >= 65:
        return "promising_candidate_for_editor_review"
    return "low_priority_candidate"


def _checkpoint_fingerprint(candidate: dict[str, Any], model_name: str) -> str:
    return _sha256({
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
        "candidate": candidate,
    })


def evaluate_one_candidate(
    candidate: dict[str, Any],
    episode: dict[str, Any],
    analysis: dict[str, Any] | None = None,
    *,
    request_fn: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    lookup = _event_lookup(episode)
    validation_errors = validate_candidate_source(candidate, episode, lookup)
    base = {
        "candidate_id": candidate.get("candidate_id"),
        "episode_id": candidate.get("episode_id"),
        "duration_seconds": candidate.get("duration_seconds"),
        "segments": candidate.get("segments"),
        "structural_validation_passed": not validation_errors,
        "structural_validation_errors": validation_errors,
    }
    if validation_errors:
        base.update({
            "evaluation_status": "rejected_structural",
            "weighted_score": None,
            "recommendation": "rejected_or_fix_source",
            "scores": None,
            "reason": "Candidate failed deterministic source/timestamp validation; no LLM call was made.",
        })
        return base, None, None

    context = build_candidate_context(candidate, episode, analysis)
    system_prompt, user_prompt = build_evaluation_prompts(context)
    if request_fn is None:
        from src.llm_client import request_json_object
        request_fn = request_json_object
    raw_text, llm_metadata = request_fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=max_completion_tokens,
    )
    model_result = _parse_json_object(raw_text)
    allowed_ids = {
        str(item.get("event_id"))
        for item in context["clip_events"] + context["post_cut_context_not_in_promo"]
        if item.get("event_id")
    }
    evaluation, warnings = normalize_evaluation_response(model_result, allowed_event_ids=allowed_ids)
    weighted = calculate_weighted_score(evaluation["scores"])
    recommendation = classify_candidate(weighted, evaluation["scores"], [], warnings)
    base.update({
        "evaluation_status": "evaluated",
        "scores": evaluation["scores"],
        "weighted_score": weighted,
        "recommendation": recommendation,
        "hook_assessment": evaluation["hook_assessment"],
        "ending_assessment": evaluation["ending_assessment"],
        "cliffhanger_type": evaluation["cliffhanger_type"],
        "standalone_clarity_explanation": evaluation["standalone_clarity_explanation"],
        "why_it_works": evaluation["why_it_works"],
        "strengths": evaluation["strengths"],
        "risks": evaluation["risks"],
        "evidence_event_ids": evaluation["evidence_event_ids"],
        "confidence": evaluation["confidence"],
        "evaluator_warnings": warnings,
        "editorial_review_required": True,
        "model_metadata": llm_metadata or {},
    })
    return base, raw_text, llm_metadata


def evaluate_candidates(
    candidate_pack: dict[str, Any],
    episode: dict[str, Any],
    analysis: dict[str, Any] | None = None,
    *,
    output_dir: str | Path = "outputs/evaluations",
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    delay_between_calls: float = 25.0,
    limit: int | None = None,
    resume: bool = True,
    request_fn: Callable[..., tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    candidates = candidate_pack.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate file must contain a 'candidates' list")
    episode_id = str(episode.get("episode_id") or candidate_pack.get("episode_id") or "unknown")
    pack_episode_id = str(candidate_pack.get("episode_id") or episode_id)
    if pack_episode_id != episode_id:
        raise ValueError(f"Candidate pack episode {pack_episode_id!r} does not match parsed episode {episode_id!r}")

    out = Path(output_dir)
    checkpoints = out / "checkpoints"
    raw_dir = out / "raw_llm"
    checkpoints.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    model_name = "injected-test-model"
    if request_fn is None:
        from src.llm_client import get_model_name
        model_name = get_model_name()
    selected = candidates[:limit] if limit and limit > 0 else candidates
    results: list[dict[str, Any]] = []
    api_calls = 0
    for index, candidate in enumerate(selected, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index:03d}")
        fingerprint = _checkpoint_fingerprint(candidate, model_name)
        checkpoint_path = checkpoints / f"{_safe_id(candidate_id)}.json"
        if resume and checkpoint_path.exists():
            try:
                saved = load_json(checkpoint_path)
                if saved.get("fingerprint") == fingerprint and isinstance(saved.get("result"), dict):
                    results.append(saved["result"])
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        print(f"[{index}/{len(selected)}] Evaluating {candidate_id}...")
        try:
            result, raw_text, llm_metadata = evaluate_one_candidate(
                candidate, episode, analysis,
                request_fn=request_fn,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Promo evaluation failed at candidate {index}/{len(selected)} ({candidate_id}). "
                f"Completed checkpoints are saved under {checkpoints}; rerun the same command to resume. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        if raw_text is not None:
            api_calls += 1
            raw_path = raw_dir / f"{_safe_id(candidate_id)}_raw.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
        save_json(checkpoint_path, {
            "fingerprint": fingerprint,
            "candidate_id": candidate_id,
            "raw_output_file": str(raw_path) if raw_text is not None else None,
            "model_metadata": llm_metadata or {},
            "result": result,
        })
        results.append(result)
        if raw_text is not None and delay_between_calls > 0 and index < len(selected):
            time.sleep(delay_between_calls)

    evaluated = [item for item in results if item.get("evaluation_status") == "evaluated"]
    evaluated.sort(key=lambda item: (-(item.get("weighted_score") or 0), str(item.get("candidate_id"))))
    for rank, item in enumerate(evaluated, start=1):
        item["score_rank"] = rank
    rejected = [item for item in results if item.get("evaluation_status") != "evaluated"]

    result_pack = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "episode_id": episode_id,
        "show_id": candidate_pack.get("show_id") or (analysis or {}).get("show_id"),
        "model_calls_this_run": api_calls,
        "candidate_input_count": len(candidates),
        "candidates_selected_for_evaluation": len(selected),
        "evaluated_count": len(evaluated),
        "structurally_rejected_count": len(rejected),
        "weights": WEIGHTS,
        "evaluations": evaluated,
        "rejected_candidates": rejected,
        "notes": [
            "Weighted score is calculated in Python from integer model ratings.",
            "All evaluated candidates still require human editorial review.",
            "Script-only analysis cannot verify actual acting, camera execution, sound, or audience performance.",
            "Only continuous single-episode candidates are supported in this milestone.",
        ],
    }
    return result_pack


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PromoPilot candidate promos with source evidence.")
    parser.add_argument("--episode", required=True, help="Parsed episode JSON")
    parser.add_argument("--candidates", required=True, help="Candidate pack from src.candidate_generator")
    parser.add_argument("--analysis", help="Optional normalized story analysis JSON for logline/context")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--output-dir", default="outputs/evaluations")
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--delay-between-calls", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=10, help="Evaluate at most this many candidates; default 10")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing valid checkpoints")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    episode = load_json(args.episode)
    candidate_pack = load_json(args.candidates)
    analysis = load_json(args.analysis) if args.analysis else None
    if args.max_completion_tokens < 200:
        raise SystemExit("--max-completion-tokens must be at least 200")
    if args.delay_between_calls < 0:
        raise SystemExit("--delay-between-calls cannot be negative")
    result = evaluate_candidates(
        candidate_pack,
        episode,
        analysis,
        output_dir=args.output_dir,
        max_completion_tokens=args.max_completion_tokens,
        delay_between_calls=args.delay_between_calls,
        limit=args.limit,
        resume=not args.no_resume,
    )
    output_path = Path(args.output) if args.output else Path(args.output_dir) / f"{result['episode_id'].lower()}_evaluations.json"
    save_json(output_path, result)
    print("\nPromo evaluation completed.")
    print(f"Episode:              {result['episode_id']}")
    print(f"Candidates selected:  {result['candidates_selected_for_evaluation']}")
    print(f"Evaluated:            {result['evaluated_count']}")
    print(f"Structurally rejected:{result['structurally_rejected_count']:>3}")
    print(f"API calls this run:   {result['model_calls_this_run']}")
    if result["evaluations"]:
        print("\nTop candidates:")
        for item in result["evaluations"][:5]:
            print(f"  #{item['score_rank']} {item['candidate_id']}: {item['weighted_score']}/100 — {item['recommendation']}")
    print(f"Saved:                {output_path}")


if __name__ == "__main__":
    main()
