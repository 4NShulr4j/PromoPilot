"""Extract structured story intelligence from a parsed PromoPilot episode."""

from __future__ import annotations

import argparse
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.llm_client import request_json_object

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "story_extraction_prompt.txt"

class CharacterProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    source_labels: list[str]
    description: str
    relationships: list[str]


class DramaticMoment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moment_type: Literal[
        "hook",
        "conflict",
        "reveal",
        "emotional_peak",
        "suspense",
        "cliffhanger",
        "visual_beat",
    ]
    event_ids: list[str] = Field(min_length=1)
    description: str
    why_significant: str
    emotional_intensity: int = Field(ge=0, le=5)
    hook_potential: int = Field(ge=0, le=5)
    cliffhanger_potential: int = Field(ge=0, le=5)
    context_dependent: bool
    context_needed: str


class StoryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_summary: str
    characters: list[CharacterProfile]
    dramatic_moments: list[DramaticMoment]


def _compact_episode_for_prompt(episode: dict[str, Any]) -> str:
    """Create a compact, source-grounded representation for the LLM."""
    event_rows = []
    for event in episode.get("events", []):
        event_rows.append(
            {
                "event_id": event["event_id"],
                "scene_number": event["scene_number"],
                "scene_heading": event["scene_heading"],
                "label": event["label"],
                "speaker": event.get("speaker"),
                "event_type": event["event_type"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "content": event["content"],
            }
        )

    payload = {
        "episode_id": episode["episode_id"],
        "episode_number": episode["episode_number"],
        "logline": episode.get("logline", ""),
        "continuity_notes": episode.get("continuity_notes", ""),
        "duration_seconds": episode.get("duration_seconds"),
        "scenes": episode.get("scenes", []),
        "events": event_rows,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(episode: dict[str, Any], prompt_template: str | None = None) -> str:
    template = prompt_template or PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.strip()
        + "\n\nEPISODE ID: "
        + episode["episode_id"]
        + "\n\nSOURCE EPISODE DATA (JSON):\n"
        + _compact_episode_for_prompt(episode)
    )


def validate_and_enrich_analysis(
    raw_data: dict[str, Any], episode: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate model output and attach exact source-event evidence.

    Unknown event IDs cause the corresponding dramatic moment to be rejected.
    Model-generated quotes/timestamps are intentionally not trusted or used.
    """
    analysis = StoryAnalysis.model_validate(raw_data)

    if analysis.episode_id != episode["episode_id"]:
        raise ValueError(
            f"Model returned episode_id {analysis.episode_id!r}; "
            f"expected {episode['episode_id']!r}."
        )

    event_lookup = {event["event_id"]: event for event in episode["events"]}
    known_speakers = {
        event["speaker"]
        for event in episode["events"]
        if event.get("speaker")
    }

    warnings: list[str] = []
    rejected_moments: list[dict[str, Any]] = []
    accepted_moments: list[dict[str, Any]] = []

    for index, moment in enumerate(analysis.dramatic_moments, start=1):
        event_ids = list(dict.fromkeys(moment.event_ids))
        unknown_ids = [event_id for event_id in event_ids if event_id not in event_lookup]

        if unknown_ids:
            rejected_moments.append(
                {
                    "moment_index": index,
                    "reason": "unknown_event_ids",
                    "unknown_event_ids": unknown_ids,
                    "description": moment.description,
                }
            )
            continue

        evidence_events = sorted(
            (event_lookup[event_id] for event_id in event_ids),
            key=lambda item: (item["start_seconds"], item["end_seconds"]),
        )

        if not evidence_events:
            rejected_moments.append(
                {
                    "moment_index": index,
                    "reason": "no_source_evidence",
                    "description": moment.description,
                }
            )
            continue

        if any(e["episode_number"] != episode["episode_index"] for e in evidence_events):
            rejected_moments.append(
                {
                    "moment_index": index,
                    "reason": "cross_episode_event_reference",
                    "description": moment.description,
                }
            )
            continue

        item = moment.model_dump()
        item["event_ids"] = [event["event_id"] for event in evidence_events]
        # These fields are copied from the parsed screenplay, never generated by the LLM.
        item["source_evidence"] = [
            {
                "event_id": event["event_id"],
                "label": event["label"],
                "speaker": event.get("speaker"),
                "event_type": event["event_type"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "content": event["content"],
            }
            for event in evidence_events
        ]
        accepted_moments.append(item)

    characters = []
    for character in analysis.characters:
        valid_labels = [
            label for label in character.source_labels if label in known_speakers
        ]
        invalid_labels = [
            label for label in character.source_labels if label not in known_speakers
        ]
        if invalid_labels:
            warnings.append(
                f"Character {character.canonical_name!r} referenced unknown "
                f"speaker labels: {invalid_labels}"
            )
        character_data = character.model_dump()
        character_data["source_labels"] = valid_labels
        characters.append(character_data)

    validated = {
        "episode_id": analysis.episode_id,
        "episode_summary": analysis.episode_summary,
        "characters": characters,
        "dramatic_moments": accepted_moments,
        "source_event_count": len(episode["events"]),
        "accepted_moment_count": len(accepted_moments),
    }

    report = {
        "episode_id": episode["episode_id"],
        "input_event_count": len(episode["events"]),
        "model_moment_count": len(analysis.dramatic_moments),
        "accepted_moment_count": len(accepted_moments),
        "rejected_moment_count": len(rejected_moments),
        "rejected_moments": rejected_moments,
        "warnings": warnings,
    }
    return validated, report


def select_story_events(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep narrative-bearing events for LLM analysis.

    Pure CAMERA framing blocks are retained in the parsed source data and will
    be available to later modules through event timestamps. They are omitted
    from this analysis prompt because they dominate the source token count and
    usually repeat shot/framing information rather than story facts.
    """
    events = [
        event for event in episode.get("events", [])
        if str(event.get("label", "")).strip().upper() != "CAMERA"
    ]
    # Very unusual scripts may contain only camera blocks. Preserve a fallback
    # so the tool remains transparent rather than submitting an empty prompt.
    return events or list(episode.get("events", []))


def chunk_events(
    events: list[dict[str, Any]], *, chunk_size: int = 10, overlap: int = 2
) -> list[list[dict[str, Any]]]:
    """Split chronological events into overlapping, bounded chunks."""
    if chunk_size < 4:
        raise ValueError("chunk_size must be at least 4")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if not events:
        return []

    ordered = sorted(
        events,
        key=lambda e: (
            e.get("start_seconds", 0),
            e.get("end_seconds", 0),
            e.get("event_id", ""),
        ),
    )
    step = chunk_size - overlap
    chunks: list[list[dict[str, Any]]] = []
    for start in range(0, len(ordered), step):
        chunk = ordered[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(ordered):
            break
    return chunks


def _make_chunk_episode(
    episode: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create a shallow episode view containing only this chunk's events."""
    scene_numbers = {event.get("scene_number") for event in events}
    chunk_episode = dict(episode)
    chunk_episode["events"] = events
    chunk_episode["event_count"] = len(events)
    chunk_episode["scenes"] = [
        scene for scene in episode.get("scenes", [])
        if scene.get("scene_number") in scene_numbers
    ]
    return chunk_episode


def _chunk_fingerprint(episode_id: str, events: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"episode_id": episode_id, "events": events},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_chunk_results(
    episode: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    chunk_reports: list[dict[str, Any]],
    model_metadata: list[dict[str, Any]],
    story_input_event_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge overlapping chunk analyses without fabricating new evidence."""
    character_map: dict[str, dict[str, Any]] = {}
    moment_map: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    summaries: list[str] = []
    warnings: list[str] = []
    rejected: list[dict[str, Any]] = []
    model_moment_count = 0

    for chunk_index, (result, report) in enumerate(
        zip(chunk_results, chunk_reports), start=1
    ):
        summary = str(result.get("episode_summary", "")).strip()
        if summary and summary not in summaries:
            summaries.append(summary)

        for warning in report.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        rejected.extend(report.get("rejected_moments", []))
        model_moment_count += int(report.get("model_moment_count", 0))

        for character in result.get("characters", []):
            key = str(character.get("canonical_name", "")).strip().casefold()
            if not key:
                continue
            if key not in character_map:
                character_map[key] = {
                    "canonical_name": character.get("canonical_name", ""),
                    "source_labels": [],
                    "description": character.get("description", ""),
                    "relationships": [],
                }
            merged = character_map[key]
            for label in character.get("source_labels", []):
                if label not in merged["source_labels"]:
                    merged["source_labels"].append(label)
            for relationship in character.get("relationships", []):
                if relationship and relationship not in merged["relationships"]:
                    merged["relationships"].append(relationship)
            if len(character.get("description", "")) > len(merged["description"]):
                merged["description"] = character.get("description", "")

        for moment in result.get("dramatic_moments", []):
            event_ids = list(dict.fromkeys(moment.get("event_ids", [])))
            key = (moment.get("moment_type", ""), tuple(sorted(event_ids)))
            if key not in moment_map:
                moment_map[key] = moment
            else:
                existing = moment_map[key]
                # Preserve the more specific explanation/stronger ratings.
                for score in (
                    "emotional_intensity",
                    "hook_potential",
                    "cliffhanger_potential",
                ):
                    existing[score] = max(existing.get(score, 0), moment.get(score, 0))
                if len(moment.get("why_significant", "")) > len(existing.get("why_significant", "")):
                    existing["why_significant"] = moment["why_significant"]
                if len(moment.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = moment["description"]

    all_events = {event["event_id"]: event for event in episode.get("events", [])}
    moments = list(moment_map.values())
    moments.sort(
        key=lambda item: min(
            (all_events[eid]["start_seconds"] for eid in item.get("event_ids", []) if eid in all_events),
            default=float("inf"),
        )
    )

    # Check merged character source labels against the full episode, not a chunk.
    known_speakers = {
        event.get("speaker") for event in episode.get("events", [])
        if event.get("speaker")
    }
    for character in character_map.values():
        character["source_labels"] = [
            label for label in character["source_labels"] if label in known_speakers
        ]

    summary_text = " ".join(summaries)
    validated = {
        "episode_id": episode["episode_id"],
        "episode_summary": summary_text,
        "chunk_summaries": summaries,
        "characters": list(character_map.values()),
        "dramatic_moments": moments,
        "source_event_count": len(episode.get("events", [])),
        "story_input_event_count": story_input_event_count,
        "accepted_moment_count": len(moments),
        "model_metadata": {
            "provider": "groq",
            "model": model_metadata[0].get("model") if model_metadata else None,
            "chunk_count": len(chunk_results),
            "chunk_metadata": model_metadata,
        },
    }

    report = {
        "episode_id": episode["episode_id"],
        "input_event_count": len(episode.get("events", [])),
        "story_input_event_count": story_input_event_count,
        "chunk_count": len(chunk_results),
        "model_moment_count": model_moment_count,
        "accepted_moment_count": len(moments),
        "rejected_moment_count": len(rejected),
        "rejected_moments": rejected,
        "warnings": warnings,
        "status": "success_with_rejections" if rejected else "success",
    }
    return validated, report


def analyze_episode(
    episode: dict[str, Any],
    *,
    output_dir: str | Path = "outputs/story_analysis",
    raw_dir: str | Path = "outputs/raw_llm",
    max_completion_tokens: int = 900,
    chunk_size: int = 10,
    chunk_overlap: int = 2,
    delay_between_chunks: float = 18.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Analyse one episode in small overlapping chunks to fit free-tier TPM.

    Pure CAMERA events are deliberately excluded from LLM story analysis to
    reduce prompt size. They remain in the original parsed episode and can be
    retrieved later when a candidate's source timestamps are known.
    """
    episode_id = episode["episode_id"].lower()
    output_path = Path(output_dir)
    raw_path = Path(raw_dir)
    chunk_path = output_path / "chunks"
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)
    chunk_path.mkdir(parents=True, exist_ok=True)

    story_events = select_story_events(episode)
    chunks = chunk_events(story_events, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No events available for story analysis: {episode_id}")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    system_prompt = (
        "You are a careful microdrama story analyst. Analyse only the event "
        "chunk supplied in this request. Do not assume that it represents the "
        "entire episode. Follow source-evidence rules and return one valid JSON "
        "object matching the exact structure described in the user prompt. "
        "Do not use Markdown fences."
    )

    chunk_results: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    last_api_call_at: float | None = None

    for index, events in enumerate(chunks, start=1):
        chunk_episode = _make_chunk_episode(episode, events)
        fingerprint = _chunk_fingerprint(episode_id, events)
        validated_cache = chunk_path / f"{episode_id}_chunk_{index:02d}.json"
        raw_file = raw_path / f"{episode_id}_story_chunk_{index:02d}_raw.txt"

        # Resume a completed chunk if its source events have not changed.
        if resume and validated_cache.exists():
            cached = json.loads(validated_cache.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                chunk_results.append(cached["validated"])
                chunk_reports.append(cached["report"])
                metadata_rows.append(cached.get("model_metadata", {}))
                print(f"[{index}/{len(chunks)}] Reusing completed chunk checkpoint")
                continue

        chunk_prompt = build_user_prompt(chunk_episode, prompt_template)
        chunk_prompt += (
            f"\n\nANALYSIS CHUNK: {index} OF {len(chunks)}. "
            "This is a partial chronological view of one episode. "
            "Return no more than 3 distinct high-signal dramatic moments. "
            "Prefer concise descriptions and cite only event IDs in this chunk. "
            "A moment may use multiple events only when they jointly support it."
        )

        if last_api_call_at is not None and delay_between_chunks > 0:
            elapsed = time.monotonic() - last_api_call_at
            wait_seconds = max(0.0, delay_between_chunks - elapsed)
            if wait_seconds > 0:
                print(f"Waiting {wait_seconds:.1f}s to respect free-tier token limits...")
                time.sleep(wait_seconds)

        print(
            f"[{index}/{len(chunks)}] Analysing {len(events)} narrative events "
            f"(max output {max_completion_tokens} tokens)..."
        )
        try:
            raw_text, metadata = request_json_object(
                system_prompt=system_prompt,
                user_prompt=chunk_prompt,
                max_completion_tokens=max_completion_tokens,
            )
            last_api_call_at = time.monotonic()
        except Exception as exc:
            # Previous completed chunks are checkpointed and will be reused on
            # the next run. This avoids wasting free-tier quota after a failure.
            raise RuntimeError(
                f"Story analysis failed at chunk {index}/{len(chunks)}. "
                f"Completed chunks are saved under {chunk_path}; rerun the same "
                f"command to resume. Original error: {type(exc).__name__}: {exc}"
            ) from exc

        # Save raw model output before parsing or validating it.
        raw_file.write_text(raw_text, encoding="utf-8")

        try:
            raw_data = json.loads(raw_text)
            validated_chunk, report = validate_and_enrich_analysis(
                raw_data, chunk_episode
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            error_path = chunk_path / f"{episode_id}_chunk_{index:02d}_error.json"
            error_path.write_text(
                json.dumps(
                    {"chunk": index, "error": str(exc), "model_metadata": metadata},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Chunk {index} returned invalid story analysis. Raw response: {raw_file}"
            ) from exc

        cache_payload = {
            "fingerprint": fingerprint,
            "validated": validated_chunk,
            "report": report,
            "model_metadata": metadata,
        }
        validated_cache.write_text(
            json.dumps(cache_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        chunk_results.append(validated_chunk)
        chunk_reports.append(report)
        metadata_rows.append(metadata)

    validated, report = _merge_chunk_results(
        episode,
        chunk_results,
        chunk_reports,
        metadata_rows,
        story_input_event_count=len(story_events),
    )

    (output_path / f"{episode_id}_story_analysis.json").write_text(
        json.dumps(validated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / f"{episode_id}_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validated

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse one parsed PromoPilot episode using Groq."
    )
    parser.add_argument(
        "--input",
        default="data/processed/ep01.json",
        help="Path to one parsed episode JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/story_analysis",
        help="Directory for validated analysis and reports.",
    )
    parser.add_argument(
        "--raw-dir",
        default="outputs/raw_llm",
        help="Directory for untouched LLM responses.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=900,
        help="Maximum output tokens per chunk (default: 900).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Narrative events per request (default: 10).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=2,
        help="Overlapping events between requests (default: 2).",
    )
    parser.add_argument(
        "--delay-between-chunks",
        type=float,
        default=18.0,
        help="Minimum seconds between API calls (default: 18).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all chunks instead of reusing saved checkpoints.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Parsed episode JSON not found: {input_path}. Run the parser first."
        )

    episode = json.loads(input_path.read_text(encoding="utf-8"))
    result = analyze_episode(
        episode,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        max_completion_tokens=args.max_completion_tokens,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        delay_between_chunks=args.delay_between_chunks,
        resume=not args.no_resume,
    )

    print(f"Story analysis completed for {result['episode_id']}.")
    print(f"Characters identified: {len(result['characters'])}")
    print(f"Dramatic moments accepted: {result['accepted_moment_count']}")
    print(f"Narrative events analysed: {result['story_input_event_count']}")
    print(f"API chunks processed: {result['model_metadata']['chunk_count']}")
    print(f"Validated output: {Path(args.output_dir) / (result['episode_id'].lower() + '_story_analysis.json')}")


if __name__ == "__main__":
    main()
