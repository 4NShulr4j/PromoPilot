"""Parser for Kuku-style timestamped microdrama screenplay TXT files.

The parser preserves overlapping camera, action, dialogue and sound events.
It does not infer or generate timestamps; all times come from the source file.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

TIMESTAMP_PATTERN = r"\d{2,}:\d{2}\.\d{2}"

METADATA_PATTERNS = {
    "episode_number": re.compile(r"(?m)^EPISODE NUMBER:\s*(.+?)\s*$"),
    "logline": re.compile(r"(?m)^EPISODE LOGLINE:\s*(.+?)\s*$"),
    "continuity_notes": re.compile(r"(?m)^CONTINUITY NOTES:\s*(.+?)\s*$"),
}

SCENE_HEADER_RE = re.compile(r"(?m)^SCENE\s+(\d+)\s*$")
EVENT_HEADER_RE = re.compile(
    rf"(?m)^\*\*(?P<label>[^*\r\n]+?):\*\*\s*\r?\n"
    rf"Start:\s*\((?P<start>{TIMESTAMP_PATTERN})\)"
    rf"\s*\|\s*End:\s*\((?P<end>{TIMESTAMP_PATTERN})\)\s*\r?\n"
)
SCENE_HEADING_RE = re.compile(r"(?m)^Scene heading:\s*(.*?)\s*$")
SCENE_START_RE = re.compile(rf"(?m)^Scene start:\s*({TIMESTAMP_PATTERN})\s*$")
SCENE_END_RE = re.compile(rf"(?m)^Scene end:\s*({TIMESTAMP_PATTERN})\s*$")

DIRECTION_LABELS = {"CAMERA", "ACTION", "SOUND FX", "SOUND EFFECT", "MUSIC"}


class ScreenplayParseError(ValueError):
    """Raised when a screenplay cannot be parsed safely."""


def timestamp_to_seconds(timestamp: str) -> Decimal:
    """Convert MM:SS.xx to Decimal seconds, avoiding float arithmetic drift."""
    try:
        minutes_text, seconds_text = timestamp.split(":", maxsplit=1)
        minutes = Decimal(minutes_text)
        seconds = Decimal(seconds_text)
    except Exception as exc:  # noqa: BLE001 - normalise malformed input errors
        raise ScreenplayParseError(f"Invalid timestamp: {timestamp!r}") from exc

    if minutes < 0 or seconds < 0 or seconds >= 60:
        raise ScreenplayParseError(f"Timestamp out of range: {timestamp!r}")

    return minutes * Decimal(60) + seconds


def _seconds_as_number(value: Decimal) -> float:
    """Return a JSON-friendly number while keeping source strings separately."""
    return float(value.quantize(Decimal("0.01")))


def _extract_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for field, pattern in METADATA_PATTERNS.items():
        match = pattern.search(text)
        metadata[field] = match.group(1).strip() if match else ""

    episode_match = re.search(r"\d+", metadata["episode_number"])
    if not episode_match:
        raise ScreenplayParseError("Could not extract episode number")

    metadata["episode_index"] = str(int(episode_match.group(0)))
    return metadata


def _clean_event_content(content: str) -> str:
    return content.strip().strip("\r\n")


def parse_screenplay_text(text: str, source_name: str = "<memory>") -> dict[str, Any]:
    """Parse screenplay text into metadata, scenes and timestamped events."""
    if not isinstance(text, str) or not text.strip():
        raise ScreenplayParseError(f"Empty screenplay: {source_name}")

    metadata = _extract_metadata(text)
    episode_index = int(metadata["episode_index"])
    episode_id = f"EP{episode_index:02d}"

    scene_matches = list(SCENE_HEADER_RE.finditer(text))
    if not scene_matches:
        raise ScreenplayParseError(f"No SCENE headings found in {source_name}")

    scenes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for scene_position, scene_match in enumerate(scene_matches):
        section_start = scene_match.end()
        section_end = (
            scene_matches[scene_position + 1].start()
            if scene_position + 1 < len(scene_matches)
            else len(text)
        )
        scene_text = text[section_start:section_end]
        scene_number = int(scene_match.group(1))

        heading_match = SCENE_HEADING_RE.search(scene_text)
        start_match = SCENE_START_RE.search(scene_text)
        end_match = SCENE_END_RE.search(scene_text)

        if not heading_match or not start_match or not end_match:
            raise ScreenplayParseError(
                f"Scene {scene_number} is missing heading/start/end in {source_name}"
            )

        scene_start = timestamp_to_seconds(start_match.group(1))
        scene_end = timestamp_to_seconds(end_match.group(1))
        if scene_end <= scene_start:
            raise ScreenplayParseError(
                f"Scene {scene_number} has invalid time range in {source_name}"
            )

        scene = {
            "scene_id": f"{episode_id}_SC{scene_number:02d}",
            "scene_number": scene_number,
            "heading": heading_match.group(1).strip(),
            "start_time": start_match.group(1),
            "end_time": end_match.group(1),
            "start_seconds": _seconds_as_number(scene_start),
            "end_seconds": _seconds_as_number(scene_end),
        }
        scenes.append(scene)

        event_matches = list(EVENT_HEADER_RE.finditer(scene_text))
        for event_position, event_match in enumerate(event_matches):
            content_start = event_match.end()
            content_end = (
                event_matches[event_position + 1].start()
                if event_position + 1 < len(event_matches)
                else len(scene_text)
            )
            # Do not attach the screenplay terminator to the final event content.
            script_end_match = re.search(
                r"(?m)^--- SCRIPT END ---\s*$", scene_text
            )
            if script_end_match:
                content_end = min(content_end, script_end_match.start())

            label = event_match.group("label").strip()
            start_text = event_match.group("start")
            end_text = event_match.group("end")
            start_seconds = timestamp_to_seconds(start_text)
            end_seconds = timestamp_to_seconds(end_text)

            if end_seconds <= start_seconds:
                raise ScreenplayParseError(
                    f"Event has non-positive duration: {label!r} in {source_name}"
                )
            if start_seconds < scene_start or end_seconds > scene_end:
                raise ScreenplayParseError(
                    f"Event {label!r} ({start_text}-{end_text}) lies outside "
                    f"scene {scene_number} ({scene['start_time']}-{scene['end_time']}) "
                    f"in {source_name}"
                )

            base_label = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip()
            normalised_base = re.sub(r"\s+", " ", base_label).upper()
            is_direction = normalised_base in DIRECTION_LABELS

            event_number = event_position + 1
            event = {
                "event_id": f"{episode_id}_SC{scene_number:02d}_EV{event_number:03d}",
                "episode_number": episode_index,
                "scene_id": scene["scene_id"],
                "scene_number": scene_number,
                "scene_heading": scene["heading"],
                "label": label,
                "speaker": None if is_direction else base_label,
                "event_type": "direction" if is_direction else "dialogue",
                "start_time": start_text,
                "end_time": end_text,
                "start_seconds": _seconds_as_number(start_seconds),
                "end_seconds": _seconds_as_number(end_seconds),
                "duration_seconds": _seconds_as_number(end_seconds - start_seconds),
                "content": _clean_event_content(scene_text[content_start:content_end]),
            }
            events.append(event)

    # Keep overlapping events. Sorting only establishes deterministic timeline order.
    events.sort(
        key=lambda item: (
            item["start_seconds"],
            item["end_seconds"],
            item["event_id"],
        )
    )

    if not events:
        raise ScreenplayParseError(f"No timestamped events found in {source_name}")

    return {
        "source_file": source_name,
        "episode_number": metadata["episode_number"],
        "episode_index": episode_index,
        "episode_id": episode_id,
        "logline": metadata["logline"],
        "continuity_notes": metadata["continuity_notes"],
        "duration_seconds": max(scene["end_seconds"] for scene in scenes),
        "scenes": scenes,
        "events": events,
        "event_count": len(events),
    }


def parse_screenplay_file(file_path: str | Path) -> dict[str, Any]:
    """Read and parse a UTF-8 screenplay TXT file."""
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ScreenplayParseError(
            f"Could not decode {path.name} as UTF-8. Save it as UTF-8 and retry."
        ) from exc
    except OSError as exc:
        raise ScreenplayParseError(f"Could not read {path}: {exc}") from exc

    return parse_screenplay_text(text, source_name=path.name)


def save_parsed_episode(parsed: dict[str, Any], output_dir: str | Path) -> Path:
    """Save one parsed episode as structured JSON."""
    output_path = Path(output_dir) / f"{parsed['episode_id'].lower()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def parse_input_directory(input_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    """Parse all TXT files in a directory and save one JSON per episode."""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt screenplay files found in {input_path}")

    parsed_episodes: list[dict[str, Any]] = []
    errors: list[str] = []

    for file_path in files:
        try:
            parsed = parse_screenplay_file(file_path)
            save_parsed_episode(parsed, output_dir)
            parsed_episodes.append(parsed)
        except (ScreenplayParseError, OSError) as exc:
            errors.append(f"{file_path.name}: {exc}")

    if errors:
        error_path = Path(output_dir) / "parser_errors.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("\n".join(errors), encoding="utf-8")

    if not parsed_episodes:
        raise ScreenplayParseError(
            "No episodes parsed successfully. See parser_errors.txt for details."
        )

    return parsed_episodes


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Parse PromoPilot screenplay TXT files")
    cli.add_argument("--input", default="data/input", help="Directory containing TXT files")
    cli.add_argument("--output", default="data/processed", help="Output directory for JSON")
    args = cli.parse_args()

    parsed = parse_input_directory(args.input, args.output)
    print(f"Successfully parsed {len(parsed)} episode(s).")
    for episode in parsed:
        print(
            f"{episode['episode_id']}: {len(episode['scenes'])} scene(s), "
            f"{episode['event_count']} event(s), "
            f"{episode['duration_seconds']:.2f}s"
        )
