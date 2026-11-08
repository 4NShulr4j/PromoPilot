from decimal import Decimal

from src.parser import parse_screenplay_text, timestamp_to_seconds, ScreenplayParseError


def sample_script():
    return '''EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A short test episode.
CONTINUITY NOTES: Test continuity.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM – DAY
Scene start: 00:00.00
Scene end: 00:30.00

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
A close-up.

**ACTION:**
Start: (00:01.00) | End: (00:05.00)
A person enters.

**JULIA (Angry, Loud):**
Start: (00:05.00) | End: (00:07.50)
"You lied!"

**SOUND FX:**
Start: (00:07.50) | End: (00:08.00)
A loud bang.

--- SCRIPT END ---
'''


def test_timestamp_to_seconds():
    assert timestamp_to_seconds("00:11.30") == Decimal("11.30")
    assert timestamp_to_seconds("02:18.00") == 138


def test_parser_extracts_metadata_and_events():
    parsed = parse_screenplay_text(sample_script(), "sample.txt")
    assert parsed["episode_id"] == "EP07"
    assert parsed["episode_index"] == 7
    assert len(parsed["scenes"]) == 1
    assert parsed["event_count"] == 4
    assert parsed["duration_seconds"] == 30.0


def test_overlapping_events_are_preserved():
    parsed = parse_screenplay_text(sample_script())
    camera = parsed["events"][0]
    action = next(e for e in parsed["events"] if e["label"] == "ACTION")
    assert camera["start_seconds"] == 0.0
    assert action["start_seconds"] == 1.0
    assert action["end_seconds"] == 5.0


def test_dialogue_speaker_normalisation():
    parsed = parse_screenplay_text(sample_script())
    dialogue = next(e for e in parsed["events"] if e["event_type"] == "dialogue")
    assert dialogue["speaker"] == "JULIA"
    assert dialogue["content"] == '"You lied!"'


def test_script_end_marker_is_not_part_of_last_event():
    parsed = parse_screenplay_text(sample_script())
    assert "SCRIPT END" not in parsed["events"][-1]["content"]


def test_rejects_event_outside_scene():
    text = sample_script().replace(
        "End: (00:08.00)\nA loud bang.",
        "End: (00:35.00)\nA loud bang."
    )
    try:
        parse_screenplay_text(text)
    except ScreenplayParseError:
        pass
    else:
        raise AssertionError("Expected invalid event to be rejected")
