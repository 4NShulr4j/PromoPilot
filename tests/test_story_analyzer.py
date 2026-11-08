import pytest
from pydantic import ValidationError

from src.story_analyzer import (
    StoryAnalysis,
    validate_and_enrich_analysis,
)


@pytest.fixture
def parsed_episode():
    return {
        "episode_number": "Episode 1",
        "episode_index": 1,
        "episode_id": "EP01",
        "events": [
            {
                "event_id": "EP01_SC01_EV001",
                "episode_number": 1,
                "scene_number": 1,
                "label": "MATRIARCH (Cold, Commanding)",
                "speaker": "MATRIARCH",
                "event_type": "dialogue",
                "start_time": "00:11.30",
                "end_time": "00:13.10",
                "start_seconds": 11.3,
                "end_seconds": 13.1,
                "content": '"You should turn around, Juliana."',
            },
            {
                "event_id": "EP01_SC01_EV002",
                "episode_number": 1,
                "scene_number": 1,
                "label": "LADY JULIANA (Confused, Breathless)",
                "speaker": "LADY JULIANA",
                "event_type": "dialogue",
                "start_time": "00:13.50",
                "end_time": "00:14.30",
                "start_seconds": 13.5,
                "end_seconds": 14.3,
                "content": '"Why?"',
            },
        ],
    }


@pytest.fixture
def valid_payload():
    return {
        "episode_id": "EP01",
        "episode_summary": "A bride discovers a betrayal at the altar.",
        "characters": [
            {
                "canonical_name": "Matriarch",
                "source_labels": ["MATRIARCH"],
                "description": "A stern family authority figure.",
                "relationships": ["Part of Juliana's family."],
            }
        ],
        "dramatic_moments": [
            {
                "moment_type": "hook",
                "event_ids": ["EP01_SC01_EV001"],
                "description": "A warning stops Juliana in the aisle.",
                "why_significant": "It immediately creates a question.",
                "emotional_intensity": 3,
                "hook_potential": 5,
                "cliffhanger_potential": 2,
                "context_dependent": False,
                "context_needed": "",
            }
        ],
    }


def test_valid_analysis_is_enriched_with_source_evidence(parsed_episode, valid_payload):
    validated, report = validate_and_enrich_analysis(valid_payload, parsed_episode)

    assert validated["episode_id"] == "EP01"
    assert validated["accepted_moment_count"] == 1
    evidence = validated["dramatic_moments"][0]["source_evidence"][0]
    assert evidence["start_time"] == "00:11.30"
    assert evidence["content"] == '"You should turn around, Juliana."'
    assert report["rejected_moment_count"] == 0


def test_unknown_event_id_rejects_moment(parsed_episode, valid_payload):
    valid_payload["dramatic_moments"][0]["event_ids"] = ["MADE_UP_EVENT"]

    validated, report = validate_and_enrich_analysis(valid_payload, parsed_episode)

    assert validated["dramatic_moments"] == []
    assert report["rejected_moment_count"] == 1
    assert report["rejected_moments"][0]["reason"] == "unknown_event_ids"


def test_unknown_speaker_label_is_removed_with_warning(parsed_episode, valid_payload):
    valid_payload["characters"][0]["source_labels"] = ["MATRIARCH", "INVENTED LABEL"]

    validated, report = validate_and_enrich_analysis(valid_payload, parsed_episode)

    assert validated["characters"][0]["source_labels"] == ["MATRIARCH"]
    assert len(report["warnings"]) == 1


def test_score_outside_range_is_rejected(valid_payload):
    valid_payload["dramatic_moments"][0]["hook_potential"] = 8

    with pytest.raises(ValidationError):
        StoryAnalysis.model_validate(valid_payload)


def test_wrong_episode_id_is_rejected(parsed_episode, valid_payload):
    valid_payload["episode_id"] = "EP99"

    with pytest.raises(ValueError, match="expected 'EP01'"):
        validate_and_enrich_analysis(valid_payload, parsed_episode)
