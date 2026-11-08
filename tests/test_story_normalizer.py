import copy

import pytest

from src.story_normalizer import (
    empty_series_registry,
    normalize_speaker_label,
    normalize_story,
)


def make_episode(episode_id="EP77"):
    return {
        "episode_id": episode_id,
        "episode_number": 77,
        "logline": "Mira discovers that her mentor has hidden the truth.",
        "continuity_notes": "Mira and Dev are introduced.",
        "duration_seconds": 40.0,
        "events": [
            {
                "event_id": f"{episode_id}_SC01_EV001",
                "episode_number": 77,
                "scene_id": f"{episode_id}_SC01",
                "label": "MIRA (Shocked)",
                "speaker": "MIRA",
                "event_type": "dialogue",
                "start_time": "00:02.00",
                "end_time": "00:04.00",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "content": "You knew the whole time?",
            },
            {
                "event_id": f"{episode_id}_SC01_EV002",
                "episode_number": 77,
                "scene_id": f"{episode_id}_SC01",
                "label": "DEV (Cold) (O.S.)",
                "speaker": "DEV",
                "event_type": "dialogue",
                "start_time": "00:05.00",
                "end_time": "00:07.00",
                "start_seconds": 5.0,
                "end_seconds": 7.0,
                "content": "I did what I had to do.",
            },
            {
                "event_id": f"{episode_id}_SC01_EV003",
                "episode_number": 77,
                "scene_id": f"{episode_id}_SC01",
                "label": "CAMERA",
                "speaker": None,
                "event_type": "direction",
                "start_time": "00:35.00",
                "end_time": "00:40.00",
                "start_seconds": 35.0,
                "end_seconds": 40.0,
                "content": "Push in as Mira turns toward the door; cut to black.",
            },
        ],
    }


def make_analysis(episode_id="EP77"):
    return {
        "episode_id": episode_id,
        "episode_summary": "Mira confronts Dev.",
        "chunk_summaries": [],
        "characters": [
            {
                "canonical_name": "Mira",
                "source_labels": ["MIRA (Shocked)"],
                "description": "The protagonist.",
                "relationships": ["sister of Nia"],
            },
            {
                "canonical_name": "DEV",
                "source_labels": [],
                "description": "Mira's mentor.",
                "relationships": ["mentor of Mira"],
            },
            {
                "canonical_name": "The Mentor",
                "source_labels": ["DEV (Cold) (O.S.)"],
                "description": "A mysterious mentor.",
                "relationships": ["mentor of Mira"],
            },
            {
                "canonical_name": "Nia",
                "source_labels": [],
                "description": "A character mentioned in the story.",
                "relationships": ["sister of Mira"],
            },
        ],
        "dramatic_moments": [
            {
                "moment_type": "hook",
                "event_ids": [f"{episode_id}_SC01_EV001"],
                "description": "Mira confronts Dev.",
                "hook_potential": 4,
                "cliffhanger_potential": 1,
                "source_evidence": [{"event_id": "FAKE", "content": "Do not trust model evidence"}],
            },
            {
                "moment_type": "cliffhanger",
                "event_ids": [f"{episode_id}_SC01_EV999"],
                "description": "Invented event.",
            },
        ],
        "model_metadata": {"provider": "test"},
    }


def test_normalize_speaker_label_removes_delivery_notes_and_os():
    assert normalize_speaker_label("MATRIARCH (O.S.) (Firm)") == "MATRIARCH"
    assert normalize_speaker_label("LADY JULIANA (Heartbroken)") == "LADY JULIANA"
    assert normalize_speaker_label("DEV [whispering]") == "DEV"


def test_generic_new_show_and_profile_merging_by_same_source_speaker():
    normalized, report, registry = normalize_story(
        make_episode(), make_analysis(), show_id="new_show_abc"
    )
    by_label = {
        label: character
        for character in normalized["characters"]
        for label in character["source_labels"]
    }
    # DEV and The Mentor both resolve to the same parsed speaker label DEV.
    dev = by_label["DEV"]
    assert dev["canonical_name"] == "DEV" or dev["canonical_name"] == "The Mentor"
    assert dev["identity_evidence"]["profile_count"] == 2
    assert report["accepted_moment_count"] == 1
    assert report["rejected_moment_count"] == 1
    assert registry["show_id"] == "new_show_abc"


def test_different_speaker_labels_are_not_fuzzy_merged_without_approval():
    episode = make_episode()
    episode["events"].append({
        "event_id": "EP77_SC01_EV004", "episode_number": 77,
        "scene_id": "EP77_SC01", "label": "THE MENTOR", "speaker": "THE MENTOR",
        "event_type": "dialogue", "start_time": "00:08.00", "end_time": "00:09.00",
        "start_seconds": 8.0, "end_seconds": 9.0, "content": "Leave now."
    })
    analysis = make_analysis()
    analysis["characters"].append({
        "canonical_name": "Mentor", "source_labels": ["THE MENTOR"],
        "description": "A second speaker label.", "relationships": []
    })
    normalized, _, _ = normalize_story(episode, analysis, show_id="new_show_abc")
    dev_group = next(c for c in normalized["characters"] if "DEV" in c["source_labels"])
    mentor_group = next(c for c in normalized["characters"] if "THE MENTOR" in c["source_labels"])
    assert dev_group["character_id"] != mentor_group["character_id"]


def test_human_approved_alias_merges_distinct_labels_across_episodes():
    registry = empty_series_registry("show_one")
    registry["approved_aliases"]["THE MENTOR"] = {
        "character_id": "char_mentor_001",
        "canonical_name": "Dev",
        "approved": True,
        "evidence_event_ids": ["EP76_SC02_EV004"],
        "approved_by": "human reviewer",
    }
    registry["approved_aliases"]["DEV"] = {
        "character_id": "char_mentor_001",
        "canonical_name": "Dev",
        "approved": True,
        "evidence_event_ids": ["EP77_SC01_EV002"],
        "approved_by": "human reviewer",
    }
    episode = make_episode()
    episode["events"].append({
        "event_id": "EP77_SC01_EV004", "episode_number": 77,
        "scene_id": "EP77_SC01", "label": "THE MENTOR", "speaker": "THE MENTOR",
        "event_type": "dialogue", "start_time": "00:08.00", "end_time": "00:09.00",
        "start_seconds": 8.0, "end_seconds": 9.0, "content": "Leave now."
    })
    normalized, _, _ = normalize_story(episode, make_analysis(), registry, show_id="show_one")
    mentor_groups = [c for c in normalized["characters"] if c["character_id"] == "char_mentor_001"]
    assert len(mentor_groups) == 1
    assert set(mentor_groups[0]["source_labels"]) == {"DEV", "THE MENTOR"}
    assert mentor_groups[0]["identity_status"] == "human_approved_alias"


def test_relationship_claims_are_unverified_unless_in_registry():
    normalized, report, _ = normalize_story(make_episode(), make_analysis(), show_id="show_x")
    mira = next(c for c in normalized["characters"] if "MIRA" in c["source_labels"])
    assert mira["confirmed_relationships"] == []
    assert any(c["claim"] == "sister of Nia" for c in mira["unverified_relationship_claims"])
    assert report["model_relationship_claim_count"] >= 1


def test_moment_evidence_is_rebuilt_from_parser_and_invalid_ids_rejected():
    normalized, report, _ = normalize_story(make_episode(), make_analysis(), show_id="show_x")
    moment = normalized["dramatic_moments"][0]
    assert moment["source_evidence"][0]["event_id"] == "EP77_SC01_EV001"
    assert moment["source_evidence"][0]["content"] == "You knew the whole time?"
    assert moment["source_evidence"][0]["event_id"] != "FAKE"
    assert report["rejected_moment_count"] == 1


def test_visual_only_hook_is_flagged_for_review_not_reclassified():
    episode = make_episode()
    analysis = make_analysis()
    analysis["dramatic_moments"] = [{
        "moment_type": "hook", "event_ids": ["EP77_SC01_EV003"],
        "description": "Visual ending beat.", "hook_potential": 5,
    }]
    normalized, report, _ = normalize_story(episode, analysis, show_id="show_x")
    moment = normalized["dramatic_moments"][0]
    assert moment["moment_type"] == "hook"
    assert moment["editorial_review_flags"][0]["code"] == "visual_only_hook_review"
    assert report["moment_review_flag_count"] == 1


def test_registry_is_show_scoped():
    registry = empty_series_registry("show_a")
    with pytest.raises(ValueError, match="does not match"):
        normalize_story(make_episode(), make_analysis(), registry, show_id="show_b")


def test_episode_ending_context_is_derived_from_timeline():
    normalized, _, _ = normalize_story(make_episode(), make_analysis(), show_id="show_x")
    ending = normalized["episode_ending_context"]
    assert ending["status"].startswith("derived_from_source_timeline")
    assert ending["window_start_seconds"] == 25.0
    assert ending["events"][-1]["event_id"] == "EP77_SC01_EV003"


def test_new_show_has_no_hardcoded_character_names_or_episode_ids():
    episode = make_episode("S99E02")
    analysis = make_analysis("S99E02")
    normalized, report, registry = normalize_story(episode, analysis, show_id="a_completely_new_show")
    assert normalized["episode_id"] == "S99E02"
    assert normalized["show_id"] == "a_completely_new_show"
    assert registry["show_id"] == "a_completely_new_show"
    assert report["status"] in {"success", "success_with_review_items"}
    assert all("Juliana" not in c["canonical_name"] for c in normalized["characters"])


def test_registry_only_persists_observed_labels_not_model_inferences():
    normalized, _, registry = normalize_story(make_episode(), make_analysis(), show_id="show_x")
    assert "MIRA" in registry["observed_speaker_labels"]
    assert "DEV" in registry["observed_speaker_labels"]
    assert registry["approved_aliases"] == {}
    assert registry["approved_relationships"] == []


def test_unlinked_profile_can_receive_review_only_alias_suggestion():
    episode = make_episode()
    analysis = make_analysis()
    analysis["characters"].append({
        "canonical_name": "Nia",
        "source_labels": [],
        "description": "A woman wearing a silver crown and blue cloak.",
        "relationships": [],
    })
    episode["events"].append({
        "event_id": "EP77_SC01_EV004", "episode_number": 77,
        "scene_id": "EP77_SC01", "label": "THE WOMAN IN SILVER CROWN",
        "speaker": "THE WOMAN IN SILVER CROWN", "event_type": "dialogue",
        "start_time": "00:08.00", "end_time": "00:09.00",
        "start_seconds": 8.0, "end_seconds": 9.0, "content": "I am Nia."
    })
    normalized, _, _ = normalize_story(episode, analysis, show_id="show_x")
    suggestion = next(
        item for item in normalized["identity_resolution_candidates"]
        if item["profile_name"] == "Nia"
    )
    assert suggestion["status"] == "requires_human_review"
    assert "THE WOMAN IN SILVER CROWN" in suggestion["candidate_source_labels"]
    # The alias is only suggested; no auto-merge occurs.
    nia = next(c for c in normalized["characters"] if c["canonical_name"] == "Nia")
    assert nia["identity_status"] == "unlinked_model_profile"


def test_registry_accumulates_speaker_observations_across_episodes():
    registry = empty_series_registry("same_show")
    first_episode = make_episode("EP01")
    first_analysis = make_analysis("EP01")
    _, _, registry = normalize_story(first_episode, first_analysis, registry, show_id="same_show")

    second_episode = make_episode("EP02")
    second_analysis = make_analysis("EP02")
    _, _, registry = normalize_story(second_episode, second_analysis, registry, show_id="same_show")

    assert registry["observed_speaker_labels"]["MIRA"]["episodes_seen"] == ["EP01", "EP02"]
    assert registry["observed_speaker_labels"]["DEV"]["episodes_seen"] == ["EP01", "EP02"]
