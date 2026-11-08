import copy

import pytest

from src.candidate_generator import generate_candidates, validate_candidate


def make_episode():
    events = [
        {"event_id": "EP01_SC01_EV001", "episode_number": 1, "start_time": "00:00.00", "end_time": "00:02.00", "start_seconds": 0.0, "end_seconds": 2.0, "event_type": "dialogue", "speaker": "A", "label": "A", "content": '"Stop!"'},
        {"event_id": "EP01_SC01_EV002", "episode_number": 1, "start_time": "00:02.00", "end_time": "00:18.00", "start_seconds": 2.0, "end_seconds": 18.0, "event_type": "direction", "speaker": None, "label": "CAMERA", "content": "Close-up and reaction."},
        {"event_id": "EP01_SC01_EV003", "episode_number": 1, "start_time": "00:18.00", "end_time": "00:22.00", "start_seconds": 18.0, "end_seconds": 22.0, "event_type": "dialogue", "speaker": "B", "label": "B", "content": '"Why did you do it?"'},
        {"event_id": "EP01_SC01_EV004", "episode_number": 1, "start_time": "00:22.00", "end_time": "00:25.00", "start_seconds": 22.0, "end_seconds": 25.0, "event_type": "dialogue", "speaker": "A", "label": "A", "content": '"You know why."'},
        {"event_id": "EP01_SC01_EV005", "episode_number": 1, "start_time": "00:25.00", "end_time": "00:30.00", "start_seconds": 25.0, "end_seconds": 30.0, "event_type": "direction", "speaker": None, "label": "CAMERA", "content": "Cut to black."},
    ]
    return {"episode_id": "EP01", "duration_seconds": 30.0, "events": events}


def make_analysis():
    return {
        "show_id": "show_alpha",
        "episode_id": "EP01",
        "dramatic_moments": [
            {
                "moment_type": "hook",
                "hook_potential": 5,
                "cliffhanger_potential": 0,
                "source_event_ids": ["EP01_SC01_EV001"],
                "description": "A shocking command.",
                "editorial_review_flags": [],
            },
            {
                "moment_type": "conflict",
                "hook_potential": 1,
                "cliffhanger_potential": 4,
                "source_event_ids": ["EP01_SC01_EV003"],
                "description": "An unanswered question.",
            },
        ],
        "episode_ending_context": {
            "events": [{"event_id": "EP01_SC01_EV005"}]
        },
    }


def test_generates_valid_20_to_60_second_candidates():
    result = generate_candidates(make_episode(), make_analysis())
    assert result["candidate_count"] >= 1
    for candidate in result["candidates"]:
        assert 20 <= candidate["duration_seconds"] <= 60
        assert candidate["validation_passed"] is True


def test_keeps_all_overlapping_timeline_events():
    result = generate_candidates(make_episode(), make_analysis())
    candidate = result["candidates"][0]
    ids = candidate["source_event_ids"]
    assert "EP01_SC01_EV002" in ids


def test_question_cut_is_flagged_for_review():
    result = generate_candidates(make_episode(), make_analysis())
    analysis = make_analysis()
    # Isolate the question-cut heuristic from the model cliffhanger anchor.
    analysis["dramatic_moments"][1]["cliffhanger_potential"] = 0
    result = generate_candidates(make_episode(), analysis)
    question_candidates = [
        c for c in result["candidates"]
        if c["ending"]["ending_basis"] == "question_cut_requires_review"
    ]
    assert question_candidates
    assert question_candidates[0]["ending"]["confirmed_cliffhanger"] is False


def test_episode_end_is_not_claimed_as_confirmed_cliffhanger():
    result = generate_candidates(make_episode(), make_analysis())
    ending_candidates = [
        c for c in result["candidates"]
        if c["ending"]["ending_basis"] == "episode_ending_candidate"
    ]
    assert ending_candidates
    assert ending_candidates[0]["ending"]["confirmed_cliffhanger"] is False


def test_rejects_wrong_episode_analysis():
    analysis = make_analysis()
    analysis["episode_id"] = "EP99"
    with pytest.raises(ValueError, match="does not match"):
        generate_candidates(make_episode(), analysis)


def test_no_candidate_when_duration_cannot_fit():
    episode = make_episode()
    result = generate_candidates(
        episode, make_analysis(), min_duration=40, max_duration=60
    )
    assert result["candidate_count"] == 0


def test_visual_only_hook_is_flagged():
    analysis = make_analysis()
    analysis["dramatic_moments"][0]["source_event_ids"] = ["EP01_SC01_EV002"]
    analysis["dramatic_moments"][0]["editorial_review_flags"] = [
        {"code": "visual_only_hook_review", "severity": "advisory"}
    ]
    result = generate_candidates(make_episode(), analysis)
    # Visual-only anchor is not silently presented as a proven scroll-stopping hook.
    if result["candidate_count"]:
        assert any(
            flag["code"] == "visual_hook_requires_video_review"
            for candidate in result["candidates"]
            for flag in candidate["editorial_review_flags"]
        )


def test_invalid_source_event_ids_are_ignored_as_anchors():
    analysis = make_analysis()
    analysis["dramatic_moments"][0]["source_event_ids"] = ["MISSING"]
    result = generate_candidates(make_episode(), analysis)
    assert result["hook_anchor_count"] == 0


def test_validator_rejects_out_of_range_duration():
    episode = make_episode()
    analysis = make_analysis()
    result = generate_candidates(episode, analysis)
    candidate = copy.deepcopy(result["candidates"][0])
    candidate["segments"][0]["end_seconds"] = 61.0
    errors = validate_candidate(candidate, episode)
    assert any("Duration" in error for error in errors)


def test_new_show_works_without_episode_specific_names():
    episode = make_episode()
    episode["episode_id"] = "EP77"
    for index, event in enumerate(episode["events"], start=1):
        event["event_id"] = f"EP77_SC01_EV{index:03d}"
    analysis = make_analysis()
    analysis["episode_id"] = "EP77"
    analysis["show_id"] = "entirely_new_show"
    analysis["dramatic_moments"][0]["source_event_ids"] = ["EP77_SC01_EV001"]
    analysis["dramatic_moments"][1]["source_event_ids"] = ["EP77_SC01_EV003"]
    analysis["episode_ending_context"]["events"] = [{"event_id": "EP77_SC01_EV005"}]
    result = generate_candidates(episode, analysis)
    assert result["episode_id"] == "EP77"
    assert result["show_id"] == "entirely_new_show"
    assert result["candidate_count"] >= 1


def test_model_calls_are_zero():
    result = generate_candidates(make_episode(), make_analysis())
    assert result["model_calls_made"] == 0
