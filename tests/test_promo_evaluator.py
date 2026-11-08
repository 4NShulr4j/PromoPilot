import json
from pathlib import Path

import pytest

from src.promo_evaluator import (
    WEIGHTS,
    build_candidate_context,
    calculate_weighted_score,
    evaluate_candidates,
    evaluate_one_candidate,
    normalize_evaluation_response,
    validate_candidate_source,
)


def make_episode():
    events = [
        {"event_id": "EP01_EV001", "start_time": "00:00.00", "end_time": "00:03.00", "start_seconds": 0.0, "end_seconds": 3.0, "event_type": "dialogue", "speaker": "A", "label": "A", "content": '"Stop!"'},
        {"event_id": "EP01_EV002", "start_time": "00:03.00", "end_time": "00:20.00", "start_seconds": 3.0, "end_seconds": 20.0, "event_type": "direction", "speaker": None, "label": "CAMERA", "content": "Camera follows the characters."},
        {"event_id": "EP01_EV003", "start_time": "00:20.00", "end_time": "00:24.00", "start_seconds": 20.0, "end_seconds": 24.0, "event_type": "dialogue", "speaker": "B", "label": "B", "content": '"Why did you do it?"'},
        {"event_id": "EP01_EV004", "start_time": "00:24.00", "end_time": "00:27.00", "start_seconds": 24.0, "end_seconds": 27.0, "event_type": "dialogue", "speaker": "A", "label": "A", "content": '"You know why."'},
        {"event_id": "EP01_EV005", "start_time": "00:27.00", "end_time": "00:30.00", "start_seconds": 27.0, "end_seconds": 30.0, "event_type": "direction", "speaker": None, "label": "CAMERA", "content": "Cut to black."},
    ]
    return {"episode_id": "EP01", "duration_seconds": 30.0, "logline": "A tense confrontation.", "events": events}


def make_candidate():
    return {
        "candidate_id": "EP01_PROMO_001",
        "episode_id": "EP01",
        "promo_type": "continuous_single_episode",
        "segments": [{
            "episode_id": "EP01", "start_seconds": 0.0, "end_seconds": 24.0,
            "start_time": "00:00.00", "end_time": "00:24.00",
            "start_event_id": "EP01_EV001", "end_event_id": "EP01_EV003",
            "ending_basis": "model_cliffhanger_anchor",
        }],
        "duration_seconds": 24.0,
        "hook": {"event_id": "EP01_EV001", "text": '"Stop!"'},
        "ending": {"event_id": "EP01_EV003", "text": '"Why did you do it?"'},
        "timeline_events": [
            {"event_id": e["event_id"], "start_time": e["start_time"], "end_time": e["end_time"],
             "start_seconds": e["start_seconds"], "end_seconds": e["end_seconds"],
             "event_type": e["event_type"], "speaker": e["speaker"], "label": e["label"], "content": e["content"]}
            for e in make_episode()["events"][:3]
        ],
        "source_event_ids": ["EP01_EV001", "EP01_EV002", "EP01_EV003"],
        "editorial_review_flags": [],
        "validation_passed": True,
    }


def make_model_response():
    return {
        "scores": {
            "hook_strength": 4,
            "cliffhanger_strength": 4,
            "standalone_clarity": 3,
            "emotional_engagement": 4,
            "pacing": 4,
            "clip_integrity": 5,
        },
        "hook_assessment": "The abrupt command creates immediate curiosity.",
        "ending_assessment": "The question leaves an outcome unresolved.",
        "cliffhanger_type": "open_question",
        "standalone_clarity_explanation": "The basic confrontation is understandable.",
        "why_it_works": "It opens with conflict and cuts before an answer.",
        "risks": ["The relationship between speakers is not explained."],
        "strengths": ["Immediate conflict", "Question cut"],
        "evidence_event_ids": ["EP01_EV001", "EP01_EV003"],
        "confidence": 0.85,
    }


def fake_request(**kwargs):
    return json.dumps(make_model_response()), {"provider": "mock", "model": "offline-test"}


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_weighted_score_is_calculated_in_python():
    result = calculate_weighted_score(make_model_response()["scores"])
    assert result == 77.0


def test_candidate_source_validation_accepts_valid_candidate():
    assert validate_candidate_source(make_candidate(), make_episode()) == []


def test_candidate_source_validation_rejects_bad_duration():
    candidate = make_candidate()
    candidate["segments"][0]["end_seconds"] = 70.0
    assert any("Duration" in error for error in validate_candidate_source(candidate, make_episode()))


def test_candidate_source_validation_rejects_unknown_event():
    candidate = make_candidate()
    candidate["segments"][0]["start_event_id"] = "MISSING"
    assert any("Unknown start_event_id" in error for error in validate_candidate_source(candidate, make_episode()))


def test_candidate_context_reconstructs_timeline_from_source():
    context = build_candidate_context(make_candidate(), make_episode(), {"logline": "A tense confrontation."})
    ids = [event["event_id"] for event in context["clip_events"]]
    assert ids == ["EP01_EV001", "EP01_EV002", "EP01_EV003"]
    assert context["post_cut_context_not_in_promo"][0]["event_id"] == "EP01_EV004"


def test_normalizer_rejects_out_of_range_score():
    model = make_model_response()
    model["scores"]["hook_strength"] = 7
    with pytest.raises(ValueError, match="between 0 and 5"):
        normalize_evaluation_response(model, allowed_event_ids={"EP01_EV001", "EP01_EV003"})


def test_normalizer_discards_unverified_evidence_ids():
    model = make_model_response()
    model["evidence_event_ids"].append("FAKE_EVENT")
    normalized, warnings = normalize_evaluation_response(
        model, allowed_event_ids={"EP01_EV001", "EP01_EV003"}
    )
    assert "FAKE_EVENT" not in normalized["evidence_event_ids"]
    assert any("Ignored invalid" in warning for warning in warnings)


def test_evaluate_one_candidate_with_mocked_llm():
    result, raw, metadata = evaluate_one_candidate(
        make_candidate(), make_episode(), request_fn=fake_request
    )
    assert result["evaluation_status"] == "evaluated"
    assert result["weighted_score"] == 77.0
    assert result["recommendation"] == "promising_candidate_for_editor_review"
    assert raw and metadata["provider"] == "mock"


def test_structurally_invalid_candidate_skips_llm():
    candidate = make_candidate()
    candidate["segments"][0]["end_seconds"] = 70.0
    def should_not_call(**kwargs):
        raise AssertionError("LLM should not be called for invalid candidate")
    result, raw, metadata = evaluate_one_candidate(
        candidate, make_episode(), request_fn=should_not_call
    )
    assert result["evaluation_status"] == "rejected_structural"
    assert raw is None and metadata is None


def test_evaluate_candidates_writes_checkpoint_and_resumes(tmp_path):
    pack = {"episode_id": "EP01", "show_id": "show_test", "candidates": [make_candidate()]}
    result = evaluate_candidates(
        pack, make_episode(), output_dir=tmp_path, delay_between_calls=0,
        request_fn=fake_request,
    )
    assert result["evaluated_count"] == 1
    assert result["model_calls_this_run"] == 1
    second = evaluate_candidates(
        pack, make_episode(), output_dir=tmp_path, delay_between_calls=0,
        request_fn=fake_request,
    )
    assert second["evaluated_count"] == 1
    assert second["model_calls_this_run"] == 0


def test_no_model_calls_for_structurally_invalid_candidate_pack(tmp_path):
    candidate = make_candidate()
    candidate["segments"][0]["end_seconds"] = 70.0
    pack = {"episode_id": "EP01", "candidates": [candidate]}
    result = evaluate_candidates(
        pack, make_episode(), output_dir=tmp_path, delay_between_calls=0,
        request_fn=fake_request,
    )
    assert result["structurally_rejected_count"] == 1
    assert result["model_calls_this_run"] == 0
