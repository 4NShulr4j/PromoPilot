from src.story_analyzer import (
    _make_chunk_episode,
    _merge_chunk_results,
    chunk_events,
    select_story_events,
)


def _event(i, label="DIALOGUE", speaker="CHARACTER"):
    return {
        "event_id": f"EP01_SC01_EV{i:03d}",
        "episode_number": 1,
        "scene_number": 1,
        "scene_id": "EP01_SC01",
        "scene_heading": "INT. ROOM - DAY",
        "label": label,
        "speaker": speaker if label != "CAMERA" else None,
        "event_type": "direction" if label == "CAMERA" else "dialogue",
        "start_time": f"00:{i:02d}.00",
        "end_time": f"00:{i:02d}.50",
        "start_seconds": float(i),
        "end_seconds": float(i) + 0.5,
        "content": f"Event {i}",
    }


def test_select_story_events_excludes_pure_camera_but_keeps_source_data():
    all_events = [_event(1), _event(2, "CAMERA", None), _event(3, "ACTION", None)]
    episode = {"events": all_events}
    selected = select_story_events(episode)
    assert [e["event_id"] for e in selected] == [
        "EP01_SC01_EV001", "EP01_SC01_EV003"
    ]
    assert len(episode["events"]) == 3


def test_chunk_events_has_expected_overlap():
    events = [_event(i) for i in range(1, 24)]
    chunks = chunk_events(events, chunk_size=10, overlap=2)
    assert [len(c) for c in chunks] == [10, 10, 7]
    assert chunks[0][-2]["event_id"] == chunks[1][0]["event_id"]
    assert chunks[1][-2]["event_id"] == chunks[2][0]["event_id"]


def test_make_chunk_episode_only_includes_chunk_scenes():
    episode = {
        "episode_id": "EP01",
        "episode_index": 1,
        "scenes": [
            {"scene_number": 1},
            {"scene_number": 2},
        ],
        "events": [],
    }
    chunk = _make_chunk_episode(episode, [_event(1)])
    assert len(chunk["events"]) == 1
    assert chunk["scenes"] == [{"scene_number": 1}]
    assert len(episode["scenes"]) == 2


def test_merge_chunk_results_deduplicates_overlap_moments():
    episode = {
        "episode_id": "EP01",
        "episode_index": 1,
        "events": [_event(1), _event(2)],
    }
    moment = {
        "moment_type": "hook",
        "event_ids": ["EP01_SC01_EV001"],
        "description": "An immediate warning.",
        "why_significant": "It creates curiosity.",
        "emotional_intensity": 2,
        "hook_potential": 5,
        "cliffhanger_potential": 1,
        "context_dependent": False,
        "context_needed": "",
        "source_evidence": [],
    }
    chunk_result = {
        "episode_summary": "A character gives a warning.",
        "characters": [],
        "dramatic_moments": [moment],
    }
    report = {
        "model_moment_count": 1,
        "rejected_moments": [],
        "rejected_moment_count": 0,
        "warnings": [],
    }
    merged, merged_report = _merge_chunk_results(
        episode,
        [chunk_result, chunk_result],
        [report, report],
        [{"model": "test"}, {"model": "test"}],
        story_input_event_count=2,
    )
    assert merged["accepted_moment_count"] == 1
    assert merged_report["model_moment_count"] == 2
    assert merged_report["chunk_count"] == 2


def test_analyze_episode_chunks_and_resumes_without_api_calls(tmp_path, monkeypatch):
    import json
    from src import story_analyzer as analyzer

    events = [_event(i) for i in range(1, 16)]
    episode = {
        "episode_id": "EP01",
        "episode_index": 1,
        "episode_number": "Episode 1",
        "logline": "A short test story.",
        "continuity_notes": "",
        "duration_seconds": 20.0,
        "scenes": [{"scene_number": 1, "heading": "INT. ROOM - DAY"}],
        "events": events,
        "event_count": len(events),
    }
    fake_response = json.dumps({
        "episode_id": "EP01",
        "episode_summary": "This chunk contains dialogue from a test scene.",
        "characters": [],
        "dramatic_moments": [],
    })
    calls = []

    def fake_request(**kwargs):
        calls.append(kwargs)
        return fake_response, {"provider": "groq", "model": "mock", "usage": {}}

    monkeypatch.setattr(analyzer, "request_json_object", fake_request)
    output_dir = tmp_path / "story_analysis"
    raw_dir = tmp_path / "raw_llm"

    first = analyzer.analyze_episode(
        episode,
        output_dir=output_dir,
        raw_dir=raw_dir,
        chunk_size=10,
        chunk_overlap=2,
        delay_between_chunks=0,
        resume=True,
    )
    assert first["model_metadata"]["chunk_count"] == 2
    assert len(calls) == 2
    assert (raw_dir / "ep01_story_chunk_01_raw.txt").exists()
    assert (output_dir / "ep01_story_analysis.json").exists()

    calls.clear()
    second = analyzer.analyze_episode(
        episode,
        output_dir=output_dir,
        raw_dir=raw_dir,
        chunk_size=10,
        chunk_overlap=2,
        delay_between_chunks=0,
        resume=True,
    )
    assert second["model_metadata"]["chunk_count"] == 2
    assert calls == []
