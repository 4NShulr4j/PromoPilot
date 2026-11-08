import unittest
from src.evaluation_quality_gate import run_gate


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.episode = {"episode_id": "ep01", "events": [
            {"event_id": "E1", "start_time": 10.0, "end_time": 11.0, "speaker": "A", "content": '"Why did you do it?"'},
            {"event_id": "E2", "start_time": 11.3, "end_time": 12.0, "speaker": "B", "content": '"Because I had to."'},
            {"event_id": "E3", "start_time": 20.0, "end_time": 21.0, "speaker": "A", "content": '"I cannot believe this..."'},
            {"event_id": "E4", "start_time": 21.2, "end_time": 22.0, "speaker": "A", "content": '"It changes everything!"'},
        ]}
        self.candidates = {"candidates": [
            {"candidate_id": "C1", "start_time": 10.0, "end_time": 11.0, "hook": {"event_id": "E1"}, "ending": {"event_id": "E1"}, "start_event_id": "E1", "end_event_id": "E1"},
            {"candidate_id": "C2", "start_time": 20.0, "end_time": 21.0, "hook": {"event_id": "E3"}, "ending": {"event_id": "E3"}, "start_event_id": "E3", "end_event_id": "E3"},
        ]}
        self.evaluations = {"evaluations": [
            {"candidate_id": "C1", "total_score": 80, "scores": {"hook_strength": 4}},
            {"candidate_id": "C2", "total_score": 70, "scores": {"hook_strength": 0}},
        ]}

    def test_flags_quick_question_followup(self):
        report = run_gate(self.episode, self.candidates, self.evaluations)
        c1 = next(x for x in report["results"] if x["candidate_id"] == "C1")
        self.assertTrue(any(f["code"] == "QUESTION_FOLLOWED_QUICKLY" for f in c1["flags"]))

    def test_quoted_ellipsis_detects_same_speaker_continuation(self):
        report = run_gate(self.episode, self.candidates, self.evaluations)
        c2 = next(x for x in report["results"] if x["candidate_id"] == "C2")
        self.assertEqual(c2["original_model_score"], 70)
        self.assertEqual(c2["adjusted_score"], 60)
        self.assertEqual(c2["adjustment_points"], -10)
        flag = next(f for f in c2["flags"] if f["code"] == "CUT_MID_THOUGHT")
        self.assertEqual(flag["evidence"]["event_id"], "E4")

    def test_original_score_is_not_overwritten(self):
        report = run_gate(self.episode, self.candidates, self.evaluations)
        c2 = next(x for x in report["results"] if x["candidate_id"] == "C2")
        self.assertEqual(c2["original_model_score"], 70)
        self.assertEqual(c2["adjusted_score"], 60)

    def test_no_llm_calls(self):
        report = run_gate(self.episode, self.candidates, self.evaluations)
        self.assertEqual(report["mode"], "deterministic_source_checks_no_llm_calls")


if __name__ == "__main__":
    unittest.main()
