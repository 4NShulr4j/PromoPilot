import unittest
from src.ui_helpers import format_timecode, index_candidate_details, select_best_promo


class TestPromoPresentationHelpers(unittest.TestCase):
    def test_selects_highest_scoring_recommended_candidate(self):
        ranking = {
            "shortlist": [
                {"candidate_id": "A", "ranking_score": 70, "overall_rank": 2},
                {"candidate_id": "B", "ranking_score": 85, "overall_rank": 1},
            ],
            "full_ranking": [],
        }
        self.assertEqual(select_best_promo(ranking)["candidate_id"], "B")

    def test_does_not_select_blocked_or_ineligible_shortlist_item(self):
        ranking = {
            "shortlist": [
                {"candidate_id": "A", "ranking_score": 95, "blocking_flags": ["CUT_MID_THOUGHT"]},
                {"candidate_id": "B", "ranking_score": 60, "shortlist_eligible": False},
            ],
            "full_ranking": [
                {"candidate_id": "C", "ranking_score": 55, "blocking_flags": []},
            ],
        }
        self.assertEqual(select_best_promo(ranking)["candidate_id"], "C")

    def test_falls_back_to_best_nonblocked_candidate_when_shortlist_empty(self):
        ranking = {
            "shortlist": [],
            "full_ranking": [
                {"candidate_id": "A", "ranking_score": 20, "blocking_flags": []},
                {"candidate_id": "B", "ranking_score": 50, "blocking_flags": ["CUT_MID_THOUGHT"]},
            ],
        }
        self.assertEqual(select_best_promo(ranking)["candidate_id"], "A")

    def test_no_candidate_when_everything_is_blocked(self):
        ranking = {"shortlist": [], "full_ranking": [{"candidate_id": "A", "ranking_score": 99, "blocking_flags": ["CUT_MID_THOUGHT"]}]}
        self.assertIsNone(select_best_promo(ranking))

    def test_candidate_index_maps_details(self):
        payload = {"candidates": [{"candidate_id": "A", "hook": {"text": "Opening"}}]}
        self.assertEqual(index_candidate_details(payload)["A"]["hook"]["text"], "Opening")

    def test_timecode_format(self):
        self.assertEqual(format_timecode(74.3), "01:14.3")
        self.assertEqual(format_timecode(3601.25), "1:00:01.2")


if __name__ == "__main__":
    unittest.main()
