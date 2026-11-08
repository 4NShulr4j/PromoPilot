import unittest
from copy import deepcopy
from src.shortlist_quality_policy import apply_shortlist_policy


class ShortlistPolicyTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "ranking_policy": {"top_n": 5},
            "summary": {"shortlist_count": 2},
            "shortlist": [
                {"candidate_id": "A", "ranking_score": 75, "adjusted_score": 75,
                 "quality_flags": ["QUESTION_FOLLOWED_QUICKLY"], "shortlist_selected": True},
                {"candidate_id": "B", "ranking_score": 20, "adjusted_score": 20,
                 "quality_flags": ["HOOK_SCORE_REVIEW"], "shortlist_selected": True},
            ],
            "full_ranking": [
                {"candidate_id": "A", "ranking_score": 75, "quality_flags": ["QUESTION_FOLLOWED_QUICKLY"], "shortlist_selected": True},
                {"candidate_id": "B", "ranking_score": 20, "quality_flags": ["HOOK_SCORE_REVIEW"], "shortlist_selected": True},
                {"candidate_id": "C", "ranking_score": 10, "quality_flags": ["CUT_MID_THOUGHT"], "shortlist_selected": False},
            ],
        }

    def test_low_score_removed_from_shortlist_but_kept_in_full_ranking(self):
        result = apply_shortlist_policy(self.data, 40)
        self.assertEqual([x["candidate_id"] for x in result["shortlist"]], ["A"])
        b = next(x for x in result["full_ranking"] if x["candidate_id"] == "B")
        self.assertFalse(b["shortlist_selected"])
        self.assertIn("Below minimum shortlist score", b["shortlist_exclusion_reason"])
        self.assertEqual(len(result["editorial_review_excluded"]), 1)

    def test_editorial_flag_does_not_auto_exclude_candidate(self):
        result = apply_shortlist_policy(self.data, 40)
        a = result["shortlist"][0]
        self.assertTrue(a["shortlist_selected"])
        self.assertTrue(a["editorial_review_required"])
        self.assertEqual(a["selection_status"], "provisional_editorial_review")

    def test_scores_and_input_are_not_mutated(self):
        original = deepcopy(self.data)
        result = apply_shortlist_policy(self.data, 40)
        self.assertEqual(self.data, original)
        self.assertEqual(result["shortlist"][0]["adjusted_score"], 75)
        self.assertEqual(result["shortlist"][0]["ranking_score"], 75)

    def test_min_score_validation(self):
        with self.assertRaises(ValueError):
            apply_shortlist_policy(self.data, 101)

    def test_all_shortlist_items_can_be_filtered(self):
        result = apply_shortlist_policy(self.data, 80)
        self.assertEqual(result["shortlist"], [])
        self.assertEqual(result["summary"]["shortlist_count"], 0)
        self.assertEqual(result["summary"]["shortlist_slots_unfilled"], 5)


if __name__ == "__main__":
    unittest.main()
