import unittest

from src.ui_helpers import candidate_table_rows, flag_counts, safe_episode_id, validate_ranking


class UIHelpersTests(unittest.TestCase):
    def test_valid_ranking(self):
        data = {"episode_id": "EP01", "shortlist": [{"candidate_id": "C1"}], "full_ranking": [{"candidate_id": "C1"}]}
        self.assertEqual(validate_ranking(data), [])

    def test_invalid_ranking(self):
        self.assertGreaterEqual(len(validate_ranking({"shortlist": []})), 2)

    def test_safe_episode_id(self):
        self.assertEqual(safe_episode_id("ep 01/part"), "EP_01_PART")
        with self.assertRaises(ValueError):
            safe_episode_id(" /// ")

    def test_flag_counts(self):
        counts = flag_counts([{"quality_flags": ["A", "B"]}, {"quality_flags": ["A"]}])
        self.assertEqual(counts, {"A": 2, "B": 1})

    def test_table_projection(self):
        rows = candidate_table_rows([{"candidate_id": "C1", "ranking_score": 80, "quality_flags": ["FLAG"]}])
        self.assertEqual(rows[0]["Candidate"], "C1")
        self.assertEqual(rows[0]["Score"], 80)
        self.assertEqual(rows[0]["Flags"], "FLAG")


if __name__ == "__main__":
    unittest.main()
