import unittest
from src.ranking_engine import run_ranking, parse_time, extract_span


def gate(results):
    return {"episode_id": "EP01", "results": results}


def result(cid, score, flags=None):
    return {
        "candidate_id": cid,
        "original_model_score": score + (10 if flags and any(x.get("code") == "CUT_MID_THOUGHT" for x in flags) else 0),
        "adjusted_score": score,
        "quality_status": "needs_review" if flags else "pass",
        "flags": flags or [],
    }


class RankingEngineTests(unittest.TestCase):
    def test_time_parser(self):
        self.assertAlmostEqual(parse_time("01:47.50"), 107.5)
        self.assertAlmostEqual(parse_time("00:00:35.40"), 35.4)

    def test_rank_uses_adjusted_score(self):
        report = run_ranking(gate([result("EP01_PROMO_A", 65), result("EP01_PROMO_B", 75)]))
        self.assertEqual(report["full_ranking"][0]["candidate_id"], "EP01_PROMO_B")
        self.assertEqual(report["full_ranking"][0]["ranking_score"], 75)

    def test_mid_thought_blocked_but_score_preserved(self):
        r = result("EP01_PROMO_A", 55, [{"code": "CUT_MID_THOUGHT", "severity": "high"}])
        report = run_ranking(gate([r]))
        item = report["full_ranking"][0]
        self.assertEqual(item["ranking_score"], 55)
        self.assertFalse(item["shortlist_eligible"])
        self.assertEqual(report["shortlist"], [])

    def test_review_only_flag_does_not_block_or_reduce_score(self):
        r = result("EP01_PROMO_A", 75, [{"code": "QUESTION_FOLLOWED_QUICKLY", "severity": "review"}])
        report = run_ranking(gate([r]))
        self.assertEqual(report["full_ranking"][0]["ranking_score"], 75)
        self.assertEqual(report["shortlist"][0]["candidate_id"], "EP01_PROMO_A")

    def test_overlap_selects_highest_scoring_eligible_representative(self):
        candidates = {"candidates": [
            {"candidate_id": "EP01_PROMO_A", "start_seconds": 10, "end_seconds": 40},
            {"candidate_id": "EP01_PROMO_B", "start_seconds": 10, "end_seconds": 60},
            {"candidate_id": "EP01_PROMO_C", "start_seconds": 100, "end_seconds": 130},
        ]}
        report = run_ranking(gate([result("EP01_PROMO_A", 70), result("EP01_PROMO_B", 80), result("EP01_PROMO_C", 65)]), candidates)
        selected = [x["candidate_id"] for x in report["shortlist"]]
        self.assertEqual(selected, ["EP01_PROMO_B", "EP01_PROMO_C"])
        self.assertEqual(report["summary"]["overlap_cluster_count"], 2)

    def test_cut_mid_thought_alternative_does_not_displace_eligible_variant(self):
        candidates = {"candidates": [
            {"candidate_id": "EP01_PROMO_A", "start_seconds": 1, "end_seconds": 23.5},
            {"candidate_id": "EP01_PROMO_B", "start_seconds": 1, "end_seconds": 56.9},
        ]}
        results = [result("EP01_PROMO_A", 59), result("EP01_PROMO_B", 45, [{"code": "CUT_MID_THOUGHT", "severity": "high"}])]
        report = run_ranking(gate(results), candidates)
        self.assertEqual([x["candidate_id"] for x in report["shortlist"]], ["EP01_PROMO_A"])

    def test_missing_span_is_reported_without_failure(self):
        report = run_ranking(gate([result("EP01_PROMO_A", 60)]), {"candidates": [{"candidate_id": "EP01_PROMO_A"}]})
        self.assertEqual(report["candidate_metadata_missing"], ["EP01_PROMO_A"])
        self.assertEqual(len(report["full_ranking"]), 1)


if __name__ == "__main__":
    unittest.main()
