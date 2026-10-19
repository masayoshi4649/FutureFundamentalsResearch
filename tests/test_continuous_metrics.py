"""未確定対象の除外可能性を含む正答率の意味を検証する。"""

import unittest

from research_support.continuous_metrics import summarize_scores


def score(target: str, status: str, lower: bool = False, upper: bool = False, unanswered: bool = False) -> dict:
    """独立した採点条件を集計検証へ渡す。

    引数: target/status (str) は対象と採点、lower/upper/unanswered (bool) は確定・可能正解・無回答。
    返り値: dict は検証用の採点。
    """
    return {"target_status": target, "status": status, "correct_lower": lower, "correct_upper": upper,
            "denominator": target != "対象外", "unanswered": unanswered}


class ContinuousMetricsTests(unittest.TestCase):
    """実際の対象分母が未確定な場合に成績を過大・過小断定しない。"""

    def test_target_unknown_can_raise_conditional_upper(self):
        """対象外の可能性が固定分母と条件付き上限を異ならせることを確認する。

        引数: self (ContinuousMetricsTests) は検証インスタンス。
        返り値: None。
        """
        result = summarize_scores([score("対象", "正解", True, True), score("対象", "不正解"),
                                   score("未確定", "未確定", upper=True), score("未確定", "不正解")])
        self.assertEqual(result["fixed_denominator_lower_pct"], 25)
        self.assertEqual(result["fixed_denominator_possible_pct"], 50)
        self.assertAlmostEqual(result["conditional_target_optimistic_upper_pct"], 200 / 3)

    def test_unanswered_certain_target_remains_in_denominator(self):
        """対象内無回答を不正解として分母へ残す。

        引数: self (ContinuousMetricsTests) は検証インスタンス。
        返り値: None。
        """
        result = summarize_scores([score("対象", "正解", True, True), score("対象", "不正解", unanswered=True),
                                   score("対象外", "対象外", unanswered=True)])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["unanswered"], 2)
        self.assertEqual(result["unanswered_in_target_denominator"], 1)
        self.assertEqual(result["conditional_target_optimistic_upper_pct"], 50)

    def test_no_possible_target_is_not_a_measured_rate(self):
        """確定対象も正解可能対象もない場合、対象内の実測率を作らない。

        引数: self (ContinuousMetricsTests) は検証インスタンス。
        返り値: None。
        """
        self.assertIsNone(summarize_scores([])["conditional_target_optimistic_upper_pct"])
        self.assertIsNone(summarize_scores([score("未確定", "不正解")])["conditional_target_optimistic_upper_pct"])

    def test_invalid_certain_correct_is_rejected(self):
        """対象未確定の確定正解への繰上げを拒否する。

        引数: self (ContinuousMetricsTests) は検証インスタンス。
        返り値: None。
        """
        with self.assertRaises(ValueError):
            summarize_scores([score("未確定", "正解", lower=True, upper=True)])


if __name__ == "__main__":
    unittest.main()
