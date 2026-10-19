"""材料点数の減衰と推定可能性の境界を検証する。"""

import unittest

from fundamentals.common import timestamp
from fundamentals.model import fit_predict, material_features
from fundamentals.reactions import reaction_statistics


class ModelTests(unittest.TestCase):
    """結果に合わせない縮小推定と、観測の分母を検証する。"""

    def test_directional_relation_is_learned_with_shrinkage(self) -> None:
        """十分な符号付き材料例では、過度に外挿せず方向関係を推定する。

        引数: self (ModelTests) は試験対象。
        返り値: None。
        """
        rows = [{"features": {"景況／通常": sign}, "actual_pct": 2 * sign} for sign in (-1, 1) for _ in range(12)]
        upward = fit_predict(rows, {"景況／通常": 1})
        downward = fit_predict(rows, {"景況／通常": -1})
        self.assertGreater(upward["prediction_pct"], 1)
        self.assertLess(upward["prediction_pct"], 2)
        self.assertAlmostEqual(upward["prediction_pct"], -downward["prediction_pct"])

    def test_unseen_or_rare_category_does_not_receive_a_zero_prediction(self) -> None:
        """未学習や少数の材料分類を、ゼロ効果の確立として表示しない。

        引数: self (ModelTests) は試験対象。
        返り値: None。
        """
        rows = [{"features": {"既知": 1}, "actual_pct": 0.2} for _ in range(24)]
        result = fit_predict(rows, {"新分類": 1})
        self.assertIsNone(result["prediction_pct"])
        self.assertEqual(result["unseen_categories"], ["新分類"])
        rows[0]["features"]["希少"] = 1
        self.assertIsNone(fit_predict(rows, {"希少": 1})["prediction_pct"])

    def test_long_material_features_decay_without_daily_return_extrapolation(self) -> None:
        """半減期を１日とした材料の残存特徴が１日ごとに半減する。

        引数: self (ModelTests) は試験対象。
        返り値: None。
        """
        as_of = timestamp("2026-10-15T07:45:00+09:00")
        article = {"id": 1, "revision_id": "初版", "available_at": as_of.isoformat()}
        event = {"event_id": "材料", "story_key": "事象", "category": "景況", "regime": "通常",
                 "known_at": as_of.isoformat(), "annotated_at": as_of.isoformat(), "confidence": 1,
                 "novelty": 1, "priced_in": 0, "half_life_days": 1,
                 "impacts": {"NK225": 1, "TOPIX": 1}, "evidence": [{"id": 1, "revision_id": "初版"}], "reason": "合成仮説"}
        one, _, _ = material_features([event], [article], as_of, timestamp("2026-10-16T07:45:00+09:00"), "NK225", "回想")
        two, _, _ = material_features([event], [article], as_of, timestamp("2026-10-17T07:45:00+09:00"), "NK225", "回想")
        self.assertAlmostEqual(one["景況／通常"], 0.5)
        self.assertAlmostEqual(two["景況／通常"], 0.25)

    def test_material_count_does_not_inflate_reaction_observations(self) -> None:
        """同じ予測の同分類に二つの材料があっても騰落標本を１件と数える。

        引数: self (ModelTests) は試験対象。
        返り値: None。
        """
        row = {"mode": "回想", "symbol": "NK225", "kind": "Day", "horizon": "today", "category": "景況", "regime": "通常",
               "forecast_hash": "同一予測", "remaining_score": 0.3, "actual_pct": 1.2, "as_of": "2026-10-15T07:45:00+09:00"}
        result = reaction_statistics([row, row])
        self.assertEqual(next(iter(result.values()))["forecast_observations"], 1)
