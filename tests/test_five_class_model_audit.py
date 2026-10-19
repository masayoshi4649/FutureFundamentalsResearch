"""モデル実装と別に損失の一階条件を計算する、実価格を使わない独立監査。"""

import math
import random
import unittest
from datetime import timedelta
from statistics import mean, median, pstdev

from fundamentals.common import timestamp
from research_support.five_class import CLASSES
from research_support.five_class_model import CLASSIFICATION_POLICY, FEATURE_KEYS, predict_five


class FiveClassModelAuditTests(unittest.TestCase):
    """13特徴と13欠測表示を持つ多変量の解を損失から検証する。"""

    def _sample(self) -> tuple[dict, list[dict]]:
        """複数分類・欠測・ゼロ重みを含む固定乱数の合成教師を作る。

        引数: self (FiveClassModelAuditTests) はテスト。
        返り値: tuple[dict, list[dict]] は今回特徴と32件の過去教師。
        """
        generator = random.Random(20261016)
        cutoff = "2026-10-15T07:45:00+09:00"
        current = {"symbol": "NK225", "kind": "Day", "as_of": cutoff,
                   "features": {key: (0.0 if key == "is_night" else 0.15) for key in FEATURE_KEYS}}
        criteria = {"target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
                    "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5",
                    "reversal_requires_reference_cross": False, "range_threshold_pct": "1",
                    "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
                    "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10}
        rows = []
        for index in range(32):
            end = timestamp("2026-09-01T15:45:00+09:00") + timedelta(days=index)
            features = {key: (None if (index + offset) % 5 == 0 else generator.uniform(-3, 3))
                        for offset, key in enumerate(FEATURE_KEYS)}
            features["is_night"] = 0.0
            if index % 6 < 2:
                certain, uncertain = [CLASSES[index % 5]], []
            elif index % 6 == 5:
                certain, uncertain = [], []
            else:
                certain, uncertain = [], [CLASSES[(index + offset) % 5] for offset in range(2 + index % 3)]
            evaluation = {"symbol": "NK225", "classification_policy": CLASSIFICATION_POLICY,
                          "session": end.date().isoformat() + "_Day", "target_end": end.isoformat(),
                          "target_status": "対象", "correct_candidates": certain,
                          "uncertain_candidates": uncertain, "contract": dict(criteria)}
            rows.append({"symbol": "NK225", "kind": "Day", "as_of": end.replace(hour=7, minute=45).isoformat(),
                         "label_end": end.isoformat(), "features": features,
                         "five_class_evaluation": evaluation})
        return current, rows

    def test_multivariate_weighted_normal_equations_and_common_scaler(self) -> None:
        """両方式の26列回帰が、非罰則切片と罰則４の損失を最小にすることを確認する。

        引数: self (FiveClassModelAuditTests) はテスト。
        返り値: None。
        """
        current, rows = self._sample()
        columns, current_columns = {}, {}
        for key in FEATURE_KEYS:
            observed = [row["features"][key] for row in rows if row["features"][key] is not None]
            fill = median(observed) if observed else 0.0
            values = [fill if row["features"][key] is None else row["features"][key] for row in rows]
            indicators = [float(row["features"][key] is None) for row in rows]
            for name, data, value in ((key, values, current["features"][key]),
                                     (key + "__missing", indicators, 0.0)):
                average, deviation = mean(data), pstdev(data)
                scale = deviation or 1.0
                columns[name] = [(point - average) / scale for point in data]
                current_columns[name] = (value - average) / scale
        results = []
        for mode in ("確定教師", "候補配分"):
            result = predict_five(current, rows, current["as_of"], mode)
            results.append(result)
            self.assertFalse(result["cold_start"])
            weights, targets = [], []
            for row in rows:
                label = row["five_class_evaluation"]
                chosen = label["correct_candidates"] if mode == "確定教師" else label["correct_candidates"] + label["uncertain_candidates"]
                count = len(chosen)
                weights.append(0.0 if not count else 1.0 if mode == "確定教師" else 1.0 / count)
                targets.append({name: 1.0 / count if name in chosen else 0.0 for name in CLASSES})
            self.assertEqual(weights, result["weights"])
            self.assertEqual(targets, result["targets"])
            for name in CLASSES:
                beta = result["coefficients"][name]
                residuals = [beta["intercept"] + math.fsum(beta[key] * values[index] for key, values in columns.items())
                             - targets[index][name] for index in range(len(rows))]
                intercept_gradient = math.fsum(weight * residual for weight, residual in zip(weights, residuals))
                self.assertAlmostEqual(intercept_gradient, 0.0, places=10)
                for key, values in columns.items():
                    gradient = math.fsum(weight * value * residual for weight, value, residual in zip(weights, values, residuals)) + 4 * beta[key]
                    self.assertAlmostEqual(gradient, 0.0, places=10)
                direct_score = beta["intercept"] + math.fsum(beta[key] * value for key, value in current_columns.items())
                self.assertAlmostEqual(result["class_scores"][name], direct_score, places=12)
                self.assertAlmostEqual(sum(result["contributions"][name].values()), direct_score, places=12)
        self.assertEqual(results[0]["scaler"], results[1]["scaler"])
        self.assertEqual(results[0]["expanded_feature_keys"], list(columns))


if __name__ == "__main__":
    unittest.main()
