"""回想研究の逐次推定で、未来混入・標本水増し・銘柄混同を検証する。"""

import copy
import unittest

from fundamentals.pdca import CHANNELS, feature_vector, predict_learned, predict_rule
from fundamentals.common import digest


class PdcaTests(unittest.TestCase):
    """係数学習の情報境界と、定性判断・回想成績の区別を検証する。"""

    def setUp(self) -> None:
        """実データや未来結果を含まない２先物の入力を用意する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        self.packet = {"as_of": "2026-10-15T07:45:00+09:00", "kind": "Day", "instruments": {}}
        self.annotation = {"instruments": {}}
        for symbol in ("NK225", "TOPIX"):
            self.packet["instruments"][symbol] = {
                "reference": {"time": "2026-10-15T05:55:00+09:00", "session": "2026-10-14_Night"},
                "recent_sessions": [{"session": "2026-10-14_Night", "return_pct": -0.4}]}
            self.annotation["instruments"][symbol] = {"channels": {
                key: {"score": 0.6, "confidence": 0.8, "novelty": 0.7, "priced_in": 0.2,
                      "evidence": ["合成記事"], "reason": "実績と無関係な固定注釈"}
                for key in CHANNELS}}

    def _history(self, symbol: str = "NK225", size: int = 8, actual_pct: float = 0.8) -> list[dict]:
        """評価終点を重複させず、受信記録を仮定しない回想履歴を作る。

        引数: self (PdcaTests) はテスト、symbol (str) は先物、size (int) は件数、
        actual_pct (float) は合成した実績騰落率。
        返り値: list[dict] は特徴・実績・満了時刻だけを持つ回想学習行。
        """
        features = feature_vector(self.annotation, self.packet, symbol)
        rows = []
        for index in range(size):
            row_features = dict(features)
            row_features["is_night"] = float(index % 2)
            rows.append({"symbol": symbol, "features": row_features, "actual_pct": actual_pct,
                         "label_end": f"2026-10-{index + 1:02d}T15:45:00+09:00"})
        return rows

    # ----------------------------------------

    def test_future_label_is_rejected_before_unknown_values_are_read(self) -> None:
        """未来の未知価格や特徴は解釈せず、満了時刻だけで先に拒否する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        for symbol in ("NK225", "TOPIX"):
            with self.subTest(symbol=symbol):
                future = {"symbol": symbol, "label_end": "2026-10-15T15:45:00+09:00",
                          "features": "未開示", "actual_pct": object()}
                with self.assertRaisesRegex(ValueError, "情報締切より後に満了"):
                    predict_learned(self.annotation, self.packet, "NK225", self._history() + [future])

    def test_other_futures_outcomes_do_not_change_prediction(self) -> None:
        """他先物の実績を反転しても、対象先物の係数・予測・証跡を変えない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        own = self._history()
        other = self._history("TOPIX", actual_pct=5.0)
        original = predict_learned(self.annotation, self.packet, "NK225", own + other)
        modified = copy.deepcopy(other)
        for row in modified:
            row["actual_pct"] = -40.0
            row["features"]["growth"] = -1.0
        changed = predict_learned(self.annotation, self.packet, "NK225", own + modified)
        self.assertEqual(original, changed)
        self.assertEqual(original["fitted_n"], len(own))

    def test_duplicate_outcome_cannot_make_cold_start_look_fitted(self) -> None:
        """同じ満了実績の複製で最低学習件数を満たせない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=7)
        duplicate = copy.deepcopy(rows[0])
        duplicate["actual_pct"] = -2.0
        duplicate["features"]["growth"] = -0.8
        with self.assertRaisesRegex(ValueError, "重複"):
            predict_learned(self.annotation, self.packet, "NK225", rows + [duplicate])

    def test_cold_start_has_no_fitted_rows_or_predicted_return(self) -> None:
        """履歴が存在しても必要件数未満なら未学習の固定判断として返す。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rule = predict_rule(self.annotation, self.packet, "NK225")
        result = predict_learned(self.annotation, self.packet, "NK225", self._history(size=7))
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["eligible_n"], 7)
        self.assertEqual(result["fitted_n"], 0)
        self.assertEqual(result["coefficients"], rule["coefficients"])
        self.assertEqual(result["direction"], rule["direction"])
        self.assertIsNone(result["prediction_pct"])
        self.assertIn("冷開始", result["status"])
        self.assertEqual(result["statistical_advantage"], "未検証")

    def test_changed_past_outcomes_change_next_coefficients(self) -> None:
        """満了済み結果を変えれば次回の係数・予測が変わり、更新が実在する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        positive = self._history(actual_pct=1.5)
        negative = copy.deepcopy(positive)
        for row in negative:
            row["actual_pct"] = -1.5
        first = predict_learned(self.annotation, self.packet, "NK225", positive)
        second = predict_learned(self.annotation, self.packet, "NK225", negative)
        self.assertFalse(first["cold_start"])
        self.assertEqual(first["fitted_n"], 8)
        self.assertNotEqual(first["coefficients"], second["coefficients"])
        self.assertGreater(first["prediction_pct"], second["prediction_pct"])
        self.assertEqual(first["direction"], "上昇")
        self.assertEqual(second["direction"], "下落")
        self.assertNotEqual(first["training_labels_hash"], second["training_labels_hash"])

    # ----------------------------------------

    def test_history_order_does_not_change_fit_or_label_hashes(self) -> None:
        """同じ満了履歴は入力順を変えても同じ係数と学習証跡を持つ。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        first = predict_learned(self.annotation, self.packet, "NK225", rows)
        second = predict_learned(self.annotation, self.packet, "NK225", list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(set(first["training_label_hashes"])), 8)

    def test_predictions_are_marked_as_replay_without_observation_contract(self) -> None:
        """受信・評価記録時刻を持たない履歴を前向き運用の実績に見せない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rule = predict_rule(self.annotation, self.packet, "NK225")
        learned = predict_learned(self.annotation, self.packet, "NK225", self._history())
        for value in (rule, learned):
            with self.subTest(model=value["model"]):
                self.assertEqual(value["mode"], "回想・開発再生")
                self.assertIn("label_endのみ", value["history_time_contract"])
                self.assertIn("受信・評価記録時刻は未検証", value["history_time_contract"])
                self.assertEqual(value["statistical_advantage"], "未検証")

    def test_prediction_does_not_mutate_fixed_inputs(self) -> None:
        """係数推定後も固定注釈・可視入力・過去ラベルを上書きしない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        original = copy.deepcopy((self.annotation, self.packet, rows))
        predict_learned(self.annotation, self.packet, "NK225", rows)
        self.assertEqual((self.annotation, self.packet, rows), original)

    # ----------------------------------------

    def test_existing_rule_and_ridge_output_hashes_are_unchanged(self) -> None:
        """類例方式追加前に固定した出力ハッシュを既存３経路で再現する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        cases = [
            (predict_rule(self.annotation, self.packet, "NK225"), "38006e660fe788076825da97a0c079deec6a9ecb87a2d78eb7c3491ef1cb933d"),
            (predict_learned(self.annotation, self.packet, "NK225", self._history(size=7)), "47662eb85a2b1ca8c504ce3e1d176dcefd1a89c5af8f785bb76381af323cedcc"),
            (predict_learned(self.annotation, self.packet, "NK225", self._history()), "a3e63ddc0cd767fa0c2fed7b091ee2cb424e48389a41f0eb7ba941bc8b3f3202"),
            (predict_learned(self.annotation, self.packet, "NK225", self._history(size=7), mode="analog"), "081d10220110fc1b98a9e27f9abf10a4cd9f012eae967410265d09aac08527d4"),
            (predict_learned(self.annotation, self.packet, "NK225", self._history(), mode="analog"), "9a0ac206ea95eaff11d0f1dab57d5e5eb2c656d3b7438c11ac17fbfcd2f38e1b")]
        for value, expected in cases:
            self.assertEqual(digest(value), expected)

    def test_analog_uses_close_reactions_instead_of_distant_opposite_outcomes(self) -> None:
        """近い５例が下落なら、遠い例の大幅上昇に方向を引きずられない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history(actual_pct=-1.0)
        current = feature_vector(self.annotation, self.packet, "NK225")
        for index, row in enumerate(rows):
            row["features"] = dict(current)
            if index < 3:
                for key in CHANNELS:
                    row["features"][key] = -1.0
                row["actual_pct"] = 30.0
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="analog")
        self.assertEqual(result["direction"], "下落")
        self.assertEqual(result["analogs_n"], 5)
        self.assertTrue(all(row["actual_pct"] == -1 for row in result["analogs"]))
        self.assertEqual(result["coefficients"], {})
        self.assertIsNone(result["prediction_pct"])
        self.assertIn("確率校正済みではない", result["weight_share_status"])
        own = copy.deepcopy(rows)
        for row in own[:3]:
            row["actual_pct"] = -40.0
        changed = predict_learned(self.annotation, self.packet, "NK225", own, mode="analog")
        self.assertEqual(result["direction_weights"], changed["direction_weights"])
        self.assertNotEqual(result["training_labels_hash"], changed["training_labels_hash"])

    def test_analog_equal_distances_prefer_newer_labels(self) -> None:
        """距離が同じなら新しい５件を使い、入力順に依存しない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        current = feature_vector(self.annotation, self.packet, "NK225")
        for row in rows:
            row["features"] = dict(current)
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="analog")
        self.assertEqual([row["label_end"] for row in result["analogs"]], [row["label_end"] for row in reversed(rows[-5:])])
        self.assertEqual(result, predict_learned(self.annotation, self.packet, "NK225", list(reversed(rows)), mode="analog"))

    def test_analog_calendar_week_halves_equal_distance_weight(self) -> None:
        """終了時刻からの端数時間ではなく暦日差で７日半減を適用する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        current = feature_vector(self.annotation, self.packet, "NK225")
        for row, day in zip(rows, (14, 7, 6, 5, 4, 3, 2, 1)):
            row["features"] = dict(current)
            row["label_end"] = f"2026-10-{day:02d}T15:45:00+09:00"
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="analog")
        newest, previous_week = result["analogs"][:2]
        self.assertEqual(newest["age_calendar_days"], 1)
        self.assertEqual(previous_week["age_calendar_days"], 8)
        self.assertAlmostEqual(newest["weight"], 2 * previous_week["weight"])

    def test_analog_direction_ties_use_fixed_rule_then_fixed_order(self) -> None:
        """方向重み同点時は固定規則を優先し、集合外なら固定順で選ぶ。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        for score, expected in ((0.6, "上昇"), (-0.6, "下落"), (0.0, "上昇")):
            with self.subTest(score=score):
                annotation = copy.deepcopy(self.annotation)
                for channel in annotation["instruments"]["NK225"]["channels"].values():
                    channel["score"] = score
                current = feature_vector(annotation, self.packet, "NK225")
                rows = self._history()
                for index, row in enumerate(rows):
                    row["features"] = dict(current)
                    if index < 3:
                        row["features"]["previous_return"] = 1.0
                    else:
                        row["label_end"] = f"2026-10-14T15:0{index - 3}:00+09:00"
                        row["actual_pct"] = (1.0, 1.0, -1.0, -1.0, 0.0)[index - 3]
                result = predict_learned(annotation, self.packet, "NK225", rows, mode="analog")
                self.assertEqual(result["direction"], expected)
                self.assertEqual(result["direction_weights"]["上昇"], result["direction_weights"]["下落"])

    def test_analog_preserves_cold_start_and_future_label_guards(self) -> None:
        """類例方式でも履歴不足を隠さず、未来の未知実績を解析前に拒否する。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        result = predict_learned(self.annotation, self.packet, "NK225", self._history(size=7), mode="analog")
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["fitted_n"], 0)
        self.assertEqual(result["analogs_n"], 0)
        self.assertEqual(result["direction"], predict_rule(self.annotation, self.packet, "NK225")["direction"])
        self.assertIsNone(result["prediction_pct"])
        future = {"symbol": "TOPIX", "label_end": "2026-10-15T15:45:00+09:00", "actual_pct": object()}
        with self.assertRaisesRegex(ValueError, "情報締切より後"):
            predict_learned(self.annotation, self.packet, "NK225", self._history() + [future], mode="analog")

    def test_analog_ignores_other_futures_and_rejects_duplicate_rows(self) -> None:
        """他先物は参照せず、同先物の同じ終点は類例として重ねない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        original = predict_learned(self.annotation, self.packet, "NK225", rows, mode="analog")
        mixed = predict_learned(self.annotation, self.packet, "NK225", rows + self._history("TOPIX", actual_pct=-50.0), mode="analog")
        self.assertEqual(original, mixed)
        with self.assertRaisesRegex(ValueError, "重複"):
            predict_learned(self.annotation, self.packet, "NK225", rows + [rows[0]], mode="analog")

    # ----------------------------------------

    def test_scaled_statistics_use_past_features_without_current_input(self) -> None:
        """現在の材料点数や直前騰落を変えても、過去の標準化統計を変えない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for index, row in enumerate(rows):
            row["features"]["growth"] = index * 0.01
            row["features"]["previous_return"] = -0.7 + index * 0.2
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="ridge_scaled")
        changed_annotation, changed_packet = copy.deepcopy(self.annotation), copy.deepcopy(self.packet)
        changed_annotation["instruments"]["NK225"]["channels"]["growth"]["score"] = -0.9
        changed_packet["instruments"]["NK225"]["recent_sessions"][-1]["return_pct"] = 2.7
        changed = predict_learned(changed_annotation, changed_packet, "NK225", rows, mode="ridge_scaled")
        for key in ("means", "scales", "zero_variance_features", "train_min", "train_max"):
            self.assertEqual(result["scaling"][key], changed["scaling"][key])
        self.assertTrue(result["scaling"]["trained_on_past_only"])
        self.assertEqual(result["scaling"]["train_min"]["growth"], 0)
        self.assertEqual(result["scaling"]["train_max"]["growth"], 0.07)
        self.assertNotEqual(result["features"], changed["features"])

    def test_scaled_prediction_is_invariant_to_zero_prior_feature_units(self) -> None:
        """事前係数０の特徴だけ単位を倍にしても、標準化後の予測を変えない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for index, row in enumerate(rows):
            value = -0.35 + index * 0.1
            row["features"]["previous_return"] = value
            row["actual_pct"] = value * 3
        first = predict_learned(self.annotation, self.packet, "NK225", rows, mode="ridge_scaled")
        changed_rows, changed_packet = copy.deepcopy(rows), copy.deepcopy(self.packet)
        for row in changed_rows:
            row["features"]["previous_return"] *= 2
        changed_packet["instruments"]["NK225"]["recent_sessions"][-1]["return_pct"] *= 2
        second = predict_learned(self.annotation, changed_packet, "NK225", changed_rows, mode="ridge_scaled")
        self.assertAlmostEqual(first["prediction_pct"], second["prediction_pct"], places=12)
        self.assertEqual(first["direction"], second["direction"])
        self.assertAlmostEqual(first["standardized_coefficients"]["previous_return"], second["standardized_coefficients"]["previous_return"], places=12)
        self.assertAlmostEqual(first["coefficients"]["previous_return"], 2 * second["coefficients"]["previous_return"], places=12)

    def test_scaled_raw_and_standardized_representations_match(self) -> None:
        """元尺度・標準化尺度の予測と、変換前後の事前予測を一致させる。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for index, row in enumerate(rows):
            row["features"]["growth"] = 0.01 * index
            row["actual_pct"] = -1.0 + index * 0.4
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="ridge_scaled")
        scaling, features = result["scaling"], result["features"]
        standardized = {key: (features[key] - mean) / scaling["scales"][key] for key, mean in scaling["means"].items()}
        standardized["intercept"] = 1.0
        raw_prediction = 3 * sum(result["coefficients"][key] * value for key, value in features.items())
        z_prediction = 3 * sum(result["standardized_coefficients"][key] * value for key, value in standardized.items())
        self.assertAlmostEqual(result["prediction_pct"], raw_prediction, places=12)
        self.assertAlmostEqual(result["prediction_pct"], z_prediction, places=12)
        raw_prior = sum(result["prior_coefficients"][key] * value for key, value in features.items())
        z_prior = sum(result["standardized_prior_coefficients"][key] * value for key, value in standardized.items())
        self.assertAlmostEqual(raw_prior, z_prior, places=12)

    def test_scaled_zero_variance_and_extrapolation_are_visible_without_clipping(self) -> None:
        """定数特徴は安全に扱い、小さな学習域からの外挿を隠したりクリップしない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for index, row in enumerate(rows):
            row["features"]["growth"] = (-3.5 + index) * 0.0001
            row["actual_pct"] = -1.5 + index * (3 / 7)
        result = predict_learned(self.annotation, self.packet, "NK225", rows, mode="ridge_scaled")
        for key in ("discount", "fx", "cost_risk", "previous_return"):
            self.assertIn(key, result["scaling"]["zero_variance_features"])
            self.assertEqual(result["scaling"]["scales"][key], 1.0)
        self.assertIn("growth", result["scaling"]["out_of_range_features"])
        self.assertIn("未校正の外挿", result["scaling"]["extrapolation_status"])
        self.assertGreater(abs(result["prediction_pct"]), 3.0)

    def test_scaled_future_duplicate_and_cold_start_guards_remain_active(self) -> None:
        """標準化方式も未来・重複を拒否し、冷開始では統計を計算しない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        cold = predict_learned(self.annotation, self.packet, "NK225", self._history(size=7), mode="ridge_scaled")
        self.assertTrue(cold["cold_start"])
        self.assertEqual(cold["scaling"]["means"], {})
        self.assertEqual(cold["standardized_coefficients"], {})
        self.assertIsNone(cold["prediction_pct"])
        rows = self._history()
        future = {"symbol": "TOPIX", "label_end": "2026-10-15T15:45:00+09:00", "actual_pct": object()}
        with self.assertRaisesRegex(ValueError, "情報締切より後"):
            predict_learned(self.annotation, self.packet, "NK225", rows + [future], mode="ridge_scaled")
        with self.assertRaisesRegex(ValueError, "重複"):
            predict_learned(self.annotation, self.packet, "NK225", rows + [rows[0]], mode="ridge_scaled")

    def test_scaled_teacher_clipping_is_identical_for_larger_shocks(self) -> None:
        """教師の固定クリップを超える衝撃の増幅で係数を変えない。

        引数: self (PdcaTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for index, row in enumerate(rows):
            row["features"]["previous_return"] = -0.7 + index * 0.2
            row["actual_pct"] = -3.0 if index < 4 else 3.0
        original = predict_learned(self.annotation, self.packet, "NK225", rows, mode="ridge_scaled")
        changed_rows = copy.deepcopy(rows)
        for row in changed_rows:
            row["actual_pct"] *= 10
        changed = predict_learned(self.annotation, self.packet, "NK225", changed_rows, mode="ridge_scaled")
        self.assertEqual(original["coefficients"], changed["coefficients"])
        self.assertEqual(original["prediction_pct"], changed["prediction_pct"])
        self.assertNotEqual(original["training_labels_hash"], changed["training_labels_hash"])


if __name__ == "__main__":
    unittest.main()
