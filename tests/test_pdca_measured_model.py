"""実測特徴の方向分類で、未来混入・欠測混同・銘柄混同を検証する。"""

import copy
from datetime import datetime, timedelta, timezone
import unittest

from fundamentals.pdca_measured_model import DIRECTIONS, MODE_FEATURES, predict_measured


class PdcaMeasuredModelTests(unittest.TestCase):
    """実価格を読まず、合成した既知特徴と結果だけで分類契約を検証する。"""

    def setUp(self) -> None:
        """固定した情報締切と今回入力を用意する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        self.as_of = "2026-10-15T07:45:00+09:00"
        self.current = {
            "symbol": "NK225", "kind": "Day", "as_of": self.as_of,
            "features": {key: 0.0 for key in MODE_FEATURES["raw_news"]},
        }

    def _history(self, size: int = 12, symbol: str = "NK225") -> list[dict]:
        """評価終点が重ならず、特徴と方向が対応する合成履歴を作る。

        引数: self (PdcaMeasuredModelTests) はテスト、size (int) は件数、symbol (str) は先物。
        返り値: list[dict] は実際の価格結果を含まないテスト履歴。
        """
        rows = []
        start = datetime(2026, 9, 1, 15, 45, tzinfo=timezone(timedelta(hours=9)))
        for index in range(size):
            value = (-1.0, 0.0, 1.0)[index % 3]
            features = dict(self.current["features"])
            features["previous_return_pct"] = value
            rows.append({
                "symbol": symbol, "kind": "Day", "features": features,
                "actual_pct": value, "label_end": (start + timedelta(days=index)).isoformat(),
            })
        return rows

    # ----------------------------------------

    def test_direction_is_learned_directly_and_outcome_magnitudes_do_not_change_coefficients(self) -> None:
        """同じ方向内で実績の大きさを変えても、分類係数を変えない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=18)
        self.current["features"]["previous_return_pct"] = 1.0
        positive = predict_measured(self.current, rows, self.as_of)
        self.assertEqual(positive["direction"], "上昇")
        self.assertEqual(positive["fitted_n"], 18)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["actual_pct"] *= 100
        same_classes = predict_measured(self.current, changed, self.as_of)
        self.assertEqual(positive["class_scores"], same_classes["class_scores"])
        self.assertEqual(positive["coefficients"], same_classes["coefficients"])
        self.current["features"]["previous_return_pct"] = -1.0
        self.assertEqual(predict_measured(self.current, rows, self.as_of)["direction"], "下落")

    def test_all_future_times_are_checked_before_any_features_or_outcomes(self) -> None:
        """末尾に未来行があれば、その前の壊れた特徴・実績より先に拒否する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        rows[0]["features"] = object()
        rows[0]["actual_pct"] = object()
        for symbol in ("NK225", "TOPIX"):
            future = {"symbol": symbol, "label_end": "2026-10-15T15:45:00+09:00",
                      "features": object(), "actual_pct": object()}
            with self.subTest(symbol=symbol):
                with self.assertRaisesRegex(ValueError, "情報締切以降に満了"):
                    predict_measured(self.current, rows + [future], self.as_of)

    def test_label_ending_exactly_at_cutoff_is_not_available(self) -> None:
        """締切と同時刻の評価満了を、既知の教師として使わない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        rows[0]["label_end"] = self.as_of
        with self.assertRaisesRegex(ValueError, "情報締切以降に満了"):
            predict_measured(self.current, rows, self.as_of)

    def test_other_symbol_outcomes_and_features_are_not_interpreted(self) -> None:
        """他先物の実績や特徴が未知値でも、対象先物の結果を変えない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        own, other = self._history(), self._history(symbol="TOPIX")
        for row in other:
            row["features"] = object()
            row["actual_pct"] = object()
        result = predict_measured(self.current, own, self.as_of)
        self.assertEqual(result, predict_measured(self.current, own + other, self.as_of))

    def test_duplicate_end_cannot_inflate_training_size_even_with_different_kind(self) -> None:
        """同一先物・同一終点は場種別を変えても別標本にしない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=7)
        duplicate = copy.deepcopy(rows[0])
        duplicate["kind"] = "Night"
        with self.assertRaisesRegex(ValueError, "重複"):
            predict_measured(self.current, rows + [duplicate], self.as_of)

    def test_no_history_returns_hold_without_fabricated_scores(self) -> None:
        """履歴０件では過去多数方向を作らず保留する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        result = predict_measured(self.current, [], self.as_of)
        self.assertEqual(result["direction"], "保留")
        self.assertEqual(result["class_scores"], dict.fromkeys(DIRECTIONS))
        self.assertEqual(result["training_n"], 0)
        self.assertEqual(result["fitted_n"], 0)
        self.assertEqual(result["coefficients"], {})

    def test_few_rows_use_majority_with_fixed_tie_order(self) -> None:
        """８件未満では係数を推定せず、同数なら固定した方向順で決める。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=6)
        tied = predict_measured(self.current, rows, self.as_of)
        self.assertEqual(tied["direction"], "上昇")
        self.assertEqual(tied["training_n"], 6)
        self.assertEqual(tied["fitted_n"], 0)
        self.assertEqual(tied["coefficients"], {})
        rows[1]["actual_pct"] = -1.0
        self.assertEqual(predict_measured(self.current, rows, self.as_of)["direction"], "下落")

    def test_constant_features_keep_unpenalized_intercept_and_do_not_make_fit_singular(self) -> None:
        """全特徴が定数でも特異行列で失敗せず、過去構成比を保つ。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for row in rows:
            row["features"] = dict(self.current["features"])
            row["actual_pct"] = 1.0
        result = predict_measured(self.current, rows, self.as_of)
        self.assertEqual(result["class_scores"], {"上昇": 1.0, "下落": 0.0, "横ばい": 0.0})
        self.assertEqual(result["coefficients"]["上昇"]["intercept"], 1.0)
        self.assertTrue(all(value == 0 for key, value in result["coefficients"]["上昇"].items() if key != "intercept"))

    def test_missing_value_and_actual_zero_are_learned_as_different_states(self) -> None:
        """中央値が０でも、欠測フラグにより実ゼロと欠測を混同しない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=16)
        for index, row in enumerate(rows):
            row["features"]["previous_return_pct"] = None if index % 2 else 0.0
            row["actual_pct"] = -1.0 if index % 2 else 1.0
        known = predict_measured(self.current, rows, self.as_of)
        self.current["features"]["previous_return_pct"] = None
        missing = predict_measured(self.current, rows, self.as_of)
        self.assertEqual(known["direction"], "上昇")
        self.assertEqual(missing["direction"], "下落")
        self.assertEqual(known["feature_stats"]["previous_return_pct"]["median"], 0.0)
        self.assertEqual(missing["feature_stats"]["previous_return_pct"]["current_imputed"], 0.0)
        self.assertNotEqual(known["expanded_features"]["previous_return_pct__missing"],
                            missing["expanded_features"]["previous_return_pct__missing"])

    def test_preprocessing_uses_only_past_and_does_not_clip_outside_values(self) -> None:
        """今回の極端値を学習統計へ入れず、域外フラグを付けて値を保持する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        first = predict_measured(self.current, rows, self.as_of)
        self.current["features"]["previous_return_pct"] = 100.0
        second = predict_measured(self.current, rows, self.as_of)
        stats = second["feature_stats"]["previous_return_pct"]
        self.assertEqual(stats["mean"], first["feature_stats"]["previous_return_pct"]["mean"])
        self.assertEqual(stats["std"], first["feature_stats"]["previous_return_pct"]["std"])
        self.assertIn("previous_return_pct", second["out_of_range_features"])
        self.assertGreater(second["expanded_features"]["previous_return_pct"], 100)
        self.assertGreater(second["class_scores"]["上昇"], 1.0)
        self.assertLess(second["class_scores"]["下落"], 0.0)

    def test_all_missing_feature_retains_unknown_state_and_flags_first_observation(self) -> None:
        """過去の全欠測を実ゼロの観測とみなさず、今回の初観測を表示する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        for row in rows:
            row["features"]["late_change_pct"] = None
        result = predict_measured(self.current, rows, self.as_of)
        stats = result["feature_stats"]["late_change_pct"]
        self.assertTrue(stats["all_missing"])
        self.assertIsNone(stats["median"])
        self.assertEqual(stats["observed_n"], 0)
        self.assertIn("late_change_pct", result["newly_observed_features"])
        self.assertIn("late_change_pct__missing", result["out_of_range_features"])

    # ----------------------------------------

    def test_each_mode_ignores_unselected_extra_features_without_parsing(self) -> None:
        """全方式の特徴を同じ辞書に保存しても、方式外の未知値を解釈しない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        for mode, keys in MODE_FEATURES.items():
            with self.subTest(mode=mode):
                current, rows = copy.deepcopy(self.current), self._history()
                for value in [current] + rows:
                    value["features"]["future_unselected"] = object()
                    for key in set().union(*MODE_FEATURES.values()) - set(keys):
                        value["features"][key] = object()
                result = predict_measured(current, rows, self.as_of, mode=mode)
                self.assertEqual(result["feature_keys"], list(keys))
                self.assertNotIn("future_unselected", result["features"])

    def test_news_mode_can_learn_from_news_without_observed_direction_signal(self) -> None:
        """市場観測が定数でも、採用したニュース列に対応する方向を学習する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        for mode, feature in (("raw_news", "growth"), ("market_news", "us_nasdaq_return_pct")):
            with self.subTest(mode=mode):
                current, rows = copy.deepcopy(self.current), self._history(size=18)
                for row in rows:
                    row["features"][feature] = row["actual_pct"]
                    row["features"]["previous_return_pct"] = 0.0
                current["features"][feature] = -1.0
                self.assertEqual(predict_measured(current, rows, self.as_of, mode)["direction"], "下落")
                current["features"][feature] = 1.0
                self.assertEqual(predict_measured(current, rows, self.as_of, mode)["direction"], "上昇")

    def test_boundary_values_are_flat_and_not_changed_by_classification(self) -> None:
        """正負０．０５％ちょうどを横ばいとして固定する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history(size=8)
        for index, row in enumerate(rows):
            row["actual_pct"] = 0.05 if index % 2 else -0.05
        result = predict_measured(self.current, rows, self.as_of)
        self.assertEqual(result["training_counts"], {"上昇": 0, "下落": 0, "横ばい": 8})
        self.assertEqual(result["direction"], "横ばい")

    def test_contributions_reconstruct_each_class_score(self) -> None:
        """係数と今回特徴の寄与を足すと、表示した各クラススコアに一致する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        result = predict_measured(self.current, self._history(), self.as_of)
        for direction in DIRECTIONS:
            self.assertAlmostEqual(sum(result["contributions"][direction].values()), result["class_scores"][direction])
        self.assertIn("確率ではなく", result["score_semantics"])
        self.assertEqual(result["statistical_advantage"], "未検証")

    def test_input_order_and_time_zone_representation_do_not_change_prediction(self) -> None:
        """同じ満了時刻の履歴なら、配列順や時差表記で係数を変えない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        first = predict_measured(self.current, rows, self.as_of)
        changed = copy.deepcopy(list(reversed(rows)))
        for row in changed:
            row["label_end"] = datetime.fromisoformat(row["label_end"]).astimezone(timezone.utc).isoformat()
        self.assertEqual(first, predict_measured(self.current, changed, self.as_of))

    def test_prediction_does_not_mutate_inputs(self) -> None:
        """予測を実行しても固定入力や過去特徴を補完値で上書きしない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        rows[0]["features"]["previous_return_pct"] = None
        original = copy.deepcopy((self.current, rows))
        predict_measured(self.current, rows, self.as_of)
        self.assertEqual((self.current, rows), original)

    def test_invalid_selected_values_and_mismatched_identity_are_rejected(self) -> None:
        """採用列の非有限値や場種別・情報締切の不一致を拒否する。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        cases = []
        for value in (float("nan"), float("inf"), True, "0"):
            current = copy.deepcopy(self.current)
            current["features"]["previous_return_pct"] = value
            cases.append(current)
        for field, value in (("symbol", "現物"), ("kind", "不明"), ("as_of", "2026-10-15T16:00:00+09:00")):
            current = copy.deepcopy(self.current)
            current[field] = value
            cases.append(current)
        current = copy.deepcopy(self.current)
        current["features"]["is_night"] = 1.0
        cases.append(current)
        for current in cases:
            with self.subTest(current=current):
                with self.assertRaises(ValueError):
                    predict_measured(current, self._history(), self.as_of)

    def test_history_prediction_time_must_precede_its_target_end(self) -> None:
        """履歴に情報締切が付く場合、満了後に作った特徴を受け入れない。

        引数: self (PdcaMeasuredModelTests) はテスト。
        返り値: None。
        """
        rows = self._history()
        rows[0]["as_of"] = rows[0]["label_end"]
        with self.assertRaisesRegex(ValueError, "情報締切as_ofは満了時刻"):
            predict_measured(self.current, rows, self.as_of)


if __name__ == "__main__":
    unittest.main()
