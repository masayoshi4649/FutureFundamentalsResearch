"""実価格を使わず、５分類候補の作業用学習と未来遮断を合成履歴で検証する。"""

import copy
import math
import unittest
from datetime import timedelta

from fundamentals.common import digest, timestamp
from research_support.five_class import CLASSES
from research_support.five_class_model import CLASSIFICATION_POLICY, FEATURE_KEYS, _choose, predict_five


class FiveClassModelTests(unittest.TestCase):
    """正解候補・行重み・目的値・共通標準化を別々に点検する。"""

    def setUp(self) -> None:
        """数値と時点が明確な13特徴の合成入力を用意する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        self.as_of = "2026-10-15T07:45:00+09:00"
        self.current = {"symbol": "NK225", "kind": "Day", "as_of": self.as_of,
                        "features": dict.fromkeys(FEATURE_KEYS, 0.0)}
        self.criteria = {"target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
                         "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5",
                         "reversal_requires_reference_cross": False, "range_threshold_pct": "1",
                         "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
                         "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10}

    def _rows(self, count: int = 12, symbol: str = "NK225") -> list[dict]:
        """満了時点が重ならない、真の相場と無関係な教師を作る。

        引数: self (FiveClassModelTests) はテスト、count (int) は教師件数、symbol (str) は先物。
        返り値: list[dict] は初期状態ですべて上がるを一意正解とした履歴。
        """
        rows = []
        for index in range(count):
            end = timestamp("2026-09-01T15:45:00+09:00") + timedelta(days=index)
            evaluation = {"classification_policy": CLASSIFICATION_POLICY, "symbol": symbol,
                          "session": end.date().isoformat() + "_Day", "target_end": end.isoformat(), "target_status": "対象",
                          "correct_candidates": ["上がる"], "uncertain_candidates": [], "contract": copy.deepcopy(self.criteria)}
            rows.append({"symbol": symbol, "kind": "Day", "as_of": end.replace(hour=7, minute=45).isoformat(),
                         "label_end": end.isoformat(), "features": dict(self.current["features"]), "actual_pct": None,
                         "five_class_evaluation": evaluation})
        return rows

    def _candidates(self, row: dict, certain: list[str], uncertain: list[str]) -> None:
        """合成教師の確定集合と未確定集合だけを明示的に変更する。

        引数: self (FiveClassModelTests) はテスト、row (dict) は教師、certain/uncertain (list[str]) は集合。
        返り値: None。
        """
        row["five_class_evaluation"]["correct_candidates"] = certain
        row["five_class_evaluation"]["uncertain_candidates"] = uncertain

    # ----------------------------------------

    def test_all_future_rows_are_rejected_before_any_feature_or_candidate_parsing(self) -> None:
        """末尾の他先物未来教師も、先頭の壊れた特徴を読む前に拒否する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows()
        rows[0]["features"] = object()
        rows[0]["five_class_evaluation"] = object()
        for symbol in ("NK225", "TOPIX"):
            future = {"symbol": symbol, "label_end": "2026-10-15T15:45:00+09:00", "features": object()}
            with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, "数値解釈前"):
                predict_five(self.current, [*rows, future], self.as_of, "確定教師")

    def test_exact_cutoff_is_not_past_and_current_time_mismatch_is_rejected(self) -> None:
        """満了と締切の同時刻を使わず、今回入力の別締切も拒否する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows()
        rows[0]["label_end"] = self.as_of
        with self.assertRaisesRegex(ValueError, "情報締切以降"):
            predict_five(self.current, rows, self.as_of, "確定教師")
        changed = {**self.current, "as_of": "2026-10-15T07:44:00+09:00"}
        with self.assertRaisesRegex(ValueError, "一致"):
            predict_five(changed, [], self.as_of, "確定教師")

    def test_other_symbol_fields_are_not_parsed_after_time_check(self) -> None:
        """他先物の未知特徴・候補で対象先物の予測を変えない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        own, other = self._rows(), self._rows(symbol="TOPIX")
        for row in other:
            row["features"] = row["five_class_evaluation"] = object()
        expected = predict_five(self.current, own, self.as_of, "確定教師")
        self.assertEqual(expected, predict_five(self.current, own + other, self.as_of, "確定教師"))

    def test_duplicate_end_is_rejected_before_kind_change_can_make_another_sample(self) -> None:
        """同じ先物・同じ満了時刻の教師を場名変更でも増やせない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows()
        extra = {**rows[0], "kind": "Night"}
        with self.assertRaisesRegex(ValueError, "重複"):
            predict_five(self.current, rows + [extra], self.as_of, "候補配分")

    def test_no_history_produces_none_and_does_not_fabricate_scores(self) -> None:
        """教師なしは無回答Noneとして採点へ渡せる値を返す。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        result = predict_five(self.current, [], self.as_of, "確定教師")
        self.assertIsNone(result["direction"])
        self.assertEqual(result["class_scores"], dict.fromkeys(CLASSES))
        self.assertEqual(result["fitted_n"], 0)
        self.assertEqual(result["effective_weight"], 0)
        self.assertEqual(result["coefficients"], {})
        self.assertEqual(result["hash"], digest({key: value for key, value in result.items() if key != "hash"}))

    def test_candidate_targets_and_row_weights_are_separate_inverse_counts(self) -> None:
        """候補２個は目的値1/2と行重み1/2に分け、各寄与1/4を記録する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=1)
        self._candidates(rows[0], [], ["上がる", "下がる"])
        result = predict_five(self.current, rows, self.as_of, "候補配分")
        self.assertEqual(result["weights"], [0.5])
        self.assertEqual(result["targets"][0]["上がる"], 0.5)
        self.assertEqual(result["label_counts"]["weighted_class_mass"]["上がる"], 0.25)
        self.assertEqual(result["effective_weight"], 0.5)
        self.assertEqual(result["class_scores"]["上がる"], 0.5)
        self.assertEqual(result["direction"], "上がる")

    def test_certain_mode_ignores_uncertain_singleton_as_teacher_but_keeps_scaler(self) -> None:
        """候補が一個でも未確定なら確定教師へ格上げしない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=1)
        self._candidates(rows[0], [], ["上がる"])
        certain = predict_five(self.current, rows, self.as_of, "確定教師")
        shared = predict_five(self.current, rows, self.as_of, "候補配分")
        self.assertIsNone(certain["direction"])
        self.assertEqual(certain["weights"], [0.0])
        self.assertEqual(shared["weights"], [1.0])
        self.assertEqual(certain["scaler"], shared["scaler"])

    def test_zero_candidate_rows_have_zero_weight_without_disappearing(self) -> None:
        """候補なし教師を真のクラスへ割り当てず、全履歴と前処理に残す。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=3)
        for row in rows:
            self._candidates(row, [], [])
        for mode in ("確定教師", "候補配分"):
            result = predict_five(self.current, rows, self.as_of, mode)
            self.assertIsNone(result["direction"])
            self.assertEqual(result["training_n"], 3)
            self.assertEqual(result["positive_weight_n"], 0)
            self.assertEqual(result["weights"], [0.0] * 3)
            self.assertEqual(len(result["scaler"]), 26)

    def test_fallback_uses_weighted_class_mass_not_raw_membership_counts(self) -> None:
        """少数時は確定１行と候補配分１行の質量を区別して構成比を作る。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=2)
        self._candidates(rows[1], [], ["下がる", "上がってから下がる"])
        result = predict_five(self.current, rows, self.as_of, "候補配分")
        self.assertAlmostEqual(result["class_scores"]["上がる"], 2 / 3)
        self.assertAlmostEqual(result["class_scores"]["下がる"], 1 / 6)
        self.assertEqual(result["positive_weight_n"], 2)
        self.assertEqual(result["fitted_n"], 0)

    def test_fitting_threshold_is_positive_row_count_not_effective_weight(self) -> None:
        """候補２個の８行は有効重み４でも８行としてfitへ進む。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=8)
        for row in rows:
            self._candidates(row, [], ["上がる", "下がる"])
        result = predict_five(self.current, rows, self.as_of, "候補配分")
        self.assertEqual(result["effective_weight"], 4)
        self.assertEqual(result["fitted_n"], 8)
        self.assertFalse(result["cold_start"])
        certain = predict_five(self.current, rows, self.as_of, "確定教師")
        self.assertEqual(certain["positive_weight_n"], 0)
        self.assertEqual(certain["fitted_n"], 0)

    def test_scaler_is_shared_across_modes_and_includes_zero_weight_rows(self) -> None:
        """不採用教師の特徴を含めた同じ前処理で２方式の目的値処理だけを比較する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=10)
        rows[-1]["features"]["growth"] = 1000.0
        self._candidates(rows[-1], [], ["上がる", "下がる"])
        certain = predict_five(self.current, rows, self.as_of, "確定教師")
        shared = predict_five(self.current, rows, self.as_of, "候補配分")
        self.assertEqual(certain["scaler"], shared["scaler"])
        self.assertEqual(certain["scaler"]["growth"]["mean"], 100)
        self.assertEqual(certain["scaler"]["growth"]["observed_n"], 10)
        self.assertEqual(certain["positive_weight_n"], 9)
        self.assertEqual(shared["positive_weight_n"], 10)

    def test_weighted_ridge_matches_independent_scalar_normal_equation(self) -> None:
        """重み付き中心化と非罰則切片を、１変数の独立計算に照合する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=8)
        for index, row in enumerate(rows):
            row["features"]["growth"] = float(index)
            if index >= 4:
                self._candidates(row, [], ["下がる", "もみ合い"])
        self.current["features"]["growth"] = 9
        result = predict_five(self.current, rows, self.as_of, "候補配分")
        weights, target = [1.0] * 4 + [0.5] * 4, [1.0] * 4 + [0.0] * 4
        xs = [(index - 3.5) / math.sqrt(5.25) for index in range(8)]
        effective = sum(weights)
        xmean = sum(weight * value for weight, value in zip(weights, xs)) / effective
        ymean = sum(weight * value for weight, value in zip(weights, target)) / effective
        numerator = sum(weight * (x - xmean) * (y - ymean) for weight, x, y in zip(weights, xs, target))
        denominator = sum(weight * (x - xmean) ** 2 for weight, x in zip(weights, xs)) + 4
        slope = numerator / denominator
        intercept = ymean - slope * xmean
        self.assertAlmostEqual(result["coefficients"]["上がる"]["growth"], slope, places=12)
        self.assertAlmostEqual(result["coefficients"]["上がる"]["intercept"], intercept, places=12)
        self.assertAlmostEqual(result["class_scores"]["上がる"], intercept + slope * (9 - 3.5) / math.sqrt(5.25), places=12)

    def test_one_class_constant_features_keep_unpenalized_intercept(self) -> None:
        """１クラス・全列定数でも特異行列を起こさず切片１を保持する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        result = predict_five(self.current, self._rows(), self.as_of, "確定教師")
        self.assertEqual(result["class_scores"], {name: float(name == "上がる") for name in CLASSES})
        self.assertEqual(result["coefficients"]["上がる"]["intercept"], 1)
        self.assertTrue(all(value == 0 for key, value in result["coefficients"]["上がる"].items() if key != "intercept"))

    def test_missing_flags_distinguish_absent_features_from_real_zero(self) -> None:
        """中央値補完が０でも欠測フラグで異なる教師状態を保持する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=16)
        for index, row in enumerate(rows):
            if index % 2:
                row["features"]["growth"] = None
                self._candidates(row, ["下がる"], [])
        known = predict_five(self.current, rows, self.as_of, "確定教師")
        self.current["features"]["growth"] = None
        missing = predict_five(self.current, rows, self.as_of, "確定教師")
        self.assertEqual(known["direction"], "上がる")
        self.assertEqual(missing["direction"], "下がる")
        self.assertEqual(known["scaler"]["growth"]["median"], 0)
        self.assertNotEqual(known["expanded_features"]["growth__missing"], missing["expanded_features"]["growth__missing"])

    def test_extreme_current_is_not_used_to_fit_scaler_or_clipped_into_probability(self) -> None:
        """今回の極端値を学習統計へ混ぜず、域外スコアを確率範囲へ切り詰めない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=12)
        for index, row in enumerate(rows):
            row["features"]["growth"] = float(index % 2)
            if index % 2 == 0:
                self._candidates(row, ["下がる"], [])
        first = predict_five(self.current, rows, self.as_of, "確定教師")
        self.current["features"]["growth"] = 1000
        large = predict_five(self.current, rows, self.as_of, "確定教師")
        self.assertEqual(first["scaler"]["growth"]["mean"], large["scaler"]["growth"]["mean"])
        self.assertIn("growth", large["out_of_range_features"])
        self.assertGreater(large["class_scores"]["上がる"], 1)
        self.assertLess(large["class_scores"]["下がる"], 0)
        self.assertIn("予測確率ではない", large["score_semantics"])
        for name in CLASSES:
            self.assertAlmostEqual(large["class_scores"][name], math.fsum(large["contributions"][name].values()))

    def test_tie_uses_weighted_mass_then_fixed_order(self) -> None:
        """スコア同点時は重み付き質量、なお同点なら固定した５種類順を使う。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        scores, mass = dict.fromkeys(CLASSES, 0.0), dict.fromkeys(CLASSES, 0.0)
        scores["上がる"] = scores["下がる"] = 1
        mass["下がる"] = 2
        self.assertEqual(_choose(scores, mass), "下がる")
        mass["上がる"] = 2
        self.assertEqual(_choose(scores, mass), "上がる")

    def test_outcome_magnitude_and_unparsed_terminal_fields_do_not_define_target(self) -> None:
        """終点が欠測・未知でも５分類候補だけから学習し、旧終点方向へ戻さない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows()
        expected = predict_five(self.current, rows, self.as_of, "確定教師")
        for row in rows:
            row["actual_pct"] = object()
            row["five_class_evaluation"]["metrics"] = object()
        self.assertEqual(expected, predict_five(self.current, rows, self.as_of, "確定教師"))

    def test_invalid_candidate_sets_and_old_classification_policy_are_rejected(self) -> None:
        """重複・未知語・複数確定・確定と不明の混在や旧方式を拒否する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        cases = ((["上がる", "上がる"], []), (["横ばい"], []), (["上がる", "下がる"], []),
                 (["上がる"], ["下がる"]), (["上がる"], ["上がる"]), ([], ["下がる", "下がる"]))
        for certain, uncertain in cases:
            rows = self._rows(count=1)
            self._candidates(rows[0], certain, uncertain)
            with self.subTest(certain=certain, uncertain=uncertain), self.assertRaises(ValueError):
                predict_five(self.current, rows, self.as_of, "候補配分")
        rows = self._rows(count=1)
        rows[0]["five_class_evaluation"]["classification_policy"] = "終点だけの旧分類"
        with self.assertRaisesRegex(ValueError, "新規約"):
            predict_five(self.current, rows, self.as_of, "確定教師")

    def test_mixed_contract_and_label_identity_are_rejected(self) -> None:
        """違う閾値・観測契約・先物・満了・場名を持つ評価を混ぜない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        for key, value in (("symbol", "TOPIX"), ("target_end", "2026-09-02T15:45:00+09:00"),
                           ("session", "2026-09-01_Night")):
            rows = self._rows(count=1)
            rows[0]["five_class_evaluation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "一致"):
                predict_five(self.current, rows, self.as_of, "確定教師")
        rows = self._rows(count=2)
        rows[1]["five_class_evaluation"]["contract"]["reference_max_lag_minutes"] = 9
        with self.assertRaisesRegex(ValueError, "異なる５分類契約"):
            predict_five(self.current, rows, self.as_of, "確定教師")
        rows[1]["five_class_evaluation"]["contract"]["movement_threshold_pct"] = "1"
        with self.assertRaisesRegex(ValueError, "規約が一致"):
            predict_five(self.current, rows, self.as_of, "確定教師")

    def test_history_order_does_not_change_hash_or_prediction(self) -> None:
        """過去の入力順序に依存せず、満了順へ正規化して同じ結果を返す。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows()
        self.assertEqual(predict_five(self.current, rows, self.as_of, "確定教師"),
                         predict_five(self.current, list(reversed(rows)), self.as_of, "確定教師"))

    def test_malformed_evaluation_metadata_is_explicitly_rejected(self) -> None:
        """欠落した満了・場や対象状態・不正な候補型を黙って受け付けない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        for key, value in (("target_end", None), ("session", None), ("session", "不明_Day"),
                           ("target_status", "無視"), ("correct_candidates", "上がる"), ("uncertain_candidates", [None])):
            rows = self._rows(count=1)
            rows[0]["five_class_evaluation"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                predict_five(self.current, rows, self.as_of, "確定教師")

    def test_nonfinite_features_and_kind_mismatch_are_rejected(self) -> None:
        """特徴の無限・NaN・真偽値や場との矛盾を有限の学習行へ入れない。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        for bad in (float("nan"), float("inf"), True, "1"):
            rows = self._rows(count=1)
            rows[0]["features"]["growth"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                predict_five(self.current, rows, self.as_of, "確定教師")
        changed = copy.deepcopy(self.current)
        changed["features"]["is_night"] = 1
        with self.assertRaisesRegex(ValueError, "一致"):
            predict_five(changed, [], self.as_of, "確定教師")

    def test_all_missing_past_features_do_not_invent_observed_zero_and_flag_new_value(self) -> None:
        """過去全欠測の列を観測済みゼロとせず、今回初観測を明示する。

        引数: self (FiveClassModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(count=8)
        for row in rows:
            row["features"] = dict.fromkeys(FEATURE_KEYS)
        self.current["features"]["us_nasdaq_return_pct"] = 1
        result = predict_five(self.current, rows, self.as_of, "確定教師")
        stats = result["scaler"]["us_nasdaq_return_pct"]
        self.assertIsNone(stats["median"])
        self.assertTrue(stats["all_missing"])
        self.assertEqual(stats["observed_n"], 0)
        self.assertIn("us_nasdaq_return_pct", result["newly_observed_features"])
        self.assertEqual(result["class_scores"]["上がる"], 1)


if __name__ == "__main__":
    unittest.main()
