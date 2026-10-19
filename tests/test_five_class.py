"""実価格や旧予測を開かず、５分類の境界・経路不明・分母を合成足で確認する。"""

import unittest
from dataclasses import replace
from datetime import timedelta

from fundamentals.common import timestamp
from fundamentals.prices import Candle
from research_support.five_class import CLASSES, evaluate_path, evaluate_priority_path, score_prediction, validate_contract


class FiveClassTests(unittest.TestCase):
    """420本の合成デイ足を使い、結果に依存しない採点契約を検証する。"""

    def setUp(self) -> None:
        """直前ナイト100と全分を収録する合成デイを作る。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.start, self.end = timestamp("2026-10-01T08:45:00+09:00"), timestamp("2026-10-01T15:45:00+09:00")
        self.reference = Candle(timestamp("2026-10-01T05:59:00+09:00"), timestamp("2026-10-01T06:00:00+09:00"),
                                100, 100, 100, 100, 1, "2026-09-30_Night", "NK225", "202612")
        self.bars = [Candle(self.start + timedelta(minutes=index), self.start + timedelta(minutes=index + 1),
                            100, 100, 100, 100, 1, "2026-10-01_Day", "NK225", "202612") for index in range(420)]
        self.contract = {"target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
                         "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5",
                         "reversal_requires_reference_cross": False, "range_threshold_pct": "1",
                         "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
                         "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10}

    def _run(self, bars: list[Candle] | None = None, **settings: object) -> dict:
        """当回だけの契約変更を明示して合成経路を評価する。

        引数: self (FiveClassTests) はテスト、bars (list[Candle] | None) は置換足、
            settings (object) は当回の契約変更。
        返り値: dict は合成経路の評価。
        """
        return evaluate_path(self.reference, self.bars if bars is None else bars, "2026-10-01_Day", self.end,
                             {**self.contract, **settings})

    # ----------------------------------------

    def test_all_contract_fields_are_required_and_numeric_boundaries_are_decimal(self) -> None:
        """契約の省略・追加や不正な数値を拒否し、十進保存を再現する。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.assertEqual(validate_contract(self.contract).to_dict(), self.contract)
        for changed in ({key: value for key, value in self.contract.items() if key != "target_basis"},
                        {**self.contract, "未知": 1}, {**self.contract, "reversal_threshold_pct": True},
                        {**self.contract, "movement_threshold_pct": "NaN"}, {**self.contract, "interval_minutes": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_contract(changed)

    def test_flat_uses_actual_session_high_low_without_adding_reference(self) -> None:
        """窓を開けた場中高安に前終値を加えて幅を広げない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.4, high=100.6, low=100.4, close=100.5) for bar in self.bars]
        value = self._run(bars)
        self.assertEqual(value["metrics"]["session_range_pct"], 0.2)
        self.assertEqual(value["metrics"]["low_pct"], 0.4)
        self.assertIn("もみ合い", value["correct_candidates"])
        self.assertIn("上がる", value["correct_candidates"])

    def test_flat_and_target_boundaries_include_exact_half_and_one_percent(self) -> None:
        """終点0.5％と場中幅1％の等号を両方含める。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[1] = replace(self.bars[1], high=100.5, low=99.5)
        self.bars[-1] = replace(self.bars[-1], high=100.5, close=100.5)
        value = self._run()
        self.assertEqual(value["target_status"], "対象")
        self.assertIn("もみ合い", value["correct_candidates"])
        self.assertEqual(value["metrics"]["session_range_pct"], 1)
        self.bars[1] = replace(self.bars[1], high=100.500001)
        self.assertNotIn("もみ合い", self._run()["correct_candidates"])

    def test_target_basis_distinguishes_intraday_arrival_from_terminal_move(self) -> None:
        """途中0.5％到達後に戻る場合を、明示した対象基準で分ける。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], high=100.5)
        self.assertEqual(self._run()["target_status"], "対象")
        self.assertEqual(self._run(target_basis="終点")["target_status"], "対象外")

    def test_zero_direction_threshold_uses_strict_sign_without_labeling_zero_twice(self) -> None:
        """通常方向閾値０は正負を判定し、変化なしを上昇・下落へ含めない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        zero = self._run(direction_threshold_pct="0")
        self.assertNotIn("上がる", zero["correct_candidates"])
        self.assertNotIn("下がる", zero["correct_candidates"])
        self.bars[10] = replace(self.bars[10], high=101)
        self.bars[-1] = replace(self.bars[-1], high=100.1, close=100.1)
        value = self._run(direction_threshold_pct="0")
        self.assertEqual(value["target_status"], "対象")
        self.assertIn("上がる", value["correct_candidates"])

    def test_distinct_bars_confirm_both_reversals_and_preserve_class_overlap(self) -> None:
        """異なる時刻の上昇・下落・上昇では両反転を確定し、一意化しない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], high=101, close=101)
        self.bars[20] = replace(self.bars[20], low=99, close=99)
        value = self._run()
        self.assertIn("上がってから下がる", value["correct_candidates"])
        self.assertIn("下がってから上がる", value["correct_candidates"])
        self.assertNotIn("もみ合い", value["correct_candidates"])

    def test_intrabar_high_low_order_remains_uncertain(self) -> None:
        """同一足内の高安順序だけで成立する反転を確定正解にしない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.4, high=100.4, low=100.4, close=100.4) for bar in self.bars]
        bars[-1] = replace(bars[-1], high=100.6, low=100, close=100.4)
        value = self._run(bars)
        self.assertNotIn("上がってから下がる", value["correct_candidates"])
        self.assertIn("上がってから下がる", value["uncertain_candidates"])
        scored = score_prediction("上がってから下がる", value)
        self.assertEqual(scored["status"], "未確定")
        self.assertFalse(scored["correct_lower"])
        self.assertTrue(scored["correct_upper"])

    def test_intrabar_order_branches_do_not_leak_flags_into_each_other(self) -> None:
        """高先行候補の反転成立を低先行候補へ誤って引き継がない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.4, high=100.4, low=100.4, close=100.4) for bar in self.bars]
        bars[10] = replace(bars[10], high=100.6, low=100, close=100.4)
        value = self._run(bars)
        self.assertIn("上がってから下がる", value["uncertain_candidates"])
        self.assertNotIn("上がってから下がる", value["correct_candidates"])

    def test_close_can_make_intrabar_reversal_certain_in_both_orders(self) -> None:
        """高値到達後に終値が十分低ければ高安順序によらず反転を確定する。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], high=100.5)
        self.assertIn("上がってから下がる", self._run()["correct_candidates"])

    def test_optional_reference_cross_requires_final_cross_including_equality(self) -> None:
        """終点の前終値跨ぎ指定を、ピークからの反転幅とは別に扱う。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.2, high=100.2, low=100.2, close=100.2) for bar in self.bars]
        bars[10] = replace(bars[10], high=101, close=101)
        self.assertIn("上がってから下がる", self._run(bars)["correct_candidates"])
        crossed = self._run(bars, reversal_requires_reference_cross=True)
        self.assertNotIn("上がってから下がる", crossed["correct_candidates"] + crossed["uncertain_candidates"])
        bars[-1] = replace(bars[-1], low=100, close=100)
        self.assertIn("上がってから下がる", self._run(bars, reversal_requires_reference_cross=True)["correct_candidates"])

    def test_missing_path_does_not_assert_flat_or_absence_of_reversals(self) -> None:
        """欠測中の高安や反転を推測せず、対象不明も分母へ残す。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        value = self._run(self.bars[:10] + self.bars[11:])
        self.assertEqual(value["quality"]["missing_rows"], 1)
        self.assertEqual(value["target_status"], "未確定")
        self.assertNotIn("もみ合い", value["correct_candidates"])
        self.assertIn("もみ合い", value["uncertain_candidates"])
        for label in ("上がってから下がる", "下がってから上がる"):
            self.assertIn(label, value["uncertain_candidates"])
        score = score_prediction("もみ合い", value)
        self.assertTrue(score["denominator"])
        self.assertFalse(score["correct_lower"])
        self.assertTrue(score["correct_upper"])

    def test_known_reversal_survives_unrelated_missing_bar(self) -> None:
        """欠測があっても観測時刻順で既に成立した反転は確定として残す。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], high=101)
        value = self._run(self.bars[:20] + self.bars[21:])
        self.assertIn("上がってから下がる", value["correct_candidates"])
        self.assertEqual(value["target_status"], "対象")

    def test_unclosed_bar_prices_are_not_interpreted(self) -> None:
        """未確定足の異常価格を読まず、足の不足として未確定へ移す。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], high="未確定の将来値", closed=False)
        value = self._run()
        self.assertEqual(value["quality"]["unclosed_rows"], 1)
        self.assertEqual(value["target_status"], "未確定")

    def test_incomplete_volume_is_disclosed_separately_from_price_closure(self) -> None:
        """出来高不足だけを価格足未確定に読み替えない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[10] = replace(self.bars[10], volume_complete=False)
        value = self._run()
        self.assertEqual(value["quality"]["incomplete_volume_rows"], 1)
        self.assertEqual(value["quality"]["missing_rows"], 0)
        self.assertIn("もみ合い", value["correct_candidates"])

    def test_endpoint_proxy_is_explicit_and_late_missing_endpoint_stays_unknown(self) -> None:
        """許容遅延内の観測終値代用と、遅延超過で終点不明を分ける。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[-6] = replace(self.bars[-6], high=101, close=101)
        value = self._run(self.bars[:-5], target_basis="終点")
        self.assertEqual(value["metrics"]["actual_pct"], 1)
        self.assertEqual(value["quality"]["endpoint_lag_minutes"], 5)
        self.assertEqual(value["quality"]["endpoint_status"], "最終観測終値の代用")
        late = self._run(self.bars[:-11], target_basis="終点")
        self.assertIsNone(late["metrics"]["actual_pct"])
        self.assertEqual(late["target_status"], "未確定")
        self.assertIn("上がる", late["uncertain_candidates"])

    def test_unanswered_is_wrong_even_when_target_or_path_is_unknown(self) -> None:
        """無回答を確定対象でも対象不明でも正解上限へ加えない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        for value in (self._run([]), self._run([replace(bar, high=101, close=101) for bar in self.bars])):
            score = score_prediction(None, value)
            self.assertEqual(score["status"], "不正解")
            self.assertTrue(score["denominator"])
            self.assertFalse(score["correct_lower"])
            self.assertFalse(score["correct_upper"])

    def test_target_outside_is_visible_and_unknown_answer_is_rejected(self) -> None:
        """対象外の結果を明示し、回答語彙の後付け追加を認めない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        value = self._run()
        score = score_prediction("もみ合い", value)
        self.assertEqual(score["status"], "対象外")
        self.assertFalse(score["denominator"])
        with self.assertRaises(ValueError):
            score_prediction("横ばい", value)

    def test_mixed_symbol_session_contract_or_duplicate_is_rejected(self) -> None:
        """別先物・別場・別限月や重複足で結果を作らない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        for changed in (replace(self.bars[0], symbol="TOPIX"), replace(self.bars[0], session="2026-10-02_Day"),
                        replace(self.bars[0], contract="202703")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self._run([changed, *self.bars[1:]])
        with self.assertRaises(ValueError):
            self._run([self.bars[0], *self.bars])

    def test_reference_and_target_time_mismatch_is_rejected(self) -> None:
        """現物や当場を基準にせず、基準遅延や対象終点の相違を拒否する。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        for changed in (replace(self.reference, session="2026-09-30_Day"), replace(self.reference, closed=False),
                        replace(self.reference, time=self.reference.time - timedelta(minutes=11),
                                end=self.reference.end - timedelta(minutes=11))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                evaluate_path(changed, self.bars, "2026-10-01_Day", self.end, self.contract)
        with self.assertRaises(ValueError):
            evaluate_path(self.reference, self.bars, "2026-10-01_Day", self.end - timedelta(minutes=1), self.contract)

    def test_nonfinite_bad_ohlc_wrong_grid_or_future_bar_is_rejected(self) -> None:
        """境界比較前に異常値・不整合な足・セッション後の足を拒否する。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        for changed in (replace(self.bars[0], high=float("inf")), replace(self.bars[0], low=101),
                        replace(self.bars[0], time=self.start + timedelta(seconds=30), end=self.start + timedelta(seconds=90)),
                        replace(self.bars[-1], end=self.end + timedelta(minutes=1))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self._run([changed])

    def test_downward_case_and_empty_path_expose_all_uncertainty(self) -> None:
        """下落0.5％も等号で対象とし、価格なしを対象外へ落とさない。

        引数: self (FiveClassTests) はテスト。
        返り値: None。
        """
        self.bars[-1] = replace(self.bars[-1], low=99.5, close=99.5)
        value = self._run()
        self.assertEqual(score_prediction("下がる", value)["status"], "正解")
        empty = self._run([])
        self.assertEqual(empty["target_status"], "未確定")
        self.assertEqual(empty["correct_candidates"], [])
        self.assertEqual(set(empty["uncertain_candidates"]), set(CLASSES))


class PriorityClassTests(unittest.TestCase):
    """もみ合い優先と最初の到達方向を固定する新規約を確認する。"""

    def setUp(self) -> None:
        """一般方式と同じ合成足生成だけを再利用する。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.sample = FiveClassTests()
        self.sample.setUp()

    def _run(self, bars: list[Candle] | None = None, **settings: object) -> dict:
        """もみ合い優先方式で合成入力だけを評価する。

        引数: self (PriorityClassTests) はテスト、bars (list[Candle] | None) は置換足、settings (object) は明示的契約変更。
        返り値: dict は一意正解または未確定候補。
        """
        return evaluate_priority_path(self.sample.reference, self.sample.bars if bars is None else bars,
                                      "2026-10-01_Day", self.sample.end, {**self.sample.contract, **settings})

    def _fixed(self, value: float) -> list[Candle]:
        """全観測価格が同じ合成場を作る。

        引数: self (PriorityClassTests) はテスト、value (float) は固定価格。
        返り値: list[Candle] は全420分の足。
        """
        return [replace(bar, open=value, high=value, low=value, close=value) for bar in self.sample.bars]

    # ----------------------------------------

    def test_flat_has_priority_over_both_reversals_and_direction_boundary(self) -> None:
        """もみ合いの条件成立時は他の経路・終点候補を正解に残さない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        bars = self.sample.bars
        bars[1] = replace(bars[1], high=100.5, low=99.5)
        bars[-1] = replace(bars[-1], high=100.5, close=100.5)
        value = self._run()
        self.assertEqual(value["correct_candidates"], ["もみ合い"])
        self.assertEqual(value["uncertain_candidates"], [])
        self.assertEqual(score_prediction("上がる", value)["status"], "不正解")
        self.assertEqual(score_prediction("上がってから下がる", value)["status"], "不正解")

    def test_unknown_flat_suppresses_otherwise_confirmed_direction(self) -> None:
        """欠測でもみ合いを否定できない時に上がるを確定正解にしない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        bars = self._fixed(100.5)
        value = self._run(bars[:10] + bars[11:])
        self.assertEqual(value["flat_status"], "未確定")
        self.assertEqual(value["correct_candidates"], [])
        self.assertEqual(set(value["uncertain_candidates"]), {"もみ合い", "上がる", "上がってから下がる"})
        self.assertEqual(score_prediction("上がる", value)["status"], "未確定")

    def test_first_up_then_down_is_not_replaced_by_later_down_then_up(self) -> None:
        """複数往復でも最初に上へ到達した経路の反転分類を維持する。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.sample.bars[10] = replace(self.sample.bars[10], high=101, close=101)
        self.sample.bars[20] = replace(self.sample.bars[20], low=99, close=99)
        value = self._run()
        self.assertEqual(value["correct_candidates"], ["上がってから下がる"])
        self.assertEqual(value["uncertain_candidates"], [])
        self.assertEqual(score_prediction("下がってから上がる", value)["status"], "不正解")

    def test_first_down_then_up_is_not_replaced_by_later_up_then_down(self) -> None:
        """初回が下なら、後で上から下へ動いても下落後上昇のまま扱う。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.sample.bars[10] = replace(self.sample.bars[10], low=99, close=99)
        self.sample.bars[20] = replace(self.sample.bars[20], high=101, close=101)
        value = self._run()
        self.assertEqual(value["correct_candidates"], ["下がってから上がる"])

    def test_same_bar_first_arrival_order_remains_two_uncertain_labels(self) -> None:
        """同一足で上下へ到達した順序を勝手に決めて一つへ採点しない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.sample.bars[10] = replace(self.sample.bars[10], high=100.6, low=99.4)
        value = self._run()
        self.assertEqual(value["flat_status"], "不成立")
        self.assertEqual(value["correct_candidates"], [])
        self.assertEqual(set(value["uncertain_candidates"]), {"上がってから下がる", "下がってから上がる"})

    def test_no_reversal_uses_first_direction_instead_of_terminal_threshold(self) -> None:
        """通常上昇は初回上到達後に反転しない経路として一意に扱う。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        value = self._run(self._fixed(100.6))
        self.assertEqual(value["correct_candidates"], ["上がる"])
        self.assertEqual(value["uncertain_candidates"], [])
        changed_legacy_threshold = self._run(self._fixed(100.6), direction_threshold_pct="50")
        self.assertEqual(changed_legacy_threshold["correct_candidates"], ["上がる"])

    def test_no_reversal_down_is_unique(self) -> None:
        """初回下到達から反転しない場合は下がる一種類にする。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.assertEqual(self._run(self._fixed(99.4))["correct_candidates"], ["下がる"])

    def test_reversal_half_percent_boundary_is_included_without_final_cross(self) -> None:
        """前終値を跨がなくてもピークから0.5％戻れば反転とする。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        bars = self._fixed(100.7)
        bars[10] = replace(bars[10], high=101.2, close=101.2)
        value = self._run(bars)
        self.assertEqual(value["correct_candidates"], ["上がってから下がる"])
        self.assertGreater(value["metrics"]["actual_pct"], 0)

    def test_no_arrival_remains_outside_target_with_flat_diagnostic(self) -> None:
        """0.5％未到達の場は対象外を明示し、もみ合いの補助判定を保存する。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        value = self._run()
        self.assertEqual(value["target_status"], "対象外")
        self.assertEqual(value["correct_candidates"], ["もみ合い"])
        self.assertFalse(score_prediction("もみ合い", value)["denominator"])

    def test_missing_before_first_arrival_cannot_confirm_first_direction(self) -> None:
        """先頭欠測中の先行下落を否定せず、その後の上昇で下落後上昇を候補に残す。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        value = self._run(self._fixed(100.6)[1:])
        self.assertEqual(value["correct_candidates"], [])
        self.assertEqual(set(value["uncertain_candidates"]), {"上がる", "上がってから下がる", "下がってから上がる"})
        self.assertNotIn("下がる", value["uncertain_candidates"])

    def test_missing_after_known_first_arrival_cannot_change_first_direction(self) -> None:
        """上への先行到達を観測済みなら後の欠測で初回下落候補を追加しない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        bars = self._fixed(100.6)
        value = self._run(bars[:10] + bars[11:])
        self.assertEqual(set(value["uncertain_candidates"]), {"上がる", "上がってから下がる"})

    def test_endpoint_proxy_gap_may_hide_reversal_but_not_change_first_direction(self) -> None:
        """末尾代理の後の経路不足を反転なしの確定へ使わない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        value = self._run(self._fixed(100.6)[:-5])
        self.assertEqual(value["correct_candidates"], [])
        self.assertEqual(set(value["uncertain_candidates"]), {"上がる", "上がってから下がる"})
        self.assertEqual(value["quality"]["endpoint_lag_minutes"], 5)

    def test_reversal_already_observed_remains_unique_after_missing_path(self) -> None:
        """もみ合い不成立・初回反転が既知なら後の欠測で確定分類を消さない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        self.sample.bars[10] = replace(self.sample.bars[10], high=101.1, close=101.1)
        value = self._run(self.sample.bars[:20] + self.sample.bars[21:])
        self.assertEqual(value["correct_candidates"], ["上がってから下がる"])
        self.assertEqual(score_prediction("上がってから下がる", value)["status"], "正解")

    def test_unknown_path_and_unanswered_still_count_as_wrong_lower_bound(self) -> None:
        """全経路不明でも無回答を精度下限・上限の正解へ加えない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        value = self._run([])
        self.assertEqual(set(value["uncertain_candidates"]), set(CLASSES))
        score = score_prediction(None, value)
        self.assertTrue(score["denominator"])
        self.assertFalse(score["correct_upper"])

    def test_new_policy_rejects_incompatible_contract(self) -> None:
        """終点対象・初回到達幅相違・終点跨ぎ必須を新方式へ混ぜない。

        引数: self (PriorityClassTests) はテスト。
        返り値: None。
        """
        for changed in ({"target_basis": "終点"}, {"initial_excursion_pct": "1"},
                        {"reversal_requires_reference_cross": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self._run(**changed)


if __name__ == "__main__":
    unittest.main()
