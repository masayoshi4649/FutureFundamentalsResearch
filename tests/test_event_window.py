"""実ニュースや実価格を開かず、臨時材料前の別評価を合成足で確認する。"""

import copy
import unittest
from dataclasses import replace
from datetime import timedelta

from fundamentals.common import timestamp
from fundamentals.prices import Candle
from research_support.event_window import evaluate_event_window, select_event
from research_support.five_class import CLASSES, evaluate_priority_path, score_prediction


class EventWindowTests(unittest.TestCase):
    """予定材料と臨時材料、時点、足の安全な切断、未確定を検証する。"""

    def setUp(self) -> None:
        """前先物終値100と全分の合成足・証跡付き臨時材料を用意する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        self.as_of = timestamp("2026-10-01T07:45:00+09:00")
        self.start, self.end = timestamp("2026-10-01T08:45:00+09:00"), timestamp("2026-10-01T15:45:00+09:00")
        self.session = "2026-10-01_Day"
        self.reference = Candle(timestamp("2026-10-01T05:59:00+09:00"), timestamp("2026-10-01T06:00:00+09:00"),
                                100, 100, 100, 100, 1, "2026-09-30_Night", "NK225", "202612")
        self.bars = [Candle(self.start + timedelta(minutes=index), self.start + timedelta(minutes=index + 1),
                            100, 100, 100, 100, 1, self.session, "NK225", "202612") for index in range(420)]
        self.contract = {"target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
                         "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5",
                         "reversal_requires_reference_cross": False, "range_threshold_pct": "1",
                         "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
                         "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10}
        self.event = {"event_id": "合成材料１", "assessment_as_of": self.as_of.isoformat(),
                      "first_public_at": "2026-10-01T09:00:30+09:00", "article_ids": ["合成記事１"],
                      "source_revision_hash": "a" * 64, "decision": "採用", "schedule_status": "臨時・予見不能",
                      "affected_symbols": ["NK225", "TOPIX"], "novelty_evidence": "予測時点には公表されていない。",
                      "schedule_evidence": "当該締切の予定表と事前予告を点検済み。",
                      "impact_evidence": "独立した材料認定者が影響根拠を記録。",
                      "timing_evidence": {"初報": "一次公表の時計を確認。"}, "judge_id": "合成人定者"}

    def _run(self, events: list[dict] | None = None, bars: list[Candle] | None = None, **settings: object) -> dict:
        """合成入力だけで当該窓を評価する。

        引数: self (EventWindowTests) はテスト、events (list[dict] | None) は材料台帳、
            bars (list[Candle] | None) は価格足、settings (object) は契約の置換。
        返り値: dict は窓の評価。
        """
        return evaluate_event_window(self.reference, self.bars if bars is None else bars, self.session, self.as_of,
                                     [self.event] if events is None else events, {**self.contract, **settings})

    def _select(self, events: list[dict], symbol: str = "NK225") -> dict | None:
        """当該締切・先物に対する材料選択を行う。

        引数: self (EventWindowTests) はテスト、events (list[dict]) は台帳、symbol (str) は先物。
        返り値: dict | None は選択材料。
        """
        return select_event(events, self.as_of, symbol, self.end)

    # ----------------------------------------

    def test_no_event_preserves_existing_full_session_evaluation(self) -> None:
        """材料がない場合の既存各項目を一切変えない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        expected = evaluate_priority_path(self.reference, self.bars, self.session, self.end, self.contract)
        result = self._run([])
        window = result.pop("evaluation_window")
        self.assertEqual(result, expected)
        self.assertEqual(window["status"], "全セッション")
        self.assertEqual(window["label_available_at"], self.end.isoformat())
        self.assertFalse(window["adjudication_unknown"])

    def test_planned_or_unknown_schedule_cannot_be_adopted(self) -> None:
        """予定指標の意外な結果や予定不明を理由に窓を短縮しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for status in ("予定済み", "不明"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self._select([{**self.event, "schedule_status": status}])
        planned = {**self.event, "decision": "棄却", "schedule_status": "予定済み"}
        self.assertEqual(self._run([planned])["evaluation_window"]["status"], "全セッション")

    def test_known_news_and_outside_session_events_are_not_used(self) -> None:
        """締切同時以前と終了同時以降の材料は適用しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for public in (self.as_of, self.as_of - timedelta(minutes=1), self.end, self.end + timedelta(seconds=1)):
            with self.subTest(public=public):
                self.assertIsNone(self._select([{**self.event, "first_public_at": public.isoformat()}]))

    def test_assessment_is_specific_to_forecast_cutoff_and_symbol(self) -> None:
        """別締切や他先物の臨時認定を流用しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        self.assertIsNone(self._select([{**self.event, "assessment_as_of": "2026-10-01T07:44:00+09:00"}]))
        self.assertIsNone(self._select([{**self.event, "affected_symbols": ["TOPIX"]}]))
        self.assertEqual(self._select([{**self.event, "affected_symbols": ["TOPIX"]}], "TOPIX")["event_id"], "合成材料１")

    def test_first_event_is_selected_by_clock_then_identifier(self) -> None:
        """後の材料へ差し替えて採点に有利な終点を選ばない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        first = {**self.event, "event_id": "Ａ", "first_public_at": "2026-10-01T08:59:00+09:00"}
        tied = {**first, "event_id": "Ｂ"}
        for events in ([self.event, tied, first], [first, self.event, tied]):
            self.assertEqual(self._select(events)["event_id"], "Ａ")
        selected = self._select([first])
        selected["article_ids"].append("別記事")
        self.assertEqual(first["article_ids"], ["合成記事１"])

    def test_missing_evidence_revision_clock_and_duplicate_adoption_are_rejected(self) -> None:
        """未証明の認定や不明時刻へ架空の値を入れた台帳を受理しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        variants = [{**self.event, "source_revision_hash": "未固定"}, {**self.event, "impact_evidence": " "},
                    {**self.event, "schedule_evidence": {"本文": ""}}, {**self.event, "article_ids": []},
                    {**self.event, "first_public_at": None},
                    {key: value for key, value in self.event.items() if key != "assessment_as_of"}]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._select([value])
        with self.assertRaises(ValueError):
            self._select([self.event, copy.deepcopy(self.event)])

    def test_event_clock_is_floored_without_interpreting_crossing_or_later_prices(self) -> None:
        """公表分を跨ぐ足と後発の異常価格・確定指定は解釈しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = self.bars[:15] + [replace(bar, high="後発の異常値", low=None, closed="未解釈") for bar in self.bars[15:]]
        value = self._run(bars=bars)
        self.assertEqual(value["target_end"], "2026-10-01T09:00:00+09:00")
        self.assertEqual(value["quality"]["observed_rows"], 15)
        window = value["evaluation_window"]
        self.assertEqual(window["source_event_time"], "2026-10-01T09:00:30+09:00")
        self.assertEqual(window["scheduled_full_end"], self.end.isoformat())
        self.assertEqual(window["effective_quote_time"], "2026-10-01T09:00:00+09:00")
        self.assertEqual(window["event_to_nominal_seconds"], 30)
        self.assertEqual(window["event_to_quote_seconds"], 30)
        self.assertEqual(window["label_available_at"], "2026-10-01T09:00:30+09:00")

    def test_full_session_also_filters_future_numeric_values_before_interpretation(self) -> None:
        """公式終了後の異常足を結果の価格ハッシュや高安へ混ぜない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        extra = replace(self.bars[-1], time=self.end, end=self.end + timedelta(minutes=1), high="未来値", closed="未来指定")
        expected = self._run([])
        result = self._run([], [*self.bars, extra])
        self.assertEqual(result["path_hash"], expected["path_hash"])
        self.assertEqual(result["metrics"], expected["metrics"])
        self.assertEqual(result["evaluation_window"]["excluded_after_window_rows"], 1)

    def test_preopen_expiration_has_no_price_score_and_remains_planned(self) -> None:
        """開始以前の臨時材料を無回答の不正解へ置き換えず別件数に残す。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for public in ("2026-10-01T08:30:15+09:00", self.start.isoformat()):
            result = self._run([{**self.event, "first_public_at": public}])
            self.assertEqual(result["evaluation_window"]["status"], "開始前失効")
            self.assertFalse(result["evaluation_window"]["scoreable"])
            self.assertTrue(result["evaluation_window"]["planned_count"])
            self.assertEqual(result["correct_candidates"], [])
            self.assertEqual(result["uncertain_candidates"], [])
            self.assertEqual(result["quality"]["observed_rows"], 0)
            self.assertIsNone(result["metrics"]["actual_pct"])

    def test_event_in_first_minute_is_not_preopen_and_keeps_unknown_denominator(self) -> None:
        """開始直後に完全足がない場合は未確定の価格期間として保持する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        result = self._run([{**self.event, "first_public_at": "2026-10-01T08:45:30+09:00"}])
        self.assertEqual(result["evaluation_window"]["status"], "材料前まで")
        self.assertTrue(result["evaluation_window"]["scoreable"])
        self.assertEqual(result["target_status"], "未確定")
        self.assertEqual(set(result["uncertain_candidates"]), set(CLASSES))
        self.assertTrue(score_prediction(None, result)["denominator"])
        self.assertFalse(score_prediction(None, result)["correct_upper"])

    def test_prefix_recovery_is_measured_from_previous_futures_close(self) -> None:
        """夜間下落後の回復も同じ先物前終値100に対する上昇で判定する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.6, high=100.6, low=100.6, close=100.6) for bar in self.bars]
        result = self._run(bars=bars)
        self.assertEqual(result["correct_candidates"], ["上がる"])
        self.assertEqual(result["metrics"]["actual_pct"], 0.6)
        self.assertEqual(result["metrics"]["reference_price"], 100)

    def test_flat_priority_uses_prefix_high_low_not_full_session(self) -> None:
        """材料前に幅1％・終点0.5％以内なら後発大幅変動にかかわらずもみ合い。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = self.bars[:]
        bars[1] = replace(bars[1], high=100.5, low=99.5)
        bars[14] = replace(bars[14], high=100.5, close=100.5)
        bars[20] = replace(bars[20], high=110, close=110)
        result = self._run(bars=bars)
        self.assertEqual(result["correct_candidates"], ["もみ合い"])
        self.assertEqual(result["metrics"]["session_range_pct"], 1)
        self.assertEqual(result["target_status"], "対象")

    def test_reversal_without_crossing_reference_is_preserved(self) -> None:
        """反転0.5％を材料前に観測したら前終値を越えずとも反転クラスにする。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.7, high=100.7, low=100.7, close=100.7) for bar in self.bars]
        bars[10] = replace(bars[10], high=101.2, close=101.2)
        result = self._run(bars=bars)
        self.assertEqual(result["correct_candidates"], ["上がってから下がる"])
        self.assertGreater(result["metrics"]["actual_pct"], 0)

    def test_intrabar_order_and_missing_path_remain_uncertain(self) -> None:
        """材料前でも同足順序や欠測を使って一意の正解を作らない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = self.bars[:]
        bars[10] = replace(bars[10], high=100.6, low=99.4)
        result = self._run(bars=bars)
        self.assertEqual(result["correct_candidates"], [])
        self.assertEqual(set(result["uncertain_candidates"]), {"上がってから下がる", "下がってから上がる"})
        missing = self._run(bars=self.bars[:10] + self.bars[11:])
        self.assertEqual(missing["target_status"], "未確定")
        self.assertEqual(missing["quality"]["missing_rows"], 1)
        self.assertFalse(score_prediction("もみ合い", missing)["correct_lower"])

    def test_prefix_endpoint_lag_is_relative_to_window_end(self) -> None:
        """材料前の末尾観測と分切下げ時刻の遅延を公式終了遅延と分ける。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        result = self._run(bars=self.bars[:10])
        self.assertEqual(result["quality"]["endpoint_lag_minutes"], 5)
        self.assertTrue(result["quality"]["endpoint_available"])
        self.assertEqual(result["evaluation_window"]["quote_to_nominal_seconds"], 300)
        self.assertEqual(result["evaluation_window"]["event_to_quote_seconds"], 330)
        unknown = self._run(bars=self.bars[:4])
        self.assertIsNone(unknown["metrics"]["quote_time"])
        self.assertFalse(unknown["quality"]["endpoint_available"])

    def test_five_minute_partial_interval_is_a_missing_observation(self) -> None:
        """公表の分切下げが５分格子外なら跨ぐ足を除外して残余を不明にする。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        bars = [replace(self.bars[index], end=self.bars[index].time + timedelta(minutes=5)) for index in range(0, 420, 5)]
        event = {**self.event, "first_public_at": "2026-10-01T09:03:10+09:00"}
        result = self._run([event], bars, interval_minutes=5)
        self.assertEqual(result["target_end"], "2026-10-01T09:03:00+09:00")
        self.assertEqual(result["quality"]["observed_rows"], 3)
        self.assertEqual(result["quality"]["expected_rows"], 4)
        self.assertEqual(result["quality"]["missing_rows"], 1)
        self.assertEqual(result["quality"]["partial_interval_seconds"], 180)
        self.assertEqual(result["target_status"], "未確定")

    def test_actual_clock_maturity_is_explicitly_left_to_caller(self) -> None:
        """合成将来日付にも適用できる純粋評価と実時計満了責務を分離する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        result = self._run()
        self.assertEqual(result["evaluation_window"]["maturity_required_at"], self.event["first_public_at"])
        self.assertIn("呼出側", result["evaluation_window"]["actual_clock_maturity_check"])

    def test_pending_earlier_material_keeps_label_unknown_and_denominator(self) -> None:
        """先行する臨時材料の認定保留を材料なしの確定正解として扱わない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        pending = {**self.event, "event_id": "保留材料", "decision": "保留", "schedule_status": "不明",
                   "first_public_at": "2026-10-01T08:55:00+09:00"}
        for events in ([pending], [self.event, pending]):
            with self.subTest(events=events):
                result = self._run(events)
                self.assertTrue(result["evaluation_window"]["adjudication_unknown"])
                self.assertEqual(result["target_status"], "未確定")
                self.assertEqual(result["correct_candidates"], [])
                self.assertEqual(set(result["uncertain_candidates"]), set(CLASSES))
                self.assertEqual(result["evaluation_window"]["pending_events"][0]["event_id"], "保留材料")
                self.assertTrue(score_prediction("もみ合い", result)["denominator"])
                self.assertFalse(score_prediction("もみ合い", result)["correct_lower"])
                self.assertFalse(score_prediction(None, result)["correct_upper"])
                self.assertIn("provisional_evaluation", result)

    def test_later_or_planned_pending_and_rejected_events_do_not_block_prefix(self) -> None:
        """採用打切り後の保留や予定済み・棄却の材料は当該窓を不明にしない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for pending in ({**self.event, "event_id": "後の保留", "decision": "保留", "first_public_at": "2026-10-01T09:01:00+09:00"},
                        {**self.event, "event_id": "予定保留", "decision": "保留", "schedule_status": "予定済み",
                         "first_public_at": "2026-10-01T08:55:00+09:00"},
                        {**self.event, "event_id": "棄却", "decision": "棄却", "first_public_at": "2026-10-01T08:55:00+09:00"}):
            with self.subTest(pending=pending):
                result = self._run([pending, self.event])
                self.assertFalse(result["evaluation_window"]["adjudication_unknown"])
                self.assertEqual(result["evaluation_window"]["pending_events"], [])

    def test_invalid_prefix_prices_reference_and_timing_are_rejected(self) -> None:
        """適用窓内の異常足・限月違いと開始後締切を拒否する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for bad in (replace(self.bars[0], high="不正な既知価格"), replace(self.bars[0], contract="202703"),
                    replace(self.bars[0], symbol="TOPIX")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._run(bars=[bad, *self.bars[1:]])
        with self.assertRaises(ValueError):
            evaluate_event_window(self.reference, self.bars, self.session, self.start, [], self.contract)

    def _unknown_public(self, upper: str, lower: str | None = None) -> dict:
        """記事公開上限だけ分かる、初報時計nullの合成保留を作る。

        引数: self (EventWindowTests) はテスト、upper (str) は記事公開上限、lower (str | None) は任意の公表下限。
        返り値: dict は材料時計を捏造しない保留レコード。
        """
        timing = {"article_public_upper_bound": upper, "根拠": "記事は材料の初報時計そのものを証明していない。"}
        if lower is not None:
            timing["material_public_lower_bound"] = lower
        return {**copy.deepcopy(self.event), "event_id": "時計未確認の保留", "decision": "保留", "schedule_status": "不明",
                "first_public_at": None, "timing_evidence": timing}

    def test_null_first_public_is_pending_without_using_article_clock_as_censor(self) -> None:
        """初報時計不明を記事時刻で打ち切らず、窓自体の認定不明として残す。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        event = self._unknown_public("2026-10-01T09:30:00+09:00")
        original = copy.deepcopy(event)
        result = self._run([event])
        window = result["evaluation_window"]
        self.assertIsNone(window["source_event_time"])
        self.assertIsNone(window["selected_event"])
        self.assertEqual(window["nominal_window_end"], self.end.isoformat())
        self.assertEqual(window["pending_events"], [original])
        self.assertIsNone(window["pending_events"][0]["first_public_at"])
        self.assertEqual(result["target_status"], "未確定")
        self.assertEqual(result["correct_candidates"], [])
        self.assertEqual(set(result["uncertain_candidates"]), set(CLASSES))
        self.assertEqual(event, original)

    def test_null_upper_at_or_before_forecast_is_known_and_not_pending(self) -> None:
        """記事上限が締切同時以前なら、材料は既知として不明窓へ含めない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for upper in (self.as_of, self.as_of - timedelta(seconds=1)):
            with self.subTest(upper=upper):
                result = self._run([self._unknown_public(upper.isoformat())])
                self.assertFalse(result["evaluation_window"]["adjudication_unknown"])
                self.assertEqual(result["evaluation_window"]["pending_events"], [])

    def test_null_upper_at_or_after_full_end_does_not_rule_out_intraday_publication(self) -> None:
        """終了同時・終了後の記事上限しかなくても、場中初報の可能性を消さない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for upper in (self.end, self.end + timedelta(hours=4)):
            with self.subTest(upper=upper):
                result = self._run([self._unknown_public(upper.isoformat())])
                self.assertTrue(result["evaluation_window"]["adjudication_unknown"])
                self.assertEqual(result["target_status"], "未確定")
                self.assertEqual(result["correct_candidates"], [])

    def test_null_lower_boundary_is_checked_against_earlier_adopted_event(self) -> None:
        """先行採用の公表以前に初報があり得る場合だけ、その窓を認定不明にする。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        limit = timestamp(self.event["first_public_at"])
        for lower, expected in ((limit - timedelta(seconds=1), True), (limit, False), (limit + timedelta(seconds=1), False)):
            with self.subTest(lower=lower):
                pending = self._unknown_public("2026-10-01T18:00:00+09:00", lower.isoformat())
                result = self._run([pending, self.event])
                self.assertEqual(result["evaluation_window"]["adjudication_unknown"], expected)
                self.assertEqual(result["evaluation_window"]["selected_event"]["event_id"], self.event["event_id"])

    def test_null_lower_at_or_after_full_end_excludes_only_with_that_evidence(self) -> None:
        """公式終了以前に出ていない下限根拠がある場合だけ、未確認初報を除外する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        for lower in (self.end, self.end + timedelta(seconds=1)):
            pending = self._unknown_public("2026-10-01T18:00:00+09:00", lower.isoformat())
            self.assertFalse(self._run([pending])["evaluation_window"]["adjudication_unknown"])

    def test_null_bounds_require_upper_and_lower_not_above_upper(self) -> None:
        """初報nullの上限欠落・下限上限逆転・採用への格上げを拒否する。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        invalid = self._unknown_public("2026-10-01T09:30:00+09:00")
        invalid["timing_evidence"] = {"初報未確認": "上限も未確認"}
        inverted = self._unknown_public("2026-10-01T09:30:00+09:00", "2026-10-01T09:31:00+09:00")
        adopted = {**self._unknown_public("2026-10-01T09:30:00+09:00"), "decision": "採用", "schedule_status": "臨時・予見不能"}
        for event in (invalid, inverted, adopted):
            with self.subTest(event=event), self.assertRaises(ValueError):
                self._run([event])

    def test_null_planned_and_other_forecast_or_symbol_are_not_applicable(self) -> None:
        """初報不明でも予定済み・別締切・別先物の材料を当回へ流用しない。

        引数: self (EventWindowTests) はテスト。
        返り値: None。
        """
        base = self._unknown_public("2026-10-01T18:00:00+09:00")
        for changes in ({"schedule_status": "予定済み"}, {"assessment_as_of": "2026-10-01T07:44:00+09:00"},
                        {"affected_symbols": ["TOPIX"]}):
            with self.subTest(changes=changes):
                result = self._run([{**base, **changes}])
                self.assertFalse(result["evaluation_window"]["adjudication_unknown"])


if __name__ == "__main__":
    unittest.main()
