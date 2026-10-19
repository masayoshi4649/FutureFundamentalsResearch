"""臨時材料前の教師窓と旧推定式の同値性を、実相場とは無関係の合成履歴で確認する。"""

import copy
import hashlib
import unittest
from pathlib import Path

import test_five_class_model as fixture_module
import test_five_class as price_fixture_module
from fundamentals.common import timestamp
from research_support.five_class import CLASSES
from research_support.five_class_model import predict_five
from research_support.five_event_model import BASE_MODEL_SHA256, ESTIMATION_FORMULA_SHA256, predict_event_five
from research_support.event_window import evaluate_event_window


class FiveEventModelTests(unittest.TestCase):
    """公表時計と切下げ時計を区別し、使えない教師の目的値だけを抑止する。"""

    def setUp(self) -> None:
        """旧版の合成特徴生成だけを再利用し、全場窓を追加する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        self.fixture = fixture_module.FiveClassModelTests()
        self.fixture.setUp()
        self.current, self.as_of = self.fixture.current, self.fixture.as_of

    def _rows(self, count: int = 12, symbol: str = "NK225") -> list[dict]:
        """旧版と同じ候補・特徴に、材料なしの公式終点窓を付ける。

        引数: self (FiveEventModelTests) はテスト、count (int) は件数、symbol (str) は先物。
        返り値: list[dict] は公表時計を捏造しない全場教師。
        """
        rows = self.fixture._rows(count, symbol)
        for row in rows:
            end = row["label_end"]
            row["five_class_evaluation"]["evaluation_window"] = {
                "version": "臨時材料前１", "status": "全セッション", "as_of": row["as_of"],
                "scheduled_full_end": end, "source_event_time": None, "nominal_window_end": end,
                "scoreable": True, "planned_count": True, "selected_event": None, "pending_events": [],
                "adjudication_unknown": False, "label_available_at": end, "maturity_required_at": end}
        return rows

    def _event(self, row: dict, clock: str = "11:59:45", decision: str = "採用") -> dict:
        """元予測時点に対して認定した合成材料を作る。

        引数: self (FiveEventModelTests) はテスト、row (dict) は教師、clock (str) は同日の公表時分秒、
            decision (str) は採用または保留。
        返り値: dict は現実の記事を用いない、全必須根拠付き材料。
        """
        return {"event_id": "合成臨時材料", "assessment_as_of": row["as_of"],
                "first_public_at": row["as_of"][:10] + "T" + clock + "+09:00", "article_ids": ["合成記事"],
                "source_revision_hash": "a" * 64, "decision": decision,
                "schedule_status": "臨時・予見不能" if decision == "採用" else "不明", "affected_symbols": [row["symbol"]],
                "novelty_evidence": "合成の新規性根拠", "schedule_evidence": "合成の予定外根拠",
                "impact_evidence": "合成の影響根拠", "timing_evidence": "合成の公表時計根拠", "judge_id": "合成判定者"}

    def _prefix(self, row: dict, clock: str = "11:59:45", label: str = "もみ合い") -> None:
        """窓の前半だけを示す合成ラベルと、独立した公表時計を設定する。

        引数: self (FiveEventModelTests) はテスト、row (dict) は教師、clock (str) は公表時計、label (str) は合成の窓内分類。
        返り値: None。
        """
        event = self._event(row, clock)
        published = timestamp(event["first_public_at"])
        nominal = published.replace(second=0, microsecond=0).isoformat()
        evaluation = row["five_class_evaluation"]
        evaluation.update(target_end=nominal, correct_candidates=[label], uncertain_candidates=[])
        evaluation["evaluation_window"].update(status="材料前まで", source_event_time=published.isoformat(),
            nominal_window_end=nominal, label_available_at=published.isoformat(), maturity_required_at=published.isoformat(),
            selected_event=event)
        row["label_end"] = published.isoformat()

    # ----------------------------------------

    def test_common_full_session_inputs_keep_old_coefficients_scores_and_scaler(self) -> None:
        """全場で同じ特徴・候補なら、両方式の旧版と数値結果を完全一致させる。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(16)
        for index, row in enumerate(rows):
            row["features"]["growth"] = None if index % 3 == 0 else float(index)
            if index % 4 == 0:
                self.fixture._candidates(row, [], ["下がる", "もみ合い"])
            elif index % 4 == 1:
                self.fixture._candidates(row, [], [])
        for mode in ("確定教師", "候補配分"):
            old = predict_five(self.current, rows, self.as_of, mode)
            new = predict_event_five(self.current, rows, self.as_of, mode)
            for key in ("direction", "class_scores", "scaler", "features", "weights", "targets", "coefficients",
                        "contributions", "effective_weight", "training_n", "positive_weight_n", "fitted_n", "label_counts"):
                with self.subTest(mode=mode, key=key):
                    self.assertEqual(old[key], new[key])

    def test_estimation_formula_block_is_byte_identical_to_frozen_base(self) -> None:
        """罰則・切片・重み付き方程式のブロックを旧版と同じハッシュで照合する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        original_path, event_path = Path("research_support/five_class_model.py"), Path("research_support/five_event_model.py")
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).hexdigest(), BASE_MODEL_SHA256)
        blocks = []
        for path in (original_path, event_path):
            source = path.read_text(encoding="utf-8")
            blocks.append(source[source.index("    size = len(expanded)\n"):])
        self.assertEqual(blocks[0], blocks[1])
        self.assertEqual(hashlib.sha256(blocks[1].encode("utf-8")).hexdigest(), ESTIMATION_FORMULA_SHA256)

    def test_prefix_label_uses_public_availability_and_does_not_recreate_terminal_class(self) -> None:
        """分切下げの終点と公表時計を分け、窓内の分類を終点騰落から作り直さない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        self._prefix(rows[0])
        rows[0]["actual_pct"] = object()
        rows[0]["five_class_evaluation"]["metrics"] = object()
        result = predict_event_five(self.current, rows, self.as_of, "確定教師")
        self.assertEqual(result["direction"], "もみ合い")
        self.assertEqual(result["last_label_end"], "2026-09-01T11:59:45+09:00")
        self.assertEqual(result["targets"][0]["もみ合い"], 1)

    def test_nominal_rounding_cannot_make_teacher_available_before_publication(self) -> None:
        """11:59:45公表を11:59:00から既知の教師と扱うことを拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        self._prefix(rows[0])
        rows[0]["label_end"] = rows[0]["five_class_evaluation"]["target_end"]
        with self.assertRaisesRegex(ValueError, "label_available_at"):
            predict_event_five(self.current, rows, self.as_of, "確定教師")

    def test_all_available_times_are_checked_before_features_and_other_symbol_filter(self) -> None:
        """他先物の未来利用可能時計も、先頭の壊れた特徴より前に拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows, other = self._rows(1), self._rows(1, "TOPIX")[0]
        rows[0]["features"] = object()
        other["five_class_evaluation"]["evaluation_window"]["label_available_at"] = self.as_of
        with self.assertRaisesRegex(ValueError, "利用可能.*数値解釈前"):
            predict_event_five(self.current, rows + [other], self.as_of, "候補配分")

    def test_preopen_expiration_has_zero_weight_and_stays_in_feature_pool(self) -> None:
        """開始前の失効を候補なし・行重み０として残し、前処理から除外しない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(2)
        self._prefix(rows[0], clock="08:15:30")
        evaluation = rows[0]["five_class_evaluation"]
        evaluation["correct_candidates"] = []
        evaluation["evaluation_window"].update(status="開始前失効", scoreable=False)
        rows[0]["features"]["growth"] = 100
        for mode in ("確定教師", "候補配分"):
            result = predict_event_five(self.current, rows, self.as_of, mode)
            self.assertEqual(result["training_n"], 2)
            self.assertEqual(result["weights"], [0.0, 1.0])
            self.assertEqual(result["scaler"]["growth"]["mean"], 50)
            self.assertEqual(result["label_exclusion_counts"]["開始前失効"], 1)

    def test_pending_material_is_not_a_uniform_five_class_teacher(self) -> None:
        """材料認定が不明な全５候補は候補配分でも０重みとし、欠測癖を教師にしない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        evaluation = rows[0]["five_class_evaluation"]
        evaluation.update(target_status="未確定", correct_candidates=[], uncertain_candidates=list(CLASSES))
        evaluation["evaluation_window"].update(adjudication_unknown=True,
            pending_events=[self._event(rows[0], clock="10:00:00", decision="保留")])
        for mode in ("確定教師", "候補配分"):
            result = predict_event_five(self.current, rows, self.as_of, mode)
            self.assertIsNone(result["direction"])
            self.assertEqual(result["weights"], [0.0])
            self.assertEqual(result["targets"], [dict.fromkeys(CLASSES, 0.0)])
            self.assertEqual(result["label_exclusion_counts"]["材料認定不明"], 1)

    def test_ordinary_prefix_path_uncertainty_still_allows_candidate_distribution(self) -> None:
        """材料認定が済んだ窓の価格経路不明は、従来の候補配分を許容する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        self._prefix(rows[0])
        self.fixture._candidates(rows[0], [], ["上がる", "上がってから下がる"])
        result = predict_event_five(self.current, rows, self.as_of, "候補配分")
        self.assertEqual(result["weights"], [0.5])
        self.assertEqual(result["targets"][0]["上がる"], 0.5)

    def test_scheduled_event_and_wrong_assessment_cutoff_cannot_censor_label(self) -> None:
        """予定済み発表や別の予測締切向け認定を使った打切りを拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        for key, value in (("schedule_status", "予定済み"), ("assessment_as_of", "2026-09-01T07:44:00+09:00")):
            rows = self._rows(1)
            self._prefix(rows[0])
            rows[0]["five_class_evaluation"]["evaluation_window"]["selected_event"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict_event_five(self.current, rows, self.as_of, "確定教師")

    def test_expired_or_pending_labels_cannot_contain_confirmed_class(self) -> None:
        """失効・保留を確定正解へ変換して正重み教師にする改変を拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        self._prefix(rows[0], clock="08:15:30")
        rows[0]["five_class_evaluation"]["evaluation_window"].update(status="開始前失効", scoreable=False)
        with self.assertRaisesRegex(ValueError, "開始前失効.*分類候補"):
            predict_event_five(self.current, rows, self.as_of, "候補配分")
        rows = self._rows(1)
        rows[0]["five_class_evaluation"]["evaluation_window"].update(adjudication_unknown=True,
            pending_events=[self._event(rows[0], clock="10:00:00", decision="保留")])
        with self.assertRaisesRegex(ValueError, "認定不明.*確定"):
            predict_event_five(self.current, rows, self.as_of, "確定教師")

    def test_current_future_event_metadata_is_rejected_instead_of_read(self) -> None:
        """当回入力へ未来材料や評価窓を渡すと、値を解釈せず拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        for key in ("events", "evaluation_window", "selected_event"):
            changed = {**self.current, key: object()}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "指定外"):
                predict_event_five(changed, [], self.as_of, "確定教師")
        changed = copy.deepcopy(self.current)
        changed["features"]["future_event"] = object()
        with self.assertRaisesRegex(ValueError, "指定外"):
            predict_event_five(changed, [], self.as_of, "候補配分")

    def test_full_end_and_nominal_or_availability_mismatch_is_rejected(self) -> None:
        """公式終点・名目窓・公表後利用可能の別々の時計を勝手に置換しない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        for key, value in (("scheduled_full_end", "2026-09-01T15:40:00+09:00"),
                           ("nominal_window_end", "2026-09-01T11:59:01+09:00"),
                           ("source_event_time", "2026-09-01T11:59:00+09:00")):
            rows = self._rows(1)
            self._prefix(rows[0])
            rows[0]["five_class_evaluation"]["evaluation_window"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict_event_five(self.current, rows, self.as_of, "確定教師")

    def test_helper_prefix_output_connects_without_changing_candidate(self) -> None:
        """実装した窓ヘルパーの合成価格出力をそのまま教師として受け付ける。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        price = price_fixture_module.FiveClassTests()
        price.setUp()
        row = {"symbol": "NK225", "kind": "Day", "as_of": "2026-10-01T07:45:00+09:00",
               "features": dict(self.current["features"]), "actual_pct": None}
        event = self._event(row, "09:00:30")
        evaluation = evaluate_event_window(price.reference, price.bars, "2026-10-01_Day", timestamp(row["as_of"]),
                                           [event], price.contract)
        row.update(five_class_evaluation=evaluation, label_end=evaluation["evaluation_window"]["label_available_at"])
        result = predict_event_five(self.current, [row], self.as_of, "確定教師")
        self.assertEqual(evaluation["correct_candidates"], ["もみ合い"])
        self.assertEqual(result["direction"], "もみ合い")
        self.assertEqual(result["last_label_end"], event["first_public_at"])

    def test_helper_pending_and_preopen_outputs_have_no_label_weight(self) -> None:
        """ヘルパーが返す認定保留と開始前失効を両方式で不採用教師にする。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        price = price_fixture_module.FiveClassTests()
        price.setUp()
        base = {"symbol": "NK225", "kind": "Day", "as_of": "2026-10-01T07:45:00+09:00",
                "features": dict(self.current["features"]), "actual_pct": None}
        for clock, decision in (("09:00:30", "保留"), ("08:15:30", "採用")):
            event = self._event(base, clock, decision)
            evaluation = evaluate_event_window(price.reference, price.bars, "2026-10-01_Day", timestamp(base["as_of"]),
                                               [event], price.contract)
            row = {**base, "five_class_evaluation": evaluation, "label_end": evaluation["evaluation_window"]["label_available_at"]}
            for mode in ("確定教師", "候補配分"):
                with self.subTest(clock=clock, mode=mode):
                    result = predict_event_five(self.current, [row], self.as_of, mode)
                    self.assertEqual(result["weights"], [0.0])
                    self.assertIsNone(result["direction"])

    def test_unknown_first_public_upper_after_session_remains_zero_weight(self) -> None:
        """記事上限が終了後の初報未確認保留も、両方式の確定・弱教師へ入れない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        price = price_fixture_module.FiveClassTests()
        price.setUp()
        row = {"symbol": "NK225", "kind": "Day", "as_of": "2026-10-01T07:45:00+09:00",
               "features": dict(self.current["features"]), "actual_pct": None}
        event = self._event(row, decision="保留")
        event.update(first_public_at=None, timing_evidence={"article_public_upper_bound": "2026-10-01T18:00:00+09:00"})
        evaluation = evaluate_event_window(price.reference, price.bars, "2026-10-01_Day", timestamp(row["as_of"]),
                                           [event], price.contract)
        row.update(five_class_evaluation=evaluation, label_end=evaluation["evaluation_window"]["label_available_at"])
        for mode in ("確定教師", "候補配分"):
            result = predict_event_five(self.current, [row], self.as_of, mode)
            self.assertEqual(result["weights"], [0.0])
            self.assertIsNone(result["direction"])
            self.assertEqual(result["label_exclusion_counts"]["材料認定不明"], 1)

    def test_pending_flag_must_match_empty_list_in_both_directions(self) -> None:
        """一覧なしの認定不明も、一覧ありの認定済みも整合しない教師として拒否する。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        rows = self._rows(1)
        rows[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"] = True
        for mode in ("確定教師", "候補配分"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "一覧が一致"):
                predict_event_five(self.current, rows, self.as_of, mode)

    def test_same_session_with_different_forecast_cutoff_and_public_clock_is_not_two_teachers(self) -> None:
        """同じ場を元締切と公表時計だけ変えて別教師へ水増ししない。

        引数: self (FiveEventModelTests) はテスト。
        返り値: None。
        """
        first = self._rows(1)[0]
        second = copy.deepcopy(first)
        second["as_of"] = "2026-09-01T07:44:00+09:00"
        second["five_class_evaluation"]["evaluation_window"]["as_of"] = second["as_of"]
        self._prefix(first, "10:00:30")
        self._prefix(second, "11:00:30")
        for mode in ("確定教師", "候補配分"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "元予測締切または対象セッション"):
                predict_event_five(self.current, [first, second], self.as_of, mode)


if __name__ == "__main__":
    unittest.main()
