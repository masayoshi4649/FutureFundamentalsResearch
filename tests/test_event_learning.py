"""未知臨時材料前の比較手順を、合成価格・合成来歴だけで接続検証する。"""

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_event_window as price_fixture_module
import test_five_class_learning as fixture_module
from fundamentals.common import digest, read_json, timestamp, write_json
from fundamentals.replay import _append_audit, _audit, _file_hash
from research_support import event_learning as implementation
from research_support.five_class import CLASSES
from research_support.five_class_learning import VARIANTS
from research_support.five_event_model import FEATURE_KEYS, predict_event_five


class EventLearningTests(unittest.TestCase):
    """結果開示順・公表後の教師訂正・主評価と参考評価の分離を確認する。"""

    def setUp(self) -> None:
        """正しい来歴の合成２回を再利用し、結果専用の材料台帳を追加する。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        self.sample = fixture_module.FiveClassLearningTests()
        self.sample.setUp()
        self.addCleanup(self.sample.doCleanups)
        self.root, self.prices = self.sample.root, self.sample.prices
        self.inputs, self.measured = self.sample.fixture.inputs, self.sample.fixture.measured
        self.identities = self.sample.fixture.identities
        self.rules = self.sample.rules
        (self.root / "臨時材料規約.json").write_bytes(Path("research_support/臨時材料規約.json").read_bytes())
        self.evidence = self.sample.fixture.base / "独立材料認定.md"
        self.evidence.write_text("# 結果を含まない合成材料認定\n", encoding="utf-8")
        self.ledger = self.sample.fixture.base / "材料台帳.json"
        self.origins = [read_json(self.inputs / "masked" / (identity + ".json"))["as_of"] for identity in self.identities]
        self.model_calls, self.current_label_reads = [], []
        self._save_ledger([self._event(origin) for origin in self.origins])

    def _event(self, origin: str, clock: str = "10:00:30", decision: str = "採用") -> dict:
        """結果側だけで使用する、根拠付きの合成材料を作る。

        引数: self (EventLearningTests) はテスト、origin (str) は認定対象締切、clock (str) は同日の公表時計、
            decision (str) は材料判定。
        返り値: dict は実ニュースを含まない材料レコード。
        """
        public = origin[:10] + "T" + clock + "+09:00"
        article = "合成記事_" + origin[:10]
        return {"event_id": "合成材料_" + origin[:10], "assessment_as_of": origin, "first_public_at": public,
                "article_ids": [article], "source_revision_hash": "a" * 64, "decision": decision,
                "schedule_status": "臨時・予見不能" if decision == "採用" else "不明", "affected_symbols": ["NK225", "TOPIX"],
                "novelty_evidence": "合成の新規性根拠", "schedule_evidence": "合成の予定外根拠",
                "impact_evidence": "合成の影響根拠", "timing_evidence": {"article_public_upper_bound": public,
                    "article_clocks": [{"article_id": article, "published_at": public}]}, "judge_id": "合成判定者"}

    def _save_ledger(self, events: list[dict]) -> None:
        """認定原本SHAと自己ハッシュを持つ合成台帳を保存する。

        引数: self (EventLearningTests) はテスト、events (list[dict]) は結果専用材料。
        返り値: None。
        """
        article_clocks = {str(article["article_id"]): article["published_at"] for event in events
                          for article in event["timing_evidence"]["article_clocks"]}
        value = {"events": events, "source_files": {str(self.evidence.resolve()): _file_hash(self.evidence)},
                 "article_public_clocks": article_clocks, "article_available_clocks": dict(article_clocks)}
        value["ledger_hash"] = digest(value)
        write_json(self.ledger, value, exclusive=False)

    def _evaluation(self, origin: str, symbol: str, answer: str, full: bool = False, pending: list[dict] | None = None) -> dict:
        """主窓・参考全場または認定不明の合成評価を作る。

        引数: self (EventLearningTests) はテスト、origin/symbol/answer (str) は締切・先物・正解、
            full (bool) は公式全場、pending (list[dict] | None) は窓内の保留。
        返り値: dict は本物のヘルパーと同じ時計・分類の接続形式を持つ評価。
        """
        end = origin[:10] + "T15:45:00+09:00"
        event = None if full else self._event(origin)
        available = end if full else event["first_public_at"]
        nominal = end if full else timestamp(available).replace(second=0, microsecond=0).isoformat()
        pending = [] if pending is None else copy.deepcopy(pending)
        window = {"version": "臨時材料前１", "status": "全セッション" if full else "材料前まで", "as_of": origin,
                  "scheduled_full_end": end, "source_event_time": None if full else available, "nominal_window_end": nominal,
                  "effective_quote_time": nominal, "scoreable": True, "planned_count": True, "selected_event": event,
                  "pending_events": pending, "adjudication_unknown": bool(pending), "label_available_at": available,
                  "maturity_required_at": available}
        return {"classification_policy": "もみ合い優先・初回到達方向と最初の反転", "symbol": symbol,
                "session": origin[:10] + "_Day", "target_end": nominal, "target_status": "未確定" if pending else "対象",
                "correct_candidates": [] if pending else [answer], "uncertain_candidates": list(CLASSES) if pending else [],
                "evaluation_window": window, "metrics": {"actual_pct": 0.8 if answer == "上がる" else -0.8,
                    "session_range_pct": 1.2, "quote_time": nominal}, "flat_status": "材料認定未了" if pending else "不成立",
                "quality": {}, "contract": copy.deepcopy(self.rules["criteria"])}

    def _rows(self, origin: str, full: bool = False, pending: list[dict] | None = None) -> list[dict]:
        """固定した正反対の主結果・参考結果を両先物分返す。

        引数: self (EventLearningTests) はテスト、origin (str) は締切、full (bool) は全場、pending (list[dict] | None) は保留。
        返り値: list[dict] は元価格へのアクセスを伴わない結果。
        """
        answer = "下がる" if full else "上がる"
        result = []
        for symbol in ("NK225", "TOPIX"):
            evaluation = self._evaluation(origin, symbol, answer, full, pending)
            result.append({"symbol": symbol, "label_end": evaluation["evaluation_window"]["label_available_at"],
                           "evaluation": evaluation, "source_file": "合成", "source_file_hash": "合成"})
        return result

    def _model(self, current: dict, history: list[dict], as_of: str, mode: str) -> dict:
        """モデルの受取値を検査し、未来材料を読まない固定回答を返す。

        引数: self (EventLearningTests) はテスト、current (dict) は今回13特徴、history (list[dict]) は過去教師、
            as_of/mode (str) は締切と方式。
        返り値: dict は接続に必要な診断を持つ合成モデル。
        """
        self.assertEqual(set(current), {"symbol", "kind", "as_of", "features"})
        self.assertEqual(set(current["features"]), set(FEATURE_KEYS))
        self.assertTrue(all(timestamp(row["label_end"]) < timestamp(as_of) for row in history))
        for row in history:
            window = row["five_class_evaluation"]["evaluation_window"]
            for event in window["pending_events"] + ([window["selected_event"]] if window["selected_event"] else []):
                self.assertLess(implementation._event_evidence_time(event), timestamp(as_of))
        self.model_calls.append({"as_of": as_of, "mode": mode, "symbol": current["symbol"], "history": copy.deepcopy(history)})
        count = sum(row["symbol"] == current["symbol"] for row in history)
        return {"direction": "上がる", "class_scores": {name: float(name == "上がる") for name in CLASSES},
                "training_n": count, "positive_weight_n": count, "fitted_n": 0, "contributions": {}, "variant": mode}

    def _labels(self, prices: Path, origin: str, session: str, rules: dict, events: list[dict],
                references: dict | None = None, full_only: bool = False) -> list[dict]:
        """過去再構成を許容し、当回結果は両方式のseal後にだけ返す。

        引数: self (EventLearningTests) はテスト、prices (Path) は未読価格、origin/session (str) は対象、
            rules/events (dict/list[dict]) は規約と結果専用台帳、references (dict | None) は基準、full_only (bool) は参考評価。
        返り値: list[dict] は順序検査後に開示する合成結果。
        """
        if references is not None:
            identity = self.identities[self.origins.index(origin)]
            for folder in VARIANTS.values():
                path = self.root / folder / identity
                self.assertEqual(_audit(path)[-1]["phase"], "seal")
                self.assertTrue((path / "予測.json").is_file())
                self.assertFalse((path / "result.json").exists())
            self.current_label_reads.append((origin, full_only))
            return self._rows(origin, full_only)
        relevant = [event for event in events if event["assessment_as_of"] == origin]
        pending = [event for event in relevant if event["decision"] == "保留"]
        return self._rows(origin, full=not any(event["decision"] == "採用" for event in relevant), pending=pending)

    # ----------------------------------------

    def test_immature_window_blocks_price_read_before_any_candle_access(self) -> None:
        """公表直前は窓の満了判定で止まり、原価格を読み始めない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        origin = self.origins[0]
        with patch("research_support.event_learning.now", return_value="2026-10-01T10:00:29+09:00"), \
                patch("research_support.event_learning.load_candles") as reader:
            with self.assertRaisesRegex(ValueError, "実時計で評価窓が未満了"):
                implementation._labels(self.prices, origin, "2026-10-01_Day", self.rules, [self._event(origin)])
            reader.assert_not_called()

    def test_planned_surprise_is_not_a_censor_and_reads_official_full_window(self) -> None:
        """予定済み指標の予想外の結果は、主窓を公式終了より前へ短縮しない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        price = price_fixture_module.EventWindowTests()
        price.setUp()
        event = {**self._event(self.origins[0], decision="棄却"), "schedule_status": "予定済み", "impact_evidence": "結果が市場予想外でも予定発表。"}

        def prices(root: Path, symbol: str, cutoff: object) -> list:
            """公式終点指定を確認して合成価格を返す。

            引数: root (Path) は未使用価格、symbol (str) は先物、cutoff (object) は読取上限。
            返り値: list は指定商品へ置き換えた合成足。
            """
            self.assertEqual(cutoff, price.end)
            return [replace(bar, symbol=symbol) for bar in [price.reference, *price.bars]]

        with patch("research_support.event_learning.load_candles", side_effect=prices):
            rows = implementation._labels(self.prices, self.origins[0], price.session, self.rules, [event])
        self.assertTrue(all(row["label_end"] == price.end.isoformat() for row in rows))
        self.assertTrue(all(row["evaluation"]["evaluation_window"]["status"] == "全セッション" for row in rows))

    def test_expired_and_unanswered_are_separate_states_and_counts(self) -> None:
        """開始前失効を予定件数に残し、対象内・材料不明の無回答は不正解にする。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        known = self._rows(self.origins[0])[0]["evaluation"]
        self.assertEqual(implementation._score(None, known)["status"], "不正解")
        expired = copy.deepcopy(known)
        expired["evaluation_window"].update(scoreable=False, status="開始前失効")
        expired["correct_candidates"] = []
        score = implementation._score("上がる", expired)
        group = implementation._group()
        implementation._add(group, "上がる", expired, score)
        self.assertEqual(score["status"], "開始前失効")
        self.assertFalse(score["denominator"])
        self.assertEqual((group["scheduled"], group["expired_preopen"], group["unanswered"]), (1, 1, 0))
        pending = self._rows(self.origins[0], full=True, pending=[self._event(self.origins[0], decision="保留")])[0]["evaluation"]
        unknown = implementation._score(None, pending)
        self.assertEqual(unknown["status"], "不正解")
        self.assertFalse(unknown["correct_upper"])
        implementation._add(group, None, pending, unknown)
        self.assertEqual((group["scheduled"], group["material_unknown"], group["unanswered"]), (2, 1, 1))

    def test_review_keeps_primary_teacher_separate_from_opposite_full_reference(self) -> None:
        """同じ予測の主窓一致・全場外れを別保存し、教師へ参考全場を混ぜない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        identity, origin = self.identities[0], self.origins[0]
        current = read_json(self.measured / identity / "入力.json")
        path = self.root / "単独振り返り"
        write_json(path / "入力.json", current)
        _append_audit(path, "prepare", timestamp(origin), [path / "入力.json"])
        prediction = {"instruments": {symbol: self._model(current["instruments"][symbol], [], origin, "確定教師")
                                       for symbol in ("NK225", "TOPIX")}}
        prediction["forecast_hash"] = digest(prediction)
        write_json(path / "予測.json", prediction)
        _append_audit(path, "seal", timestamp(origin), [path / "予測.json"])
        primary, full = self._rows(origin), self._rows(origin, full=True)
        teachers = implementation._review(path, current, primary, full, prediction)
        saved = read_json(path / "result.json")
        self.assertTrue(all(row["score"]["correct_lower"] for row in saved["labels"]))
        self.assertTrue(all(not row["full_session_reference"]["score"]["correct_lower"] for row in saved["labels"]))
        self.assertTrue(all(row["label_end"] == primary[0]["label_end"] for row in teachers))
        self.assertTrue(all(row["five_class_evaluation"]["correct_candidates"] == ["上がる"] for row in teachers))
        review = read_json(path / "review.json")
        self.assertNotIn("next_training", review)
        self.assertEqual(review["final_primary_labels"], teachers)
        self.assertEqual([row["phase"] for row in _audit(path)], ["prepare", "seal", "reveal", "review"])

    def test_run_seals_both_modes_before_each_current_window_and_hides_future_events(self) -> None:
        """当回の主・参考結果を両方式の固定後に読み、モデルに未来材料を渡さない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        with patch("research_support.event_learning._labels", side_effect=self._labels), \
                patch("research_support.event_learning.predict_event_five", side_effect=self._model):
            summary = implementation.run_event_comparison(self.root, self.inputs, self.measured, self.prices, self.ledger)
        self.assertEqual(self.current_label_reads, [(origin, full) for origin in self.origins for full in (False, True)])
        self.assertEqual([len(call["history"]) for call in self.model_calls], [0, 0, 0, 0, 2, 2, 2, 2])
        for origin in self.origins:
            histories = [call["history"] for call in self.model_calls if call["as_of"] == origin]
            self.assertTrue(all(history == histories[0] for history in histories))
        for result in summary.values():
            self.assertEqual(result["groups"]["NK225"]["correct_lower"], 2)
            self.assertEqual(result["full_session_reference_groups"]["NK225"]["correct_lower"], 0)

    def test_history_is_corrected_only_after_article_evidence_becomes_known(self) -> None:
        """過去材料の根拠記事出現前は全場教師を保ち、出現後だけ保留へ訂正する。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        origin = "2026-09-30T07:45:00+09:00"
        source = {"symbol": "NK225", "kind": "Day", "as_of": origin, "features": dict.fromkeys(FEATURE_KEYS, 0.0)}
        pending = self._event(origin, decision="保留")
        upper = "2026-10-01T12:00:00+09:00"
        pending.update(first_public_at=None, timing_evidence={"article_public_upper_bound": upper,
            "article_clocks": [{"article_id": pending["article_ids"][0], "published_at": upper}]})
        cache = {}
        with patch("research_support.event_learning._labels", side_effect=self._labels) as labels:
            first = implementation._history_as_of([source], self.prices, self.rules, [pending], self.origins[0], cache)
            second = implementation._history_as_of([source], self.prices, self.rules, [pending], self.origins[1], cache)
        self.assertEqual(labels.call_count, 2)
        self.assertFalse(first[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertTrue(second[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertEqual(first[0]["features"], second[0]["features"])
        for mode in ("確定教師", "候補配分"):
            current = read_json(self.measured / self.identities[1] / "入力.json")["instruments"]["NK225"]
            model = predict_event_five(current, second, self.origins[1], mode)
            self.assertEqual(model["weights"], [0.0])

    def test_article_public_exactly_at_cutoff_is_not_known_and_immature_source_is_blocked(self) -> None:
        """根拠公開の締切同時は除き、未満了の元場は価格読取前に拒否する。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        source = {"symbol": "NK225", "kind": "Day", "as_of": "2026-09-30T07:45:00+09:00",
                  "features": dict.fromkeys(FEATURE_KEYS, 0.0)}
        event = self._event(source["as_of"], decision="保留")
        event.update(first_public_at=None, timing_evidence={"article_public_upper_bound": self.origins[0],
            "article_clocks": [{"article_id": event["article_ids"][0], "published_at": self.origins[0]}]})
        with patch("research_support.event_learning._labels", side_effect=self._labels):
            rows = implementation._history_as_of([source], self.prices, self.rules, [event], self.origins[0], {})
        self.assertFalse(rows[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        source["as_of"] = self.origins[0]
        source["features"] = "まだ解釈しない値"
        with patch("research_support.event_learning._labels") as labels:
            with self.assertRaisesRegex(ValueError, "満了していない"):
                implementation._history_as_of([source], self.prices, self.rules, [], self.origins[0], {})
            labels.assert_not_called()

    def test_public_article_does_not_correct_history_until_recorded_version_is_available(self) -> None:
        """公開済みでも収録版の取得が締切後なら、その教師訂正を先取りしない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        origin = "2026-09-30T07:45:00+09:00"
        source = {"symbol": "NK225", "kind": "Day", "as_of": origin, "features": dict.fromkeys(FEATURE_KEYS, 0.0)}
        event = self._event(origin, decision="保留")
        public, acquired = "2026-10-01T07:00:00+09:00", "2026-10-01T08:00:00+09:00"
        identity = event["article_ids"][0]
        event.update(first_public_at=None, timing_evidence={"article_public_upper_bound": public,
            "article_clocks": [{"article_id": identity, "published_at": public}]})
        clocks, available = {identity: public}, {identity: acquired}
        cache = {}
        with patch("research_support.event_learning._labels", side_effect=self._labels):
            first = implementation._history_as_of([source], self.prices, self.rules, [event], self.origins[0], cache, clocks, available)
            second = implementation._history_as_of([source], self.prices, self.rules, [event], self.origins[1], cache, clocks, available)
        self.assertFalse(first[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertTrue(second[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertEqual(first[0]["features"], second[0]["features"])

    def test_all_evidence_articles_need_matching_public_and_available_clocks(self) -> None:
        """根拠の一部だけの索引や、公開より前の利用可能時計を受け付けない。

        引数: self (EventLearningTests) はテスト。
        返り値: None。
        """
        event = self._event("2026-09-30T07:45:00+09:00", decision="保留")
        public = event["first_public_at"]
        identity = event["article_ids"][0]
        event["article_ids"].append("未照合の別記事")
        with self.assertRaisesRegex(ValueError, "全根拠記事"):
            implementation._event_evidence_time(event, {identity: public}, {identity: public})
        event["article_ids"].pop()
        with self.assertRaisesRegex(ValueError, "利用可能時計"):
            implementation._event_evidence_time(event, {identity: public}, {identity: "2026-09-30T07:00:00+09:00"})


if __name__ == "__main__":
    unittest.main()
