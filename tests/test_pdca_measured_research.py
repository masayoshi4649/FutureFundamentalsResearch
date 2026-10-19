"""実測特徴の再生で、予測固定・実績開示・学習更新の順序を合成入力で検証する。"""

import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import JST, digest, read_json, timestamp, write_json
from fundamentals.pdca import CHANNELS
from fundamentals.pdca_measured_research import (
    MODES, SYMBOLS, _initial_labels, _source_labels, run_measured, visible_input,
)
from fundamentals.replay import _append_audit, _audit


class MeasuredResearchTests(unittest.TestCase):
    """実測特徴の計算と分類器を分離し、再生処理の時点契約を調べる。"""

    def setUp(self) -> None:
        """計画、可視入力、原実績を別々の一時フォルダに用意する。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.inputs = self.base / "新研究", self.base / "固定入力"
        self.source, self.prices = self.base / "原実績", self.base / "合成価格"
        self.root.mkdir()
        (self.root / "実験計画.md").write_text("# 合成した固定計画\n", encoding="utf-8")
        write_json(self.inputs / "training" / "labels.json", {"mode": "回想・開発再生", "rows": []})

    def _round(self, number: int, result_changes: dict | None = None,
               prediction_changes: dict | None = None, reference_changes: dict | None = None,
               duplicate: bool = False) -> tuple[str, dict]:
        """固定入力と、任意の不整合も含められる監査済み原実績を作成する。

        引数: self (MeasuredResearchTests) はテスト、number (int) は昇順の日、
            result_changes (dict | None) は先頭ラベル変更、prediction_changes (dict | None) は原予測変更、
            reference_changes (dict | None) は原予測の先頭先物基準変更、duplicate (bool) は重複指定。
        返り値: tuple[str, dict] は識別子と可視入力。
        """
        day = datetime(2026, 10, number, tzinfo=JST)
        as_of, end = day.replace(hour=7, minute=45), day.replace(hour=15, minute=45)
        identity = f"{number:02d}_合成回"
        previous = (day - timedelta(days=1)).date().isoformat() + "_Night"
        packet = {"as_of": as_of.isoformat(), "session": day.date().isoformat() + "_Day", "kind": "Day",
                  "targets": {"today": end.isoformat()}, "instruments": {},
                  "news": [{"id": "可視記事", "available_at": day.replace(hour=6).isoformat(),
                            "published_at": day.replace(hour=6).isoformat(),
                            "fetched_at": day.replace(hour=6).isoformat(), "body_text": "結果とは無関係な材料"}]}
        annotation, labels = {"as_of": as_of.isoformat(), "instruments": {}}, []
        for symbol in SYMBOLS:
            packet["instruments"][symbol] = {
                "reference": {"price": 100.0, "time": day.replace(hour=5, minute=55).isoformat(), "session": previous},
                "horizons": {"today": {"distribution": {"n": 0, "terminal_range_pct": None, "path_range_pct": None}}}}
            annotation["instruments"][symbol] = {"channels": {
                channel: {"score": 0.5, "confidence": 0.8, "novelty": 0.7, "priced_in": 0.2,
                          "evidence": ["可視記事"], "reason": "結果と独立した合成注釈"} for channel in CHANNELS}}
            labels.append({"symbol": symbol, "horizon": "today", "label_end": end.isoformat(),
                           "reference_price": 100.0, "quote_price": 101.0,
                           "quote_time": day.replace(hour=15, minute=40).isoformat(),
                           "actual_pct": 1.0, "low_pct": -0.3, "high_pct": 1.2,
                           "path_hash": digest([identity, symbol])})
        packet["packet_hash"] = digest(packet)
        write_json(self.inputs / "masked" / (identity + ".json"), packet)
        write_json(self.inputs / "annotations" / (identity + ".json"), annotation)

        # ----------------------------------------

        original = self.source / identity
        write_json(original / "入力.json", packet)
        _append_audit(original, "prepare", as_of, [original / "入力.json"])
        prediction = {"as_of": packet["as_of"], "session": packet["session"],
                      "instruments": copy.deepcopy(packet["instruments"])}
        prediction.update(prediction_changes or {})
        prediction["instruments"]["NK225"]["reference"].update(reference_changes or {})
        prediction["forecast_hash"] = digest(prediction)
        write_json(original / "予測.json", prediction)
        _append_audit(original, "seal", as_of, [original / "予測.json"])
        labels[0].update(result_changes or {})
        if duplicate:
            labels.append(copy.deepcopy(labels[0]))
        result = {"forecast_hash": prediction["forecast_hash"], "labels": labels}
        write_json(original / "result.json", result)
        _append_audit(original, "reveal", end, [original / "result.json"])
        write_json(original / "review.json", {"原実績": digest(result)})
        _append_audit(original, "review", end, [original / "review.json"])
        return identity, packet

    def _visible(self, packet: dict, annotation: dict, prices: Path,
                 news_articles: list | None = None) -> dict:
        """再生順序の検証に必要な実測入力だけを合成する。

        引数: self (MeasuredResearchTests) はテスト、packet (dict) は可視入力、
            annotation (dict) は注釈、prices (Path) は未使用の価格ルート、
            news_articles (list | None) は未使用の追加記事。
        返り値: dict は結果を一切含まない共通入力。
        """
        return {"as_of": packet["as_of"], "session": packet["session"],
                "states": {symbol: {"status": "利用可能", "target_session": packet["session"],
                                     "reference": copy.deepcopy(packet["instruments"][symbol]["reference"]),
                                     "previous_session": {"session": packet["instruments"][symbol]["reference"]["session"],
                                                          "return_pct": -0.4, "close_location": 0.4, "range_pct": 1.0,
                                                          "late_window": {"observed_candles": 40, "expected_candles": 60,
                                                                          "complete": False, "change_pct": -0.1}}}
                           for symbol in SYMBOLS},
                "instruments": {symbol: {"symbol": symbol, "kind": packet["kind"], "as_of": packet["as_of"],
                                         "features": {"previous_return_pct": -0.4}} for symbol in SYMBOLS}}

    def _predict(self, current: dict, history: list[dict], as_of: str, mode: str) -> dict:
        """現在回の実績が未追加であることを確認する合成分類器。

        引数: self (MeasuredResearchTests) はテスト、current (dict) は今回特徴、history (list[dict]) は過去教師、
            as_of (str) は締切、mode (str) は方式。
        返り値: dict は短い予測整形に必要な固定スコア。
        """
        self.assertTrue(all(timestamp(row["label_end"]) < timestamp(as_of) for row in history))
        own = [row for row in history if row["symbol"] == current["symbol"]]
        return {"direction": "上昇", "training_n": len(own), "fitted_n": len(own),
                "class_scores": {"上昇": 1.0, "下落": 0.0, "横ばい": 0.0}, "out_of_range_features": []}

    def _run(self, root: Path | None = None) -> dict:
        """実測入力と分類器だけをモックして、実際の保存・開示処理を実行する。

        引数: self (MeasuredResearchTests) はテスト、root (Path | None) は独立した出力計画。
        返り値: dict は再生集計。
        """
        selected = root or self.root
        selected.mkdir(parents=True, exist_ok=True)
        plan = selected / "実験計画.md"
        if not plan.exists():
            plan.write_text("# 合成固定計画\n", encoding="utf-8")
        with patch("fundamentals.pdca_measured_research.visible_input", side_effect=self._visible), \
                patch("fundamentals.pdca_measured_research.predict_measured", side_effect=self._predict):
            return run_measured(selected, self.inputs, self.source, self.prices, "observed")

    # ----------------------------------------

    def test_both_predictions_are_sealed_before_source_and_next_round_follows_review(self) -> None:
        """両先物の予測固定後だけ原結果を読み、次回には前回の両教師だけを渡す。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identities = [self._round(number)[0] for number in (1, 2)]
        originals = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        output = self.root / MODES["observed"]
        seen = []

        def guarded_source(source: Path, packet: dict) -> tuple[list[dict], str]:
            """原結果の読取直前に両予測と前回振り返りの固定を検査する。

            引数: source (Path) は当回原実績、packet (dict) は可視入力。
            返り値: tuple[list[dict], str] は通常の原結果読取。
            """
            current = output / source.name
            prediction = read_json(current / "予測.json")
            self.assertEqual(set(prediction["instruments"]), set(SYMBOLS))
            self.assertTrue((current / "予測.md").is_file())
            self.assertEqual(_audit(current)[-1]["phase"], "seal")
            if seen:
                self.assertEqual(_audit(output / seen[-1])[-1]["phase"], "review")
                self.assertTrue((output / seen[-1] / "振り返り.md").is_file())
            seen.append(source.name)
            return _source_labels(source, packet)

        with patch("fundamentals.pdca_measured_research._source_labels", side_effect=guarded_source):
            summary = self._run()
        self.assertEqual(seen, identities)
        self.assertEqual(summary["groups"]["NK225"]["n"], 2)
        for symbol in SYMBOLS:
            first = read_json(output / identities[0] / "予測.json")["instruments"][symbol]
            second = read_json(output / identities[1] / "予測.json")["instruments"][symbol]
            self.assertEqual((first["training_n"], second["training_n"]), (0, 1))
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_same_cutoff_cannot_read_first_symbol_label_before_second_prediction(self) -> None:
        """同時刻の片先物を先に採点して他方の予測へ入れる経路がない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        events = []

        def record_prediction(current: dict, history: list[dict], as_of: str, mode: str) -> dict:
            """ラベル読取前に両先物が同じ空履歴を受けたことを記録する。

            引数: current (dict) は今回特徴、history (list[dict]) は履歴、as_of (str) は締切、mode (str) は方式。
            返り値: dict は合成予測。
            """
            self.assertEqual(history, [])
            events.append(current["symbol"])
            return self._predict(current, history, as_of, mode)

        def record_source(source: Path, packet: dict) -> tuple[list[dict], str]:
            """両予測が済んだ後でだけ原実績を開く。

            引数: source (Path) は原実績、packet (dict) は可視入力。
            返り値: tuple[list[dict], str] は原ラベル。
            """
            self.assertEqual(events, list(SYMBOLS))
            events.append("結果開示")
            return _source_labels(source, packet)

        with patch("fundamentals.pdca_measured_research.visible_input", side_effect=self._visible), \
                patch("fundamentals.pdca_measured_research.predict_measured", side_effect=record_prediction), \
                patch("fundamentals.pdca_measured_research._source_labels", side_effect=record_source):
            run_measured(self.root, self.inputs, self.source, self.prices, "observed")
        self.assertEqual(events, [*SYMBOLS, "結果開示"])

    def test_future_initial_labels_are_checked_before_any_unknown_value(self) -> None:
        """全初期教師の時刻を先に検査し、未来・同時刻なら未開示数値を解釈しない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        path = self.inputs / "training" / "labels.json"
        for ending in ("2026-10-01T07:45:00+09:00", "2026-10-01T15:45:00+09:00"):
            with self.subTest(ending=ending):
                write_json(path, {"mode": "回想・開発再生", "rows": [
                    {"symbol": "NK225", "label_end": "2026-09-30T15:45:00+09:00", "actual_pct": "未開示"},
                    {"symbol": "TOPIX", "label_end": ending, "actual_pct": "未開示"},
                ]}, exclusive=False)
                with self.assertRaisesRegex(ValueError, "最初の締切より前"):
                    _initial_labels(path, "2026-10-01T07:45:00+09:00")

    def test_duplicate_initial_labels_cannot_inflate_history(self) -> None:
        """同じ先物と終点の初期教師を重ねて学習件数を水増しできない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        row = {"symbol": "NK225", "label_end": "2026-09-30T15:45:00+09:00", "actual_pct": 1.0}
        write_json(self.inputs / "training" / "labels.json",
                   {"mode": "回想・開発再生", "rows": [row, copy.deepcopy(row)]}, exclusive=False)
        with patch("fundamentals.pdca_measured_research._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "重複"):
                self._run()
            source_read.assert_not_called()

    def test_source_wrong_reference_price_session_and_timestamp_are_rejected(self) -> None:
        """原予測の基準価格・セッション・時刻のすり替えを個別に拒否する。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        changes = [{"price": 99}, {"session": "2026-09-29_Night"}, {"time": "2026-10-01T05:54:00+09:00"}]
        for number, change in enumerate(changes, start=1):
            with self.subTest(change=change):
                identity, packet = self._round(number, reference_changes=change)
                with self.assertRaisesRegex(ValueError, "基準"):
                    _source_labels(self.source / identity, packet)

    def test_source_target_and_duplicate_labels_are_rejected_after_seal(self) -> None:
        """別の対象終点とラベル重複を予測固定後に検知し、次回教師へ渡さない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identity, _ = self._round(1, result_changes={"label_end": "2026-10-02T15:45:00+09:00"})
        with self.assertRaisesRegex(ValueError, "終点"):
            self._run()
        self.assertEqual(_audit(self.root / MODES["observed"] / identity)[-1]["phase"], "seal")
        other, packet = self._round(2, duplicate=True)
        with self.assertRaisesRegex(ValueError, "各先物１行"):
            _source_labels(self.source / other, packet)

    def test_source_wrong_prediction_target_is_rejected(self) -> None:
        """元予測の対象セッションと情報締切が違えば原結果を採点しない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        changes = [{"session": "2026-10-01_Night"}, {"as_of": "2026-10-03T07:45:00+09:00"}]
        for number, change in enumerate(changes, start=1):
            with self.subTest(change=change):
                identity, packet = self._round(number, prediction_changes=change)
                with self.assertRaisesRegex(ValueError, "対象時点"):
                    _source_labels(self.source / identity, packet)

    def test_source_quote_after_nominal_end_or_over_ten_minutes_old_is_rejected(self) -> None:
        """終点後の未来価格と10分を超える古い代理を原実績から受け入れない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        for number, clock in enumerate(((15, 46), (15, 34)), start=1):
            with self.subTest(clock=clock):
                day = datetime(2026, 10, number, tzinfo=JST).replace(hour=clock[0], minute=clock[1])
                identity, packet = self._round(number, result_changes={"quote_time": day.isoformat()})
                with self.assertRaises(ValueError):
                    _source_labels(self.source / identity, packet)

    def test_output_overwrite_is_rejected_before_inputs_or_results_are_opened(self) -> None:
        """既存出力は入力・実績を読む前に拒否し、保存済み内容を変更しない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        output = self.root / MODES["observed"]
        output.mkdir()
        sentinel = output / "保存済み.md"
        sentinel.write_text("固定済み\n", encoding="utf-8")
        with patch("fundamentals.pdca_measured_research._initial_labels") as labels_read, \
                patch("fundamentals.pdca_measured_research._source_labels") as source_read:
            with self.assertRaises(FileExistsError):
                run_measured(self.root, self.inputs, self.source, self.prices, "observed")
            labels_read.assert_not_called()
            source_read.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "固定済み\n")

    def test_future_packet_articles_are_rejected_before_price_or_source_read(self) -> None:
        """記事の各時計が締切を超える場合、価格生成や原結果読取へ進まない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identity, original = self._round(1)
        for field in ("available_at", "published_at", "fetched_at", "updated_at"):
            with self.subTest(field=field):
                packet = copy.deepcopy(original)
                packet["news"][0][field] = "2026-10-01T08:00:00+09:00"
                packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
                write_json(self.inputs / "masked" / (identity + ".json"), packet, exclusive=False)
                selected = self.root / field
                selected.mkdir()
                (selected / "実験計画.md").write_text("# 固定\n", encoding="utf-8")
                with patch("fundamentals.pdca_measured_research.observed_state") as price_read, \
                        patch("fundamentals.pdca_measured_research._source_labels") as source_read:
                    with self.assertRaisesRegex(ValueError, "未来の記事"):
                        run_measured(selected, self.inputs, self.source, self.prices, "observed")
                    price_read.assert_not_called()
                    source_read.assert_not_called()

    def test_visible_state_reference_and_target_must_match_packet(self) -> None:
        """実測状態の対象・基準を固定入力と照合し、価格だけの一致で代替しない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identity, packet = self._round(1)
        annotation = read_json(self.inputs / "annotations" / (identity + ".json"))
        original = self._visible(packet, annotation, self.prices)["states"]
        cases = [("target_session", "別の対象"), ("price", 99), ("time", "2026-10-01T05:54:00+09:00"),
                 ("session", "2026-09-29_Night")]
        for key, value in cases:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                if key == "target_session":
                    changed["NK225"][key] = value
                else:
                    changed["NK225"]["reference"][key] = value
                with patch("fundamentals.pdca_measured_research.observed_state",
                           side_effect=lambda prices, symbol, as_of: changed[symbol]), \
                        patch("fundamentals.pdca_measured_research.extract_market_facts", return_value={"series": {}}):
                    with self.assertRaisesRegex(ValueError, "基準|対象セッション"):
                        visible_input(packet, annotation, self.prices)

    def test_next_round_rejects_missing_previous_review_phase(self) -> None:
        """振り返り段階が記録されなかった場合、次回予測を発行できない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        first, _ = self._round(1)
        second, _ = self._round(2)

        def drop_review(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """異常を模擬し、初回振り返りの監査追記だけを失わせる。

            引数: path (Path) は当回出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
                files (list[Path]) は固定対象。
            返り値: None。
            """
            if path.name == first and phase == "review":
                return
            _append_audit(path, phase, simulated_at, files)

        with patch("fundamentals.pdca_measured_research._append_audit", side_effect=drop_review):
            with self.assertRaisesRegex(ValueError, "振り返り|review|段階"):
                self._run()
        self.assertFalse((self.root / MODES["observed"] / second / "予測.json").exists())

    def test_raw_news_future_body_never_reaches_extractor(self) -> None:
        """原本を渡す経路でも、未来版の本文を数値抽出へ渡さない。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identity, packet = self._round(1)
        annotation = read_json(self.inputs / "annotations" / (identity + ".json"))
        states = self._visible(packet, annotation, self.prices)["states"]
        article = {"id": "過去原本", "headline": "NY株式", "body_text": "観測済みの本文",
                   "published_at": "2026-10-01T06:00:00+09:00", "fetched_at": "2026-10-01T06:01:00+09:00"}
        future = {"id": "未来原本", "published_at": "2026-10-01T08:00:00+09:00",
                  "fetched_at": "2026-10-01T08:01:00+09:00", "headline": object(), "body_text": object()}
        with patch("fundamentals.pdca_measured_research.observed_state",
                   side_effect=lambda prices, symbol, as_of: states[symbol]), \
                patch("fundamentals.pdca_measured_research.extract_market_facts",
                      return_value={"series": {}}) as extractor:
            visible_input(packet, annotation, self.prices, [article, future])
        selected, cutoff = extractor.call_args.args
        self.assertEqual([item["id"] for item in selected], ["過去原本"])
        self.assertEqual(cutoff, packet["as_of"])

    def test_raw_news_hash_change_after_seal_is_rejected_before_source(self) -> None:
        """原本ニュースを固定後に変更すると、当回結果の開示前に停止する。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        identity, _ = self._round(1)
        news_path = self.base / "版付きニュース.json"
        write_json(news_path, {"articles": []})

        def mutate_news(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """通常の固定直後に記事原本のバイトだけを変更する。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
                files (list[Path]) は固定対象。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                news_path.write_bytes(news_path.read_bytes() + b" ")

        with patch("fundamentals.pdca_measured_research.visible_input", side_effect=self._visible), \
                patch("fundamentals.pdca_measured_research.predict_measured", side_effect=self._predict), \
                patch("fundamentals.pdca_measured_research._append_audit", side_effect=mutate_news), \
                patch("fundamentals.pdca_measured_research._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                run_measured(self.root, self.inputs, self.source, self.prices, "observed", (news_path,))
            source_read.assert_not_called()
        output = self.root / MODES["observed"]
        self.assertIn(str(news_path.resolve()), read_json(output / "manifest.json")["files"])
        self.assertEqual(_audit(output / identity)[-1]["phase"], "seal")

    def test_manifest_change_after_seal_is_rejected_before_source(self) -> None:
        """固定一覧自体の変更も、原結果の読取前に検知する。

        引数: self (MeasuredResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)

        def mutate_manifest(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """通常の固定直後に一覧だけを変更する。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
                files (list[Path]) は固定対象。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                target = path.parent / "manifest.json"
                target.write_bytes(target.read_bytes() + b" ")

        with patch("fundamentals.pdca_measured_research._append_audit", side_effect=mutate_manifest), \
                patch("fundamentals.pdca_measured_research._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                self._run()
            source_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
