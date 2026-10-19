"""時点別教師の後発訂正と、元CSVの公表前切断を別の合成入力で監査する。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_event_window as fixture_module
from fundamentals.common import timestamp
from research_support import event_learning as implementation
from research_support.five_event_model import FEATURE_KEYS, predict_event_five


class EventLearningAuditTests(unittest.TestCase):
    """実ニュース・元価格を読まず、教師投入と価格解釈の境界を確認する。"""

    def setUp(self) -> None:
        """合成した過去デイと、一日後に取得された保留根拠を用意する。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        self.sample = fixture_module.EventWindowTests()
        self.sample.setUp()
        self.features = dict.fromkeys(FEATURE_KEYS, 0.0)
        self.source = {"symbol": "NK225", "kind": "Day", "as_of": self.sample.as_of.isoformat(),
                       "features": dict(self.features)}
        self.public = "2026-10-02T10:00:00+09:00"
        self.available = "2026-10-02T16:00:00+09:00"
        self.pending = {**self.sample.event, "first_public_at": None, "decision": "保留", "schedule_status": "不明",
                        "timing_evidence": {"article_public_upper_bound": self.public}}
        self.article = self.pending["article_ids"][0]

    def test_all_evidence_uses_latest_acquisition_and_extra_ids_cannot_bypass_it(self) -> None:
        """全記事の取得時計を最大化し、ID外の追加根拠で取得時計を省略させない。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        later = "2026-10-02T20:00:00+09:00"
        event = {**self.pending, "article_ids": [self.article, "追加記事"]}
        public = {self.article: self.public, "追加記事": "2026-10-02T11:00:00+09:00"}
        available = {self.article: self.available, "追加記事": later}
        self.assertEqual(implementation._event_evidence_time(event, public, available), timestamp(later))
        extra = {**self.pending, "timing_evidence": {"article_public_upper_bound": self.public,
                 "related_article_clocks": [{"article_id": "追加記事", "published_at": public["追加記事"]}]}}
        with self.assertRaises(ValueError):
            implementation._event_evidence_time(extra, public, available)

    def test_late_pending_is_invisible_until_acquired_then_changes_only_later_history(self) -> None:
        """後発保留が存在する場合も取得前・取得同時は旧教師と完全一致し、以後だけ０重み。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        bars = [replace(bar, open=100.6, high=100.6, low=100.6, close=100.6) for bar in self.sample.bars]

        def synthetic_prices(prices: Path, symbol: str, cutoff: object) -> list:
            """対象過去場より後の価格を要求しないことを確認して合成足を返す。

            引数: prices (Path) は未使用、symbol (str) は先物、cutoff (object) は読取上限。
            返り値: list は過去デイとその前終値だけの合成足。
            """
            self.assertEqual(cutoff, self.sample.end)
            return [replace(bar, symbol=symbol) for bar in [self.sample.reference, *bars]]

        cache = {}
        clocks, available = {self.article: self.public}, {self.article: self.available}
        early, boundary, later = "2026-10-02T07:45:00+09:00", self.available, "2026-10-05T07:45:00+09:00"
        with patch("research_support.event_learning.load_candles", side_effect=synthetic_prices):
            baseline = implementation._history_as_of([self.source], Path("未使用合成価格"),
                {"criteria": self.sample.contract}, [], early, {}, clocks, available)
            before = implementation._history_as_of([self.source], Path("未使用合成価格"),
                {"criteria": self.sample.contract}, [self.pending], early, cache, clocks, available)
            exact = implementation._history_as_of([self.source], Path("未使用合成価格"),
                {"criteria": self.sample.contract}, [self.pending], boundary, cache, clocks, available)
            after = implementation._history_as_of([self.source], Path("未使用合成価格"),
                {"criteria": self.sample.contract}, [self.pending], later, cache, clocks, available)
        self.assertEqual(baseline, before)
        self.assertEqual(before, exact)
        self.assertFalse(before[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertTrue(after[0]["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"])
        self.assertEqual(before[0]["features"], after[0]["features"])
        current = {"symbol": "NK225", "kind": "Day", "as_of": later, "features": dict(self.features)}
        for mode in ("確定教師", "候補配分"):
            self.assertEqual(predict_event_five(current, before, later, mode)["weights"], [1.0])
            self.assertEqual(predict_event_five(current, after, later, mode)["weights"], [0.0])

    def test_session_maturity_cannot_be_replaced_by_an_early_label_end(self) -> None:
        """元場が未満了なら、偽の早いlabel_endがあっても価格や特徴の解釈前に拒否する。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        source = {**self.source, "label_end": "2026-10-01T09:00:00+09:00", "features": object()}
        with patch("research_support.event_learning._labels") as reader:
            with self.assertRaises(ValueError):
                implementation._history_as_of([source], Path("未使用"), {"criteria": self.sample.contract}, [],
                    "2026-10-01T10:00:00+09:00", {})
            reader.assert_not_called()

    def test_raw_csv_after_event_is_skipped_before_price_or_flag_parsing(self) -> None:
        """本物のCSV読込でも公表を跨ぐ不正価格・不正フラグを解釈しない。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as temporary:
            prices = Path(temporary)
            header = "time,open,high,low,close,volume,volume_complete,closed\n"
            for symbol in ("NK225", "TOPIX"):
                folder = prices / "candle_1min" / symbol
                folder.mkdir(parents=True)
                (folder / "2026-09-30_Night.csv").write_text(header +
                    "2026-10-01T05:59:00+09:00,100,100,100,100,1,true,true\n", encoding="utf-8")
                (folder / "2026-10-01_Day.csv").write_text(header +
                    "2026-10-01T08:45:00+09:00,100.6,100.6,100.6,100.6,1,true,true\n" +
                    "2026-10-01T08:46:00+09:00,不正,不正,不正,不正,不正,未解釈,未解釈\n", encoding="utf-8")
            event = {**self.sample.event, "first_public_at": "2026-10-01T08:46:30+09:00"}
            rows = implementation._labels(prices, self.sample.as_of.isoformat(), self.sample.session,
                {"criteria": self.sample.contract}, [event])
            for row in rows:
                self.assertEqual(row["label_end"], event["first_public_at"])
                self.assertEqual(row["evaluation"]["quality"]["observed_rows"], 1)
                self.assertEqual(row["evaluation"]["metrics"]["quote_time"], "2026-10-01T08:46:00+09:00")

    def test_freeze_preserves_originals_and_detects_later_price_changes(self) -> None:
        """開始固定で入力原本を変えず、価格CSVもmanifestへ束縛する。

        引数: self (EventLearningAuditTests) はテスト。
        返り値: None。
        """
        import test_event_learning as connection_fixture

        sample = connection_fixture.EventLearningTests()
        sample.setUp()
        self.addCleanup(sample.doCleanups)
        price_file = sample.prices / "candle_1min/NK225/2026-09-30_Night.csv"
        price_file.parent.mkdir(parents=True)
        price_file.write_text("時刻だけの固定テスト。数値解釈は行わない。\n", encoding="utf-8")
        originals = {path: path.read_bytes() for root in (sample.inputs, sample.measured, sample.prices)
                     for path in root.rglob("*") if path.is_file()}
        originals[sample.ledger] = sample.ledger.read_bytes()
        manifest = implementation._freeze(sample.root, sample.inputs, sample.measured, sample.prices, sample.ledger)
        self.assertIn(str(price_file.resolve()), manifest["files"])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        implementation._verify(sample.root)
        price_file.write_bytes(price_file.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            implementation._verify(sample.root)


if __name__ == "__main__":
    unittest.main()
