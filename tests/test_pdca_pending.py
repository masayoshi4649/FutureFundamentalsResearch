"""結果未開示の追跡で、予測固定・満期待ち・固定原本の境界を検証する。"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import read_json, timestamp, write_json
from fundamentals.pdca import CHANNELS
from fundamentals.pdca_pending import AS_OF, evaluate, prepare, seal
from fundamentals.replay import _audit


class PdcaPendingTests(unittest.TestCase):
    """一時価格だけを使い、実績読取の順序と未満了の非採点を確認する。"""

    def setUp(self) -> None:
        """直前セッション・可知記事・過去教師を合成して用意する。

        引数: self (PdcaPendingTests) はテスト。
        返り値: None。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root, self.prices = base / "追跡", base / "価格"
        self.root.mkdir()
        (self.root / "実験計画.md").write_text("# 合成した結果未開示の追跡\n", encoding="utf-8")
        self.news, self.history = base / "可知記事.json", base / "過去教師.json"
        article = {"id": "既知", "revision_id": "初版", "headline": "合成材料", "canonical_url": "https://example.com/known",
                   "published_at": "2026-10-16T06:00:00+09:00", "fetched_at": "2026-10-16T06:01:00+09:00",
                   "body_text": "予測時点に利用できる合成本文"}
        future = dict(article, id="未来秘密", canonical_url="https://example.com/future", published_at="2026-10-16T09:00:00+09:00")
        write_json(self.news, [article, future])
        rows = []
        for symbol in ("NK225", "TOPIX"):
            self._price(symbol, "2026-10-15_Day", "2026-10-15T15:39:00+09:00", 100.0)
            self._price(symbol, "2026-10-15_Night", "2026-10-16T05:54:00+09:00", 105.0)
            future_path = self.prices / "candle_1min" / symbol / "2026-10-16_Day.csv"
            future_path.write_text("time\n2026-10-16T15:39:00+09:00\n", encoding="utf-8")
            for day in range(1, 9):
                features = {key: 0.01 * day for key in CHANNELS}
                features.update(previous_return=0.1 * day, is_night=float(day % 2), intercept=1.0)
                rows.append({"symbol": symbol, "features": features, "actual_pct": 0.5,
                             "label_end": f"2026-10-{day:02d}T15:45:00+09:00"})
        write_json(self.history, {"mode": "回想・開発再生", "rows": rows})

    def _price(self, symbol: str, session: str, started: str, price: float) -> None:
        """一時領域へ確定済みの合成１分足を保存する。

        引数: self (PdcaPendingTests) はテスト、symbol/session/started (str) は先物・セッション・開始時刻、
        price (float) は価格。
        返り値: None。
        """
        path = self.prices / "candle_1min" / symbol / (session + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": started, "open": price, "high": price + 1, "low": price - 1,
               "close": price, "volume": 1, "closed": "true", "volume_complete": "true"}
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def _prepare_and_seal(self) -> dict:
        """合成注釈だけを追加し、14時の時計で予測を固定する。

        引数: self (PdcaPendingTests) はテスト。
        返り値: dict は３候補の固定予測。
        """
        prepare(self.root, self.news, self.history, self.prices)
        annotation = {"as_of": AS_OF, "instruments": {
            symbol: {"channels": {key: {"score": 0.5, "confidence": 0.8, "novelty": 0.7, "priced_in": 0.2,
                                       "evidence": ["既知"], "reason": "合成の可知材料"} for key in CHANNELS}}
            for symbol in ("NK225", "TOPIX")}}
        write_json(self.root / "annotation.json", annotation)
        with patch("fundamentals.pdca_pending.now", return_value="2026-10-16T14:00:00+09:00"):
            return seal(self.root)

    # ----------------------------------------

    def test_prediction_is_fixed_before_read_and_pending_creates_no_review(self) -> None:
        """未来記事を遮断して予測を先に固定し、14時には結果や振り返りを作らない。

        引数: self (PdcaPendingTests) はテスト。
        返り値: None。
        """
        with patch("fundamentals.pdca_pending.evaluate_prices") as read_results:
            prediction = self._prepare_and_seal()
            read_results.assert_not_called()
        packet = read_json(self.root / "masked" / "packet.json")
        self.assertEqual([article["id"] for article in packet["news"]], ["既知"])
        self.assertEqual(packet["schema_version"], 2)
        self.assertEqual(_audit(self.root)[-1]["phase"], "seal")
        self.assertLess(timestamp(prediction["issued_at"]), timestamp(prediction["target"]))
        with patch("fundamentals.pdca_pending.now", return_value="2026-10-16T14:30:00+09:00"):
            result = evaluate(self.root, timestamp("2026-10-16T14:00:00+09:00"))
            with self.assertRaisesRegex(ValueError, "未来の観測時刻"):
                evaluate(self.root, timestamp("2026-10-16T15:45:00+09:00"))
        self.assertEqual(result["status"], "満期/原本待ち")
        self.assertFalse((self.root / "結果.json").exists())
        self.assertFalse((self.root / "振り返り.md").exists())
        self.assertEqual(_audit(self.root)[-1]["phase"], "seal")

    def test_changed_prediction_is_rejected_before_any_result_read(self) -> None:
        """固定済みMDの変更時は結果価格を読む前に拒否する。

        引数: self (PdcaPendingTests) はテスト。
        返り値: None。
        """
        self._prepare_and_seal()
        target = self.root / "予測.md"
        target.write_bytes(target.read_bytes() + b" ")
        with patch("fundamentals.pdca_pending.now", return_value="2026-10-16T14:30:00+09:00"), patch("fundamentals.pdca_pending.evaluate_prices") as read_results:
            with self.assertRaisesRegex(ValueError, "固定ファイルが変更"):
                evaluate(self.root, timestamp("2026-10-16T14:00:00+09:00"))
            read_results.assert_not_called()

    def test_synthetic_matured_two_labels_are_scored_once_after_seal(self) -> None:
        """一時領域で満了を模擬し、固定した６方向だけを一度採点する。

        引数: self (PdcaPendingTests) はテスト。
        返り値: None。
        """
        prediction = self._prepare_and_seal()
        for symbol in ("NK225", "TOPIX"):
            self._price(symbol, "2026-10-16_Day", "2026-10-16T15:39:00+09:00", 110.0)
        with patch("fundamentals.pdca_pending.now", return_value="2026-10-16T16:10:00+09:00"):
            result = evaluate(self.root, timestamp("2026-10-16T16:00:00+09:00"))
            with self.assertRaisesRegex(FileExistsError, "上書き"):
                evaluate(self.root, timestamp("2026-10-16T16:01:00+09:00"))
        self.assertEqual(result["forecast_hash"], prediction["forecast_hash"])
        self.assertEqual(len(result["labels"]), 2)
        self.assertEqual(len(result["comparisons"]), 6)
        for item in result["comparisons"]:
            self.assertAlmostEqual(item["actual_pct"], (110 / 105 - 1) * 100)
            self.assertEqual(item["actual_direction"], "上昇")
            self.assertEqual(item["direction_hit"], item["direction"] == "上昇")
        self.assertEqual(_audit(self.root)[-1]["phase"], "review")
        self.assertEqual(read_json(self.root / "予測.json"), prediction)


if __name__ == "__main__":
    unittest.main()
