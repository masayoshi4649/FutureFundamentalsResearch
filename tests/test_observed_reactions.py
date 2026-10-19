"""分類正解と材料の公表時刻の先後を混同しないための観測補足を検査する。"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import JST
from fundamentals.prices import Candle
from research_support import observed_reactions as observed


def candle(hour: int, minute: int, high: float, low: float, *, closed: bool = True) -> Candle:
    """観測時計を明示した合成足を作る。

    引数: hour/minute (int) は足開始、high/low (float) は高安、closed (bool) は確定状態。
    返り値: Candle は検査用の足。始終値は高安の中点。
    """
    start = datetime(2026, 9, 24, hour, minute, tzinfo=JST)
    price = (high + low) / 2
    return Candle(start, start + timedelta(minutes=1), price, high, low, price,
                  1, "2026-09-24_Night", "NK225", closed=closed)


class ObservedReactionTests(unittest.TestCase):
    """欠測・足内順序・公表前反転・開示前遮断を検査する。"""

    def test_one_bar_does_not_prove_intrabar_order(self):
        """同じ足の両閾値到達を順序付き反転にしない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        result = observed.observed_path([candle(18, 0, 100.6, 99.4)], 100)
        self.assertTrue(all(row["first_observed_touch"] for row in result.values()))
        self.assertTrue(all(row["later_bar_reversal"] is None for row in result.values()))

    def test_gap_does_not_move_observation_to_announcement(self):
        """欠測後の反転確認を予定指標の時計へ付け替えない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        bars = [candle(18, 43, 99.5, 99.4), candle(21, 20, 100, 99.9)]
        value = observed.observed_path(bars, 100)["down"]["later_bar_reversal"]
        self.assertEqual(value["prior_extreme"]["start"], bars[0].time.isoformat())
        self.assertEqual(value["reversal_bar"]["start"], bars[1].time.isoformat())
        self.assertLess(bars[1].end, datetime(2026, 9, 24, 21, 30, tzinfo=JST))

    def test_exact_threshold_is_inclusive(self):
        """P0比0.5％の到達・反転を境界で落とさない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        bars = [candle(18, 0, 99.5, 99.5), candle(18, 1, 100, 100)]
        value = observed.observed_path(bars, 100)
        self.assertIsNotNone(value["down"]["later_bar_reversal"])
        self.assertIsNone(value["up"]["first_observed_touch"])

    def test_empty_and_uncrossed_do_not_invent_events(self):
        """空観測や閾値未到達をイベントへ補完しない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        for bars in ([], [candle(18, 0, 100.1, 99.9)]):
            value = observed.observed_path(bars, 100)
            self.assertTrue(all(row["first_observed_touch"] is None for row in value.values()))

    def test_invalid_reference_is_rejected(self):
        """ゼロ・非有限のP0で閾値を作らない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        for reference in (0, -100, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                observed.observed_path([], reference)

    def test_unclosed_and_reversed_bars_are_rejected(self):
        """未確定足や時間逆転を反応の証拠にしない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        for bars in ([candle(18, 0, 100, 100, closed=False)],
                     [candle(18, 1, 100, 100), candle(18, 0, 100, 100)]):
            with self.assertRaises(ValueError):
                observed.observed_path(bars, 100)

    def test_before_reveal_does_not_read_result(self):
        """予測固定だけの段階では結果や価格を開かない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        with patch.object(observed, "_audit", return_value=[{"phase": "seal"}]), \
                patch.object(observed, "read_json") as read, patch.object(observed, "_read_file") as prices:
            with self.assertRaises(ValueError):
                observed.build_observed_supplement(Path("合成"))
            read.assert_not_called()
            prices.assert_not_called()

    def test_changed_price_source_is_not_opened(self):
        """開示時のSHAと異なるCSVを解析しない。

        引数: self (ObservedReactionTests)。返り値: None。
        """
        result = {"labels": [{"symbol": "NK225", "full_reference": {
            "source_file": "合成.csv", "source_file_hash": "元のSHA"}}]}
        with patch.object(observed, "_audit", return_value=[{"phase": "reveal"}]), \
                patch.object(observed, "read_json", return_value=result), \
                patch.object(observed, "_file_hash", side_effect=["結果のSHA", "異なるSHA"]), \
                patch.object(observed, "_read_file") as prices:
            with self.assertRaises(ValueError):
                observed.build_observed_supplement(Path("合成"))
            prices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
