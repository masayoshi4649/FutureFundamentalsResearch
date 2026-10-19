"""予測時刻の境界、価格異常と価格データの欠測候補を検証する。"""

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fundamentals.prices import inventory, load_candles, read_book_sample, session_groups


class PriceTests(unittest.TestCase):
    """一時フォルダの価格だけを使い、読み取り境界を検証する。"""

    def setUp(self):
        """
        一時的な価格ルートを準備する。

        引数: なし。
        返り値: None。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_rows(self, rows, session="2026-10-15_Day", interval=1):
        """
        テスト用 CSV を一時フォルダに書く。

        引数:
            rows (list[dict]): 足データ。
            session (str): ファイルのセッション名。
            interval (int): 足間隔。
        返り値:
            Path: 書き込んだ CSV。
        """
        directory = self.root / f"candle_{interval}min" / "NK225"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{session}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "time", "open", "high", "low", "close", "volume", "volume_complete", "closed"
            ])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, time="2026-10-15T08:45:00+09:00", **changes):
        """
        標準のテスト用価格行を作る。

        引数:
            time (str): 開始時刻。
            changes (dict): 上書きする列。
        返り値:
            dict: CSV に使う価格行。
        """
        row = dict(time=time, open=100, high=102, low=99, close=101,
                   volume=10, volume_complete="true", closed="true")
        row.update(changes)
        return row

    def test_cutoff_requires_bar_end_and_closed_flag(self):
        """
        末尾足と未確定足を予測入力に混入させない。

        引数: なし。
        返り値: None。
        """
        self.write_rows([
            self.row(), self.row("2026-10-15T08:46:00+09:00"),
            self.row("2026-10-15T08:47:00+09:00", closed="false"),
        ])
        before = load_candles(self.root, "NK225", cutoff=datetime.fromisoformat("2026-10-15T08:45:59+09:00"))
        self.assertEqual(before, [])
        exact = load_candles(self.root, "NK225", cutoff=datetime.fromisoformat("2026-10-14T23:46:00+00:00"))
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].end.isoformat(), "2026-10-15T08:46:00+09:00")
        self.assertIsNone(exact[0].contract)
        self.assertEqual(len(load_candles(self.root, "NK225")), 2)

    def test_five_minute_end_is_not_start(self):
        """
        5 分足が終了する前に取得されないことを検証する。

        引数: なし。
        返り値: None。
        """
        self.write_rows([self.row()], interval=5)
        self.assertEqual(load_candles(self.root, "NK225", 5, datetime.fromisoformat("2026-10-15T08:49:00+09:00")), [])
        self.assertEqual(len(load_candles(self.root, "NK225", 5, datetime.fromisoformat("2026-10-15T08:50:00+09:00"))), 1)

    def test_future_prices_do_not_change_past_input(self):
        """
        予測時点より後の価格異常を過去入力の変更や中断につなげない。

        引数: なし。
        返り値: None。
        """
        limit = datetime.fromisoformat("2026-10-15T08:46:00+09:00")
        self.write_rows([self.row(), self.row("2026-10-15T08:46:00+09:00")])
        original = load_candles(self.root, "NK225", cutoff=limit)
        self.write_rows([
            self.row(),
            self.row("2026-10-15T08:46:00+09:00", open="将来の不正価格", volume="NaN", closed="不明"),
        ])
        self.assertEqual(load_candles(self.root, "NK225", cutoff=limit), original)
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225")
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225", cutoff=datetime.fromisoformat("2026-10-15T08:47:00+09:00"))
        with self.assertRaises(ValueError):
            inventory(self.root)

    def test_future_sessions_are_not_opened(self):
        """
        未来セッションの壊れた CSV を過去予測で開かない。

        引数: なし。
        返り値: None。
        """
        self.write_rows([self.row()])
        limit = datetime.fromisoformat("2026-10-15T16:00:00+09:00")
        original = load_candles(self.root, "NK225", cutoff=limit)
        night = self.root / "candle_1min" / "NK225" / "2026-10-15_Night.csv"
        future = self.root / "candle_1min" / "NK225" / "2026-10-16_Day.csv"
        night.write_text("価格の必須列も日時も存在しない\n", encoding="utf-8")
        future.write_text("価格の必須列も日時も存在しない\n", encoding="utf-8")
        self.assertEqual(load_candles(self.root, "NK225", cutoff=limit), original)
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225")

    def test_future_only_rows_do_not_require_price_columns(self):
        """
        対象セッションでも全行が未来なら価格列を検証しない。

        引数: なし。
        返り値: None。
        """
        path = self.write_rows([self.row()])
        path.write_text("time\n2026-10-15T08:46:00+09:00\n", encoding="utf-8")
        limit = datetime.fromisoformat("2026-10-15T08:46:00+09:00")
        self.assertEqual(load_candles(self.root, "NK225", cutoff=limit), [])
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225")
        path.write_text("time\n日時の構文が不正\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225", cutoff=limit)

    def test_session_gap_is_not_a_missing_candle(self):
        """
        日中と夜間の間を欠測扱いせず、セッション内の空白を数える。

        引数: なし。
        返り値: None。
        """
        self.write_rows([self.row(), self.row("2026-10-15T08:48:00+09:00", volume_complete="false")])
        self.write_rows([self.row("2026-10-15T17:00:00+09:00")], session="2026-10-15_Night")
        candles = load_candles(self.root, "NK225")
        self.assertEqual([len(group) for group in session_groups(list(reversed(candles)))], [2, 1])
        facts = inventory(self.root)["symbols"]["NK225"]["1min"]
        self.assertEqual(facts["missing_interval_slots"], 2)
        self.assertEqual(facts["gap_count"], 1)
        self.assertEqual(facts["incomplete_volume_rows"], 1)

    def test_timezone_is_required_and_utc_is_normalized(self):
        """
        タイムゾーン省略を拒否し、UTC 入力を日本時間に揃える。

        引数: なし。
        返り値: None。
        """
        self.write_rows([self.row("2026-10-14T23:45:00+00:00")])
        self.assertEqual(load_candles(self.root, "NK225")[0].time.isoformat(), "2026-10-15T08:45:00+09:00")
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225", cutoff=datetime(2026, 10, 15, 9))
        self.write_rows([self.row("2026-10-15T08:45:00")])
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225")

    def test_invalid_values_are_rejected(self):
        """
        不正 OHLC・非有限数・フラグ・重複時刻を拒否する。

        引数: なし。
        返り値: None。
        """
        cases = [dict(high=100), dict(low=103), dict(open="NaN"), dict(close="inf"),
                 dict(volume=-1), dict(closed="yes"), dict(volume_complete="yes")]
        for change in cases:
            with self.subTest(change=change):
                self.write_rows([self.row(**change)])
                with self.assertRaises(ValueError):
                    load_candles(self.root, "NK225")
        self.write_rows([self.row(), self.row()])
        with self.assertRaises(ValueError):
            load_candles(self.root, "NK225")

    def test_unknown_symbol_and_interval_are_rejected(self):
        """
        対象外の銘柄と足間隔を拒否する。

        引数: なし。
        返り値: None。
        """
        for symbol, interval in (("NK225mini", 1), ("../TOPIX", 1), ("NK225", 3), ("NK225", True)):
            with self.subTest(symbol=symbol, interval=interval):
                with self.assertRaises(ValueError):
                    load_candles(self.root, symbol, interval)

    def test_book_sample_stops_without_reading_future_lines(self):
        """
        生板を指定件数だけ読み、後続の全ファイル読込を行わない。

        引数: なし。
        返り値: None。
        """
        path = self.root / "book.jsonl"
        path.write_text('{"CurrentPriceTime":null}\n不正な後続行\n', encoding="utf-8")
        self.assertEqual(read_book_sample(path), [{"CurrentPriceTime": None}])
        self.assertEqual(read_book_sample(path, 0), [])
        with self.assertRaises(ValueError):
            read_book_sample(path, -1)


if __name__ == "__main__":
    unittest.main()
