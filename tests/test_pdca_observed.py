"""合成した１分足で既反応量の基準・締切・欠測を検証する。"""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.pdca_observed import observed_state


class ObservedStateTests(unittest.TestCase):
    """実際の予測対象価格を使わず、観測特徴の契約を検証する。"""

    def setUp(self) -> None:
        """合成足だけを置く一時的な価格ルートを用意する。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "candle_1min" / "NK225").mkdir(parents=True)

    def _row(self, started: str, price: float = 100, **changes) -> dict:
        """任意の開始時刻の正常な１分足を作成する。

        引数: self (ObservedStateTests) はテスト、started (str) は開始時刻、
            price (float) は標準価格、changes (dict) は列の変更。
        返り値: dict は CSV 用の行。
        """
        value = {"time": started, "open": price, "high": price, "low": price,
                 "close": price, "volume": 1, "volume_complete": "true", "closed": "true"}
        value.update(changes)
        return value

    def _write(self, session: str, rows: list[dict], symbol: str = "NK225", interval: int = 1) -> Path:
        """合成した足を対象銘柄の CSV に保存する。

        引数: self (ObservedStateTests) はテスト、session (str) はセッション、rows (list[dict]) は足、
            symbol (str) は先物名、interval (int) は足間隔。
        返り値: Path は保存先。
        """
        directory = self.root / f"candle_{interval}min" / symbol
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (session + ".csv")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self._row("未使用")))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _pair(self, night_end: str = "2026-10-02T05:54:00+09:00") -> None:
        """直前DayとNightに、異なる終値とNight高安を作成する。

        引数: self (ObservedStateTests) はテスト、night_end (str) は Night 最終足開始。
        返り値: None。
        """
        self._write("2026-10-01_Day", [self._row("2026-10-01T15:39:00+09:00", 100)])
        self._write("2026-10-01_Night", [
            self._row("2026-10-01T17:00:00+09:00", 95, low=90, high=96),
            self._row(night_end, 97, high=99, low=96, close=98),
        ])

    # ----------------------------------------

    def test_same_futures_previous_close_and_two_sessions(self) -> None:
        """直前Nightの変化を直前Day基準で算出し、次Day基準をNight終値にする。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair()
        state = observed_state(self.root, "NK225", "2026-10-02T07:45:00+09:00")
        self.assertEqual(state["status"], "利用可能")
        self.assertEqual(state["reference"]["price"], 98)
        self.assertEqual(state["reference"]["time"], "2026-10-02T05:55:00+09:00")
        self.assertEqual(state["reference"]["tail_missing_minutes"], 5)
        self.assertFalse(state["reference"]["is_official_close"])
        previous = state["previous_session"]
        self.assertEqual(previous["reference"]["price"], 100)
        self.assertAlmostEqual(previous["return_pct"], -2)
        self.assertEqual((previous["high"], previous["low"]), (99, 90))
        self.assertAlmostEqual(previous["low_to_last_pct"], (98 / 90 - 1) * 100)
        self.assertAlmostEqual(previous["close_location"], 8 / 9)
        self.assertAlmostEqual(previous["range_pct"], 9)
        self.assertEqual(state["preceding_session"]["session"], "2026-10-01_Day")
        self.assertIsNone(state["preceding_session"]["return_pct"])

    def test_missing_slots_and_late_window_are_explicit(self) -> None:
        """末尾窓の少数観測を完全な60分変化と扱わず、公式末尾不足も保存する。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair()
        previous = observed_state(self.root, "NK225", "2026-10-02T07:45:00+09:00")["previous_session"]
        self.assertEqual(previous["expected_candles"], 780)
        self.assertEqual(previous["observed_candles"], 2)
        self.assertEqual(previous["missing_interval_slots"], 778)
        self.assertEqual(previous["tail_missing_minutes"], 5)
        self.assertEqual(previous["internal_missing_minutes"], 773)
        window = previous["late_window"]
        self.assertEqual(window["nominal_start"], "2026-10-02T05:00:00+09:00")
        self.assertEqual(window["observed_start"], "2026-10-02T05:54:00+09:00")
        self.assertEqual(window["observed_candles"], 1)
        self.assertEqual(window["missing_interval_slots"], 59)
        self.assertFalse(window["complete"])
        self.assertAlmostEqual(window["change_pct"], (98 / 97 - 1) * 100)

    def test_complete_late_window_and_neutral_flat_range(self) -> None:
        """連続60足だけを完全窓とし、平坦高安の終値位置は null にする。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        start = datetime.fromisoformat("2026-10-02T05:00:00+09:00")
        rows = [self._row((start + timedelta(minutes=index)).isoformat()) for index in range(60)]
        self._write("2026-10-01_Night", rows)
        previous = observed_state(self.root, "NK225", "2026-10-02T07:45:00+09:00")["previous_session"]
        self.assertIsNone(previous["close_location"])
        self.assertIsNone(previous["range_pct"])
        self.assertEqual(previous["low_to_last_pct"], 0)
        self.assertTrue(previous["late_window"]["complete"])
        self.assertEqual(previous["late_window"]["change_pct"], 0)
        self.assertEqual(previous["tail_missing_minutes"], 0)
        self.assertEqual(previous["leading_missing_minutes"], 720)

    def test_ten_minute_proxy_boundary_and_no_five_minute_substitution(self) -> None:
        """10分代理境界を厳密に守り、５分足があっても不足した１分基準を補わない。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair("2026-10-02T05:49:00+09:00")
        cutoff = "2026-10-02T07:45:00+09:00"
        self.assertEqual(observed_state(self.root, "NK225", cutoff)["reference"]["price"], 98)
        self._pair("2026-10-02T05:48:00+09:00")
        self._write("2026-10-01_Night", [self._row("2026-10-02T05:50:00+09:00", 1000)], interval=5)
        state = observed_state(self.root, "NK225", cutoff)
        self.assertEqual(state["status"], "基準不足")
        self.assertIsNone(state["reference"])
        self.assertEqual(state["previous_session"]["tail_missing_minutes"], 11)

    def test_future_files_are_not_opened_and_future_rows_do_not_change_features(self) -> None:
        """未来ファイルを開かず、同ファイル内の締切後価格変更も過去特徴に反映しない。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair()
        cutoff = "2026-10-02T07:45:00+09:00"
        original = observed_state(self.root, "NK225", cutoff)
        future = self._write("2026-10-02_Day", [])
        future.write_text("壊れた未来本文", encoding="utf-8")
        night_path = self.root / "candle_1min" / "NK225" / "2026-10-01_Night.csv"
        with night_path.open("a", encoding="utf-8") as stream:
            stream.write("2026-10-02T07:45:00+09:00,未来価格,未来高値,未来安値,未来終値,NaN,不明,不明\n")
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            """未来ファイルを実際に開こうとした場合に検証を失敗させる。

            引数: path (Path) は対象、args (tuple) と kwargs (dict) は読取オプション。
            返り値: object は元のファイルハンドル。
            """
            if path == future:
                raise AssertionError("未来セッションを開いた")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            self.assertEqual(observed_state(self.root, "NK225", cutoff), original)

    def test_incomplete_candles_and_not_yet_ended_session_are_excluded(self) -> None:
        """未確定足と公式終了前セッションを完了済み特徴へ混入させない。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair()
        self._write("2026-10-02_Day", [
            self._row("2026-10-02T15:39:00+09:00", 100),
            self._row("2026-10-02T15:43:00+09:00", 1000, closed="false"),
            self._row("2026-10-02T15:45:00+09:00", 1000),
        ])
        state = observed_state(self.root, "NK225", "2026-10-02T15:45:00+09:00")
        self.assertEqual(state["target_session"], "2026-10-02_Night")
        self.assertEqual(state["reference"]["price"], 100)
        self.assertEqual(state["previous_session"]["observed_candles"], 1)
        self.assertAlmostEqual(state["previous_session"]["return_pct"], (100 / 98 - 1) * 100)

    def test_missing_previous_session_empty_data_and_wrong_kind(self) -> None:
        """未収録と期待セッション欠落を明示し、古い同種セッションへ補完しない。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        cutoff = "2026-10-02T07:45:00+09:00"
        state = observed_state(self.root, "NK225", cutoff)
        self.assertEqual(state["status"], "基準不足")
        self.assertIsNone(state["previous_session"])
        self.assertIsNone(state["preceding_session"])
        self._write("2026-09-30_Night", [self._row("2026-10-01T05:54:00+09:00")])
        self._write("2026-10-01_Day", [self._row("2026-10-01T15:39:00+09:00")])
        state = observed_state(self.root, "NK225", cutoff)
        self.assertEqual(state["status"], "基準不足")
        self.assertIsNone(state["reference"])
        self.assertIn("期待セッション", state["reason"])

    def test_same_symbol_and_timezone_are_preserved(self) -> None:
        """別銘柄価格へ依存せず、UTCを日本時間へ統一し、時差の省略を拒否する。

        引数: self (ObservedStateTests) はテスト。
        返り値: None。
        """
        self._pair()
        cutoff = "2026-10-02T07:45:00+09:00"
        original = observed_state(self.root, "NK225", cutoff)
        self._write("2026-10-01_Night", [self._row("2026-10-02T05:54:00+09:00", 5000)], symbol="TOPIX")
        self.assertEqual(observed_state(self.root, "NK225", "2026-10-01T22:45:00Z"), original)
        self.assertEqual(observed_state(self.root, "TOPIX", cutoff)["reference"]["price"], 5000)
        with self.assertRaises(ValueError):
            observed_state(self.root, "NK225", "2026-10-02T07:45:00")
        with self.assertRaises(ValueError):
            observed_state(self.root, "現物", cutoff)


if __name__ == "__main__":
    unittest.main()
