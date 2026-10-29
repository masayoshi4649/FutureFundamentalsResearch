"""スキルの価格点検で未来・欠測・閾値・出力保護の境界を確認する。"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / ".agents" / "skills"
          / "forecast-index-fundamentals" / "scripts" / "inspect_session.py")
SPEC = importlib.util.spec_from_file_location("skill_inspect_session", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SkillInspectSessionTests(unittest.TestCase):
    """１分足を一時生成し、価格点検の情報境界を検証する。"""

    def rows(self, count=15, close=101):
        """
        機能: 連続する確定足の入力を作る。
        引数: count (int): 本数。close (float): 終値。
        返り値: list[dict]: テスト用１分足。
        """
        start = datetime.fromisoformat("2026-10-19T08:45:00+09:00")
        return [{"time": (start + timedelta(minutes=index)).isoformat(),
                 "open": 100, "high": max(102, close), "low": min(99, close),
                 "close": close, "closed": True, "volumeComplete": True}
                for index in range(count)]

    def inspect(self, rows, as_of="2026-10-19T09:00:00+09:00", rules=None):
        """
        機能: 共通のセッション条件で集計を実行する。
        引数: rows (list[dict]): 足。as_of (str): 情報時点。rules (list[dict] | None): 条件。
        返り値: dict: 集計結果。
        """
        return MODULE.inspect(rows, as_of, "2026-10-19T08:45:00+09:00", 100, rules=rules)

    def rule(self, **changes):
        """
        機能: ２本連続の厳密超過条件を作る。
        引数: changes (dict): 変更する条件。
        返り値: dict: 事前条件の入力。
        """
        return {"id": "上方反証", "operator": "above", "threshold": 100,
                "consecutive_bars": 2, "bar_minutes": 5, **changes}

    def test_future_and_unclosed_are_excluded(self):
        """
        機能: 未確定足・未来足を価格や反証へ混入させない。
        引数: なし。
        返り値: None。
        """
        rows = self.rows(16)
        rows[14]["closed"] = False
        rows[15].update(open=500, high=501, low=499, close=500)
        result = self.inspect(rows)
        self.assertEqual(result["prices"]["latest_close"], 101)
        self.assertEqual(result["prices"]["high"], 102)
        self.assertEqual(result["quality"]["excluded"], {
            "before_session": 0, "not_finished_as_of": 1, "not_closed": 1})
        self.assertEqual(result["quality"]["missing_completed_bar_count"], 1)
        self.assertEqual(result["quality"]["latest_close_age_seconds"], 60)

    def test_missing_bucket_breaks_consecutive_condition(self):
        """
        機能: 完全足の間に欠測がある場合の連続成立を防ぐ。
        引数: なし。
        返り値: None。
        """
        rows = self.rows()
        del rows[7]
        result = self.inspect(rows, rules=[self.rule()])
        self.assertEqual(result["quality"]["complete_five_minute_bar_count"], 2)
        self.assertIsNone(result["rules"][0]["first_match"])
        self.assertFalse(result["recent_windows"][1]["fully_observed"])

    def test_strict_threshold_and_latest_missing_are_distinct(self):
        """
        機能: 閾値一致では成立せず、古い成立を現在成立と扱わない。
        引数: なし。
        返り値: None。
        """
        equal = self.inspect(self.rows(close=100), rules=[self.rule(), self.rule(id="下方", operator="below")])
        self.assertTrue(all(rule["first_match"] is None for rule in equal["rules"]))
        rows = self.rows(10)
        result = self.inspect(rows, rules=[self.rule()])
        rule = result["rules"][0]
        self.assertEqual(rule["first_match"]["established_at"], "2026-10-19T08:55:00+09:00")
        self.assertFalse(rule["holds_at_latest_completed_five_minute_boundary"])
        self.assertEqual(len(rule["first_match"]["evidence_bars"]), 2)

    def test_timezone_duplicate_and_invalid_ohlc_rejected(self):
        """
        機能: 時計不明・重複・不正価格で計算しない。
        引数: なし。
        返り値: None。
        """
        with self.assertRaisesRegex(ValueError, "タイムゾーン"):
            self.inspect(self.rows(), "2026-10-19T09:00:00")
        rows = self.rows(1)
        duplicate = dict(rows[0], time="2026-10-18T23:45:00Z")
        with self.assertRaisesRegex(ValueError, "重複"):
            self.inspect(rows + [duplicate])
        rows[0]["low"] = 102
        with self.assertRaisesRegex(ValueError, "OHLC"):
            self.inspect(rows)
        with self.assertRaisesRegex(ValueError, "JSON配列"):
            self.inspect([], rules={})

    def test_missing_open_stays_unknown_and_window_uses_asof(self):
        """
        機能: 最初の観測始値で寄付欠測を埋めず、古い足を直近窓へ移動しない。
        引数: なし。
        返り値: None。
        """
        result = self.inspect(self.rows(5)[1:])
        self.assertIsNone(result["session_open"])
        self.assertEqual(result["prices"]["first_observed_open"], 100)
        self.assertIsNone(result["recent_windows"][0]["prices"])
        self.assertEqual(result["recent_windows"][0]["missing_bar_count"], 5)

    def test_json_snapshot_ignored_and_output_never_overwritten(self):
        """
        機能: 未来のsnapshotで足を追加せず、既存出力を保護する。
        引数: なし。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "入力.json"
            target = Path(directory) / "点検.json"
            source.write_text(json.dumps({"data": self.rows(16),
                "snapshot": {"time": "2099-01-01T00:00:00+09:00", "close": 999}}), encoding="utf-8")
            original = source.read_bytes()
            args = ["--input", str(source), "--as-of", "2026-10-19T09:00:00+09:00",
                    "--session-start", "2026-10-19T08:45:00+09:00", "--p0", "100", "--output", str(target)]
            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main(args)
            first = target.read_bytes()
            result = json.loads(first)
            self.assertEqual(result["quality"]["included_bar_count"], 15)
            self.assertEqual(result["prices"]["latest_close"], 101)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                MODULE.main(args)
            self.assertEqual(target.read_bytes(), first)
            self.assertEqual(source.read_bytes(), original)

    def test_csv_auto_format_and_volume_quality(self):
        """
        機能: CSVを自動識別し、出来高未完了を価格未確定と混同しない。
        引数: なし。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "入力.csv"
            source.write_text("time,open,high,low,close,closed,volume_complete\n"
                              "2026-10-19T08:45:00+09:00,100,102,99,101,true,false\n", encoding="utf-8")
            rows, metadata = MODULE.read_input(source)
            result = self.inspect(rows, "2026-10-19T08:46:00+09:00")
            self.assertEqual(metadata["format"], "csv")
            self.assertEqual(result["quality"]["included_bar_count"], 1)
            self.assertEqual(result["quality"]["volume_incomplete_bar_count"], 1)
            self.assertAlmostEqual(result["prices"]["recovery_from_low_pct_p0"], 2)
            self.assertAlmostEqual(result["prices"]["close_position_pct_observed_range"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
