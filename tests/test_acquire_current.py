"""満了後の価格取得で、時計・確定足・原本保護を検証する。"""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_support.acquire_current import acquire, acquire_archived, validate_response


class CurrentAcquisitionTests(unittest.TestCase):
    """結果を早めたり、未確定足を終点へ使うことを防ぐ。"""

    def setUp(self) -> None:
        """最終１分足と取得時計を持つ合成応答を用意する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        self.session = "2026-10-16_Day"
        self.payload = {"snapshot": {"sessionId": self.session, "code": "NK225",
                        "asOf": "2026-10-16T15:45:02+09:00",
                        "session": {"endsAt": "2026-10-16T15:45:00+09:00"}},
                        "data": [{"time": "2026-10-16T15:39:00+09:00", "open": 100, "high": 101,
                                  "low": 99, "close": 100, "volume": 1, "volumeComplete": False, "closed": True}]}
        self.observed = "2026-10-16T15:45:03+09:00"

    def test_completion_and_volume_flags_remain_distinct(self) -> None:
        """確定足でも出来高不完全のフラグをそのまま保持する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        rows = validate_response(self.payload, "NK225", self.session, self.observed)
        self.assertEqual(rows[0]["closed"], "true")
        self.assertEqual(rows[0]["volume_complete"], "false")

    def test_future_price_is_not_interpreted(self) -> None:
        """終点後の足は価格が不正でも解釈せず取り除く。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        self.payload["data"].append({"time": "2026-10-16T15:45:00+09:00", "close": object()})
        self.assertEqual(len(validate_response(self.payload, "NK225", self.session, self.observed)), 1)

    def test_session_symbol_end_and_snapshot_clocks_are_checked(self) -> None:
        """違う場・先物・終了予定・未来の収集時計を拒否する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        for key, value in (("sessionId", "2026-10-15_Night"), ("code", "TOPIX"),
                           ("asOf", "2026-10-16T15:44:59+09:00"), ("asOf", "2026-10-16T15:46:00+09:00")):
            with self.subTest(key=key, value=value):
                payload = copy.deepcopy(self.payload)
                payload["snapshot"][key] = value
                with self.assertRaises(ValueError):
                    validate_response(payload, "NK225", self.session, self.observed)
        self.payload["snapshot"]["session"]["endsAt"] = "2026-10-16T15:40:00+09:00"
        with self.assertRaises(ValueError):
            validate_response(self.payload, "NK225", self.session, self.observed)

    def test_unclosed_candle_and_string_boolean_cannot_become_a_result(self) -> None:
        """未確定足だけの応答と文字列の真偽値を拒否する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        for flag in (False, "true"):
            payload = copy.deepcopy(self.payload)
            payload["data"][0]["closed"] = flag
            with self.assertRaises(ValueError):
                validate_response(payload, "NK225", self.session, self.observed)

    def test_before_maturity_never_calls_network_or_creates_output(self) -> None:
        """満了前は通信にも書込みにも進まない。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "新規"
            with patch("research_support.acquire_current.now", return_value="2026-10-16T15:44:59+09:00"), patch("research_support.acquire_current.urlopen") as request:
                with self.assertRaises(ValueError):
                    acquire(Path(temporary), output, self.session)
                request.assert_not_called()
            self.assertFalse(output.exists())

    def test_source_and_existing_output_are_never_overwritten(self) -> None:
        """原本内部と既存保存先への上書きを通信前に拒否する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with patch("research_support.acquire_current.now", return_value=self.observed), patch("research_support.acquire_current.urlopen") as request:
                with self.assertRaises(FileExistsError):
                    acquire(source, source, self.session)
                with self.assertRaises(ValueError):
                    acquire(source, source / "内部", self.session)
                request.assert_not_called()

    def test_another_sessions_candle_is_rejected(self) -> None:
        """異なる場の過去足を当日の経路へ混ぜない。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        self.payload["data"][0]["time"] = "2026-10-16T05:54:00+09:00"
        with self.assertRaises(ValueError):
            validate_response(self.payload, "NK225", self.session, self.observed)

    def test_completed_capture_copies_originals_without_changing_them(self) -> None:
        """２先物の取得・列変換・証跡と、元CSV不変をまとめて確認する。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "元", Path(temporary) / "新規"
            originals = []
            for symbol in ("NK225", "TOPIX"):
                path = source / "candle_1min" / symbol / "2026-10-15_Night.csv"
                path.parent.mkdir(parents=True)
                path.write_text("time,open,high,low,close,volume,volume_complete,closed\n"
                                "2026-10-16T05:54:00+09:00,100,101,99,100,1,true,true\n", encoding="utf-8")
                originals.append((path, path.read_bytes()))

            def response_for(url: str, timeout: int) -> io.BytesIO:
                """要求先物に一致する合成のHTTP本文を返す。

                引数: url (str) は要求先、timeout (int) は制限秒数。
                返り値: io.BytesIO はJSONを保持する読取ストリーム。
                """
                payload = copy.deepcopy(self.payload)
                payload["snapshot"]["code"] = "TOPIX" if "/TOPIX/" in url else "NK225"
                return io.BytesIO(json.dumps(payload).encode("utf-8"))

            with patch("research_support.acquire_current.now", return_value=self.observed), patch("research_support.acquire_current.urlopen", side_effect=response_for):
                manifest = acquire(source, output, self.session)
            self.assertEqual(len(manifest["copied"]), 2)
            self.assertTrue((output / "取得証跡.json").is_file())
            for symbol in ("NK225", "TOPIX"):
                self.assertEqual(manifest["acquired"][symbol]["last_end"], "2026-10-16T15:40:00+09:00")
                self.assertFalse(manifest["acquired"][symbol]["is_official_close"])
            for path, original in originals:
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual((output / path.relative_to(source)).read_bytes(), original)

    def test_archived_capture_checks_maturity_and_preserves_sources(self) -> None:
        """API切替後も確定保存CSVの読取だけで評価ルートを作れる。

        引数: self (CurrentAcquisitionTests) はテスト。
        返り値: None。
        """
        with tempfile.TemporaryDirectory() as temporary:
            source, archive, output = (Path(temporary) / name for name in ("元", "共有", "新規"))
            saved = {}
            for symbol in ("NK225", "TOPIX"):
                for root, session, time in ((source, "2026-10-15_Night", "05:54"), (archive, self.session, "15:39")):
                    path = root / "candle_1min" / symbol / (session + ".csv")
                    path.parent.mkdir(parents=True)
                    path.write_text("time,open,high,low,close,volume,volume_complete,closed\n"
                                    f"2026-10-16T{time}:00+09:00,100,101,99,100,1,true,true\n", encoding="utf-8")
                    saved[path] = path.read_bytes()
            with patch("research_support.acquire_current.now", return_value="2026-10-16T15:44:59+09:00"):
                with self.assertRaises(ValueError):
                    acquire_archived(source, output, self.session, archive)
            self.assertFalse(output.exists())
            with patch("research_support.acquire_current.now", return_value=self.observed):
                result = acquire_archived(source, output, self.session, archive)
            self.assertEqual(result["method"], "満了後の確定保存CSV")
            self.assertEqual(result["acquired"]["NK225"]["last_end"], "2026-10-16T15:40:00+09:00")
            for path, body in saved.items():
                self.assertEqual(path.read_bytes(), body)


if __name__ == "__main__":
    unittest.main()
