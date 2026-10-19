"""過去教師の時点検査とコピー先の分離を合成入力で確認する。"""

import unittest
from unittest.mock import patch

import test_pdca_analyst as fixture_module
from fundamentals.common import now, read_json, write_json
from research_support.prepare_five_history import prepare_history


class PrepareFiveHistoryTests(unittest.TestCase):
    """元価格を開く前の検査と、旧入力を変更しない別コピーを検証する。"""

    def setUp(self) -> None:
        """既存テストの入力生成だけを再利用し、一時的な規約と教師を作る。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        self.fixture = fixture_module.AnalystReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.rules = self.fixture.base / "規約.json"
        self.output = self.fixture.base / "別の教師"
        self.prices = self.fixture.base / "元価格"
        self.prices.mkdir()
        self.contract = {"confirmed": True, "confirmed_at": now(),
                         "overlap_policy": "もみ合い優先・初回到達方向", "criteria": {}}
        write_json(self.rules, self.contract)
        self.row = {"symbol": "NK225", "as_of": "2026-09-30T07:45:00+09:00", "kind": "Day",
                    "label_end": "2026-09-30T15:45:00+09:00", "actual_pct": 0.8, "features": {}}
        write_json(self.fixture.measured / "初期教師.json", {"rows": [self.row]}, exclusive=False)

    def _run(self) -> dict:
        """一時領域の原本から別の教師コピーを作成する。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: dict は教師コピーの来歴。
        """
        return prepare_history(self.fixture.inputs, self.fixture.measured, self.prices, self.output, self.rules)

    def test_future_teacher_is_rejected_before_price_access(self) -> None:
        """初回締切以降に満了する教師は、数値や価格を読む前に拒否する。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        write_json(self.fixture.measured / "初期教師.json", {"rows": [{**self.row,
                   "label_end": "2026-10-01T15:45:00+09:00", "actual_pct": "まだ読まない"}]}, exclusive=False)
        with patch("research_support.prepare_five_history.load_candles") as reader:
            with self.assertRaisesRegex(ValueError, "初回締切"):
                self._run()
            reader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_price_root_cannot_be_used_as_output(self) -> None:
        """元価格内の未使用フォルダを出力として指定しても拒否する。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        self.output = self.prices / "新規フォルダ"
        with patch("research_support.prepare_five_history.load_candles") as reader:
            with self.assertRaisesRegex(ValueError, "重ね"):
                self._run()
            reader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unconfirmed_rules_cannot_create_history(self) -> None:
        """未確認の規約では過去結果を分類しない。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        write_json(self.rules, {**self.contract, "confirmed": False}, exclusive=False)
        with patch("research_support.prepare_five_history.load_candles") as reader:
            with self.assertRaisesRegex(ValueError, "確認済み"):
                self._run()
            reader.assert_not_called()

    def test_unknown_class_is_preserved_without_changing_originals(self) -> None:
        """未確定の経路候補を保ち、入力原本の全バイト列を保持する。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        originals = {path: path.read_bytes() for path in self.fixture.measured.rglob("*") if path.is_file()}
        evaluation = {"metrics": {"actual_pct": 0.8}, "correct_candidates": [],
                      "uncertain_candidates": ["上がる", "上がってから下がる"]}
        with patch("research_support.prepare_five_history.load_candles", return_value=[]) as reader, \
                patch("research_support.prepare_five_history.reference_quote", return_value=object()), \
                patch("research_support.prepare_five_history.evaluate_priority_path", return_value=evaluation):
            provenance = self._run()
        self.assertEqual(reader.call_args.kwargs["cutoff"].isoformat(), self.row["label_end"])
        self.assertEqual(provenance["teacher_rows"], 1)
        copied = read_json(self.output / "初期教師.json")["rows"][0]
        self.assertEqual(copied["five_class_evaluation"], evaluation)
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        self.assertTrue(any(name.endswith("five_class.py") for name in provenance["source_hashes"]))

    def test_mismatched_terminal_result_creates_no_copy(self) -> None:
        """旧教師と基準終点が違えば、新教師や可視入力を保存しない。

        引数: self (PrepareFiveHistoryTests) はテスト。
        返り値: None。
        """
        with patch("research_support.prepare_five_history.load_candles", return_value=[]), \
                patch("research_support.prepare_five_history.reference_quote", return_value=object()), \
                patch("research_support.prepare_five_history.evaluate_priority_path", return_value={"metrics": {"actual_pct": -0.8}}):
            with self.assertRaisesRegex(ValueError, "一致"):
                self._run()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
