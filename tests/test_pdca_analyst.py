"""全文直接判断の教師締切、予測固定と担当証跡を合成入力で検証する。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import digest, read_json, write_json
from fundamentals.pdca import CHANNELS
from fundamentals.pdca_analyst import prepare_analyst, review_analyst, seal_analyst
from fundamentals.replay import _audit, _file_hash


class AnalystReplayTests(unittest.TestCase):
    """結果を見ない準備と、固定後の結果開示を分けて確認する。"""

    def setUp(self) -> None:
        """２回の可視入力と結果を含まない実測状態を一時領域に作る。

        引数: self (AnalystReplayTests) はテスト。
        返り値: None。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.inputs, self.measured = (self.base / name for name in ("新実験", "可視原本", "実測原本"))
        self.root.mkdir()
        for name in ("実験計画.md", "予測担当の手順.md"):
            (self.root / name).write_text("# 合成した固定指示\n", encoding="utf-8")
        self.identities = []
        for index in (1, 2):
            identity, day = f"{index:02d}_合成", f"2026-10-0{index}"
            self.identities.append(identity)
            packet = {"as_of": day + "T07:45:00+09:00", "session": day + "_Day", "kind": "Day",
                      "targets": {"today": day + "T15:45:00+09:00"}, "instruments": {},
                      "news": [{"id": "可視記事", "available_at": day + "T06:00:00+09:00",
                                "published_at": day + "T06:00:00+09:00", "fetched_at": day + "T06:00:00+09:00",
                                "headline": "合成材料", "body_text": "結果を含まない全文"}]}
            annotation = {"as_of": packet["as_of"], "instruments": {}}
            states = {}
            for symbol in ("NK225", "TOPIX"):
                reference = {"price": 100, "time": day + "T05:55:00+09:00", "session": "前日_Night"}
                packet["instruments"][symbol] = {"reference": reference, "horizons": {
                    "today": {"distribution": {"n": 0, "terminal_range_pct": None, "path_range_pct": None}}}}
                annotation["instruments"][symbol] = {"channels": {
                    key: {"score": 0, "confidence": 0, "novelty": 0, "priced_in": 0, "evidence": [], "reason": "合成"}
                    for key in CHANNELS}}
                states[symbol] = {"reference": copy.deepcopy(reference), "previous_session": {
                    "return_pct": -0.4, "close_location": 0.4, "range_pct": 1,
                    "late_window": {"change_pct": -0.1}}}
            packet["packet_hash"] = digest(packet)
            write_json(self.inputs / "masked" / (identity + ".json"), packet)
            write_json(self.inputs / "annotations" / (identity + ".json"), annotation)
            write_json(self.measured / identity / "入力.json", {"as_of": packet["as_of"], "session": packet["session"],
                       "states": states, "market_facts": {"series": {}, "observations": {}, "flags": [],
                                                          "excluded": [{"id": "未来情報"}], "causality": "未識別"}})
        write_json(self.measured / "初期教師.json", {"rows": []})

    def _prepare(self, index: int = 1) -> dict:
        """指定回の結果を含まない入力を準備する。

        引数: self (AnalystReplayTests) はテスト、index (int) は回番号。
        返り値: dict は切出入力。
        """
        return prepare_analyst(self.root, self.inputs, self.measured, index)

    def _seal(self, index: int = 1) -> dict:
        """合成した担当の短い予測を、入力ハッシュへ束縛して固定する。

        引数: self (AnalystReplayTests) はテスト、index (int) は回番号。
        返り値: dict は固定予測。
        """
        path = self.root / self.identities[index - 1]
        write_json(path / "予測入力.json", {"instruments": {symbol: {
            "direction": "上昇", "evidence_ids": ["可視記事"], "already_reacted": "既反応の観測。",
            "additional_reaction": "次場の追加反応仮説。", "invalidation": "反対材料。"}
            for symbol in ("NK225", "TOPIX")}})
        return seal_analyst(path, {"agent_id": f"独立担当{index}", "task_name": f"合成{index}",
                                  "fork_turns": "none", "completed": True,
                                  "input_file_hash": _file_hash(path / "masked" / "入力.json")})

    def _labels(self) -> tuple[list[dict], str]:
        """初回の合成原結果だけを作成する。

        引数: self (AnalystReplayTests) はテスト。
        返り値: tuple[list[dict], str] は両先物の結果と原結果識別子。
        """
        return ([{"symbol": symbol, "label_end": "2026-10-01T15:45:00+09:00", "actual_pct": 1,
                  "quote_price": 101, "reference_price": 100} for symbol in ("NK225", "TOPIX")], "合成原結果")

    # ----------------------------------------

    def test_future_teacher_is_rejected_before_prediction_or_result(self) -> None:
        """未来教師を未開示値のまま拒否し、原結果読取へ到達しない。

        引数: self (AnalystReplayTests) はテスト。
        返り値: None。
        """
        write_json(self.measured / "初期教師.json", {"rows": [{"symbol": "NK225",
                   "as_of": "2026-10-01T07:45:00+09:00", "label_end": "2026-10-01T15:45:00+09:00",
                   "actual_pct": "未開示"}]}, exclusive=False)
        with patch("fundamentals.pdca_analyst._source_labels") as result_read:
            with self.assertRaisesRegex(ValueError, "締切より前"):
                self._prepare()
            result_read.assert_not_called()

    def test_result_requires_seal_and_next_round_requires_review(self) -> None:
        """準備中は結果を読めず、固定・開示・振り返りの後に次回教師へ渡す。

        引数: self (AnalystReplayTests) はテスト。
        返り値: None。
        """
        packet = self._prepare()
        first = self.root / self.identities[0]
        self.assertNotIn("excluded", packet["market_facts"])
        with patch("fundamentals.pdca_analyst._source_labels", return_value=self._labels()) as result_read:
            with self.assertRaisesRegex(ValueError, "固定後"):
                review_analyst(first, self.base / "原結果")
            result_read.assert_not_called()
            with self.assertRaisesRegex(ValueError, "振り返り"):
                self._prepare(2)
            self._seal()
            review_analyst(first, self.base / "原結果")
            result_read.assert_called_once()
        self.assertEqual([entry["phase"] for entry in _audit(first)], ["prepare", "seal", "reveal", "review"])
        second = self._prepare(2)
        self.assertEqual(len(second["known_history"]), 2)
        self.assertEqual(len(second["own_past_reviews"]), 1)

    def test_actor_change_after_seal_blocks_result_read(self) -> None:
        """固定後の担当証跡の改変を、原結果読取前に検知する。

        引数: self (AnalystReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        first = self.root / self.identities[0]
        actor = read_json(first / "actor.json")
        actor["agent_id"] = "差替担当"
        write_json(first / "actor.json", actor, exclusive=False)
        with patch("fundamentals.pdca_analyst._source_labels") as result_read:
            with self.assertRaisesRegex(ValueError, "固定ファイル"):
                review_analyst(first, self.base / "原結果")
            result_read.assert_not_called()

    def test_original_change_after_seal_blocks_result_read(self) -> None:
        """固定した実測原本の変更を、原結果読取前に検知する。

        引数: self (AnalystReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        original = self.measured / self.identities[0] / "入力.json"
        original.write_bytes(original.read_bytes() + b" ")
        with patch("fundamentals.pdca_analyst._source_labels") as result_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                review_analyst(self.root / self.identities[0], self.base / "原結果")
            result_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
