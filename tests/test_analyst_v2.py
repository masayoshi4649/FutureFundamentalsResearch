"""次の未使用時点向け別版で、境界検証と個別考察の固定を合成入力だけで確認する。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import test_pdca_analyst as fixture_module
import research_support.analyst_v2 as implementation
from fundamentals.common import digest, read_json, timestamp, write_json
from fundamentals.replay import _audit, _file_hash
from research_support.analyst_v2 import (
    BOUNDARY_SENTENCE, PREPARE_FILE, PROTOCOL_FILE, finalize_review_v2, prepare_v2, reveal_v2, seal_v2,
)


class AnalystV2Tests(unittest.TestCase):
    """旧実験を再実行せず、一時領域の合成２回だけを使う。"""

    def setUp(self) -> None:
        """旧版テストの入力生成だけを再利用して、一時的な別実験を作る。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self.fixture = fixture_module.AnalystReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.inputs, self.measured = self.fixture.root, self.fixture.inputs, self.fixture.measured
        self.path = self.root / self.fixture.identities[0]

    def _prepare(self) -> dict:
        """横ばいの数値境界を静的指示へ追加して初回入力を固定する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: dict は合成した当回入力。
        """
        instructions = self.root / "予測担当の手順.md"
        instructions.write_text(instructions.read_text(encoding="utf-8") + BOUNDARY_SENTENCE + "\n", encoding="utf-8")
        return prepare_v2(self.root, self.inputs, self.measured, 1)

    def _seal(self) -> dict:
        """合成担当の上昇予測を別版手順のハッシュとともに固定する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: dict は固定した合成予測。
        """
        write_json(self.path / "予測入力.json", {"instruments": {symbol: {
            "direction": "上昇", "evidence_ids": ["可視記事"], "already_reacted": "前場の下落を観測。",
            "additional_reaction": "基準100からの回復仮説。", "invalidation": "売りが継続。"}
            for symbol in ("NK225", "TOPIX")}})
        return seal_v2(self.path, {"agent_id": "当回独立担当", "task_name": "当回独立担当",
                                  "fork_turns": "none", "completed": True,
                                  "input_file_hash": _file_hash(self.path / "masked" / "入力.json")})

    def _reveal(self) -> dict:
        """原採点が不一致、当回予測は一致する合成結果を開示する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: dict は新旧採点を分離した当回結果。
        """
        labels, source_hash = self.fixture._labels()
        for row in labels:
            row["metrics"] = {"direction_hit": False, "terminal_range_hit": False}
        with patch("research_support.analyst_v2._source_labels", return_value=(labels, source_hash)):
            return reveal_v2(self.path, self.fixture.base / "合成原結果")

    def _reflection(self, result: dict) -> Path:
        """当回予測・結果に結び付いた２先物の個別考察を作る。

        引数: self (AnalystV2Tests) はテスト、result (dict) は開示済み結果。
        返り値: Path は新規の考察入力。
        """
        prediction = read_json(self.path / "予測.json")
        reflection = {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result), "instruments": {}}
        for symbol in ("NK225", "TOPIX"):
            reflection["instruments"][symbol] = {"predicted_direction": "上昇", "actual_direction": "上昇", "causality": "未識別",
                "reason": symbol + "は基準100から101へ上昇し回復仮説と整合したが、単独材料の因果は識別できない。",
                "next_action": symbol + "の次回も直前先物の戻りと新規支援材料の有無を別々に確認する。", "evidence_ids": ["可視記事"]}
        path = self.path / "考察入力.json"
        write_json(path, reflection)
        return path

    # ----------------------------------------

    def test_boundary_is_required_before_any_prepare_manifest(self) -> None:
        """静的指示の数値境界がなければ入力や固定一覧を作らない。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        with self.assertRaisesRegex(ValueError, "横ばい境界"):
            prepare_v2(self.root, self.inputs, self.measured, 1)
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertFalse((self.root / PROTOCOL_FILE).exists())
        self.assertFalse(self.path.exists())

    def test_source_metrics_are_separated_and_review_blocks_next_cutoff(self) -> None:
        """原採点と当回採点を分け、個別考察前の次回準備を拒否する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        with patch("research_support.analyst_v2._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "予測固定後"):
                reveal_v2(self.path, self.fixture.base / "原結果")
            source_read.assert_not_called()
        self._seal()
        actor = read_json(self.path / "actor.json")
        self.assertEqual(actor["protocol_file_hash"], _file_hash(self.root / PROTOCOL_FILE))
        result = self._reveal()
        for row in result["labels"]:
            self.assertTrue(row["direction_hit"])
            self.assertTrue(row["metrics"]["direction_hit"])
            self.assertFalse(row["source_metrics"]["direction_hit"])
            self.assertNotIn("terminal_range_hit", row["metrics"])
        self.assertEqual(_audit(self.path)[-1]["phase"], "reveal")
        self.assertFalse((self.path / "review.json").exists())
        with self.assertRaisesRegex(ValueError, "振り返り"):
            prepare_v2(self.root, self.inputs, self.measured, 2)

    def test_individual_reflection_must_finish_before_next_history(self) -> None:
        """予測・結果に一致する当回固有考察の固定後だけ次回へ進める。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        result = self._reveal()
        reflection = self._reflection(result)
        supplied = read_json(reflection)
        supplied["instruments"]["TOPIX"]["evidence_ids"] = []
        write_json(reflection, supplied, exclusive=False)
        review = finalize_review_v2(self.path, reflection, {"agent_id": "当回独立担当", "completed": True})
        self.assertEqual(_audit(self.path)[-1]["phase"], "review")
        self.assertIn("NK225", review["feedback"]["instruments"]["NK225"]["reason"])
        second = prepare_v2(self.root, self.inputs, self.measured, 2)
        self.assertEqual(len(second["known_history"]), 2)
        self.assertEqual(second["own_past_reviews"][0]["instruments"]["TOPIX"]["next_action"],
                         review["feedback"]["instruments"]["TOPIX"]["next_action"])
        feedback = second["own_past_reviews"][0]["instruments"]["TOPIX"]
        self.assertEqual(feedback["prediction_evidence_ids"], ["可視記事"])
        self.assertEqual(feedback["reflection_evidence_ids"], [])
        self.assertNotIn("evidence_ids", feedback)

    def test_reflection_wrong_actor_result_or_direction_is_rejected(self) -> None:
        """別担当や別結果の考察、実績方向の書き換えをreviewへ固定できない。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        result = self._reveal()
        path = self._reflection(result)
        with self.assertRaisesRegex(ValueError, "当回担当"):
            finalize_review_v2(self.path, path, {"agent_id": "別担当", "completed": True})
        reflection = read_json(path)
        reflection["instruments"]["NK225"]["actual_direction"] = "下落"
        write_json(path, reflection, exclusive=False)
        with self.assertRaisesRegex(ValueError, "実績方向"):
            finalize_review_v2(self.path, path, {"agent_id": "当回独立担当", "completed": True})
        self.assertFalse((self.path / "review.json").exists())

    def test_wrapper_change_after_seal_is_detected_before_result_read(self) -> None:
        """固定後の別版実装ハッシュ変更を原結果読取前に検出する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        wrapper = Path(implementation.__file__).resolve()

        def changed_hash(path: Path) -> str:
            """実ファイルを書き換えず、別版実装のバイト変更を模擬する。

            引数: path (Path) はハッシュ対象。
            返り値: str は別版実装だけ異なるハッシュ。
            """
            return "変更済み" if path.resolve() == wrapper else _file_hash(path)

        with patch("research_support.analyst_v2._file_hash", side_effect=changed_hash), \
                patch("research_support.analyst_v2._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "準備固定|別版実装"):
                reveal_v2(self.path, self.fixture.base / "原結果")
            source_read.assert_not_called()

    def test_protocol_bytes_change_cannot_escape_actor_binding(self) -> None:
        """自己ハッシュの値が同じでも手順固定ファイルの変更を担当束縛で検出する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        protocol = self.root / PROTOCOL_FILE
        protocol.write_bytes(protocol.read_bytes() + b" ")
        with patch("research_support.analyst_v2._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "結び付き|準備固定"):
                reveal_v2(self.path, self.fixture.base / "原結果")
            source_read.assert_not_called()

    def test_prepare_binds_protocol_before_first_prediction_seal(self) -> None:
        """準備監査に手順固定一覧を含め、初回seal前の変更も拒否する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        record = _audit(self.path)[0]
        self.assertIn(PREPARE_FILE, record["files"])
        prepared = read_json(self.path / PREPARE_FILE)
        self.assertEqual(prepared["protocol_file_hash"], _file_hash(self.root / PROTOCOL_FILE))
        protocol = self.root / PROTOCOL_FILE
        protocol.write_bytes(protocol.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "準備固定"):
            self._seal()
        self.assertFalse((self.path / "actor.json").exists())

    def test_joint_wrapper_and_protocol_change_before_first_seal_is_rejected(self) -> None:
        """実装と手順の自己ハッシュを同時更新しても準備時の固定を逃れられない。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        path = self.root / PROTOCOL_FILE
        protocol = read_json(path)
        protocol["wrapper_hash"] = "変更済み"
        protocol["protocol_hash"] = digest({key: value for key, value in protocol.items() if key != "protocol_hash"})
        write_json(path, protocol, exclusive=False)
        wrapper = Path(implementation.__file__).resolve()

        def changed_hash(target: Path) -> str:
            """手順に合わせた実装ハッシュの同時変更を実ファイル変更なしで模擬する。

            引数: target (Path) はハッシュ対象。
            返り値: str は別版実装だけ差し替えたハッシュ。
            """
            return "変更済み" if target.resolve() == wrapper else _file_hash(target)

        with patch("research_support.analyst_v2._file_hash", side_effect=changed_hash):
            with self.assertRaisesRegex(ValueError, "準備固定"):
                self._seal()
        self.assertFalse((self.path / "actor.json").exists())

    def test_future_endpoint_blocks_source_read_before_any_result_access(self) -> None:
        """実時計で終点に届く前は予測固定後でも原結果を開かない。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        with patch("research_support.analyst_v2.now", return_value="2026-10-01T15:44:59+09:00"), \
                patch("research_support.analyst_v2._source_labels") as source_read:
            with self.assertRaisesRegex(ValueError, "実時計で未満了"):
                reveal_v2(self.path, self.fixture.base / "原結果")
            source_read.assert_not_called()
        self.assertEqual(_audit(self.path)[-1]["phase"], "seal")
        self.assertFalse((self.path / "result.json").exists())

    def test_prepare_audit_records_start_and_final_commit_separately(self) -> None:
        """追加固定の完了時刻を準備監査の最終時刻とし、開始時刻も保持する。

        引数: self (AnalystV2Tests) はテスト。
        返り値: None。
        """
        committed = "2099-01-01T00:00:00+09:00"
        with patch("research_support.analyst_v2.now", return_value=committed):
            self._prepare()
        record = _audit(self.path)[0]
        self.assertEqual(record["actual_at"], committed)
        self.assertLess(timestamp(record["prepare_started_at"]), timestamp(record["actual_at"]))
        self.assertIn(PREPARE_FILE, record["files"])


if __name__ == "__main__":
    unittest.main()
