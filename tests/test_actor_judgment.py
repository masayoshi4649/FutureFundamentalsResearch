"""実ニュース・実価格を使わず、本人入力と機械記録の境界を検査する。"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import digest, read_json, write_json
from research_support.actor_judgment import REQUIRED_ANALYSIS, REQUIRED_ATTESTATIONS, assemble_actor
from research_support.material_judgment import assemble_material_judgment
from research_support.material_report import main as report_main
from test_material_points import material_fixture, material_packet


class ActorJudgmentTests(unittest.TestCase):
    """合成入力だけで読了・判断の非補完と既存入口への接続を検査する。"""

    def setUp(self) -> None:
        """独立した合成記事と合成価格を一時フォルダへ配置する。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None。一時フォルダは試験終了時に削除する。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.data = material_fixture()
        self.packet = material_packet(self.data)
        for row in self.packet["news"]:
            row["body_excerpt"] = row.pop("body", "合成本文")
        self.packet["packet_hash"] = digest({key: value for key, value in self.packet.items() if key != "packet_hash"})
        self.packet_path = self.base / "packet.json"
        self.metadata_path = self.base / "見出し.json"
        self.manifest_path = self.base / "本文選別.json"
        self.snapshot_path = self.base / "反応.json"
        self.metadata = {"news": copy.deepcopy(self.packet["news"])}
        self.manifest = {"as_of": self.packet["as_of"], "records": [{"body": "本文選別台帳の非表示本文"}]}
        self.snapshot = {"as_of": self.packet["as_of"], "version": "合成本文遮断版", "session_count": 1,
            "instrument_observation_count": 2, "independent_sample_count": 1,
            "cases": [{"body": "反応履歴の非表示本文"}], "notice": "表示しない原文"}
        self.attestations = {key: False for key in REQUIRED_ATTESTATIONS}
        self.analysis = {key: [] for key in REQUIRED_ANALYSIS}
        self.analysis["p0_reading"] = "本人が入力した解釈"
        self.analysis["uncertainty"] = "不明を明示"
        self.scopes = []
        self.confirmed = [{"id": row["id"], "revision_id": row["revision_id"]} for row in self.metadata["news"]]
        self._save()

    def _save(self) -> None:
        """当該試験の合成JSONだけを書き出す。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None。既存の実データへはアクセスしない。
        """
        for path, value in ((self.packet_path, self.packet), (self.metadata_path, self.metadata),
                            (self.manifest_path, self.manifest), (self.snapshot_path, self.snapshot)):
            write_json(path, value, exclusive=False)

    def _assemble(self) -> dict:
        """当該試験が明示した入力だけを組立APIへ渡す。

        引数: self (ActorJudgmentTests) は試験。
        返り値: dict はactorと機械出典記録。
        """
        return assemble_actor(self.packet_path, actor_name="合成担当", headline_metadata_path=self.metadata_path,
            confirmed_headline_ids=self.confirmed, read_scopes=self.scopes, curation_manifest_path=self.manifest_path,
            feedback_snapshot_path=self.snapshot_path, attestations=self.attestations, analysis=self.analysis)

    def test_copies_explicit_inputs_without_text_or_inferred_reading(self) -> None:
        """本人申告の真偽を変えず、本文・抜粋・反応本文を返さない。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は入力保持と出力遮断の検査完了。
        """
        before = copy.deepcopy((self.attestations, self.analysis, self.scopes, self.confirmed))
        result = self._assemble()
        actor, records = result["actor"], result["source_records"]
        self.assertEqual(actor["attestations"], self.attestations)
        for key, value in self.attestations.items():
            self.assertIs(actor[key], value)
        self.assertEqual(actor["analysis"], self.analysis)
        self.assertEqual(actor["read_files"], [])
        self.assertEqual(before, (self.attestations, self.analysis, self.scopes, self.confirmed))
        encoded = json.dumps(result, ensure_ascii=False)
        for forbidden in ("body_excerpt", "非表示本文", "表示しない原文"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(records["actor_hash"], digest(actor))
        self.assertEqual(records["record_hash"], digest({key: value for key, value in records.items() if key != "record_hash"}))
        actor["analysis"]["headline_batches"].append("変更")
        self.assertEqual(self.analysis["headline_batches"], [])

    def test_all_confirmed_requires_exact_explicit_list(self) -> None:
        """不足・過剰・重複を全件確認済みへ補完しない。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は確認集合の反例検査完了。
        """
        original = self.confirmed
        for values in (original[:-1], original + [{"id": "未知", "revision_id": "1"}], original + original[:1]):
            self.confirmed = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self._assemble()

    def test_same_id_different_revision_requires_composite_confirmation(self) -> None:
        """同ID別版は単純なIDリストで全件確認扱いにしない。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は版ごとの確認要求の検査完了。
        """
        row = copy.deepcopy(self.metadata["news"][0])
        row["revision_id"] = "別版"
        self.metadata["news"].append(row)
        self._save()
        self.confirmed = [entry["id"] for entry in self.metadata["news"]]
        with self.assertRaisesRegex(ValueError, "同ID別版"):
            self._assemble()
        self.confirmed = [{"id": entry["id"], "revision_id": entry["revision_id"]} for entry in self.metadata["news"]]
        self.assertTrue(self._assemble()["actor"]["headline_review"]["all_confirmed"])

    def test_missing_attestation_or_analysis_is_rejected(self) -> None:
        """本人の必須入力が欠けた場合に安全値や分析を作らない。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は全必須入力の欠損検査完了。
        """
        for collection in (self.attestations, self.analysis):
            for key in list(collection):
                value = collection.pop(key)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self._assemble()
                collection[key] = value
        self.attestations["future_result_seen"] = 0
        with self.assertRaises(ValueError):
            self._assemble()

    def test_explicit_attestations_keep_values_at_top_level(self) -> None:
        """既存ラッパーが照合する本人申告を値不変でトップへ配線する。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は全6項目のTrue・False保持の検査完了。
        """
        for value in (True, False):
            self.attestations = {key: value for key in REQUIRED_ATTESTATIONS}
            actor = self._assemble()["actor"]
            for key in REQUIRED_ATTESTATIONS:
                self.assertIs(actor[key], value)
                self.assertIs(actor["attestations"][key], value)

    def test_unrecognized_independent_sample_count_stays_null(self) -> None:
        """本文遮断snapshotの独立数未認定をnullのまま保存する。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None はnullの保持と他件数の非負整数検査完了。
        """
        self.snapshot["independent_sample_count"] = None
        self._save()
        result = self._assemble()
        self.assertIsNone(result["actor"]["reaction_snapshot"]["independent_sample_count"])
        self.assertIsNone(result["source_records"]["feedback_snapshot"]["independent_sample_count"])
        for key in ("session_count", "instrument_observation_count"):
            original = self.snapshot[key]
            for value in (None, -1, True, "1", {"body": "隠した本文"}):
                self.snapshot[key] = value
                self._save()
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self._assemble()
            self.snapshot[key] = original

    def test_read_scopes_hash_only_explicit_paths(self) -> None:
        """読んだという本人の一覧だけを固定し、機械入力を既読一覧へ足さない。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None はpath・SHA・scopeの検査完了。
        """
        self.scopes = [{"path": self.metadata_path, "scope": "見出し全件のみ"}]
        actor = self._assemble()["actor"]
        self.assertEqual(len(actor["read_files"]), 1)
        self.assertEqual(actor["read_files"][0]["scope"], "見出し全件のみ")
        self.assertEqual(actor["read_files"][0]["path"], str(self.metadata_path.resolve()))
        self.assertEqual(len(actor["read_files"][0]["sha256"]), 64)

    def test_future_clock_and_tampered_hash_are_rejected(self) -> None:
        """未来の見出し時計と自己ハッシュ改変を別々に拒否する。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は未来入力と改変の検査完了。
        """
        self.metadata["news"][0]["available_at"] = "2099-01-01T00:00:00+09:00"
        self._save()
        with self.assertRaises(ValueError):
            self._assemble()
        self.metadata = {"news": copy.deepcopy(self.packet["news"])}
        self.snapshot["snapshot_hash"] = "改変"
        self._save()
        with self.assertRaisesRegex(ValueError, "自己ハッシュ"):
            self._assemble()

    def test_nested_body_cannot_escape_through_header_fields(self) -> None:
        """見出しと反応ヘッダを装った入れ子本文の流出を拒否する。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は許可項目の型境界の検査完了。
        """
        self.metadata["news"][0]["source_name"] = {"body": "隠した本文"}
        self._save()
        with self.assertRaises(ValueError):
            self._assemble()
        self.metadata = {"news": copy.deepcopy(self.packet["news"])}
        self.snapshot["mode"] = {"body": "隠した本文"}
        self._save()
        with self.assertRaises(ValueError):
            self._assemble()

    def test_existing_report_accepts_assembled_actor_with_synthetic_judgment(self) -> None:
        """既存の正式固定入口へ合成担当記録を無改変で渡す。

        引数: self (ActorJudgmentTests) は試験。
        返り値: None は合成記事・価格だけを用いた既存入口接続の検査完了。
        """
        details = {symbol: {key: value[key] for key in ("proxy", "source")}
                   for symbol, value in self.data["references"].items()}
        uses = [{"id": row["id"], "revision_id": row["revision_id"], "mode": "excerpt",
                 "scope": {"start": 0, "end_exclusive": 4, "notice": "合成本文4文字"}}
                for row in self.packet["news"]]
        assembled = assemble_material_judgment(self.packet, article_uses=uses, reference_details=details,
            materials=self.data["materials"], forecasts=self.data["forecasts"])
        actor = self._assemble()["actor"]
        judgment_path, actor_path = self.base / "judgment.json", self.base / "actor.json"
        write_json(judgment_path, assembled["judgment"])
        write_json(actor_path, actor)
        output = self.base / "合成固定"
        arguments = ["material_report", "--packet", str(self.packet_path), "--judgment", str(judgment_path),
                     "--actor", str(actor_path), "--out", str(output)]
        with patch.object(sys, "argv", arguments), patch("builtins.print"):
            report_main()
        self.assertEqual(read_json(output / "入力固定.json")["actor"], actor)
        self.assertTrue((output / "予測.json").exists())


if __name__ == "__main__":
    unittest.main()
