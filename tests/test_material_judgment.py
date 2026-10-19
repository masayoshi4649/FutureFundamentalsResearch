"""判断の静的組立と既存入口の接続を、合成記事・合成基準だけで検査する。"""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import digest, read_json, write_json
from research_support.material_judgment import assemble_material_judgment, point_judgment
from research_support.material_points import build_material_ledger
from research_support.material_report import main as material_report_main
from test_material_points import material_fixture, material_packet


class MaterialJudgmentTests(unittest.TestCase):
    """本文指定・版時計・原本SHA・未知・両先物の明示を検査する。"""

    def setUp(self) -> None:
        """市場の実データを含まない、独立した入力と一時原本を用意する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None。一時ファイルは試験終了時に削除する。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data = material_fixture()
        for row in self.data["articles"]:
            row["body_excerpt"] = row.pop("body")
            row["updated_at"] = "2026-09-29T21:00:30+09:00"
            row["available_at"] = row["fetched_at"]
            row["retrieved_at"] = row["fetched_at"]
        self.packet = material_packet(self.data)
        self.details = {symbol: {key: row[key] for key in ("proxy", "source")}
                        for symbol, row in self.data["references"].items()}
        self.materials = copy.deepcopy(self.data["materials"])
        self.forecasts = copy.deepcopy(self.data["forecasts"])
        for material in self.materials:
            material["impacts"] = {symbol: {"today": rows["today"]}
                                   for symbol, rows in material["impacts"].items()}
        self.forecasts = {symbol: {"today": rows["today"]} for symbol, rows in self.forecasts.items()}
        self.scope = {"start": 0, "end_exclusive": 4, "notice": "合成抜粋の先頭4文字だけを指定。"}
        self.uses = [{"id": row["id"], "revision_id": row["revision_id"], "mode": "excerpt",
                      "scope": copy.deepcopy(self.scope)} for row in self.packet["news"]]

    def _assemble(self) -> dict:
        """当該試験が明示した材料と記事区間を組み立てる。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: dict は判断JSONと機械的な出典記録。
        """
        return assemble_material_judgment(self.packet, article_uses=self.uses, reference_details=self.details,
                                          materials=self.materials, forecasts=self.forecasts)

    def _raw(self, **changes) -> Path:
        """一つ目の記事と同じ版・時計を持つ、余白付き本文原本を保存する。

        引数: self (MaterialJudgmentTests) は試験、changes (dict) は反例用の上書き値。
        返り値: Path は合成取得原本のパス。
        """
        raw = {**self.packet["news"][0], "body_text": "前置合成本文後記", "body_offset": 0, **changes}
        path = self.base / "合成取得原本.json"
        path.write_text(json.dumps({"structuredContent": {"article": raw}}, ensure_ascii=False), encoding="utf-8")
        self.uses[0] = {"id": "1", "revision_id": "1", "mode": "raw", "raw_path": str(path),
                        "scope": {"start": 2, "end_exclusive": 6, "notice": "合成原本の主文4文字だけを指定。"}}
        return path

    def test_explicit_points_unknown_and_independent_directions_survive(self) -> None:
        """未知をゼロにせず、点から方向を作らず、長期だけを未算出へそろえる。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は両先物・未知伝播・コピーの検査完了。
        """
        point = point_judgment(impact_pt=None, priced_in_before_reference_pt=2,
            priced_in_after_reference_pt=None, impact_reason="総影響は未判明。", before_reason="合成観測による主観点。",
            after_reason="基準後は未判明。", transmission="合成経路。", before_observations=[], after_observations=[])
        self.materials[0]["impacts"]["NK225"]["today"] = point
        before = copy.deepcopy((self.packet, self.materials, self.forecasts, self.uses))
        result = self._assemble()
        judgment = result["judgment"]
        self.assertEqual((self.packet, self.materials, self.forecasts, self.uses), before)
        self.assertIsNone(judgment["materials"][0]["impacts"]["NK225"]["today"]["impact_pt"])
        ledger = build_material_ledger(**judgment, source_packet=self.packet)
        self.assertIsNone(ledger["aggregates"]["NK225"]["today"]["totals"]["residual_pt"])
        for symbol, expected in (("NK225", "上がる"), ("TOPIX", "下がる")):
            self.assertEqual(judgment["forecasts"][symbol]["today"]["direction"], expected)
            self.assertEqual(judgment["references"][symbol]["price"], self.packet["instruments"][symbol]["reference"]["price"])
            for horizon in ("week", "month", "quarter"):
                self.assertIsNone(judgment["forecasts"][symbol][horizon])
                self.assertIsNone(judgment["materials"][0]["impacts"][symbol][horizon])
        self.assertNotIn("actor", result)
        facts = result["source_records"]
        self.assertEqual(facts["judgment_hash"], digest(judgment))
        self.assertEqual(facts["record_hash"], digest({key: value for key, value in facts.items() if key != "record_hash"}))

    def test_raw_slice_and_file_sha_are_exact(self) -> None:
        """主文区間の文字位置と取得原本バイト列SHAを別々に固定する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は本文・原本・元メタデータの検査完了。
        """
        path = self._raw()
        result = self._assemble()
        article = result["judgment"]["articles"][0]
        self.assertEqual(article["body"], "合成本文")
        self.assertEqual(article["body_sha256"], hashlib.sha256("合成本文".encode("utf-8")).hexdigest())
        self.assertEqual(article["body_raw_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(article["body_raw_path"], str(path.resolve()))
        self.assertEqual(result["source_records"]["articles"][0]["body_raw_sha256"], article["body_raw_sha256"])
        for key, value in self.packet["news"][0].items():
            self.assertEqual(article[key], value)
        self.uses[0]["raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self._assemble()

    def test_wrong_raw_identity_revision_and_offset_are_rejected(self) -> None:
        """別記事・別版・主文先頭不明の原本を拒否する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は3種類の独立した原本反例の検査完了。
        """
        for change in ({"id": "別記事"}, {"revision_id": "別版"}, {"body_offset": 10}):
            with self.subTest(change=change):
                self._raw(**change)
                with self.assertRaises(ValueError):
                    self._assemble()

    def test_every_raw_clock_rejects_future_and_mismatch(self) -> None:
        """取得原本の各時計について、未来と可視版からの差替えを拒否する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は全6時計の未来・不一致検査完了。
        """
        for key in ("published_at", "fetched_at", "effective_at", "updated_at", "available_at", "retrieved_at"):
            for value in ("2026-09-30T07:45:00.001+09:00", "2026-09-29T20:00:00+09:00"):
                with self.subTest(clock=key, value=value):
                    self._raw(**{key: value})
                    with self.assertRaises(ValueError):
                        self._assemble()

    def test_scope_is_required_nonempty_in_bounds_and_not_inferred(self) -> None:
        """scopeの省略・空区間・範囲外・真偽値・説明欠損を拒否する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は本文範囲の反例検査完了。
        """
        for scope in (None, {"start": 0, "end_exclusive": 0, "notice": "空"},
                      {"start": 0, "end_exclusive": 999, "notice": "超過"},
                      {"start": False, "end_exclusive": 4, "notice": "真偽値"},
                      {"start": 0, "end_exclusive": 4}):
            with self.subTest(scope=scope):
                self.uses[0]["scope"] = scope
                with self.assertRaises(ValueError):
                    self._assemble()

    def test_no_body_is_explicitly_rejected_for_selected_article(self) -> None:
        """本文を使わない記事を、本文照合可能な選択記事と装わない。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は現行入口に非互換な指定の明示拒否完了。
        """
        self.uses[0] = {"id": "1", "revision_id": "1", "mode": "none"}
        with self.assertRaisesRegex(ValueError, "本文なしの記事は選択せず別記"):
            self._assemble()

    def test_missing_points_today_or_instrument_are_not_filled(self) -> None:
        """未入力の点・当日・片方の先物をゼロや他方の複製で補完しない。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は必須の意味入力が欠けた反例の検査完了。
        """
        original = copy.deepcopy(self.materials)
        for kind in ("点", "当日", "先物"):
            with self.subTest(kind=kind):
                self.materials = copy.deepcopy(original)
                impacts = self.materials[0]["impacts"]
                if kind == "点":
                    del impacts["NK225"]["today"]["impact_pt"]
                elif kind == "当日":
                    del impacts["NK225"]["today"]
                else:
                    del impacts["TOPIX"]
                with self.assertRaises(ValueError):
                    self._assemble()

    def test_unknown_p0_contract_remains_unconfirmed(self) -> None:
        """packetの限月不明を推測で確認済みに変えない。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None はP0原値と限月の不明維持の検査完了。
        """
        self.packet["instruments"]["NK225"]["reference"]["contract"] = None
        self.packet["packet_hash"] = digest({key: value for key, value in self.packet.items() if key != "packet_hash"})
        judgment = self._assemble()["judgment"]
        self.assertIsNone(judgment["references"]["NK225"]["contract"])
        self.assertEqual(judgment["references"]["NK225"]["contract_identity_status"], "未確認")

    def test_packet_hash_and_selected_revision_are_bound(self) -> None:
        """packet改変と存在しない選択版の双方を拒否する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は自己ハッシュと版指定の検査完了。
        """
        self.uses[0]["revision_id"] = "存在しない版"
        with self.assertRaises(ValueError):
            self._assemble()
        self.packet["as_of"] = "2026-09-30T16:00:00+09:00"
        with self.assertRaisesRegex(ValueError, "自己ハッシュ"):
            self._assemble()

    def test_existing_report_entry_accepts_raw_and_excerpt_without_changes(self) -> None:
        """実入口へ判断JSONを渡し、合成原本の固定と同じ台帳を確認する。

        引数: self (MaterialJudgmentTests) は試験。
        返り値: None は既存入口を改変しない接続検査完了。
        """
        self._raw()
        result = self._assemble()
        packet_path, judgment_path, actor_path = (self.base / name for name in ("packet.json", "judgment.json", "actor.json"))
        write_json(packet_path, self.packet)
        write_json(judgment_path, result["judgment"])
        write_json(actor_path, {"packet_hash": self.packet["packet_hash"], "注記": "この試験が明示した合成担当記録。"})
        output = self.base / "固定出力"
        arguments = ["material_report", "--packet", str(packet_path), "--judgment", str(judgment_path),
                     "--actor", str(actor_path), "--out", str(output)]
        with patch.object(sys, "argv", arguments), patch("builtins.print"):
            material_report_main()
        actual = read_json(output / "予測.json")
        expected = build_material_ledger(**result["judgment"], source_packet=self.packet)
        self.assertEqual(actual, expected)
        self.assertTrue((output / "入力固定.json").exists())
        self.assertTrue((output / "予測.md").exists())


if __name__ == "__main__":
    unittest.main()
