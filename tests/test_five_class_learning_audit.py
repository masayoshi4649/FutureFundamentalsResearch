"""開始前の教師来歴・元packetとの束縛を、別の合成反例で監査する。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import test_five_class_learning as fixture_module
from fundamentals.common import digest, now, read_json, write_json
from fundamentals.replay import _file_hash
from research_support.five_class_learning import run_comparison


class FiveClassLearningAuditTests(unittest.TestCase):
    """記録のSHA固定だけで不整合な入力を正当化しないことを確認する。"""

    def setUp(self) -> None:
        """元入力・コピー・教師生成の来歴を、実価格を使わず正しい構造で作る。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        self.sample = fixture_module.FiveClassLearningTests()
        self.sample.setUp()
        self.addCleanup(self.sample.doCleanups)
        self.original = self.sample.fixture.base / "教師作成前の実測原本"
        self.original.mkdir()
        self.source_teacher = self.original / "初期教師.json"
        write_json(self.source_teacher, {"rows": []})
        sources = [self.source_teacher, self.sample.root / "五分類規約.json",
                   Path("research_support/prepare_five_history.py"), Path("research_support/five_class.py")]
        first = None
        for identity in self.sample.fixture.identities:
            packet = read_json(self.sample.fixture.inputs / "masked" / (identity + ".json"))
            first = packet["as_of"] if first is None else min(first, packet["as_of"])
            target = self.sample.fixture.measured / identity / "入力.json"
            current = read_json(target)
            current["source_packet_hash"] = digest(packet)
            write_json(target, current, exclusive=False)
            original_path = self.original / identity / "入力.json"
            original_path.parent.mkdir()
            original_path.write_bytes(target.read_bytes())
            sources.append(original_path)
        teacher = self.sample.fixture.measured / "初期教師.json"
        write_json(teacher, {"rows": [], "first_as_of": first,
                   "five_class_rules_hash": _file_hash(self.sample.root / "五分類規約.json")}, exclusive=False)
        self.provenance = {"created_at": now(), "mode": "回想・過去教師の経路点検", "first_as_of": first,
            "teacher_rows": 0, "source_hashes": {str(path.resolve()): _file_hash(path) for path in sources},
            "teacher_hash": _file_hash(teacher), "implementation_hash": _file_hash(Path("research_support/prepare_five_history.py"))}
        self._save_provenance()

    def _save_provenance(self) -> None:
        """正規の自己ハッシュを付けて合成教師の来歴を保存する。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        self.provenance.pop("provenance_hash", None)
        self.provenance["provenance_hash"] = digest(self.provenance)
        write_json(self.sample.fixture.measured / "初期教師の来歴.json", self.provenance, exclusive=False)

    def _must_reject_before_model_or_price(self) -> None:
        """不整合な来歴をモデル呼出・結果読取より前に拒否する。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        with patch("research_support.five_class_learning.predict_five", side_effect=AssertionError("先にモデルが呼ばれた")) as model, \
                patch("research_support.five_class_learning._read_labels", side_effect=AssertionError("先に結果が読まれた")) as reader:
            with self.assertRaises(ValueError):
                run_comparison(self.sample.root, self.sample.fixture.inputs, self.sample.fixture.measured, self.sample.prices)
            model.assert_not_called()
            reader.assert_not_called()

    def test_teacher_bytes_must_match_generated_provenance_before_freeze(self) -> None:
        """作成後に教師のバイト列を変えても、新manifestで承認されない。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        path = self.sample.fixture.measured / "初期教師.json"
        path.write_bytes(path.read_bytes() + b" ")
        self._must_reject_before_model_or_price()

    def test_valid_provenance_keeps_separate_scores_and_adds_each_label_once(self) -> None:
        """正しい来歴は通し、同じ実績でも方式別採点・同じ分母・一度の教師追加を確認する。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        def different_model(current: dict, history: list[dict], as_of: str, mode: str) -> dict:
            """二方式が必ず逆回答になる合成モデルを作る。

            引数: current/history (dict/list[dict]) は当回と教師、as_of/mode (str) は締切と方式。
            返り値: dict は方式別の固定回答と診断。
            """
            result = self.sample._model(current, history, as_of, mode)
            answer = "上がる" if mode == "確定教師" else "下がる"
            result["direction"] = answer
            result["class_scores"] = {name: float(name == answer) for name in fixture_module.CLASSES}
            return result

        with patch("research_support.five_class_learning.predict_five", side_effect=different_model), \
                patch("research_support.five_class_learning._read_labels", side_effect=self.sample._labels):
            summary = run_comparison(self.sample.root, self.sample.fixture.inputs,
                                     self.sample.fixture.measured, self.sample.prices)
        identity = self.sample.fixture.identities[0]
        certain = read_json(self.sample.root / "12_確定教師" / identity / "result.json")
        shared = read_json(self.sample.root / "13_候補配分" / identity / "result.json")
        self.assertEqual(certain["source_result_hash"], shared["source_result_hash"])
        for first, second in zip(certain["labels"], shared["labels"]):
            self.assertEqual(first["evaluation"], second["evaluation"])
            self.assertTrue(first["score"]["correct_lower"])
            self.assertFalse(second["score"]["correct_lower"])
        state = read_json(self.sample.root / "学習状態.json")
        self.assertEqual(len(state["rows"]), 4)
        self.assertEqual(len({(row["symbol"], row["label_end"]) for row in state["rows"]}), 4)
        self.assertEqual([len(rows) for rows in self.sample.seen_histories], [0, 0, 0, 0, 2, 2, 2, 2])
        for symbol in ("NK225", "TOPIX"):
            self.assertEqual(summary["確定教師"]["groups"][symbol]["baselines"],
                             summary["候補配分"]["groups"][symbol]["baselines"])

    def test_measured_copy_must_match_original_source_hash_before_freeze(self) -> None:
        """同じ締切の特徴だけを書き換えても、コピー元との不一致で停止する。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        path = self.sample.fixture.measured / self.sample.fixture.identities[0] / "入力.json"
        current = read_json(path)
        current["instruments"]["NK225"]["features"]["growth"] = 999
        write_json(path, current, exclusive=False)
        self._must_reject_before_model_or_price()

    def test_valid_source_hash_cannot_replace_binding_to_visible_packet(self) -> None:
        """元実測コピーと来歴を揃えても、別packet由来なら予測前に拒否する。

        引数: self (FiveClassLearningAuditTests) はテスト。
        返り値: None。
        """
        identity = self.sample.fixture.identities[0]
        path = self.sample.fixture.measured / identity / "入力.json"
        current = read_json(path)
        current["source_packet_hash"] = "0" * 64
        write_json(path, current, exclusive=False)
        original = self.original / identity / "入力.json"
        original.write_bytes(path.read_bytes())
        self.provenance["source_hashes"][str(original.resolve())] = _file_hash(original)
        self._save_provenance()
        self._must_reject_before_model_or_price()


if __name__ == "__main__":
    unittest.main()
