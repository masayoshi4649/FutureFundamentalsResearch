"""当回二方式の固定・結果開示・次回学習の順序を合成入力で検証する。"""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import test_pdca_analyst as fixture_module
from fundamentals.common import digest, now, read_json, write_json
from fundamentals.replay import _audit, _file_hash
from research_support.five_class import CLASSES
from research_support.five_class_learning import VARIANTS, _read_labels, run_comparison
from research_support.five_class_model import FEATURE_KEYS


class FiveClassLearningTests(unittest.TestCase):
    """モデルの数式と分離し、結果を開く前に両方式が固定されることを確かめる。"""

    def setUp(self) -> None:
        """一時領域へ二回の可視入力・空教師・確認済み規約を置く。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        self.fixture = fixture_module.AnalystReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.base / "比較"
        self.root.mkdir()
        (self.root / "実験計画.md").write_text("# 固定した合成比較\n", encoding="utf-8")
        self.rule_path = Path("research_support/五分類規約.json")
        (self.root / "五分類規約.json").write_bytes(self.rule_path.read_bytes())
        self.rules = read_json(self.rule_path)
        self.prices = self.fixture.base / "元価格"
        self.prices.mkdir()
        self.original = self.fixture.base / "五分類教師作成前"
        self.original.mkdir()
        write_json(self.original / "初期教師.json", {"rows": []})
        sources = [self.original / "初期教師.json", self.root / "五分類規約.json",
                   Path("research_support/prepare_five_history.py"), Path("research_support/five_class.py")]
        first = None
        for identity in self.fixture.identities:
            packet = read_json(self.fixture.inputs / "masked" / (identity + ".json"))
            first = packet["as_of"] if first is None else min(first, packet["as_of"])
            path = self.fixture.measured / identity / "入力.json"
            measured = read_json(path)
            measured["source_packet_hash"] = digest(packet)
            measured["instruments"] = {symbol: {"symbol": symbol, "kind": "Day", "as_of": measured["as_of"],
                "features": dict.fromkeys(FEATURE_KEYS, 0.0)} for symbol in ("NK225", "TOPIX")}
            write_json(path, measured, exclusive=False)
            original = self.original / identity / "入力.json"
            original.parent.mkdir()
            original.write_bytes(path.read_bytes())
            sources.append(original)
        teacher = self.fixture.measured / "初期教師.json"
        write_json(teacher, {"rows": [], "first_as_of": first,
                   "five_class_rules_hash": _file_hash(self.root / "五分類規約.json")}, exclusive=False)
        provenance = {"created_at": now(), "mode": "回想・過去教師の経路点検", "first_as_of": first,
            "teacher_rows": 0, "source_hashes": {str(path.resolve()): _file_hash(path) for path in sources},
            "teacher_hash": _file_hash(teacher), "implementation_hash": _file_hash(Path("research_support/prepare_five_history.py"))}
        provenance["provenance_hash"] = digest(provenance)
        write_json(self.fixture.measured / "初期教師の来歴.json", provenance)
        self.seen_histories = []
        self.read_count = 0

    def _model(self, current: dict, history: list[dict], as_of: str, mode: str) -> dict:
        """直前教師だけで回答を変える単純な代役を使い、学習順序を観察する。

        引数: self (FiveClassLearningTests) はテスト、current (dict) は当回、history (list[dict]) は教師、
            as_of (str) は締切、mode (str) は方式。
        返り値: dict は実モデルと同じ必須診断項目を持つ合成予測。
        """
        self.seen_histories.append(copy.deepcopy(history))
        selected = [row for row in history if row["symbol"] == current["symbol"]]
        answer = selected[-1]["five_class_evaluation"]["correct_candidates"][0] if selected else "上がる"
        return {"direction": answer, "class_scores": {name: float(name == answer) for name in CLASSES},
                "training_n": len(selected), "positive_weight_n": len(selected), "fitted_n": 0,
                "contributions": {}, "as_of": as_of, "variant": mode}

    def _labels(self, prices: Path, packet: dict, rules: dict, first_class: str = "上がる",
                second_class: str = "下がる") -> list[dict]:
        """両方式のsealを先に確認してから、合成した当回結果を開示する。

        引数: self (FiveClassLearningTests) はテスト、prices (Path) は未読の元価格、packet/rules (dict) は当回規約、
            first_class/second_class (str) は各回の合成実績。
        返り値: list[dict] は両先物の経路評価。
        """
        identity = self.fixture.identities[self.read_count]
        for folder in VARIANTS.values():
            path = self.root / folder / identity
            self.assertEqual(_audit(path)[-1]["phase"], "seal")
            self.assertTrue((path / "予測.json").is_file())
            self.assertFalse((path / "result.json").exists())
        answer = first_class if self.read_count == 0 else second_class
        self.read_count += 1
        return [{"symbol": symbol, "label_end": packet["targets"]["today"], "source_file": "合成", "source_file_hash": "合成",
                 "evaluation": {"target_status": "対象", "correct_candidates": [answer], "uncertain_candidates": [],
                     "metrics": {"actual_pct": 0.8 if answer == "上がる" else -0.8, "session_range_pct": 1.2},
                     "flat_status": "不成立", "quality": {}, "contract": rules["criteria"]}}
                for symbol in ("NK225", "TOPIX")]

    def _run(self, first: str = "上がる", second: str = "下がる") -> dict:
        """合成モデル・合成結果を使って二方式の比較本体を動かす。

        引数: self (FiveClassLearningTests) はテスト、first/second (str) は各回の合成実績。
        返り値: dict は比較の集計結果。
        """
        with patch("research_support.five_class_learning.predict_five", side_effect=self._model), \
                patch("research_support.five_class_learning._read_labels",
                      side_effect=lambda prices, packet, rules: self._labels(prices, packet, rules, first, second)):
            return run_comparison(self.root, self.fixture.inputs, self.fixture.measured, self.prices)

    def _reseal_provenance(self) -> None:
        """反例で来歴の自己ハッシュだけを正しく作り直し、内容照合を逃れられないか確認する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。合成領域の来歴だけを更新する。
        """
        path = self.fixture.measured / "初期教師の来歴.json"
        value = read_json(path)
        value.pop("provenance_hash", None)
        value["source_hashes"] = {name: _file_hash(Path(name)) for name in value["source_hashes"]}
        teacher = read_json(self.fixture.measured / "初期教師.json")
        value.update(teacher_hash=_file_hash(self.fixture.measured / "初期教師.json"), teacher_rows=len(teacher["rows"]))
        value["provenance_hash"] = digest(value)
        write_json(path, value, exclusive=False)

    def _initial_row(self) -> dict:
        """初回より前に満了する１行を元教師と経路点検後の教師へ追加する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: dict は数値・特徴を保全した合成の元教師行。
        """
        row = {"symbol": "NK225", "kind": "Day", "as_of": "2026-09-30T07:45:00+09:00",
               "label_end": "2026-09-30T15:45:00+09:00", "actual_pct": 0.8,
               "features": dict.fromkeys(FEATURE_KEYS, 0.0)}
        write_json(self.original / "初期教師.json", {"rows": [row]}, exclusive=False)
        teacher = read_json(self.fixture.measured / "初期教師.json")
        evaluation = {"classification_policy": "もみ合い優先・初回到達方向と最初の反転", "symbol": "NK225",
                      "session": "2026-09-30_Day", "target_end": row["label_end"], "target_status": "対象",
                      "correct_candidates": ["上がる"], "uncertain_candidates": [], "contract": self.rules["criteria"],
                      "metrics": {"actual_pct": 0.8, "session_range_pct": 1.2}, "flat_status": "不成立", "quality": {}}
        teacher["rows"] = [{**row, "five_class_evaluation": evaluation}]
        write_json(self.fixture.measured / "初期教師.json", teacher, exclusive=False)
        self._reseal_provenance()
        return row

    def _reject_before_access(self, expression: str) -> None:
        """反例の不整合を新manifest・モデル・当回価格の作成前に拒否する。

        引数: self (FiveClassLearningTests) はテスト、expression (str) は期待する理由。
        返り値: None。
        """
        with patch("research_support.five_class_learning.predict_five") as model, \
                patch("research_support.five_class_learning._read_labels") as reader:
            with self.assertRaisesRegex(ValueError, expression):
                run_comparison(self.root, self.fixture.inputs, self.fixture.measured, self.prices)
            model.assert_not_called()
            reader.assert_not_called()
        self.assertFalse((self.root / "manifest.json").exists())

    def test_both_predictions_are_sealed_before_current_label_access(self) -> None:
        """全回で両方式の固定が結果の読取より先になる。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        original = {path: path.read_bytes() for path in self.fixture.measured.rglob("*") if path.is_file()}
        summary = self._run()
        self.assertEqual(self.read_count, 2)
        self.assertEqual([len(rows) for rows in self.seen_histories], [0, 0, 0, 0, 2, 2, 2, 2])
        for result in summary.values():
            group = result["groups"]["NK225"]
            self.assertEqual((group["scheduled"], group["denominator"], group["correct_lower"]), (2, 2, 1))
            self.assertTrue(all(item["denominator"] == 2 for item in group["baselines"].values()))
        self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_current_result_changes_only_next_prediction(self) -> None:
        """初回結果を下落へ変えても初回予測は変わらず、次回だけ教師を反映する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        self._run(first="下がる")
        for folder in VARIANTS.values():
            first = read_json(self.root / folder / self.fixture.identities[0] / "予測.json")
            second = read_json(self.root / folder / self.fixture.identities[1] / "予測.json")
            self.assertEqual(first["instruments"]["NK225"]["direction"], "上がる")
            self.assertEqual(second["instruments"]["NK225"]["direction"], "下がる")

    def test_future_teacher_blocks_model_and_price_access(self) -> None:
        """初回締切以降の教師はモデル呼出と結果読取より前に拒否する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        write_json(self.fixture.measured / "初期教師.json", {"rows": [
            {"label_end": "2026-10-01T15:45:00+09:00", "features": "未開示"}]}, exclusive=False)
        with patch("research_support.five_class_learning.predict_five") as model, \
                patch("research_support.five_class_learning._read_labels") as reader:
            with self.assertRaisesRegex(ValueError, "締切以降"):
                run_comparison(self.root, self.fixture.inputs, self.fixture.measured, self.prices)
            model.assert_not_called()
            reader.assert_not_called()

    def test_immature_target_blocks_raw_price_read(self) -> None:
        """実時計の未満了判定より前に原価格を開かない。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        packet = {"targets": {"today": "2099-01-01T15:45:00+09:00"}}
        with patch("research_support.five_class_learning.load_candles") as reader:
            with self.assertRaisesRegex(ValueError, "実時計で未満了"):
                _read_labels(self.prices, packet, self.rules)
            reader.assert_not_called()

    def test_existing_comparison_is_not_overwritten(self) -> None:
        """同じ比較先での二度目の実行は保存結果を変更しない。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        self._run()
        originals = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_preserved_initial_numeric_row_is_accepted(self) -> None:
        """正しく元特徴を保全した過去教師を初回から受け付け、一度ずつ追加する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        original = self._initial_row()
        self._run()
        state = read_json(self.root / "学習状態.json")
        self.assertEqual(len(state["rows"]), 5)
        self.assertEqual(state["rows"][0]["features"], original["features"])
        self.assertEqual(state["rows"][0]["actual_pct"], original["actual_pct"])

    def test_rehashed_teacher_cannot_change_original_features(self) -> None:
        """教師と来歴のSHAを作り直しても元教師の数値改変を許可しない。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        self._initial_row()
        path = self.fixture.measured / "初期教師.json"
        teacher = read_json(path)
        teacher["rows"][0]["features"]["growth"] = 999
        write_json(path, teacher, exclusive=False)
        self._reseal_provenance()
        self._reject_before_access("保全")

    def test_packet_self_hash_cannot_be_replaced_by_a_valid_full_hash(self) -> None:
        """packet自己ハッシュが壊れていれば、全体SHAと元実測を揃えても拒否する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        identity = self.fixture.identities[0]
        path = self.fixture.inputs / "masked" / (identity + ".json")
        packet = read_json(path)
        packet["packet_hash"] = "0" * 64
        write_json(path, packet, exclusive=False)
        current_path = self.fixture.measured / identity / "入力.json"
        current = read_json(current_path)
        current["source_packet_hash"] = digest(packet)
        write_json(current_path, current, exclusive=False)
        (self.original / identity / "入力.json").write_bytes(current_path.read_bytes())
        self._reseal_provenance()
        self._reject_before_access("自己ハッシュ")

    def test_reference_cannot_change_even_when_original_and_copy_match(self) -> None:
        """元実測とコピーを同じ別基準へ変更しても可視原本と照合して拒否する。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        identity = self.fixture.identities[0]
        path = self.fixture.measured / identity / "入力.json"
        current = read_json(path)
        current["states"]["NK225"]["reference"]["price"] = 999
        write_json(path, current, exclusive=False)
        (self.original / identity / "入力.json").write_bytes(path.read_bytes())
        self._reseal_provenance()
        self._reject_before_access("基準価格")

    def test_generator_or_engine_hash_mismatch_blocks_freeze(self) -> None:
        """正しい自己SHAでも生成器ハッシュの相違を新manifestで追認しない。

        引数: self (FiveClassLearningTests) はテスト。
        返り値: None。
        """
        path = self.fixture.measured / "初期教師の来歴.json"
        value = read_json(path)
        value.pop("provenance_hash")
        value["implementation_hash"] = "0" * 64
        value["provenance_hash"] = digest(value)
        write_json(path, value, exclusive=False)
        self._reject_before_access("生成器")


if __name__ == "__main__":
    unittest.main()
