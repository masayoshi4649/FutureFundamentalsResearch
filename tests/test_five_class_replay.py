"""五分類controllerの規約固定、時点制約、個別考察と対象分母を合成入力で検証する。"""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_pdca_analyst as fixture_module
from fundamentals.common import digest, read_json, timestamp, write_json
from fundamentals.replay import _audit, _file_hash
from research_support import five_class as evaluation_module
from research_support.five_class import score_prediction
from research_support.five_class_replay import (
    ANSWERS, OVERLAP_POLICY, PREPARE_FILE, PROTOCOL_FILE,
    finalize_five, prepare_five, reveal_five, seal_five, summarize_five,
)


class FiveClassReplayTests(unittest.TestCase):
    """実データの採点をせず、入力生成だけを旧テストから再利用する。"""

    def setUp(self) -> None:
        """確認済みと明記した合成契約を、一時的な新実験へ置く。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self.fixture = fixture_module.AnalystReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.inputs, self.measured = self.fixture.root, self.fixture.inputs, self.fixture.measured
        self.path = self.root / self.fixture.identities[0]
        self.prices = self.fixture.base / "合成価格"
        self.criteria = {
            "target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
            "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5", "reversal_requires_reference_cross": False,
            "range_threshold_pct": "1", "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
            "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10,
        }
        self.contract = {"version": "合成規約１", "confirmed": True, "confirmed_at": "2000-01-01T00:00:00+09:00",
                         "overlap_policy": OVERLAP_POLICY, "criteria": self.criteria}
        write_json(self.root / "五分類規約.json", self.contract)
        (self.root / "予測担当の手順.md").write_text("# 合成した五分類指示\n" + "／".join(ANSWERS) + "\n", encoding="utf-8")

    def _prepare(self, index: int = 1) -> dict:
        """当回入力と新規約を事前固定する。

        引数: self (FiveClassReplayTests) はテスト、index (int) は回番号。
        返り値: dict は当回入力。
        """
        return prepare_five(self.root, self.inputs, self.measured, index)

    def _seal(self, index: int = 1, first_answer: str | None = "上がる") -> dict:
        """両先物の五分類予測を異なる担当識別子で固定する。

        引数: self (FiveClassReplayTests) はテスト、index (int) は回番号、
            first_answer (str | None) は先頭先物の回答または無回答。
        返り値: dict は固定予測。
        """
        path = self.root / self.fixture.identities[index - 1]
        items = {symbol: {"direction": first_answer if symbol == "NK225" else "上がる", "evidence_ids": ["可視記事"],
                          "already_reacted": "当回先物の既反応を観測。", "additional_reaction": "次場の追加方向仮説。",
                          "invalidation": "反対材料が強まる。"} for symbol in ("NK225", "TOPIX")}
        write_json(path / "予測入力.json", {"instruments": items})
        return seal_five(path, {"agent_id": f"五分類担当{index}", "task_name": f"五分類担当{index}",
                                "fork_turns": "none", "completed": True,
                                "input_file_hash": _file_hash(path / "masked" / "入力.json")})

    def _evaluation(self, target: str = "対象", correct: list[str] | None = None,
                    uncertain: list[str] | None = None, actual: float | None = 0.6) -> dict:
        """controllerだけを調べるための分類器結果を合成する。

        引数: self (FiveClassReplayTests) はテスト、target (str) は対象状態、
            correct (list[str] | None) は確実正解、uncertain (list[str] | None) は可能正解、
            actual (float | None) は終点実績。
        返り値: dict は価格を読まない合成評価。
        """
        return {"target_status": target, "correct_candidates": ["上がる"] if correct is None else correct,
                "uncertain_candidates": [] if uncertain is None else uncertain,
                "metrics": {"actual_pct": actual}, "quality": {"has_unobserved_path": target == "未確定"}}

    def _reveal(self, evaluations: list[dict] | None = None, index: int = 1) -> tuple[dict, list[dict]]:
        """価格読取・基準選択・分類器をモックし、実際の開示保存を実行する。

        引数: self (FiveClassReplayTests) はテスト、evaluations (list[dict] | None) は両先物評価、
            index (int) は回番号。
        返り値: tuple[dict, list[dict]] は結果と価格読取引数。
        """
        path = self.root / self.fixture.identities[index - 1]
        packet = read_json(path / "masked" / "入力.json")
        reference = packet["instruments"]["NK225"]["reference"]
        quote = SimpleNamespace(close=reference["price"], end=timestamp(reference["time"]), session=reference["session"])
        values = evaluations or [self._evaluation(), self._evaluation()]
        with patch("research_support.five_class_replay.load_candles",
                   return_value=[SimpleNamespace(session=packet["session"])]) as price_read, \
                patch("research_support.five_class_replay.reference_quote", return_value=quote), \
                patch.object(evaluation_module, "evaluate_priority_path", side_effect=copy.deepcopy(values), create=True):
            result = reveal_five(path, self.prices)
        return result, [call.kwargs for call in price_read.call_args_list]

    def _reflection(self, result: dict) -> dict:
        """当回の採点状態に一致する個別考察を作成する。

        引数: self (FiveClassReplayTests) はテスト、result (dict) は当回結果。
        返り値: dict は予測・結果の識別子を持つ考察。
        """
        return {"forecast_hash": result["forecast_hash"], "result_hash": digest(result),
                "instruments": {row["symbol"]: {"predicted_direction": row["direction"],
                    "score_status": row["score"]["status"], "causality": "未識別",
                    "reason": row["symbol"] + "の今回回答と観測経路の整合を比較したが単独因果は不明。",
                    "next_action": row["symbol"] + "で次の既反応と追加材料を別確認する。", "evidence_ids": []}
                    for row in result["labels"]}}

    # ----------------------------------------

    def test_unconfirmed_wrong_policy_and_wrong_exact_threshold_are_rejected(self) -> None:
        """未確認・別優先規約・僅かに異なる閾値でも入力固定前に拒否する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        cases = [dict(confirmed=False), dict(overlap_policy="成立する回答はいずれも正解"),
                 dict(criteria={**self.criteria, "movement_threshold_pct": "0.50000000000000000001"})]
        for changes in cases:
            with self.subTest(changes=changes):
                write_json(self.root / "五分類規約.json", {**self.contract, **changes}, exclusive=False)
                with self.assertRaises(ValueError):
                    self._prepare()
                self.assertFalse((self.root / "manifest.json").exists())
                self.assertFalse((self.root / PROTOCOL_FILE).exists())

    def test_future_teacher_is_rejected_before_price_read(self) -> None:
        """未来教師は未開示のまま排除し、当回入力へ渡さない。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        write_json(self.measured / "初期教師.json", {"rows": [{"symbol": "NK225", "as_of": "2026-10-01T07:45:00+09:00",
                   "label_end": "2026-10-01T15:45:00+09:00", "actual_pct": "未開示"}]}, exclusive=False)
        with patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "締切より前"):
                self._prepare()
            price_read.assert_not_called()

    def test_future_article_is_rejected_before_price_read(self) -> None:
        """当回締切後に利用可能になる記事を、結果価格を読む前に拒否する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        path = self.inputs / "masked" / (self.fixture.identities[0] + ".json")
        packet = read_json(path)
        packet["news"][0]["available_at"] = "2026-10-01T08:00:00+09:00"
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        write_json(path, packet, exclusive=False)
        with patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "未来の記事"):
                self._prepare()
            price_read.assert_not_called()

    def test_prepare_binds_new_contract_and_latest_input(self) -> None:
        """新規約を入力へ追加した最終バイトと手順固定をprepare段階へ束縛する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        packet = self._prepare()
        record = _audit(self.path)[0]
        self.assertEqual(packet["evaluation_contract"]["overlap_policy"], OVERLAP_POLICY)
        self.assertEqual(record["files"]["masked/入力.json"], _file_hash(self.path / "masked" / "入力.json"))
        self.assertIn(PREPARE_FILE, record["files"])
        self.assertLessEqual(timestamp(record["prepare_started_at"]), timestamp(record["actual_at"]))
        protocol = self.root / PROTOCOL_FILE
        protocol.write_bytes(protocol.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "準備後"):
            self._seal()

    def test_reveal_requires_seal_and_actual_maturity_before_loading_prices(self) -> None:
        """当回予測が未固定、または実時計で未満了なら価格原本を開かない。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        with patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "予測固定後"):
                reveal_five(self.path, self.prices)
            price_read.assert_not_called()
        self._seal()
        with patch("research_support.five_class_replay.now", return_value="2026-10-01T15:44:59+09:00"), \
                patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "未満了"):
                reveal_five(self.path, self.prices)
            price_read.assert_not_called()

    def test_unimplemented_priority_classifier_never_falls_back_to_general_set(self) -> None:
        """確認済み優先分類器がなければ一般集合へ代用せず価格取得前に停止する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        with patch.object(evaluation_module, "evaluate_priority_path", None, create=True), \
                patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "優先分類器が未実装"):
                reveal_five(self.path, self.prices)
            price_read.assert_not_called()

    def test_unanswered_target_is_saved_as_incorrect_and_not_removed(self) -> None:
        """対象で無回答なら不正解・分母１を保存し、回答を捏造して埋めない。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        prediction = self._seal(first_answer=None)
        result, _ = self._reveal()
        self.assertIsNone(prediction["instruments"]["NK225"]["direction"])
        row = result["labels"][0]
        self.assertIsNone(row["direction"])
        self.assertEqual(row["score"]["status"], "不正解")
        self.assertTrue(row["score"]["denominator"])
        self.assertFalse(row["score"]["correct_upper"])

    def test_five_minute_path_keeps_one_minute_reference(self) -> None:
        """５分経路を選んでも基準は１分、対象ファイルだけは５分とする。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self.criteria["interval_minutes"] = 5
        write_json(self.root / "五分類規約.json", self.contract, exclusive=False)
        self._prepare()
        self._seal()
        result, calls = self._reveal()
        self.assertEqual([item["interval"] for item in calls], [1, 5, 1, 5])
        self.assertTrue(all("candle_5min" in row["source_file"] for row in result["labels"]))

    def test_individual_reflection_actor_and_hashes_gate_next_cutoff(self) -> None:
        """同じ担当と予測・結果の結び付く個別考察が完成するまで次回へ進まない。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        write_json(self.measured / "初期教師.json", {"rows": [{"symbol": "NK225", "as_of": "2026-09-30T07:45:00+09:00",
                   "label_end": "2026-09-30T15:45:00+09:00", "actual_pct": 0.1, "features": {}}]}, exclusive=False)
        first = self._prepare()
        self.assertEqual(len(first["known_history"]), 1)
        self._seal()
        result, _ = self._reveal()
        with self.assertRaisesRegex(ValueError, "振り返り"):
            self._prepare(2)
        reflection = self._reflection(result)
        with self.assertRaisesRegex(ValueError, "結び付き"):
            finalize_five(self.path, reflection, {"agent_id": "別担当", "completed": True})
        invalid = {**reflection, "result_hash": "別結果"}
        with self.assertRaisesRegex(ValueError, "結び付き"):
            finalize_five(self.path, invalid, {"agent_id": "五分類担当1", "completed": True})
        review = finalize_five(self.path, reflection, {"agent_id": "五分類担当1", "completed": True})
        second = self._prepare(2)
        self.assertEqual(len(second["known_history"]), 3)
        self.assertEqual(second["own_past_reviews"][0], review["feedback"])
        self.assertEqual(review["feedback"]["instruments"]["NK225"]["prediction_evidence_ids"], ["可視記事"])
        self.assertEqual(review["feedback"]["instruments"]["NK225"]["reflection_evidence_ids"], [])

    def test_actor_and_original_mutations_are_detected_before_price_result(self) -> None:
        """固定済み担当の差替えを価格原本の読取前に拒否する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        self._seal()
        actor = read_json(self.path / "actor.json")
        actor["agent_id"] = "差替え"
        write_json(self.path / "actor.json", actor, exclusive=False)
        with patch("research_support.five_class_replay.load_candles") as price_read:
            with self.assertRaisesRegex(ValueError, "固定ファイル"):
                reveal_five(self.path, self.prices)
            price_read.assert_not_called()

    def test_input_and_measured_originals_remain_unchanged(self) -> None:
        """準備から個別考察まで実行しても旧入力・実測原本のバイトを変えない。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        originals = {path: path.read_bytes() for base in (self.inputs, self.measured) for path in base.rglob("*") if path.is_file()}
        self._prepare()
        self._seal()
        result, _ = self._reveal()
        finalize_five(self.path, self._reflection(result), {"agent_id": "五分類担当1", "completed": True})
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_summary_preserves_unknown_target_and_outside_counts(self) -> None:
        """対象不明を主分母へ残し、対象外と各状態の無回答を別件数にする。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        rows = [self._evaluation("対象"), self._evaluation("未確定", [], ["上がる"], None),
                self._evaluation("対象外", ["もみ合い"], [], 0.1)]
        result = self._summary_fixture(rows, [None, "上がる", None])
        group = result["groups"]["NK225"]
        self.assertEqual((group["scheduled"], group["target"], group["target_unknown"], group["outside"]), (3, 1, 1, 1))
        self.assertEqual((group["denominator"], group["target_unanswered"], group["outside_unanswered"]), (2, 1, 1))
        self.assertEqual(group["accuracy_lower"], 0)
        self.assertEqual(group["accuracy_upper"], 0.5)

    def test_accuracy_upper_can_exclude_unknown_non_targets(self) -> None:
        """確定対象１正解と不明９件では、全不明が対象外なら真の精度は１となる。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        evaluations = [self._evaluation()] + [self._evaluation("未確定", [], [], None) for _ in range(9)]
        summary = self._summary_fixture(evaluations, ["上がる"] * 10)
        group = summary["groups"]["NK225"]
        self.assertEqual(group["denominator"], 10)
        self.assertEqual(group["accuracy_lower"], 0.1)
        self.assertEqual(group["fixed_denominator_possible_rate"], 0.1)
        self.assertEqual(group["accuracy_upper"], 1)
        self.assertEqual(group["known_target_accuracy_lower"], 1)

    def test_all_five_constant_baselines_use_same_evaluation_and_denominator(self) -> None:
        """５回答を事後選択せず全て保存し、対象不明を含む本体と同じ主分母で比較する。

        引数: self (FiveClassReplayTests) はテスト。
        返り値: None。
        """
        evaluations = [self._evaluation("対象", ["上がる"]),
                       self._evaluation("未確定", [], ["下がる"], None),
                       self._evaluation("対象外", ["もみ合い"], [], 0.1)]
        summary = self._summary_fixture(evaluations, [None, "上がる", None])
        group = summary["groups"]["NK225"]
        self.assertEqual(tuple(group["baselines"]), ANSWERS)
        for answer, baseline in group["baselines"].items():
            self.assertEqual(baseline["denominator"], group["denominator"])
            self.assertEqual((baseline["scheduled"], baseline["target_unknown"], baseline["outside"]), (3, 1, 1))
            self.assertEqual(baseline["unanswered"], 0)
        self.assertEqual(group["baselines"]["上がる"]["accuracy_lower"], 0.5)
        self.assertEqual(group["baselines"]["下がる"]["accuracy_lower"], 0)
        self.assertEqual(group["baselines"]["下がる"]["accuracy_upper"], 0.5)
        self.assertEqual(set(summary["rows"][0]["constant_scores"]), set(ANSWERS))

    def _summary_fixture(self, evaluations: list[dict], answers: list[str | None]) -> dict:
        """分類器結果から集計だけを独立検証する監査済み相当の一時入力を作る。

        引数: self (FiveClassReplayTests) はテスト、evaluations (list[dict]) は各回評価、
            answers (list[str | None]) は各回回答。
        返り値: dict は実データを使わない集計結果。
        """
        target = self.fixture.base / "集計専用"
        identities = []
        for index, (evaluation, answer) in enumerate(zip(evaluations, answers), start=1):
            identity = f"{index:02d}_集計"
            identities.append(identity)
            write_json(target / identity / "masked" / "入力.json", {"kind": "Day"})
            write_json(target / identity / "result.json", {"labels": [{"symbol": symbol, "direction": answer,
                "evaluation": evaluation, "score": score_prediction(answer, evaluation)} for symbol in ("NK225", "TOPIX")]})
        with patch("research_support.five_class_replay._verify", return_value={"rounds": identities}), \
                patch("research_support.five_class_replay._bound", return_value={}), \
                patch("research_support.five_class_replay._audit", return_value=[{"phase": "review"}]):
            return summarize_five(target)


if __name__ == "__main__":
    unittest.main()
