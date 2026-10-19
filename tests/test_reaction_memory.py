"""未来認定・未確定分類・同じ場の水増しを反例で検査する。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fundamentals.common import digest, read_json, timestamp, write_json
from fundamentals.replay import _append_audit
from research_support.reaction_memory import append_reaction_case, snapshot_memory, verify_memory


def completed_round(root: Path, *, session: str = "2026-09-22_Day", end: str = "2026-09-22T15:45:00+09:00",
                    as_of: str = "2026-09-22T07:45:00+09:00", pending: bool = False,
                    marker: str = "一") -> Path:
    """市場原本を使わず、未来の反転順序が未確定の固定済み一場を作る。

    引数: root (Path) は保存先、session/end/as_of (str) は再生場と時計、pending (bool) は保留認定、
    marker (str) は別予測を区別する合成注記。
    返り値: Path はprepareからreviewまで監査した当回。
    """
    root.mkdir(parents=True)
    forecast = {"as_of": as_of, "references": {}, "materials": [], "forecasts": {}, "marker": marker}
    material = {"material_id": "外交材料", "story_key": "共通外交", "title": "外交期待と原油低下",
                "evidence": [{"id": 1, "revision_id": "一"}], "impacts": {}}
    for symbol in ("NK225", "TOPIX"):
        forecast["references"][symbol] = {"price": 100, "proxy": True, "contract": None}
        forecast["forecasts"][symbol] = {"today": {"direction": "上がる", "reason": "合成期待"}}
        material["impacts"][symbol] = {"today": {"impact_pt": 3, "priced_in_before_reference_pt": None,
            "priced_in_after_reference_pt": 0, "reference_remaining_pt": None, "residual_pt": None,
            "transmission": "原油低下から輸入費用改善", "observations": {"before_reference": [], "after_reference": []}}}
    forecast["materials"].append(material)
    forecast["ledger_hash"] = digest(forecast)
    labels = []
    for symbol in ("NK225", "TOPIX"):
        score = {"status": "不正解", "denominator": True, "correct_lower": False, "correct_upper": False}
        evaluation = {"session": session, "target_status": "対象", "correct_candidates": [],
                      "uncertain_candidates": ["上がってから下がる", "下がってから上がる"],
                      "metrics": {"actual_pct": -0.2, "high_pct": 0.8, "low_pct": -0.3},
                      "quality": {"missing_rows": 2, "contract_identity_available": False,
                                  "has_unobserved_path": True}}
        full = {"label_end": end, "evaluation": copy.deepcopy(evaluation)}
        evaluation["evaluation_window"] = {"adjudication_unknown": pending, "scoreable": True,
            "nominal_window_end": end, "effective_quote_time": end,
            "pending_events": [{"event_id": "後日候補", "timing_evidence": {
                "article_public_upper_bound": "2026-09-23T10:00:00+09:00"}}] if pending else []}
        labels.append({"symbol": symbol, "answer": "上がる", "primary": {"label_end": end, "evaluation": evaluation},
                       "score": {**score, "status": "未確定"} if pending else score,
                       "full_reference": full, "full_reference_score": score})
    result = {"forecast_hash": forecast["ledger_hash"], "labels": labels}
    review = {"forecast_hash": forecast["ledger_hash"], "result_hash": digest(result),
              "reviews": {symbol: {"reason": "独自の後日理由", "next_action": "次回は継続と反転を比較",
                                   "causality": "後日認定に依存する仮説"} for symbol in ("NK225", "TOPIX")}}
    write_json(root / "入力固定.json", {"合成": True})
    _append_audit(root, "prepare", timestamp(as_of), [root / "入力固定.json"])
    write_json(root / "予測.json", forecast)
    _append_audit(root, "seal", timestamp(as_of), [root / "予測.json"])
    write_json(root / "result.json", result)
    _append_audit(root, "reveal", timestamp(end), [root / "result.json"])
    write_json(root / "振り返り入力.json", review)
    (root / "振り返り.md").write_text("- 合成振り返り\n", encoding="utf-8")
    _append_audit(root, "review", timestamp(end), [root / "振り返り入力.json", root / "振り返り.md"])
    return root


class ReactionMemoryTests(unittest.TestCase):
    """時間遮断と追記不変性を、合成した別々の場で検査する。"""

    def setUp(self) -> None:
        """試験ごとに独立した合成当回と台帳先を用意する。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.round = completed_round(self.root / "当回")
        self.memory = self.root / "台帳"

    def snapshot(self, cutoff: str = "2026-09-22T16:00:00+09:00", **kwargs) -> dict:
        """未使用の一時保存先へ時点入力を作る。

        引数: self (ReactionMemoryTests) は試験、cutoff (str) は情報締切、kwargs (dict) は追加条件。
        返り値: dict は保存したsnapshot。
        """
        out = self.root / ("切出" + str(len(list(self.root.glob("切出*")))))
        return snapshot_memory(self.memory, cutoff, out, **kwargs)

    # ----------------------------------------

    def test_material_unknown_and_ambiguous_reversal_are_preserved(self) -> None:
        """材料の未知と反転２候補を、単一方向へ補完しない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        case = append_reaction_case(self.round, self.memory)
        row = case["observations"][0]
        self.assertIsNone(row["materials"][0]["points"]["reference_remaining_pt"])
        self.assertEqual(len(row["primary"]["uncertain_candidates"]), 2)
        self.assertEqual(row["full_reference"]["score"]["status"], "不正解")
        self.assertEqual(case["causal_status"], "因果未識別")

    def test_unknown_evidence_exposes_observation_without_hindsight_review(self) -> None:
        """根拠時計不明なら全場観測だけ渡し、後日理由を本文に混ぜない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        result = self.snapshot()
        row = result["cases"][0]["observations"][0]
        self.assertIsNone(row["primary"])
        self.assertIsNone(row["review"])
        self.assertNotIn("独自の後日理由", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["session_count"], 1)
        self.assertEqual(result["instrument_observation_count"], 2)
        self.assertIsNone(result["independent_sample_count"])

    def test_future_pending_evidence_does_not_leak_before_available(self) -> None:
        """後日判明した臨時材料保留を、前日の入力へ遡及させない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        later = completed_round(self.root / "保留当回", pending=True)
        append_reaction_case(later, self.memory, evidence_available_at="2026-09-23T10:05:00+09:00")
        before = self.snapshot()
        self.assertNotIn("後日候補", json.dumps(before, ensure_ascii=False))
        after = self.snapshot("2026-09-23T11:00:00+09:00")
        row = after["cases"][0]["observations"][0]
        self.assertTrue(row["primary"]["evaluation_window"]["adjudication_unknown"])
        self.assertEqual(row["primary"]["score"]["status"], "未確定")
        self.assertEqual(row["causal_status"], "因果未識別")

    def test_declared_clock_cannot_precede_explicit_article_evidence(self) -> None:
        """明示した記事公開時計より前を根拠利用可能時計に指定できない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        later = completed_round(self.root / "保留当回", pending=True)
        with self.assertRaises(ValueError):
            append_reaction_case(later, self.memory, evidence_available_at="2026-09-22T16:00:00+09:00")

    def test_unmatured_session_is_not_returned(self) -> None:
        """予測時点が過去でも対象場の満了前は事例を渡さない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        self.assertEqual(self.snapshot("2026-09-22T15:44:59+09:00")["session_count"], 0)

    def test_current_forecast_is_not_returned(self) -> None:
        """当回自身または未来予測を当回の教師にしない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        self.assertEqual(self.snapshot("2026-09-22T07:45:00+09:00")["session_count"], 0)

    def test_review_completion_has_its_own_simulated_gate(self) -> None:
        """価格満了後も、再生上の考察完了前は次回へ渡さない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory, simulated_review_at="2026-09-22T17:00:00+09:00")
        self.assertEqual(self.snapshot()["session_count"], 0)
        self.assertEqual(self.snapshot("2026-09-22T17:00:00+09:00")["session_count"], 1)

    def test_live_mode_does_not_use_later_actual_recordings(self) -> None:
        """回想記録を過去の前向き成績・入力として利用しない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        self.assertEqual(self.snapshot(mode="運用")["session_count"], 0)
        self.assertEqual(self.snapshot(mode="回想")["session_count"], 1)

    def test_incomplete_round_cannot_be_added(self) -> None:
        """review段階を終えていない当回の結果だけを教師へ追加しない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        path = self.round / "audit.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            append_reaction_case(self.round, self.memory)

    def test_duplicate_append_cannot_change_an_existing_case(self) -> None:
        """同じ事例への再追記を拒否し、元のファイルを保持する。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        case = append_reaction_case(self.round, self.memory)
        path = self.memory / "cases" / (case["case_id"] + ".json")
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            append_reaction_case(self.round, self.memory, evidence_available_at="2026-09-22T16:00:00+09:00")
        self.assertEqual(path.read_bytes(), original)

    def test_case_tamper_is_rejected_before_snapshot(self) -> None:
        """台帳本文を変更した場合、内容を入力として返さない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        case = append_reaction_case(self.round, self.memory)
        path = self.memory / "cases" / (case["case_id"] + ".json")
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.snapshot()

    def test_source_review_markdown_tamper_is_detected(self) -> None:
        """JSON以外の監査済み振り返り原文の変更も検知する。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        (self.round / "振り返り.md").write_text("変更", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_memory(self.memory)

    def test_orphan_case_is_rejected(self) -> None:
        """監査に未登録の事例を自動採用しない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        (self.memory / "cases").mkdir(parents=True)
        write_json(self.memory / "cases" / "孤立.json", {})
        with self.assertRaises(ValueError):
            verify_memory(self.memory)

    def test_similar_material_search_and_nonmatch_filter(self) -> None:
        """キーワード・物語・伝達経路で近い材料を検索する。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        for query in ({"keywords": ["外交"]}, {"story_keys": ["共通外交"]}, {"channels": ["輸入費用"]}):
            self.assertEqual(self.snapshot(**query)["session_count"], 1)
        self.assertEqual(self.snapshot(keywords=["無関係の地震"])["session_count"], 0)

    def test_two_forecasts_of_one_session_do_not_inflate_count(self) -> None:
        """同じ場の別予測を追加しても次回入力の場数を水増ししない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        another = completed_round(self.root / "別方式", marker="二")
        append_reaction_case(another, self.memory)
        self.assertEqual(len(verify_memory(self.memory)), 2)
        self.assertEqual(self.snapshot()["session_count"], 1)

    def test_snapshot_output_is_fixed_and_has_self_hash(self) -> None:
        """時点別入力の上書きを拒否し、JSON自己ハッシュを保存する。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        append_reaction_case(self.round, self.memory)
        out = self.root / "固定切出"
        result = snapshot_memory(self.memory, "2026-09-22T16:00:00+09:00", out)
        self.assertEqual(result["snapshot_hash"], digest({key: value for key, value in result.items() if key != "snapshot_hash"}))
        self.assertEqual(read_json(out / "snapshot.json"), result)
        with self.assertRaises(FileExistsError):
            snapshot_memory(self.memory, "2026-09-22T16:00:00+09:00", out)

    def test_review_cannot_be_moved_before_maturity(self) -> None:
        """満了時刻より前へ振り返り時計を移せない。

        引数: self (ReactionMemoryTests) は試験。
        返り値: None。
        """
        with self.assertRaises(ValueError):
            append_reaction_case(self.round, self.memory, simulated_review_at="2026-09-22T15:44:00+09:00")


if __name__ == "__main__":
    unittest.main()
