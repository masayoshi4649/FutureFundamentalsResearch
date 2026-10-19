"""開発再生の固定・開示順序、原証跡、材料時点を合成データで検証する。"""

import copy
import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import JST, digest, read_json, timestamp, write_json
from fundamentals.pdca import CHANNELS
from fundamentals.pdca import feature_vector
from fundamentals.pdca_research import run_cycle
from fundamentals.replay import _append_audit, _audit


class PdcaResearchTests(unittest.TestCase):
    """予測固定前の結果読取と、改変・未来・重複実績の利用を防ぐ。"""

    def setUp(self) -> None:
        """実験入力と原実績を独立した一時フォルダに用意する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.source = self.base / "改善入力", self.base / "元実績"
        self.root.mkdir()
        (self.root / "実験計画.md").write_text("# 合成実験\n", encoding="utf-8")
        (self.root / "注釈手順.md").write_text("# 可知材料だけ採点\n", encoding="utf-8")

    def _round(self, number: int, result_changes: dict | None = None,
               prediction_changes: dict | None = None, duplicate: bool = False) -> str:
        """固定入力・注釈と、監査済みの合成原予測・原実績を作る。

        引数: self (PdcaResearchTests) はテスト、number (int) は昇順の日、result_changes
        (dict | None) は先頭実績の変更、prediction_changes (dict | None) は元予測の変更、
        duplicate (bool) は重複ラベルを意図的に追加する指定。
        返り値: str は回の識別子。
        """
        day = datetime(2026, 10, number, tzinfo=JST)
        as_of = day.replace(hour=7, minute=45)
        target = day.replace(hour=15, minute=45)
        identity = f"{number:02d}_合成回"
        previous = (day - timedelta(days=1)).date().isoformat() + "_Night"
        packet = {"as_of": as_of.isoformat(), "session": day.date().isoformat() + "_Day", "kind": "Day",
                  "targets": {"today": target.isoformat()}, "instruments": {},
                  "news": [{"id": "可視記事", "available_at": day.replace(hour=6).isoformat(),
                            "fetched_at": day.replace(hour=6).isoformat(), "body_text": "固定材料"}]}
        annotation = {"as_of": as_of.isoformat(), "instruments": {}}
        labels = []
        for symbol in ("NK225", "TOPIX"):
            packet["instruments"][symbol] = {
                "reference": {"price": 100.0, "time": day.replace(hour=5, minute=55).isoformat(), "session": previous},
                "recent_sessions": [{"session": previous, "return_pct": -0.4}],
                "horizons": {"today": {"distribution": {"n": 0, "terminal_range_pct": None, "path_range_pct": None}}}}
            annotation["instruments"][symbol] = {"channels": {
                channel: {"score": 0.5, "confidence": 0.8, "novelty": 0.7, "priced_in": 0.2,
                          "evidence": ["可視記事"], "reason": "結果と無関係な合成注釈"} for channel in CHANNELS}}
            labels.append({"symbol": symbol, "horizon": "today", "label_end": target.isoformat(),
                           "reference_price": 100.0, "quote_price": 101.0,
                           "quote_time": day.replace(hour=15, minute=40).isoformat(),
                           "actual_pct": 1.0, "low_pct": -0.3, "high_pct": 1.2,
                           "path_hash": digest([identity, symbol])})
        packet["packet_hash"] = digest(packet)
        write_json(self.root / "masked" / (identity + ".json"), packet)
        write_json(self.root / "annotations" / (identity + ".json"), annotation)

        # ----------------------------------------

        original = self.source / identity
        write_json(original / "入力.json", packet)
        _append_audit(original, "prepare", as_of, [original / "入力.json"])
        prediction = {"as_of": packet["as_of"], "session": packet["session"],
                      "instruments": copy.deepcopy(packet["instruments"])}
        prediction.update(prediction_changes or {})
        prediction["forecast_hash"] = digest(prediction)
        write_json(original / "予測.json", prediction)
        _append_audit(original, "seal", as_of, [original / "予測.json"])
        labels[0].update(result_changes or {})
        if duplicate:
            labels.append(copy.deepcopy(labels[0]))
        result = {"forecast_hash": prediction["forecast_hash"], "labels": labels}
        write_json(original / "result.json", result)
        _append_audit(original, "reveal", target, [original / "result.json"])
        write_json(original / "review.json", {"原実績": digest(result)})
        _append_audit(original, "review", target, [original / "review.json"])
        return identity

    def _rewrite_packet(self, identity: str, packet: dict) -> None:
        """負の試験用に入力の意味と自己ハッシュを同時に更新する。

        引数: self (PdcaResearchTests) はテスト、identity (str) は回、packet (dict) は変更後入力。
        返り値: None。実験開始前の一時入力だけを変更する。
        """
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        write_json(self.root / "masked" / (identity + ".json"), packet, exclusive=False)

    def _initial_teacher(self, identity: str, size: int = 13) -> Path:
        """評価開始より前の両先物の学習専用実績を一時領域へ作る。

        引数: self (PdcaResearchTests) はテスト、identity (str) は最初の評価回、
        size (int) は各先物の初期教師件数。
        返り値: Path は回想専用の教師JSON。
        """
        packet = read_json(self.root / "masked" / (identity + ".json"))
        annotation = read_json(self.root / "annotations" / (identity + ".json"))
        rows = []
        for day in range(1, size + 1):
            for symbol in ("NK225", "TOPIX"):
                rows.append({"symbol": symbol, "features": feature_vector(annotation, packet, symbol),
                             "actual_pct": 0.8, "label_end": f"2026-09-{day:02d}T15:45:00+09:00"})
        path = self.base / "学習専用教師.json"
        write_json(path, {"mode": "回想・開発再生", "rows": rows})
        return path

    # ----------------------------------------

    def test_result_is_read_only_after_prediction_seal_and_past_review(self) -> None:
        """９回目の学習まで、固定→原実績読取→振り返りの順序を確認する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identities = [self._round(number) for number in range(1, 10)]
        original_bytes = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        output = self.root / "循環"
        reads = []

        def guarded_audit(path: Path) -> list[dict]:
            """原実績の監査が呼ばれる時点で当回予測と前回振り返りの固定を確認する。

            引数: path (Path) は原実績の一回。
            返り値: list[dict] は元の検証済み監査記録。
            """
            current = output / path.name
            self.assertTrue((current / "予測.json").is_file())
            self.assertTrue((current / "予測.md").is_file())
            self.assertEqual(_audit(current)[-1]["phase"], "seal")
            index = identities.index(path.name)
            if index:
                self.assertEqual(_audit(output / identities[index - 1])[-1]["phase"], "review")
            reads.append(path.name)
            return _audit(path)

        with patch("fundamentals.pdca_research._audit", side_effect=guarded_audit):
            summary = run_cycle(self.root, self.source, "循環", "ridge")
        self.assertEqual(reads, identities)
        self.assertEqual(summary["groups"]["NK225"]["fitted_predictions"], 1)
        first = read_json(output / identities[0] / "予測.json")
        ninth = read_json(output / identities[-1] / "予測.json")
        self.assertEqual(first["instruments"]["NK225"]["fitted_n"], 0)
        self.assertEqual(ninth["instruments"]["NK225"]["fitted_n"], 8)
        self.assertEqual(ninth["instruments"]["NK225"]["training_last_label_end"], "2026-10-08T15:45:00+09:00")
        self.assertEqual(original_bytes, {path: path.read_bytes() for path in original_bytes})
        for identity in identities:
            self.assertEqual([entry["phase"] for entry in _audit(output / identity)], ["prepare", "seal", "reveal", "review"])

    def test_future_article_is_rejected_before_source_read(self) -> None:
        """利用可能・公表・取得・更新のいずれかが未来なら結果を開く前に拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identity = self._round(1)
        original = read_json(self.root / "masked" / (identity + ".json"))
        for field in ("available_at", "published_at", "fetched_at", "updated_at"):
            with self.subTest(field=field):
                packet = copy.deepcopy(original)
                packet["news"][0][field] = "2026-10-01T08:00:00+09:00"
                self._rewrite_packet(identity, packet)
                with patch("fundamentals.pdca_research._audit") as source_read:
                    with self.assertRaisesRegex(ValueError, "未来の記事"):
                        run_cycle(self.root, self.source, "未来拒否_" + field, "rule")
                    source_read.assert_not_called()

    def test_missing_article_availability_and_unknown_evidence_are_rejected(self) -> None:
        """利用可能時刻の欠損と、可視記事にない根拠IDを別々に拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identity = self._round(1)
        packet = read_json(self.root / "masked" / (identity + ".json"))
        missing = copy.deepcopy(packet)
        del missing["news"][0]["available_at"]
        self._rewrite_packet(identity, missing)
        with self.assertRaisesRegex(ValueError, "available_at"):
            run_cycle(self.root, self.source, "欠損拒否", "rule")
        self._rewrite_packet(identity, packet)
        annotation_path = self.root / "annotations" / (identity + ".json")
        annotation = read_json(annotation_path)
        annotation["instruments"]["NK225"]["channels"]["growth"]["evidence"] = ["未来秘密ID"]
        write_json(annotation_path, annotation, exclusive=False)
        with self.assertRaisesRegex(ValueError, "可視記事にない根拠"):
            run_cycle(self.root, self.source, "根拠拒否", "rule")

    def test_annotation_cutoff_mismatch_is_rejected(self) -> None:
        """注釈の締切が入力の締切と違えば予測を発行しない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identity = self._round(1)
        annotation_path = self.root / "annotations" / (identity + ".json")
        annotation = read_json(annotation_path)
        annotation["as_of"] = "2026-10-01T16:00:00+09:00"
        write_json(annotation_path, annotation, exclusive=False)
        with self.assertRaisesRegex(ValueError, "情報締切が一致"):
            run_cycle(self.root, self.source, "締切拒否", "rule")

    def test_duplicate_or_wrong_endpoint_labels_are_rejected(self) -> None:
        """重複標本と別満期の実績を、固定済み原本であっても利用しない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1, duplicate=True)
        with self.assertRaisesRegex(ValueError, "各１件"):
            run_cycle(self.root, self.source, "重複拒否", "rule")

    def test_wrong_endpoint_label_is_rejected_after_seal(self) -> None:
        """評価対象日のすり替えを予測固定後に検出し、採点しない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identity = self._round(1, result_changes={"label_end": "2026-10-02T15:45:00+09:00"})
        with self.assertRaisesRegex(ValueError, "評価終点"):
            run_cycle(self.root, self.source, "満期拒否", "rule")
        self.assertEqual(_audit(self.root / "満期拒否" / identity)[-1]["phase"], "seal")

    def test_source_prediction_cutoff_and_session_are_bound(self) -> None:
        """同じ価格水準でも元予測の対象セッションが違えば利用しない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1, prediction_changes={"session": "2026-09-30_Day"})
        with self.assertRaisesRegex(ValueError, "対象セッション"):
            run_cycle(self.root, self.source, "元予測拒否", "rule")

    def test_fixed_input_changes_after_seal_are_rejected_before_source_read(self) -> None:
        """予測固定後の計画・手順・注釈・入力のバイト変更を原実績の前に拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identity = self._round(1)
        targets = [self.root / "実験計画.md", self.root / "注釈手順.md",
                   self.root / "masked" / (identity + ".json"), self.root / "annotations" / (identity + ".json")]
        for index, target in enumerate(targets):
            with self.subTest(target=target.name):
                original_bytes = target.read_bytes()

                def mutate_after_seal(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
                    """通常の固定後に試験対象の原本バイトだけを変更する。

                    引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
                    files (list[Path]) は固定するファイル。
                    返り値: None。
                    """
                    _append_audit(path, phase, simulated_at, files)
                    if phase == "seal":
                        target.write_bytes(original_bytes + b" ")

                with patch("fundamentals.pdca_research._append_audit", side_effect=mutate_after_seal), patch("fundamentals.pdca_research._audit") as source_read:
                    with self.assertRaisesRegex(ValueError, "変更されています"):
                        run_cycle(self.root, self.source, f"固定拒否{index}", "rule")
                    source_read.assert_not_called()
                target.write_bytes(original_bytes)

    def test_completed_cycle_cannot_be_overwritten(self) -> None:
        """完了した同じ循環を再実行しても予測・振り返りを上書きしない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        run_cycle(self.root, self.source, "固定済", "rule")
        output = self.root / "固定済"
        original = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(FileExistsError, "上書き"):
            run_cycle(self.root, self.source, "固定済", "rule")
        self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_nonincreasing_cutoffs_are_rejected_before_next_prediction(self) -> None:
        """ファイル名が昇順でも締切が逆行する入力から次の予測を作らない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        second = self._round(2)
        packet = read_json(self.root / "masked" / (second + ".json"))
        packet["as_of"] = "2026-10-01T07:45:00+09:00"
        packet["instruments"]["NK225"]["reference"]["time"] = "2026-10-01T05:55:00+09:00"
        packet["instruments"]["TOPIX"]["reference"]["time"] = "2026-10-01T05:55:00+09:00"
        packet["news"][0]["available_at"] = "2026-10-01T06:00:00+09:00"
        packet["news"][0]["fetched_at"] = "2026-10-01T06:00:00+09:00"
        self._rewrite_packet(second, packet)
        annotation_path = self.root / "annotations" / (second + ".json")
        annotation = read_json(annotation_path)
        annotation["as_of"] = packet["as_of"]
        write_json(annotation_path, annotation, exclusive=False)
        with self.assertRaisesRegex(ValueError, "厳密な昇順"):
            run_cycle(self.root, self.source, "逆行拒否", "rule")
        self.assertFalse((self.root / "逆行拒否" / second / "予測.json").exists())

    def test_saved_implementation_preserves_source_bytes_and_manifest_hash(self) -> None:
        """後日のコード変更後も固定内容を復元できるよう、改行を含む原文を保存する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        (self.root / "実験計画.md").write_bytes("# 合成実験\r\n- 改行保持\r\n".encode("utf-8"))
        run_cycle(self.root, self.source, "控え確認", "rule")
        output = self.root / "控え確認"
        saved = read_json(output / "実装控え.json")
        manifest = read_json(output / "manifest.json")
        self.assertEqual(set(saved["files"]), {"fundamentals/pdca.py", "fundamentals/pdca_research.py", "実験計画.md", "注釈手順.md"})
        for item in saved["files"].values():
            self.assertEqual(hashlib.sha256(item["content"].encode("utf-8")).hexdigest(), item["file_hash"])
        self.assertEqual(digest({key: value for key, value in saved.items() if key != "snapshot_hash"}), saved["snapshot_hash"])
        self.assertEqual(digest({key: value for key, value in manifest.items() if key != "manifest_hash"}), manifest["manifest_hash"])
        self.assertEqual(hashlib.sha256((output / "実装控え.json").read_bytes()).hexdigest(), manifest["implementation_copy_hash"])

    def test_manifest_mutation_is_rejected_before_source_read(self) -> None:
        """固定一覧の自己ハッシュを更新しても、開始時の一覧との差替えを拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)

        def mutate_manifest(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """予測固定後に一覧の意味と自己ハッシュを差し替える。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
            files (list[Path]) は固定ファイル。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                target = path.parent / "manifest.json"
                manifest = read_json(target)
                manifest["method"] = "差替え"
                manifest["manifest_hash"] = digest({key: value for key, value in manifest.items() if key != "manifest_hash"})
                write_json(target, manifest, exclusive=False)

        with patch("fundamentals.pdca_research._append_audit", side_effect=mutate_manifest), patch("fundamentals.pdca_research._audit") as source_read:
            with self.assertRaisesRegex(ValueError, "固定一覧のハッシュ"):
                run_cycle(self.root, self.source, "一覧拒否", "rule")
            source_read.assert_not_called()

    def test_analog_cycle_keeps_separate_plan_and_reports_reference_reactions(self) -> None:
        """追加循環の計画を固定し、冷開始と５類例照合をMDでも区別する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        identities = [self._round(number) for number in range(1, 10)]
        (self.root / "循環03_計画.md").write_text("# 固定した類例照合\n", encoding="utf-8")
        run_cycle(self.root, self.source, "類例循環", "analog")
        output = self.root / "類例循環"
        manifest = read_json(output / "manifest.json")
        saved = read_json(output / "実装控え.json")
        self.assertEqual(manifest["cycle_plan_hash"], hashlib.sha256((self.root / "循環03_計画.md").read_bytes()).hexdigest())
        self.assertIn("循環03_計画.md", saved["files"])
        first = (output / identities[0] / "予測.md").read_text(encoding="utf-8")
        ninth = (output / identities[-1] / "予測.md").read_text(encoding="utf-8")
        review = (output / identities[-1] / "振り返り.md").read_text(encoding="utf-8")
        self.assertIn("類例照合0件", first)
        self.assertIn("類例照合5件", ninth)
        self.assertIn("方向別の参考重み比（未校正）", ninth)
        self.assertIn("参照した評価終点（JST）", ninth)
        self.assertIn("類例照合後", review)
        self.assertNotIn("係数を再計算", review)
        self.assertIn("類例照合後の発行", (output / "総括.md").read_text(encoding="utf-8"))

    def test_analog_cycle_plan_change_is_rejected_before_source_read(self) -> None:
        """類例循環の追加計画も予測固定後の原実績読取前に変更検知する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        plan = self.root / "循環03_計画.md"
        plan.write_text("# 固定計画\n", encoding="utf-8")

        def mutate_cycle_plan(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """通常の予測固定直後に追加計画を変更する。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
            files (list[Path]) は固定ファイル。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                plan.write_text("# 変更した計画\n", encoding="utf-8")

        with patch("fundamentals.pdca_research._append_audit", side_effect=mutate_cycle_plan), patch("fundamentals.pdca_research._audit") as source_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                run_cycle(self.root, self.source, "類例計画拒否", "analog")
            source_read.assert_not_called()

    # ----------------------------------------

    def test_initial_history_fits_first_prediction_without_inflating_evaluation(self) -> None:
        """各先物13件で初回からモデルを使い、初期教師を評価分母へ混ぜない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first, second = self._round(1), self._round(2)
        teacher = self._initial_teacher(first)
        (self.root / "循環03_計画.md").write_text("# 固定した類例照合\n", encoding="utf-8")
        for method in ("ridge", "analog"):
            with self.subTest(method=method):
                output = self.root / ("初期教師_" + method)
                summary = run_cycle(self.root, self.source, output.name, method, initial_history_path=teacher)
                prediction = read_json(output / first / "予測.json")
                following = read_json(output / second / "予測.json")
                self.assertEqual(summary["initial_history_endpoint_n"], 13)
                self.assertEqual(summary["initial_history_n"], 26)
                self.assertEqual(len(summary["rows"]), 4)
                for symbol in ("NK225", "TOPIX"):
                    self.assertFalse(prediction["instruments"][symbol]["cold_start"])
                    self.assertEqual(prediction["instruments"][symbol]["fitted_n"], 13)
                    self.assertEqual(following["instruments"][symbol]["fitted_n"], 14)
                    self.assertEqual(summary["groups"][symbol]["n"], 2)
                manifest = read_json(output / "manifest.json")
                saved = read_json(output / "実装控え.json")
                self.assertEqual(manifest["initial_history_file_hash"], hashlib.sha256(teacher.read_bytes()).hexdigest())
                self.assertEqual(prediction["initial_history_normalized_hash"], manifest["initial_history_normalized_hash"])
                self.assertEqual(saved["files"]["初期教師.json"]["content"].encode("utf-8"), teacher.read_bytes())
                self.assertIn("評価分母へ含めない", (output / "総括.md").read_text(encoding="utf-8"))

    def test_all_initial_cutoffs_are_checked_before_unknown_numeric_values(self) -> None:
        """初期教師の全満了時刻を先に調べ、未開示値を解釈せず未来・同時刻を拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first = self._round(1)
        teacher = self._initial_teacher(first)
        original = read_json(teacher)
        for index, ending in enumerate(("2026-10-01T07:45:00+09:00", "2026-10-01T15:45:00+09:00")):
            with self.subTest(ending=ending):
                changed = copy.deepcopy(original)
                changed["rows"][0]["actual_pct"] = "未開示で解析不可"
                changed["rows"].append({"symbol": "TOPIX", "label_end": ending, "features": "未開示", "actual_pct": "未開示"})
                write_json(teacher, changed, exclusive=False)
                with self.assertRaisesRegex(ValueError, "厳密に前"):
                    run_cycle(self.root, self.source, f"初期未来拒否{index}", "ridge", initial_history_path=teacher)

    def test_duplicate_and_invalid_initial_features_are_rejected(self) -> None:
        """初期教師を重ねた件数水増しと、検証されていない特徴量を拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first = self._round(1)
        teacher = self._initial_teacher(first)
        original = read_json(teacher)
        duplicated = copy.deepcopy(original)
        duplicated["rows"].append(copy.deepcopy(duplicated["rows"][0]))
        write_json(teacher, duplicated, exclusive=False)
        with self.assertRaisesRegex(ValueError, "重複"):
            run_cycle(self.root, self.source, "初期重複拒否", "ridge", initial_history_path=teacher)
        invalid = copy.deepcopy(original)
        invalid["rows"][0]["features"]["growth"] = 2.0
        write_json(teacher, invalid, exclusive=False)
        with self.assertRaisesRegex(ValueError, "growth"):
            run_cycle(self.root, self.source, "初期特徴拒否", "ridge", initial_history_path=teacher)

    def test_initial_teacher_changes_are_rejected_before_source_read(self) -> None:
        """開始時に固定した初期教師のバイト変更を、原実績読取の前に拒否する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first = self._round(1)
        teacher = self._initial_teacher(first)
        original = teacher.read_bytes()

        def mutate_teacher(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """予測固定後に学習専用教師の原本バイトを変更する。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
            files (list[Path]) は固定ファイル。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                teacher.write_bytes(original + b" ")

        with patch("fundamentals.pdca_research._append_audit", side_effect=mutate_teacher), patch("fundamentals.pdca_research._audit") as source_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                run_cycle(self.root, self.source, "初期改変拒否", "ridge", initial_history_path=teacher)
            source_read.assert_not_called()

    def test_no_initial_history_keeps_previous_output_schema(self) -> None:
        """初期教師を指定しない既存経路へ新しい履歴項目を追加しない。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first = self._round(1)
        summary = run_cycle(self.root, self.source, "教師なし", "ridge")
        output = self.root / "教師なし"
        for value in (summary, read_json(output / "manifest.json"), read_json(output / first / "予測.json")):
            self.assertFalse(any(key.startswith("initial_history") for key in value))
        self.assertNotIn("初期教師.json", read_json(output / "実装控え.json")["files"])

    def test_scaled_cycle_preserves_plan_and_reports_extrapolation(self) -> None:
        """尺度比較だけの追加計画を結合し、学習域外の予測をMDに明記する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        first = self._round(1)
        teacher = self._initial_teacher(first, size=12)
        initial = read_json(teacher)
        for index, row in enumerate(initial["rows"]):
            row["features"]["growth"] = 0.01 * (index // 2 + 1)
        write_json(teacher, initial, exclusive=False)
        plan = self.root / "循環06_計画.md"
        plan.write_text("# 過去行だけで標準化\n", encoding="utf-8")
        run_cycle(self.root, self.source, "尺度比較", "ridge_scaled", initial_history_path=teacher)
        output = self.root / "尺度比較"
        manifest = read_json(output / "manifest.json")
        saved = read_json(output / "実装控え.json")
        self.assertEqual(manifest["scaled_plan_hash"], hashlib.sha256(plan.read_bytes()).hexdigest())
        self.assertIn("循環06_計画.md", saved["files"])
        self.assertNotIn("循環03_計画.md", saved["files"])
        report = (output / first / "予測.md").read_text(encoding="utf-8")
        self.assertIn("学習域外の特徴：growth", report)
        self.assertIn("未校正の外挿", report)

    def test_scaled_plan_change_is_rejected_before_source_read(self) -> None:
        """尺度比較の計画も予測固定後、結果読取前に変更検知する。

        引数: self (PdcaResearchTests) はテスト。
        返り値: None。
        """
        self._round(1)
        plan = self.root / "循環06_計画.md"
        plan.write_text("# 固定した尺度比較\n", encoding="utf-8")

        def mutate_scaled_plan(path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
            """予測の固定直後に尺度比較計画だけを変える。

            引数: path (Path) は出力、phase (str) は段階、simulated_at (datetime) は再生時刻、
            files (list[Path]) は固定ファイル。
            返り値: None。
            """
            _append_audit(path, phase, simulated_at, files)
            if phase == "seal":
                plan.write_text("# 差し替えた尺度比較\n", encoding="utf-8")

        with patch("fundamentals.pdca_research._append_audit", side_effect=mutate_scaled_plan), patch("fundamentals.pdca_research._audit") as source_read:
            with self.assertRaisesRegex(ValueError, "変更されています"):
                run_cycle(self.root, self.source, "尺度計画拒否", "ridge_scaled")
            source_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
