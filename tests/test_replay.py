"""入力隔離、予測固定、実績開示、振り返り引継ぎを合成市場で検証する。"""

import copy
import csv
import json
import tempfile
import unittest
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import JST, digest, read_json, timestamp, write_json
from fundamentals.replay import prepare_round, review_round, seal_prediction, summarize_replay, track_long_term


class ReplayTests(unittest.TestCase):
    """未来本文と未満了評価を次の予測へ渡さない境界を検証する。"""

    def setUp(self) -> None:
        """過去経験範囲を作れる価格と、未来記事を一時領域へ用意する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.prices = self.base / "反復", self.base / "価格"
        self.as_of = timestamp("2026-10-15T07:45:00+09:00")
        self.observed_at = timestamp("2026-10-15T16:00:00+09:00")
        for day in range(1, 15):
            self._session(f"2026-10-{day:02d}_Day", 50000 + day)
            self._session(f"2026-10-{day:02d}_Night", 48000)
        self._session("2026-10-15_Day", 48500)
        self._session("2026-10-15_Night", 49000)
        self.articles = [self._article("可視", "07"), self._article("未来秘密", "12")]

    def _session(self, session: str, price: float) -> None:
        """両指数の最終足を公式終了予定の５分前に保存する。

        引数: self (ReplayTests) はテスト、session (str) はセッション、price (float) はNK225価格。
        返り値: None。
        """
        day = datetime.strptime(session[:10], "%Y-%m-%d").date()
        start = datetime.combine(day + timedelta(days=1), time(5, 54), JST) if session.endswith("Night") else datetime.combine(day, time(15, 39), JST)
        for symbol, scale in (("NK225", 1), ("TOPIX", 0.05)):
            directory = self.prices / "candle_1min" / symbol
            directory.mkdir(parents=True, exist_ok=True)
            row = {"time": start.isoformat(), "open": price * scale, "high": (price + 10) * scale,
                   "low": (price - 10) * scale, "close": price * scale, "volume": 1,
                   "closed": "true", "volume_complete": "true"}
            with (directory / f"{session}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

    def _article(self, identity: str, hour: str) -> dict:
        """利用可能時刻と本文を持つ記事を作る。

        引数: self (ReplayTests) はテスト、identity (str) は識別子、hour (str) は公表時刻。
        返り値: dict は記事。
        """
        return {"id": identity, "revision_id": "初版", "headline": identity,
                "canonical_url": "https://example.com/" + identity,
                "published_at": f"2026-10-15T{hour}:00:00+09:00",
                "fetched_at": f"2026-10-15T{hour}:01:00+09:00", "body_text": identity * 1000}

    def _prepare(self, identity: str = "第１回") -> Path:
        """一回の可視入力を準備する。

        引数: self (ReplayTests) はテスト、identity (str) は回名。
        返り値: Path は保存先。
        """
        prepare_round(self.root, identity, self.as_of, self.prices, self.articles, [])
        return self.root / identity

    def _input(self, run: Path, directions: tuple[str, str] = ("上昇", "下落")) -> None:
        """数値範囲を含まない定性予測を保存する。

        引数: self (ReplayTests) はテスト、run (Path) は保存先、directions (tuple[str, str]) は両指数方向。
        返り値: None。
        """
        value = {"instruments": {}, "lessons_applied": []}
        for symbol, direction in zip(("NK225", "TOPIX"), directions):
            value["instruments"][symbol] = {"direction": direction, "reasons": ["可視材料の追加反応を仮説とする。"],
                                             "risks": ["反対方向の需給。"],
                                             "long_term": {horizon: {"direction": "保留", "pointer": "追加材料と満了を確認。"} for horizon in ("week", "month", "quarter")}}
        write_json(run / "予測入力.json", value)

    # ----------------------------------------

    def test_future_news_and_prices_do_not_change_masked_packet(self) -> None:
        """未来記事の識別子も価格も切出入力へ混ぜない。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        packet = read_json(run / "masked" / "packet.json")
        self.assertEqual(set(packet), {"as_of", "session", "kind", "targets", "instruments", "news", "feedback"})
        self.assertEqual([article["id"] for article in packet["news"]], ["可視"])
        self.assertEqual(len(packet["news"][0]["body_excerpt"]), 1200)
        masked = "".join(path.read_text(encoding="utf-8") for path in (run / "masked").iterdir())
        self.assertNotIn("未来秘密", masked)
        self.assertNotIn(str(self.prices), masked)
        self.assertTrue(all(len(item["recent_sessions"]) <= 6 for item in packet["instruments"].values()))
        self._session("2026-10-15_Day", 900000)
        other = self.base / "別反復"
        modified = copy.deepcopy(self.articles)
        modified[1]["headline"] = "将来改訂後の秘密"
        second = prepare_round(other, "同時点", self.as_of, self.prices, modified, [])
        self.assertEqual(packet, second)

    def test_review_before_seal_and_next_round_before_review_are_rejected(self) -> None:
        """準備・固定・開示・振り返りの順序を飛ばせない。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        with self.assertRaisesRegex(ValueError, "予測固定後"):
            review_round(run, self.prices, self.articles, self.observed_at, {})
        self._input(run)
        seal_prediction(run)
        with self.assertRaisesRegex(ValueError, "前回の振り返り"):
            prepare_round(self.root, "第２回", self.observed_at, self.prices, [], [])
        revealed = review_round(run, self.prices, self.articles, self.observed_at)
        self.assertTrue(revealed["pending_review"])
        self.assertFalse((run / "振り返り.md").exists())
        self.assertEqual(revealed["feedback"], [])
        completed = review_round(run, self.prices, self.articles, self.observed_at, {})
        self.assertFalse(completed["pending_review"])
        audit = [json.loads(line) for line in (run / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["phase"] for item in audit], ["prepare", "seal", "reveal", "review"])
        self.assertTrue(all("actual_at" in item and "simulated_at" in item for item in audit))

    def test_prediction_and_markdown_modification_are_rejected(self) -> None:
        """固定したJSONやMDのどちらの改変も開示前に検知する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        for filename in ("予測.json", "予測.md", "masked/packet.json", "baseline.json"):
            with self.subTest(filename=filename):
                self.root = self.base / filename.replace("/", "_").replace(".", "_")
                run = self._prepare()
                self._input(run)
                seal_prediction(run)
                with (run / filename).open("a", encoding="utf-8") as stream:
                    stream.write("\n ")
                with self.assertRaisesRegex(ValueError, "変更"):
                    review_round(run, self.prices, [], self.observed_at, {})
                self.assertFalse((run / "result.json").exists())

    def test_feedback_requires_past_availability_and_fixed_origin(self) -> None:
        """未来の評価と証跡にない振り返り差替えを次回へ渡さない。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        self._input(run)
        seal_prediction(run)
        result = review_round(run, self.prices, self.articles, self.observed_at, {})
        inherited = result["feedback"]
        self.assertEqual(read_json(run / "review.json")["feedback"], inherited)
        bad = copy.deepcopy(inherited)
        bad[0]["simulated_available_at"] = "2026-10-16T16:00:00+09:00"
        with self.assertRaisesRegex(ValueError, "未来"):
            prepare_round(self.root, "第２回", self.observed_at, self.prices, [], bad)
        bad = copy.deepcopy(inherited)
        bad[0]["next_action"] = "将来結果で差し替えた規則。"
        with self.assertRaisesRegex(ValueError, "一致"):
            prepare_round(self.root, "第２回", self.observed_at, self.prices, [], bad)
        packet = prepare_round(self.root, "第２回", self.observed_at, self.prices, self.articles, inherited)
        self.assertEqual(packet["feedback"], inherited)
        self.assertEqual({row["horizon"] for row in packet["feedback"]}, {"today"})

    def test_flat_direction_stays_in_denominator_and_ranges_remain_fixed(self) -> None:
        """横ばいも発行分母へ残し、保留と未校正範囲を別採点する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        self._session("2026-10-15_Day", 48000)
        run = self._prepare()
        self._input(run, ("横ばい", "保留"))
        prediction = seal_prediction(run)
        baseline = read_json(run / "baseline.json")
        for symbol, item in prediction["instruments"].items():
            self.assertEqual(item["horizons"]["today"]["distribution"], baseline["instruments"][symbol]["horizons"]["today"]["distribution"])
        result = review_round(run, self.prices, self.articles, self.observed_at, {})
        self.assertEqual(len(result["labels"]), 2)
        self.assertEqual(result["labels"][0]["metrics"]["direction_hit"], True)
        self.assertIsNone(result["labels"][1]["metrics"]["direction_hit"])
        summary = summarize_replay(self.root)
        nk, topix = summary["groups"]["NK225／today"], summary["groups"]["TOPIX／today"]
        self.assertEqual((nk["forecasts"], nk["direction_n"], nk["direction_hits"], nk["range_n"]), (1, 1, 1, 1))
        self.assertEqual((topix["abstained"], topix["direction_n"], topix["range_n"]), (1, 0, 1))
        self.assertEqual(summary["groups"]["NK225／week"]["pending"], 1)
        for name, maximum in (("予測.md", 60), ("振り返り.md", 40)):
            self.assertLessEqual(len((run / name).read_text(encoding="utf-8").splitlines()), maximum)
        self.assertLess(len((self.root / "総括.md").read_text(encoding="utf-8").splitlines()), 100)
        self.assertIn("未校正", (run / "予測.md").read_text(encoding="utf-8"))
        self.assertIn("第１回/予測.md", (self.root / "総括.md").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            review_round(run, self.prices, [], self.observed_at, {})

    def test_range_override_and_existing_round_are_rejected(self) -> None:
        """予測担当の任意レンジ追加と保存先の再利用を拒否する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        with self.assertRaises(FileExistsError):
            self._prepare()
        self._input(run)
        value = read_json(run / "予測入力.json")
        value["instruments"]["NK225"]["range_pct"] = [-50, 50]
        write_json(run / "予測入力.json", value, exclusive=False)
        with self.assertRaisesRegex(ValueError, "項目"):
            seal_prediction(run)
        self.assertFalse((run / "予測.json").exists())

    def test_actor_evidence_is_fixed_and_early_reveal_is_rejected(self) -> None:
        """独立担当の証跡を固定し、終了前の実績開示を拒否する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        self._input(run)
        actor = {"agent_id": "検証担当", "fork_turns": "none", "allowed_input": "masked/packet.json",
                 "packet_hash": digest(read_json(run / "masked" / "packet.json")), "actual_started_at": "2026-10-16T00:00:00+09:00"}
        write_json(run / "actor.json", actor)
        prediction = seal_prediction(run)
        self.assertEqual(prediction["actor"], actor)
        with self.assertRaisesRegex(ValueError, "終了予定前"):
            review_round(run, self.prices, [], timestamp("2026-10-15T15:40:00+09:00"), {})
        self.assertFalse((run / "result.json").exists())
        actor["agent_id"] = "変更担当"
        write_json(run / "actor.json", actor, exclusive=False)
        with self.assertRaisesRegex(ValueError, "変更"):
            review_round(run, self.prices, [], self.observed_at, {})

    def test_prepared_but_unissued_round_stays_in_summary(self) -> None:
        """準備済みの未発行回を消さず、未評価や方向保留と区別する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        self._prepare()
        summary = summarize_replay(self.root)
        self.assertEqual((summary["prepared_rounds"], summary["forecast_rounds"]), (1, 0))
        self.assertEqual(summary["rounds"][0]["phase"], "prepare")
        self.assertIsNone(summary["rounds"][0]["prediction"])
        group = summary["groups"]["NK225／today"]
        self.assertEqual((group["prepared"], group["unissued"], group["forecasts"], group["pending"]), (1, 1, 0, 0))
        self.assertIn("予測未発行", (self.root / "総括.md").read_text(encoding="utf-8"))

    def test_long_tracking_requires_finished_reviews_and_reuses_existing_evaluation(self) -> None:
        """全回終了前の追跡を拒否し、同時刻評価を上書きせず再利用する。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        with self.assertRaisesRegex(ValueError, "全回の振り返り"):
            track_long_term(self.root, self.prices, self.observed_at, [])
        self.assertFalse((self.root / "長期追跡.json").exists())
        self._input(run)
        prediction = seal_prediction(run)
        review_round(run, self.prices, self.articles, self.observed_at, {})
        fixed = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
        with patch("fundamentals.replay.evaluate", side_effect=AssertionError("既存評価を再生成してはいけません。")):
            tracked = track_long_term(self.root, self.prices, self.observed_at, [])
        self.assertEqual(tracked["rounds"][0]["forecast_hash"], prediction["forecast_hash"])
        self.assertEqual(tracked["rounds"][0]["labels"], [])
        self.assertEqual(tracked["pending_counts"], {"week": 2, "month": 2, "quarter": 2})
        self.assertTrue(tracked["excluded_from_feedback"])
        self.assertTrue(all("未満了" in item["reason"] for item in tracked["rounds"][0]["pending"]))
        for path, original in fixed.items():
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            track_long_term(self.root, self.prices, self.observed_at, [])

    def test_long_tracking_scores_only_matured_long_directions_without_changing_feedback(self) -> None:
        """満了した週だけを当初方向で採点し、当日や引継ぎへ重複投入しない。

        引数: self (ReplayTests) はテスト。
        返り値: None。
        """
        run = self._prepare()
        self._input(run)
        entry = read_json(run / "予測入力.json")
        entry["instruments"]["NK225"]["long_term"]["week"]["direction"] = "上昇"
        entry["instruments"]["TOPIX"]["long_term"]["week"]["direction"] = "横ばい"
        write_json(run / "予測入力.json", entry, exclusive=False)
        prediction = seal_prediction(run)
        review_round(run, self.prices, self.articles, self.observed_at, {})
        original = {name: (run / name).read_bytes() for name in ("予測.json", "review.json", "audit.jsonl", "feedback.json")}
        self._session("2026-10-22_Day", 49000)
        cutoff = timestamp("2026-10-23T16:00:00+09:00")
        tracked = track_long_term(self.root, self.prices, cutoff, self.articles)
        record = tracked["rounds"][0]
        self.assertEqual({item["horizon"] for item in record["labels"]}, {"week"})
        self.assertEqual(len(record["labels"]), 2)
        self.assertEqual([item["metrics"]["direction_hit"] for item in record["labels"]], [True, False])
        self.assertEqual(record["input_hypothesis"]["TOPIX"]["week"], prediction["instruments"]["TOPIX"]["long_term"]["week"])
        self.assertEqual(tracked["pending_counts"], {"week": 0, "month": 2, "quarter": 2})
        self.assertEqual(tracked["groups"]["TOPIX／week"]["direction_n"], 1)
        self.assertTrue(all("today" not in key for key in tracked["groups"]))
        for name, body in original.items():
            self.assertEqual((run / name).read_bytes(), body)
        self.assertFalse((self.root / "summary.json").exists())
        document = (self.root / "長期追跡.md").read_text(encoding="utf-8")
        self.assertLess(len(document.splitlines()), 50)
        self.assertEqual(record["forecast_hash"], prediction["forecast_hash"])
        self.assertIn(cutoff.isoformat(), document)
        self.assertIn("追加材料と満了を確認。", record["input_hypothesis"]["NK225"]["week"]["pointer"])


if __name__ == "__main__":
    unittest.main()
