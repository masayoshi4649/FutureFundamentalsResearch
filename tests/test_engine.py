"""先物基準の騰落判定と、予測・評価間の情報境界を統合検証する。"""

import csv
import tempfile
import unittest
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import JST, read_json, timestamp, write_json
from fundamentals.engine import evaluate, forecast, history_rows, summarize


class EngineIntegrationTests(unittest.TestCase):
    """外部データを書き換えず、小さな合成市場の前向き処理を検証する。"""

    def setUp(self) -> None:
        """経験分布を作れる過去と、予測時点より後の価格を用意する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prices = self.root / "価格"
        self.runs = self.root / "予測"
        self.as_of = timestamp("2026-10-15T07:45:00+09:00")
        for day in range(1, 15):
            self._write_session(f"2026-10-{day:02d}_Day", 50000 + (day - 14) * 10)
            self._write_session(f"2026-10-{day:02d}_Night", 48000 + (day - 14) * 5)
        self._write_session("2026-10-15_Day", 48500)
        self._write_session("2026-10-15_Night", 49000)

    def _write_session(self, session: str, price: float) -> None:
        """両指数のセッション最終観測足を一時 CSV へ書き込む。

        引数:
            self (EngineIntegrationTests): このテストケース。
            session (str): 日付と日中・夜間の区分。
            price (float): NK225 の終点価格。TOPIX は同率になるよう縮尺する。
        返り値:
            None: なし。
        """
        day = datetime.strptime(session[:10], "%Y-%m-%d").date()
        if session.endswith("_Night"):
            started = datetime.combine(day + timedelta(days=1), time(5, 54), JST)
        else:
            started = datetime.combine(day, time(15, 39), JST)
        for symbol, scale in (("NK225", 1), ("TOPIX", 0.05)):
            directory = self.prices / "candle_1min" / symbol
            directory.mkdir(parents=True, exist_ok=True)
            row = {
                "time": started.isoformat(), "open": price * scale,
                "high": (price + 10) * scale, "low": (price - 10) * scale,
                "close": price * scale, "volume": 1, "volume_complete": "true", "closed": "true",
            }
            with (directory / f"{session}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

    def _article(self, later: bool = False) -> dict:
        """判断前または判断後のニュースを生成する。

        引数:
            self (EngineIntegrationTests): このテストケース。
            later (bool): 後発材料として生成する場合は真。
        返り値:
            dict: 改訂識別子と公表・取得時刻を持つ記事。
        """
        identity = "後発記事" if later else "当時記事"
        hour = "12" if later else "07"
        return {
            "id": identity, "revision_id": "初版",
            "canonical_url": "https://example.com/" + identity,
            "headline": "判断後の新材料" if later else "判断前に公表された材料",
            "published_at": f"2026-10-15T{hour}:00:00+09:00",
            "fetched_at": f"2026-10-15T{hour}:01:00+09:00",
        }

    def _event(self, later: bool = False) -> dict:
        """記事版を根拠とする、区別可能な分類の材料注釈を生成する。

        引数:
            self (EngineIntegrationTests): このテストケース。
            later (bool): 後発記事を根拠にする場合は真。
        返り値:
            dict: 両指数への方向点数を持つ注釈。
        """
        article = self._article(later)
        hour = "12" if later else "07"
        return {
            "event_id": article["id"] + "の材料", "story_key": article["id"],
            "known_at": article["published_at"], "annotated_at": f"2026-10-15T{hour}:02:00+09:00",
            "category": "後発分類" if later else "景況", "regime": "通常",
            "confidence": 1, "novelty": 1, "priced_in": 0, "half_life_days": 10,
            "impacts": {"NK225": 0.5, "TOPIX": 0.25},
            "evidence": [{"id": article["id"], "revision_id": article["revision_id"]}],
            "reason": "合成資料に基づく検証専用の仮説。",
        }

    # ----------------------------------------

    def test_day_rebound_uses_prior_night_future_instead_of_prior_day(self) -> None:
        """日中５万→夜間４万８千→日中４万８千５百を上昇として評価する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "日中反発"
        prediction = forecast(self.prices, [], [], self.as_of, run)
        self.assertEqual(prediction["instruments"]["NK225"]["reference"]["price"], 48000)
        self.assertEqual(prediction["instruments"]["NK225"]["reference"]["session"], "2026-10-14_Night")
        result = evaluate(run, self.prices, timestamp("2026-10-15T16:00:00+09:00"))
        self.assertEqual(len(result["labels"]), 2)
        for label in result["labels"]:
            self.assertEqual(label["horizon"], "today")
            self.assertAlmostEqual(label["actual_pct"], 1.0416666666666667)
            self.assertEqual(label["metrics"]["actual_direction"], "上昇")
            self.assertFalse(label["target_is_official_close"])

    def test_night_forecast_uses_same_day_future_close(self) -> None:
        """１６時の夜間予測は当日日中の先物終値を基準とする。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "夜間"
        prediction = forecast(self.prices, [], [], timestamp("2026-10-15T16:00:00+09:00"), run)
        reference = prediction["instruments"]["NK225"]["reference"]
        self.assertEqual(reference["price"], 48500)
        self.assertEqual(reference["session"], "2026-10-15_Day")
        result = evaluate(run, self.prices, timestamp("2026-10-16T07:00:00+09:00"))
        for label in result["labels"]:
            self.assertAlmostEqual(label["actual_pct"], (49000 / 48500 - 1) * 100)
            self.assertEqual(label["metrics"]["actual_direction"], "上昇")

    def test_future_news_and_price_changes_do_not_enter_forecast_inputs(self) -> None:
        """未来記事を材料へ入れず、将来価格の変更で入力や経験分布が変わらない。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        articles = [self._article(), self._article(later=True)]
        events = [self._event(), self._event(later=True)]
        first = forecast(self.prices, articles, events, self.as_of, self.runs / "初回")
        self._write_session("2026-10-15_Day", 900000)
        self._write_session("2026-10-15_Night", 800000)
        second = forecast(self.prices, articles, events, self.as_of, self.runs / "将来価格改訂後")
        self.assertEqual(first["snapshot_hash"], second["snapshot_hash"])
        self.assertEqual(first["instruments"], second["instruments"])
        for instrument in first["instruments"].values():
            current = instrument["horizons"]["today"]
            self.assertGreater(current["distribution"]["n"], 8)
            self.assertLessEqual(timestamp(current["distribution"]["last_label_end"]), self.as_of)
            self.assertEqual(set(current["features"]), {"景況／通常"})
            self.assertEqual([item["event_id"] for item in current["materials"]], ["当時記事の材料"])
            self.assertEqual(current["excluded_materials"][0]["event_id"], "後発記事の材料")

    def test_modified_historical_price_is_rejected_at_evaluation(self) -> None:
        """予測後に入力済み価格が変わった場合は別版の評価を拒否する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "価格改訂"
        forecast(self.prices, [], [], self.as_of, run)
        self._write_session("2026-10-14_Night", 48001)
        with self.assertRaisesRegex(ValueError, "予測入力価格が変更"):
            evaluate(run, self.prices, timestamp("2026-10-15T16:00:00+09:00"))

    def test_live_annotation_may_finish_after_cutoff_but_news_may_not(self) -> None:
        """運用の注釈処理時間を許容し、情報打切後のニュースは除外する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        issued = "2026-10-15T07:46:00+09:00"
        article = self._article()
        event = self._event()
        event["annotated_at"] = issued
        later_article = self._article(later=True)
        later_article.update(
            published_at="2026-10-15T07:45:01+09:00", fetched_at="2026-10-15T07:45:02+09:00",
        )
        later_event = self._event(later=True)
        later_event.update(known_at=later_article["published_at"], annotated_at=issued)
        with patch("fundamentals.engine.now", return_value=issued):
            prediction = forecast(
                self.prices, [article, later_article], [event, later_event], self.as_of,
                self.runs / "運用時刻", mode="運用",
            )
        self.assertEqual(timestamp(prediction["issued_at"]), timestamp(issued))
        for instrument in prediction["instruments"].values():
            current = instrument["horizons"]["today"]
            self.assertEqual(set(current["features"]), {"景況／通常"})
            self.assertFalse(current["materials"][0]["annotation_retrospective"])
            self.assertEqual(current["excluded_materials"][0]["event_id"], "後発記事の材料")

    def test_live_forecast_cannot_use_a_future_information_cutoff(self) -> None:
        """現在より先の情報打切を指定した運用予測を拒否する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        with patch("fundamentals.engine.now", return_value="2026-10-15T07:44:00+09:00"):
            with self.assertRaises(ValueError):
                forecast(self.prices, [], [], self.as_of, self.runs / "未来情報打切", mode="運用")

    def test_unmatured_horizons_remain_pending(self) -> None:
        """対象終点到来前は全期間、日中終了後も長期は評価保留にする。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "未満了"
        forecast(self.prices, [], [], self.as_of, run)
        before = evaluate(run, self.prices, timestamp("2026-10-15T15:39:00+09:00"))
        self.assertEqual(before["labels"], [])
        self.assertEqual(len(before["pending"]), 8)
        after = evaluate(run, self.prices, timestamp("2026-10-15T16:00:00+09:00"))
        self.assertEqual({label["horizon"] for label in after["labels"]}, {"today"})
        self.assertEqual(len(after["pending"]), 6)
        self.assertEqual({item["horizon"] for item in after["pending"]}, {"week", "month", "quarter"})

    def test_tampered_saved_forecast_is_rejected(self) -> None:
        """保存後に予測の内容だけが改変された場合は評価を拒否する。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "予測改変"
        forecast(self.prices, [], [], self.as_of, run)
        value = read_json(run / "forecast.json")
        value["instruments"]["NK225"]["reference"]["price"] = 1
        write_json(run / "forecast.json", value, exclusive=False)
        with self.assertRaisesRegex(ValueError, "ハッシュ"):
            evaluate(run, self.prices, timestamp("2026-10-15T16:00:00+09:00"))

    def test_unknown_later_material_does_not_exempt_failure_metrics(self) -> None:
        """後発材料が見つかっても誤方向・区間外れを成績と集計へ残す。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        run = self.runs / "後発材料"
        model = {"prediction_pct": -1.0, "status": "探索的推定", "n": 12, "contributions_pct": {}}
        with patch("fundamentals.engine.fit_predict", return_value=model):
            forecast(self.prices, [], [], self.as_of, run)
        without = evaluate(run, self.prices, timestamp("2026-10-15T16:00:00+09:00"))
        with_later = evaluate(
            run, self.prices, timestamp("2026-10-15T16:01:00+09:00"), [self._article(later=True)],
        )
        for baseline, updated in zip(without["labels"], with_later["labels"]):
            self.assertEqual(baseline["metrics"], updated["metrics"])
            self.assertIs(updated["metrics"]["direction_hit"], False)
            self.assertIs(updated["metrics"]["terminal_covered"], False)
            self.assertGreater(updated["metrics"]["absolute_error_pct"], 2)
            self.assertEqual(len(updated["later_material_candidates"]), 1)
        summary = summarize(self.runs)
        group = summary["groups"]["回想／NK225／Day／today"]
        self.assertEqual(group["forecasts"], 1)
        self.assertEqual(group["evaluated"], 1)
        self.assertEqual(group["candidate_direction_n"], 1)
        self.assertEqual(group["candidate_direction_hit_rate"], 0)

    def test_only_matured_and_already_observed_history_trains_next_forecast(self) -> None:
        """未満了ラベルと未来に行う評価を次回予測の学習履歴へ入れない。

        引数:
            self (EngineIntegrationTests): このテストケース。
        返り値:
            None: なし。
        """
        eligible_run = self.runs / "利用可能履歴"
        eligible = forecast(self.prices, [], [], timestamp("2026-10-07T07:45:00+09:00"), eligible_run)
        evaluate(eligible_run, self.prices, timestamp("2026-10-07T16:00:00+09:00"))
        future_run = self.runs / "後日評価履歴"
        future = forecast(self.prices, [], [], timestamp("2026-10-08T07:45:00+09:00"), future_run)
        evaluate(future_run, self.prices, timestamp("2026-10-16T07:00:00+09:00"))
        cutoff = timestamp("2026-10-09T07:45:00+09:00")
        rows = history_rows(self.runs, cutoff, "回想")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["forecast_hash"] for row in rows}, {eligible["forecast_hash"]})
        self.assertEqual({row["horizon"] for row in rows}, {"today"})
        self.assertTrue(all(timestamp(row["label_end"]) <= cutoff for row in rows))
        prediction = forecast(self.prices, [], [], cutoff, self.root / "次回予測", history=self.runs)
        for instrument in prediction["instruments"].values():
            self.assertEqual(instrument["horizons"]["today"]["training_forecast_hashes"], [eligible["forecast_hash"]])
            self.assertNotIn(future["forecast_hash"], instrument["horizons"]["today"]["training_forecast_hashes"])
            for horizon in ("week", "month", "quarter"):
                self.assertEqual(instrument["horizons"][horizon]["training_forecast_hashes"], [])
