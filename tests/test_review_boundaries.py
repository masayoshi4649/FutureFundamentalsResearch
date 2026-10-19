"""独立レビューで見つかった発行・価格基準・採点の境界を検証する。"""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import test_engine
from fundamentals.common import timestamp
from fundamentals.engine import endpoint, evaluate, forecast, reference_quote, session_plan, summarize
from fundamentals.model import material_features
from fundamentals.news import snapshot_payload
from fundamentals.prices import load_candles


class ReviewBoundaryTests(unittest.TestCase):
    """一時的な合成データだけを使い、研究原本へ書き込まずに検証する。"""

    def setUp(self) -> None:
        """既存の合成市場を準備し、終了時の削除を登録する。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        self.market = test_engine.EngineIntegrationTests()
        self.market.setUp()
        self.addCleanup(self.market.doCleanups)

    def _late_forecast(self, initial: str, completed: str, out: Path) -> None:
        """経験分布の計算中に時計が進む運用予測を実行する。

        引数:
            self (ReviewBoundaryTests): このテストケース。
            initial (str): 計算開始時刻と情報打切。
            completed (str): 計算開始後に観測される実時刻。
            out (Path): 新しい一時保存先。
        返り値:
            None: 発行期限を過ぎた場合は予測関数の例外をそのまま伝える。
        """
        calculated = []

        def distribution(*args: object, **kwargs: object) -> dict:
            """価格分布の計算が始まった状態を作る。

            引数:
                args (tuple[object, ...]): 分布関数へ渡された位置引数。
                kwargs (dict[str, object]): 分布関数へ渡された名前付き引数。
            返り値:
                dict: レンジ未算出の合成分布。
            """
            calculated.append(True)
            return {"n": 0, "terminal_range_pct": None, "path_range_pct": None}

        def clock() -> str:
            """分布計算の前後で異なる時刻を返す。

            引数:
                なし。
            返り値:
                str: タイムゾーン付きの合成実時刻。
            """
            return completed if calculated else initial

        with patch("fundamentals.engine.now", side_effect=clock), patch(
            "fundamentals.engine.empirical_distribution", side_effect=distribution
        ):
            forecast(self.market.prices, [], [], timestamp(initial), out, mode="運用")

    # ----------------------------------------
    # 発行完了時刻と前セッション基準

    def test_completed_forecast_after_five_minutes_is_not_issued(self) -> None:
        """計算開始時だけ期限内でも完了までに５分を超えた予測を拒否する。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        out = self.market.runs / "計算中の５分超過"
        with self.assertRaises(ValueError):
            self._late_forecast("2026-10-15T07:45:00+09:00", "2026-10-15T07:51:00+09:00", out)
        self.assertFalse((out / "forecast.json").exists())

    def test_completed_forecast_after_session_start_is_not_issued(self) -> None:
        """計算中に対象セッションが始まった予測を開始前として発行しない。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        out = self.market.runs / "計算中の立会開始"
        with self.assertRaises(ValueError):
            self._late_forecast("2026-10-15T08:44:00+09:00", "2026-10-15T08:46:00+09:00", out)
        self.assertFalse((out / "forecast.json").exists())

    def test_missing_prior_night_does_not_select_an_older_night(self) -> None:
        """直前ナイト欠損時に、その前の日中も跨いだ古いナイトを使わない。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        temporary_csv = self.market.prices / "candle_1min" / "NK225" / "2026-10-14_Night.csv"
        temporary_csv.unlink()
        candles = load_candles(self.market.prices, "NK225", cutoff=self.market.as_of)
        with self.assertRaises(ValueError):
            reference_quote(candles, self.market.as_of, session_plan(self.market.as_of))

    def test_night_forecast_requires_the_same_day_day_reference(self) -> None:
        """夜間向けの当日日中欠損を前営業日の日中終値で置き換えない。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        temporary_csv = self.market.prices / "candle_1min" / "NK225" / "2026-10-15_Day.csv"
        temporary_csv.unlink()
        as_of = timestamp("2026-10-15T16:00:00+09:00")
        candles = load_candles(self.market.prices, "NK225", cutoff=as_of)
        with self.assertRaises(ValueError):
            reference_quote(candles, as_of, session_plan(as_of))

    def test_long_endpoint_does_not_skip_an_observed_but_incomplete_session(self) -> None:
        """最初の対象セッションの末尾欠測を翌日の正常終点へすり替えない。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        candles = load_candles(self.market.prices, "NK225")
        incomplete = [
            replace(bar, time=timestamp("2026-10-14T15:00:00+09:00"),
                    end=timestamp("2026-10-14T15:01:00+09:00"))
            if bar.session == "2026-10-14_Day" else bar
            for bar in candles
        ]
        result = endpoint(incomplete, timestamp("2026-10-14T15:45:00+09:00"), "Day",
                          timestamp("2026-10-15T16:00:00+09:00"))
        self.assertIsNone(result)

    # ----------------------------------------
    # 候補方向の分母と材料注釈の更新

    def test_direction_candidate_followed_by_flat_counts_as_a_miss(self) -> None:
        """候補上昇の後に横ばいとなった満了例も候補一致率の分母へ残す。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        self.market._write_session("2026-10-15_Day", 48000)
        model = {"prediction_pct": 1.0, "status": "探索的推定", "n": 12, "contributions_pct": {}}
        run = self.market.runs / "候補上昇から横ばい"
        with patch("fundamentals.engine.fit_predict", return_value=model):
            forecast(self.market.prices, [], [], self.market.as_of, run)
        evaluation = evaluate(run, self.market.prices, timestamp("2026-10-15T16:00:00+09:00"))
        for label in evaluation["labels"]:
            self.assertEqual(label["metrics"]["actual_direction"], "横ばい")
            self.assertIs(label["metrics"]["direction_hit"], False)
        group = summarize(self.market.runs)["groups"]["回想／NK225／Day／today"]
        self.assertEqual(group["candidate_direction_n"], 1)
        self.assertEqual(group["candidate_direction_hit_rate"], 0)
        self.assertEqual(group["abstained"], 1)

    def test_latest_eligible_annotation_wins_over_identifier_sort(self) -> None:
        """同じ既知時刻なら発行までに完成した最新注釈をID順に依存せず採用する。

        引数:
            self (ReviewBoundaryTests): このテストケース。
        返り値:
            None: なし。
        """
        old = self.market._event()
        old.update(event_id="z-旧注釈", annotated_at="2026-10-15T07:02:00+09:00")
        newer = dict(old, event_id="a-更新注釈", annotated_at="2026-10-15T07:45:30+09:00",
                     impacts={"NK225": -0.5, "TOPIX": -0.25})
        too_late = dict(newer, event_id="zz-未完成注釈", annotated_at="2026-10-15T07:47:00+09:00")
        articles = snapshot_payload([self.market._article()], self.market.as_of)["articles"]
        _, details, excluded = material_features(
            [old, newer, too_late], articles, self.market.as_of,
            timestamp("2026-10-15T15:45:00+09:00"), "NK225", "運用",
            timestamp("2026-10-15T07:46:00+09:00"),
        )
        self.assertEqual([row["event_id"] for row in details], ["a-更新注釈"])
        self.assertTrue(any(row["event_id"] == "zz-未完成注釈" for row in excluded))
