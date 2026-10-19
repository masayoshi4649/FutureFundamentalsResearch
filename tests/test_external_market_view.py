"""外部観測ビューの情報締切を合成入力だけで検査する。"""

from copy import deepcopy
import json
import math
import unittest

from research_support.external_market_view import InstrumentSpec, build_external_view


SPEC = {"合成": InstrumentSpec("合成系列", "合成単位", "合成観測値の増加", "試験専用の架空系列")}


def row(clock: str, close: float = 100) -> dict:
    """
    機能: 検証用の架空の足を作る。
    引数: clock (str) は足始端、close (float) は終値。
    返り値: dict 型の架空OHLC行。
    """
    return {"Datetime": clock, "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1}


def response(rows: list[dict]) -> dict:
    """
    機能: 実データへ接続せずに架空の構造化応答を作る。
    引数: rows (list[dict]) は架空の足。
    返り値: dict 型のticker別応答。
    """
    return {"合成": {"version": "1", "provider": "合成", "dataset": "合成", "collected_at": "2026-10-17T19:00:00+09:00", "metadata": {"library": "合成", "unofficial_client": True, "future_range": "表示禁止の範囲"}, "data": rows}}


def view(data: dict, **kwargs) -> dict:
    """
    機能: 共通の架空アンカーと仕様で観測ビューを作る。
    引数: data (dict) は架空応答、kwargs (dict) は上書きする検証条件。
    返り値: dict 型の外部観測ビュー。
    """
    options = {"as_of": "2026-10-17T12:00:00+09:00", "anchors": {"開始": "2026-10-17T11:00:00+09:00", "終了": "2026-10-17T12:00:00+09:00"}, "instrument_specs": SPEC}
    options.update(kwargs)
    return build_external_view(data, **options)


class ExternalMarketViewTests(unittest.TestCase):
    """実時計、実ニュース、実価格を使わない境界検査。"""

    def test_future_rows_are_invisible_except_digest(self):
        """
        機能: 未来行の追加でダイジェスト以外の表示が変わらないことを検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        baseline = response([row("2026-10-17T10:25:00+09:00"), row("2026-10-17T11:25:00+09:00", 110)])
        expanded = deepcopy(baseline)
        expanded["合成"]["data"].extend([row("2026-10-17T11:30:00+09:00", 999991), row("2026-10-19T10:00:00+09:00", 999992)])
        before, after = view(baseline), view(expanded)
        self.assertNotEqual(before.pop("raw_responses_sha256"), after.pop("raw_responses_sha256"))
        self.assertNotEqual(before["instruments"]["合成"].pop("raw_response_sha256"), after["instruments"]["合成"].pop("raw_response_sha256"))
        self.assertEqual(before, after)
        serialized = json.dumps(after, ensure_ascii=False)
        self.assertNotIn("99999", serialized)
        self.assertNotIn("2026-10-19", serialized)
        self.assertNotIn("表示禁止の範囲", serialized)
        self.assertEqual(after["instruments"]["合成"]["comparisons"][0]["relative_change_pct"], 10)

    def test_bar_end_and_lag_boundary(self):
        """
        機能: 足終端と固定遅延の境界を秒単位で検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        data = response([row("2026-10-17T11:25:00+09:00"), row("2026-10-17T11:25:01+09:00", 120), row("2026-10-17T11:59:00+09:00", 130)])
        selected = view(data)["instruments"]["合成"]["observations"]["終了"]
        self.assertEqual(selected["bar"]["close"], 100)
        self.assertEqual(selected["bar"]["assumed_visible_at"], "2026-10-17T03:00:00+00:00")
        self.assertEqual(selected["anchor_at"], "2026-10-17T03:00:00+00:00")
        self.assertEqual(selected["age_minutes"], 30)

    def test_stale_and_missing_do_not_create_returns(self):
        """
        機能: 鮮度上限の等号と超過、および欠測時の計算抑止を検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        exact = view(response([row("2026-10-17T10:55:00+09:00")]))
        self.assertEqual(exact["instruments"]["合成"]["observations"]["終了"]["status"], "利用可")
        stale = view(response([row("2026-10-17T10:54:59+09:00")]))
        self.assertEqual(stale["instruments"]["合成"]["observations"]["終了"]["status"], "鮮度超過")
        self.assertIsNone(stale["instruments"]["合成"]["comparisons"][0]["relative_change_pct"])
        missing = view(response([]))
        self.assertIsNone(missing["instruments"]["合成"]["observations"]["終了"]["bar"])

    def test_empty_ohlc_is_not_zero_or_forward_filled(self):
        """
        機能: 全OHLCがnullの行を観測なしとして扱い、ゼロ変換や補完をしないことを検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        empty = row("2026-10-17T11:25:00+09:00")
        empty.update({key: None for key in ("Open", "High", "Low", "Close")})
        self.assertEqual(view(response([empty]))["instruments"]["合成"]["observations"]["終了"]["status"], "欠測")
        result = view(response([row("2026-10-17T10:00:00+09:00"), empty]))
        observation = result["instruments"]["合成"]["observations"]["終了"]
        self.assertEqual(observation["status"], "鮮度超過")
        self.assertEqual(observation["bar"]["close"], 100)
        self.assertIsNone(result["instruments"]["合成"]["comparisons"][0]["relative_change_pct"])

    def test_timezones_duplicates_and_input_immutability(self):
        """
        機能: 異なるUTCオフセットの同一足をまとめ、元応答を変更しないことを検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        data = response([row("2026-10-17T11:25:00+09:00"), row("2026-10-16T22:25:00-04:00")])
        original = deepcopy(data)
        result = view(data)
        self.assertEqual(data, original)
        self.assertEqual(result["instruments"]["合成"]["observations"]["終了"]["bar"]["close"], 100)
        wrapped = {"合成": {"structuredContent": data["合成"]}}
        self.assertEqual(view(wrapped)["instruments"]["合成"]["source"], result["instruments"]["合成"]["source"])
        contradictory = deepcopy(data)
        contradictory["合成"]["data"][1]["Volume"] = 2
        with self.assertRaises(ValueError):
            view(contradictory)

    def test_bad_data_and_configuration_are_rejected(self):
        """
        機能: 不正値、未知列、部分欠測、時計と構成の違反を拒否することを検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        cases = [{"Datetime": "2026-10-17T11:25:00"}, {"Close": math.nan}, {"High": math.inf}, {"Close": None}, {"Close": 0}, {"Close": -1}, {"Close": True}, {"High": 90}, {"Volume": -1}, {"未知列": 100}]
        for change in cases:
            with self.subTest(change=change):
                changed = row("2026-10-17T11:25:00+09:00")
                changed.update(change)
                with self.assertRaises(ValueError):
                    view(response([changed]))
        for options in ({"display_lag_minutes": -1}, {"interval_minutes": 0}, {"max_staleness_minutes": -1}, {"display_lag_minutes": True}, {"as_of": "2026-10-17T12:00:00"}, {"anchors": {"未来": "2026-10-17T12:00:01+09:00"}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                view(response([]), **options)

    def test_equal_value_and_yield_unit_are_explicit(self):
        """
        機能: 同値を明示し、利回りの単純差と相対変化を元単位で返すことを検査する。
        引数: self (ExternalMarketViewTests) は検査インスタンス。
        返り値: None 型。
        """
        same = view(response([row("2026-10-17T10:25:00+09:00"), row("2026-10-17T11:25:00+09:00")]))
        self.assertEqual(same["instruments"]["合成"]["comparisons"][0]["status"], "同値")
        yield_spec = {"合成": InstrumentSpec("合成利回り", "元応答の利回り表示単位", "利回り観測値の増加", "株価ではない。bp換算しない。")}
        result = view(response([row("2026-10-17T10:25:00+09:00", 4), row("2026-10-17T11:25:00+09:00", 5)]), instrument_specs=yield_spec)
        comparison = result["instruments"]["合成"]["comparisons"][0]
        self.assertEqual(comparison["absolute_change"], 1)
        self.assertEqual(comparison["relative_change_pct"], 25)
        self.assertEqual(comparison["unit"], "元応答の利回り表示単位")


if __name__ == "__main__":
    unittest.main()
