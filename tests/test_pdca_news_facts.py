"""ＮＹ引け数値の抽出と、締切・転載・欠測の扱いを検査する。"""

import copy
import unittest
from datetime import datetime

from fundamentals.pdca_news_facts import extract_market_facts


def article(article_id: int, body: str, headline: str = "NY株式：引けの報告",
            published: str = "2026-09-29T20:45:00Z", fetched: str = "2026-09-29T20:50:00Z",
            **fields) -> dict:
    """試験用の記事辞書を生成する。

    引数: article_id (int) は識別子、body/headline/published/fetched (str) は
        本文・見出し・公表・取得日時、fields (dict) は追加項目。
    返り値: dict は試験用のニュース原本。
    """
    return {"id": article_id, "revision_id": "初版", "body_text": body, "headline": headline,
            "published_at": published, "fetched_at": fetched, **fields}


class TestMarketFacts(unittest.TestCase):
    """機能と時点境界を、価格実績に触れずに検査する。"""

    cutoff = "2026-10-01T07:45:00+09:00"
    stocks = "ダウ平均は131.59ドル安の51349.92ドル、ナスダックは22.84ポイント安の26797.54で取引を終了した。"
    oil = "9月29日のNY原油先物11月限は続落。WTI先物11月限は、前営業日比－3.74ドル（－4.04％）の88.86ドルで通常取引を終了した。"
    fx = "29日のニューヨーク外為市場でドル・円は156円98銭から157円70銭まで上昇し、157円38銭で引けた。"

    def test_signed_stock_points_and_correct_denominator(self):
        """機能: 符号付き株価前日比を正しい前日終値で百分率へ変換する。

        引数: なし。返り値: None は検査完了。
        """
        value = extract_market_facts([article(1, self.stocks)], self.cutoff)
        dow, nasdaq = value["series"]["dow"], value["series"]["nasdaq"]
        self.assertEqual(dow["close"], 51349.92)
        self.assertEqual(dow["change_points"], -131.59)
        self.assertAlmostEqual(dow["change_pct"], -131.59 / (51349.92 + 131.59) * 100)
        self.assertEqual(nasdaq["change_points"], -22.84)
        self.assertIsNone(dow["market_date"])
        self.assertIsNone(dow["observed_at"])
        self.assertEqual(dow["causality"], "未識別")
        self.assertEqual(dow["evidence"][0]["excerpt"], self.stocks)

    def test_digest_binds_identical_observation_without_double_count(self):
        """機能: 同じ引けの単独記事とダイジェストを一観測へ統合する。

        引数: なし。返り値: None は検査完了。
        """
        rows = [article(1, self.stocks), article(2, self.stocks + self.fx + self.oil,
                headline="29日の米国市場ダイジェスト：結果",
                published="2026-09-29T22:00:00Z", fetched="2026-09-29T22:05:00Z")]
        before = copy.deepcopy(rows)
        value = extract_market_facts(rows, self.cutoff)
        self.assertEqual(rows, before)
        self.assertEqual(value["coverage"]["unique_observations"], 4)
        self.assertEqual(value["coverage"]["duplicate_observations"], 2)
        self.assertEqual(len(value["observations"]["dow"]), 1)
        self.assertEqual(value["series"]["dow"]["source_ids"], [1, 2])
        self.assertEqual(value["series"]["dow"]["market_date"], "2026-09-29")
        self.assertEqual(value["series"]["dow"]["available_at"], "2026-09-30T05:50:00+09:00")
        self.assertEqual(value["series"]["dow"]["latest_evidence_at"], "2026-09-30T07:05:00+09:00")
        self.assertEqual(value["series"]["usdjpy"]["close"], 157.38)
        self.assertIsNone(value["series"]["usdjpy"]["change_pct"])

    def test_future_clock_is_rejected_before_body_access(self):
        """機能: 公表・取得・更新の後発記事を本文参照より前に除外する。

        引数: なし。返り値: None は検査完了。
        """
        class Guarded(dict):
            """未来記事の本文へ触れた場合に失敗する辞書。"""

            def get(self, key, default=None):
                """本文・見出しの先読みを拒否する。

                引数: key (str) は項目、default (object) は既定値。
                返り値: object は本文以外の保存値。
                """
                if key in {"body_text", "headline"}:
                    raise AssertionError("未来本文へアクセスしました。")
                return super().get(key, default)

        rows = []
        for field in ("published_at", "fetched_at", "updated_at"):
            row = article(len(rows), "読んではいけない本文")
            row[field] = "2026-10-02T00:00:00Z"
            rows.append(Guarded(row))
        value = extract_market_facts(rows, self.cutoff)
        self.assertEqual(value["coverage"]["eligible_articles"], 0)
        self.assertEqual([row["code"] for row in value["excluded"]], ["future_timestamp"] * 3)
        self.assertTrue(all(value is None for value in value["series"].values()))

    def test_cutoff_equal_is_allowed_and_missing_clocks_are_rejected(self):
        """機能: 締切と等しい取得は認め、公表欠測や時差なしを除外する。

        引数: なし。返り値: None は検査完了。
        """
        rows = [article(1, self.stocks, fetched=self.cutoff),
                article(2, self.stocks, published=None),
                article(3, self.stocks, fetched="2026-09-29T20:50:00")]
        value = extract_market_facts(rows, datetime.fromisoformat(self.cutoff))
        self.assertEqual(value["coverage"]["eligible_articles"], 1)
        self.assertEqual(value["coverage"]["excluded_articles"], 2)
        self.assertEqual(value["series"]["dow"]["source_ids"], [1])
        with self.assertRaises(ValueError):
            extract_market_facts([], "2026-10-01T07:45:00")

    def test_fx_range_and_other_currency_are_not_close(self):
        """機能: 為替範囲の末尾やユーロ円からドル円引けを推測しない。

        引数: なし。返り値: None は検査完了。
        """
        body = "ドル・円は157円17銭から157円53銭まで上昇し、引けた。ユーロ・円は180円50銭で引けた。"
        value = extract_market_facts([article(1, body, headline="NY為替：引け")], self.cutoff)
        self.assertIsNone(value["series"]["usdjpy"])
        tokyo = extract_market_facts([article(2, self.fx, headline="東京為替：過去の説明")], self.cutoff)
        self.assertIsNone(tokyo["series"]["usdjpy"])

    def test_oil_keeps_reported_percentage_and_contract(self):
        """機能: ＷＴＩの値・限月・記事百分率を別々に保持する。

        引数: なし。返り値: None は検査完了。
        """
        value = extract_market_facts([article(1, self.oil, headline="NY原油：引け")], self.cutoff)
        fact = value["series"]["wti"]
        self.assertEqual(fact["market_date"], "2026-09-29")
        self.assertEqual(fact["contract"], "11月限")
        self.assertEqual(fact["reported_change_pct"], -4.04)
        self.assertAlmostEqual(fact["change_pct"], -3.74 / 92.60 * 100)
        inconsistent = self.oil.replace("－4.04％", "＋4.04％")
        value = extract_market_facts([article(2, inconsistent, headline="NY原油：引け")], self.cutoff)
        self.assertIn("reported_change_pct_mismatch", value["series"]["wti"]["flags"])

    def test_previous_close_inconsistency_is_flagged_without_repair(self):
        """機能: 同限月で一暦日差の前日値不整合を保持し補正しない。

        引数: なし。返り値: None は検査完了。
        """
        next_oil = "9月30日のNY原油先物11月限は反発。WTI先物11月限は、前営業日比＋1.16ドル（＋1.30％）の90.54ドルで通常取引を終了した。"
        value = extract_market_facts([
            article(1, self.oil, headline="NY原油：引け"),
            article(2, next_oil, headline="NY原油：引け", published="2026-09-30T22:00:00Z", fetched="2026-09-30T22:05:00Z"),
        ], self.cutoff)
        fact = value["series"]["wti"]
        self.assertEqual(fact["close"], 90.54)
        self.assertEqual(fact["change_points"], 1.16)
        self.assertIn("previous_observed_close_mismatch", fact["flags"])
        self.assertAlmostEqual(fact["previous_observation_comparison"]["difference"], 0.52)

    def test_latest_series_uses_available_article_without_guessing_market_date(self):
        """機能: 最新観測を記事の利用可能時計で選び市場日を捏造しない。

        引数: なし。返り値: None は検査完了。
        """
        newer = "ダウ平均は20.51ドル高の50926.56ドル、ナスダックは10.54ポイント高の26871.60で取引を終了した。"
        rows = [article(1, self.stocks), article(2, newer, published="2026-09-30T20:45:00Z", fetched="2026-09-30T20:50:00Z")]
        value = extract_market_facts(rows, self.cutoff)
        self.assertEqual(value["series"]["dow"]["close"], 50926.56)
        self.assertIsNone(value["series"]["dow"]["market_date"])
        self.assertEqual(len(value["observations"]["dow"]), 2)
        self.assertEqual(value, extract_market_facts(list(reversed(rows)), self.cutoff))

    def test_future_digest_cannot_assign_market_date_to_earlier_article(self):
        """機能: 後発ダイジェストの明示日を過去の記事へ混ぜない。

        引数: なし。返り値: None は検査完了。
        """
        rows = [article(1, self.stocks), article(2, self.stocks + self.oil,
                headline="29日の米国市場ダイジェスト：結果", fetched="2026-10-02T00:00:00Z")]
        value = extract_market_facts(rows, self.cutoff)
        self.assertIsNone(value["series"]["dow"]["market_date"])
        self.assertEqual(value["series"]["dow"]["source_ids"], [1])

    def test_conflicting_same_day_closes_are_preserved_and_not_chosen(self):
        """機能: 同日の競合値を原本通り残し最新系列を曖昧として保留する。

        引数: なし。返り値: None は検査完了。
        """
        rows = [article(1, self.oil, headline="NY原油：引け"),
                article(2, self.oil.replace("88.86", "89.86"), headline="NY原油：引け", fetched="2026-09-29T21:00:00Z")]
        value = extract_market_facts(rows, self.cutoff)
        self.assertIsNone(value["series"]["wti"])
        self.assertEqual(len(value["observations"]["wti"]), 2)
        self.assertTrue(all("same_market_date_conflict" in item["flags"] for item in value["observations"]["wti"]))

    def test_latest_revision_only_and_impossible_signed_word_rejected(self):
        """機能: 同記事旧版を二重算入せず符号と高安の矛盾を抽出しない。

        引数: なし。返り値: None は検査完了。
        """
        newer = article(1, self.stocks.replace("131.59ドル安", "+131.59ドル安"),
                        revision_id="改訂", updated_at="2026-09-30T00:00:00Z")
        value = extract_market_facts([article(1, self.stocks), newer], self.cutoff)
        self.assertIsNone(value["series"]["dow"])
        self.assertIsNotNone(value["series"]["nasdaq"])
        self.assertEqual(value["coverage"]["eligible_articles"], 1)
        self.assertEqual(value["excluded"][0]["code"], "duplicate_or_superseded_article")
        self.assertEqual(value["flags"][0]["code"], "ambiguous_numeric_value")

    def test_forecast_wording_and_intraday_quotes_are_not_actual_close(self):
        """機能: 予想された終了値や取引中の価格を引け実績にしない。

        引数: なし。返り値: None は検査完了。
        """
        rows = [article(1, self.stocks.replace("終了した", "終了する見通しだ")),
                article(2, self.oil.replace("終了した", "終了すると予想される"), headline="NY原油：予想")]
        value = extract_market_facts(rows, self.cutoff)
        self.assertTrue(all(fact is None for fact in value["series"].values()))

    def test_conflicting_values_inside_one_article_are_ambiguous(self):
        """機能: 記事内の異なる引け値を恣意的に一つ選ばない。

        引数: なし。返り値: None は検査完了。
        """
        body = self.stocks + self.stocks.replace("51349.92", "51400.00")
        value = extract_market_facts([article(1, body)], self.cutoff)
        self.assertIsNone(value["series"]["dow"])
        self.assertIsNotNone(value["series"]["nasdaq"])
        self.assertIn("ambiguous_article_values", [flag["code"] for flag in value["flags"]])

    def test_nonadjacent_dates_do_not_create_previous_close_comparison(self):
        """機能: 休日や欠測営業日の前日値を推定しない。

        引数: なし。返り値: None は検査完了。
        """
        later = self.oil.replace("9月29日", "10月1日").replace("88.86", "95.00")
        rows = [article(1, self.oil, headline="NY原油：引け"),
                article(2, later, headline="NY原油：引け", published="2026-10-01T22:00:00Z", fetched="2026-10-01T22:05:00Z")]
        value = extract_market_facts(rows, "2026-10-02T07:45:00+09:00")
        self.assertNotIn("previous_observation_comparison", value["series"]["wti"])
        self.assertNotIn("previous_observed_close_mismatch", value["series"]["wti"]["flags"])


if __name__ == "__main__":
    unittest.main()
