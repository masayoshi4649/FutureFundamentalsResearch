"""材料入力の締切遮断と、初期セッションを欠測として保持する動作を検査する。"""

import copy
import csv
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from fundamentals.common import digest, read_json, timestamp, write_json
from fundamentals.prices import Candle, load_candles
from research_support.prepare_material_snapshot import prepare_material_snapshot


class MaterialSnapshotTests(unittest.TestCase):
    """実際の市場結果を使わず、意図的な未来汚染を含む合成入力を点検する。"""

    def setUp(self) -> None:
        """ニュースと両先物用の空の合成原本フォルダを作る。

        引数: self (MaterialSnapshotTests) は試験インスタンス。
        返り値: None は初期化完了。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.prices = self.base / "合成価格"
        self.news = self.base / "合成ニュース.json"
        self.output = self.base / "材料入力"
        self.as_of = "2026-09-22T07:45:00+09:00"
        for symbol in ("NK225", "TOPIX"):
            (self.prices / "candle_1min" / symbol).mkdir(parents=True)
        write_json(self.news, {"articles": []})

    def _article(self, identity: str, **changes) -> dict:
        """締切内の合成記事を作り、指定項目だけを変更する。

        引数: self (MaterialSnapshotTests) は試験、identity (str) は識別子、
            changes (dict) は上書きする記事項目。
        返り値: dict は市場結果を含まない記事原本。
        """
        value = {
            "id": identity, "revision_id": "合成初版", "headline": "入力境界を調べる合成記事",
            "canonical_url": "https://example.invalid/" + identity,
            "published_at": "2026-09-22T06:00:00+09:00",
            "fetched_at": "2026-09-22T06:05:00+09:00",
            "effective_at": "2026-09-22T06:00:00+09:00",
            "body_excerpt": "締切内の既知材料だけ。価格結果はない。",
            "body_text": "全文の合成データ。", "source_id": 18, "source_name": "合成ニュース",
        }
        value.update(changes)
        return value

    def _save_articles(self, articles: list[dict]) -> None:
        """合成ニュース原本を試験用の一時ファイルへ保存する。

        引数: self (MaterialSnapshotTests) は試験、articles (list[dict]) は合成記事。
        返り値: None は保存完了。
        """
        write_json(self.news, {"articles": articles}, exclusive=False)

    def _csv(self, symbol: str, session: str, times: list[str], close: str = "101") -> Path:
        """指定セッションの合成１分足だけを作る。

        引数: self (MaterialSnapshotTests) は試験、symbol/session (str) は先物と場名、
            times (list[str]) は開始時計、close (str) は数値または不正値の検査用文字列。
        返り値: Path は一時CSV原本。
        """
        path = self.prices / "candle_1min" / symbol / (session + ".csv")
        fields = ("time", "open", "high", "low", "close", "volume", "volume_complete", "closed")
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for clock in times:
                writer.writerow({"time": clock, "open": close, "high": close, "low": close,
                                 "close": close, "volume": 1, "volume_complete": "true", "closed": "true"})
        return path

    def _prior_night(self) -> None:
        """両先物の直前ナイト末尾に、同じ仮の確定足を用意する。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は合成CSVの作成完了。
        """
        for symbol in ("NK225", "TOPIX"):
            self._csv(symbol, "2026-09-21_Night", ["2026-09-22T05:54:00+09:00"])

    def _prepare(self, as_of: str | None = None) -> dict:
        """一つの材料入力を合成原本から切り出す。

        引数: self (MaterialSnapshotTests) は試験、as_of (str | None) は任意の締切。
        返り値: dict は新規に保存された切出入力。
        """
        return prepare_material_snapshot(as_of or self.as_of, self.news, self.prices, self.output)

    # ----------------------------------------
    # ニュースは値の内容へ進む前に各時計で排除し、除外本文も渡さない。

    def test_future_news_clocks_and_poison_text_do_not_survive(self) -> None:
        """未来の公開・取得・更新・effective時計と危険な本文を出力へ残さない。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は４種類の未来時計と情報非残存の検査完了。
        """
        self._prior_night()
        articles = [self._article("可視")]
        markers = []
        for index, field in enumerate(("effective_at", "published_at", "fetched_at", "updated_at")):
            marker = f"未来の秘密本文{index}と未公表数値987654321.{index}"
            markers.append(marker)
            articles.append(self._article(
                "後発" + str(index), **{field: "2026-09-22T07:45:00.001+09:00"},
                headline=marker, body_excerpt=marker, body_text=marker,
            ))
        before = copy.deepcopy(articles)
        self._save_articles(articles)
        packet = self._prepare()
        self.assertEqual(articles, before)
        self.assertEqual([article["id"] for article in packet["news"]], ["可視"])
        exported = (self.output / "packet.json").read_text(encoding="utf-8")
        for marker in markers:
            self.assertNotIn(marker, exported)
        self.assertNotIn("987654321", exported)
        self.assertNotIn("後発", exported)
        self.assertNotIn("excluded", packet)

    def test_future_effective_article_is_not_passed_to_news_selector(self) -> None:
        """effectiveが後発の記事を、本文版の選択処理より前に落とす。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は選択処理へ渡す記事集合の独立点検完了。
        """
        self._save_articles([
            self._article("可視"),
            self._article("effectiveだけ後発", effective_at="2026-09-23T00:00:00+09:00",
                          body_excerpt="未公表999999999.999"),
        ])
        from fundamentals.news import select_articles
        with patch("research_support.prepare_material_snapshot.select_articles", wraps=select_articles) as selected:
            packet = self._prepare()
        candidates, cutoff = selected.call_args.args
        self.assertEqual([article["id"] for article in candidates], ["可視"])
        self.assertEqual(cutoff, timestamp(self.as_of))
        self.assertEqual([article["id"] for article in packet["news"]], ["可視"])

    def test_news_exact_cutoff_is_visible_but_old_or_updated_version_is_not(self) -> None:
        """境界と等しい記事を残し、古すぎる本文や締切後更新の版を混入しない。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は時刻境界と版の独立点検完了。
        """
        exact = self._article("境界", published_at=self.as_of, fetched_at=self.as_of, effective_at=self.as_of)
        old = self._article("期限外", published_at="2026-09-19T07:44:59+09:00",
                            fetched_at="2026-09-19T07:44:59+09:00", effective_at="2026-09-19T07:44:59+09:00",
                            body_excerpt="古い秘密123456789")
        original = self._article("改訂記事", body_excerpt="締切内の原版")
        revised = {**original, "revision_id": "後発更新版", "updated_at": "2026-09-22T08:00:00+09:00",
                   "body_excerpt": "未来改訂本文9988776655"}
        self._save_articles([revised, old, exact, original])
        packet = self._prepare()
        articles = {article["id"]: article for article in packet["news"]}
        self.assertEqual(set(articles), {"境界", "改訂記事"})
        self.assertEqual(articles["改訂記事"]["body_excerpt"], "締切内の原版")
        self.assertNotIn("9988776655", json.dumps(packet, ensure_ascii=False))
        self.assertNotIn("123456789", json.dumps(packet, ensure_ascii=False))

    # ----------------------------------------
    # 価格の読取上限と追加防壁を、未来ファイル・不正な戻しデータで検査する。

    def test_loader_uses_exact_cutoff_and_does_not_open_future_poison_csv(self) -> None:
        """当回と翌場のCSVが不正値でも開かず、予測入力は前場だけへ固定する。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は合成CSVを実際に読み取る境界検査の完了。
        """
        self._prior_night()
        future_paths = []
        for symbol in ("NK225", "TOPIX"):
            future_paths.append(self._csv(symbol, "2026-09-22_Day", ["2026-09-22T08:45:00+09:00"], "未公開価格999999999999"))
            future_paths.append(self._csv(symbol, "2026-09-22_Night", ["2026-09-22T17:00:00+09:00"], "不正な将来値"))
        with patch("research_support.prepare_material_snapshot.load_candles", wraps=load_candles) as loaded:
            packet = self._prepare()
        self.assertEqual(len(loaded.call_args_list), 2)
        self.assertEqual({call.args[1] for call in loaded.call_args_list}, {"NK225", "TOPIX"})
        for call in loaded.call_args_list:
            self.assertEqual(call.args[0], self.prices)
            self.assertEqual(call.kwargs, {"cutoff": timestamp(self.as_of)})
        for path in future_paths:
            self.assertNotIn(str(path.resolve()), packet["source_hashes"])
        exported = (self.output / "packet.json").read_text(encoding="utf-8")
        self.assertNotIn("999999999999", exported)
        self.assertNotIn("不正な将来値", exported)
        for item in packet["instruments"].values():
            self.assertEqual(item["reference"]["session"], "2026-09-21_Night")
            self.assertEqual({bar["session"] for bar in item["price_timeline"]}, {"2026-09-21_Night"})
            self.assertTrue(all(timestamp(bar["end"]) <= timestamp(self.as_of) for bar in item["price_timeline"]))

    def test_loader_injected_future_and_unclosed_bars_are_filtered_again(self) -> None:
        """ローダーが未来足や未確定足を返す異常でも、数値を入力へ通さない。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は読取後の追加防壁の検査完了。
        """
        self._prior_night()

        def injected(prices: Path, symbol: str, cutoff) -> list[Candle]:
            """通常の可視価格へ、未来と未確定の汚染足を意図的に追加する。

            引数: prices (Path) は合成原本、symbol (str) は先物、cutoff (datetime) は締切。
            返り値: list[Candle] は境界防壁を試すための不正な足一覧。
            """
            rows = load_candles(prices, symbol, cutoff=cutoff)
            for start, closed, marker in (("2026-09-22T07:44:00+09:00", False, 8877665544),
                                          ("2026-09-22T08:45:00+09:00", True, 9988776655)):
                clock = timestamp(start)
                rows.append(Candle(clock, clock + timedelta(minutes=1), marker, marker, marker, marker,
                                   1, "2026-09-22_Day", symbol, closed=closed))
            return rows

        with patch("research_support.prepare_material_snapshot.load_candles", side_effect=injected):
            packet = self._prepare()
        serialized = json.dumps(packet, ensure_ascii=False)
        self.assertNotIn("8877665544", serialized)
        self.assertNotIn("9988776655", serialized)
        self.assertTrue(all(len(item["price_timeline"]) == 1 for item in packet["instruments"].values()))

    def test_target_session_with_false_past_timestamp_is_rejected(self) -> None:
        """当回場名を持つ足の時計が締切前に偽装されても保存前に拒否する。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は場名による独立した混入防止の検査完了。
        """
        self._prior_night()

        def injected(prices: Path, symbol: str, cutoff) -> list[Candle]:
            """時刻フィルタを通過する、当回セッションの矛盾した足を返す。

            引数: prices (Path) は合成原本、symbol (str) は先物、cutoff (datetime) は締切。
            返り値: list[Candle] は可視価格と、偽装した当回足。
            """
            rows = load_candles(prices, symbol, cutoff=cutoff)
            clock = cutoff - timedelta(minutes=1)
            rows.append(Candle(clock, cutoff, 123456789, 123456789, 123456789, 123456789,
                               1, "2026-09-22_Day", symbol))
            return rows

        with patch("research_support.prepare_material_snapshot.load_candles", side_effect=injected):
            with self.assertRaisesRegex(ValueError, "当回対象場"):
                self._prepare()
        self.assertFalse((self.output / "packet.json").exists())

    # ----------------------------------------
    # 最初の場も、入力が不足する理由とともに母集団へ残す。

    def test_first_night_keeps_p0_even_without_preceding_return(self) -> None:
        """９月21日ナイトは前々場がなくても、同日デイ基準と入力を保持する。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は初期セッションの保持の検査完了。
        """
        for symbol in ("NK225", "TOPIX"):
            self._csv(symbol, "2026-09-21_Day", ["2026-09-21T08:45:00+09:00", "2026-09-21T15:39:00+09:00"])
        packet = self._prepare("2026-09-21T16:00:00+09:00")
        self.assertEqual(packet["session"], "2026-09-21_Night")
        self.assertEqual(set(packet["instruments"]), {"NK225", "TOPIX"})
        for item in packet["instruments"].values():
            self.assertIsNone(item["reference_error"])
            self.assertEqual(item["reference"]["session"], "2026-09-21_Day")
            self.assertEqual(item["reference"]["price"], 101)
            self.assertEqual(len(item["recent_sessions"]), 1)
            self.assertIsNone(item["recent_sessions"][0]["previous_observed_session_close"])
            self.assertEqual(len(item["price_timeline"]), 2)

    def test_no_p0_preserves_both_instruments_and_reason(self) -> None:
        """最初のデイの基準不足をNoneと理由で残し、対象を捨てない。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は欠測対象の保持の検査完了。
        """
        packet = self._prepare("2026-09-21T07:45:00+09:00")
        self.assertEqual(packet["session"], "2026-09-21_Day")
        self.assertEqual(set(packet["instruments"]), {"NK225", "TOPIX"})
        for item in packet["instruments"].values():
            self.assertIsNone(item["reference"])
            self.assertIn("前セッション", item["reference_error"])
            self.assertEqual(item["recent_sessions"], [])
            self.assertEqual(item["price_timeline"], [])
        self.assertEqual(read_json(self.output / "packet.json"), packet)
        self.assertEqual(digest({key: value for key, value in packet.items() if key != "packet_hash"}), packet["packet_hash"])

    def test_saved_snapshot_cannot_be_overwritten(self) -> None:
        """原本が変わっても既存入力を上書きせず、固定済みの内容を保持する。

        引数: self (MaterialSnapshotTests) は試験。
        返り値: None は固定後の不変性の検査完了。
        """
        self._prior_night()
        self._prepare()
        saved = (self.output / "packet.json").read_bytes()
        self._save_articles([self._article("追加された別記事")])
        with self.assertRaises(FileExistsError):
            self._prepare()
        self.assertEqual((self.output / "packet.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
