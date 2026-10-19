"""材料切出と同先物基準の境界を、市場結果なしの独立反例で検査する。"""

import copy
import tempfile
import unittest
from pathlib import Path

from fundamentals.common import digest, timestamp, write_json
from research_support.material_points import build_material_ledger
from research_support.prepare_material_snapshot import prepare_material_snapshot
from test_material_points import material_fixture


class MaterialContractAuditTests(unittest.TestCase):
    """元記事の利用可能時計とP0の意味を、既存成功例とは別に点検する。"""

    def _snapshot(self, articles: list[dict]) -> dict:
        """空の合成価格と指定記事から、締切内の入力だけを切り出す。

        引数: self (MaterialContractAuditTests) は試験、articles (list[dict]) は合成記事。
        返り値: dict は一時保存した入力。市場価格や実在記事は読まない。
        """
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            prices = base / "合成価格"
            for symbol in ("NK225", "TOPIX"):
                (prices / "candle_1min" / symbol).mkdir(parents=True)
            news = base / "合成ニュース.json"
            write_json(news, {"articles": articles})
            return prepare_material_snapshot("2026-09-30T07:45:00+09:00", news, prices, base / "入力")

    def _article(self) -> dict:
        """記事版の各時計が互いに異なる、締切内の合成記事を作る。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: dict は特定時計だけを変更できる合成記事。
        """
        return {"id": "合成記事", "revision_id": "版1", "headline": "合成材料",
                "canonical_url": "https://example.invalid/material-audit",
                "published_at": "2026-09-30T06:00:00+09:00",
                "fetched_at": "2026-09-30T06:10:00+09:00",
                "effective_at": "2026-09-30T06:00:00+09:00", "body_excerpt": "合成本文"}

    # ----------------------------------------

    def test_future_original_available_clock_does_not_become_visible(self) -> None:
        """元索引のavailable時計を切出時に繰り上げて未来本文を通さない。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None は未来の利用可能時計が拒否または除外されたこと。
        """
        article = self._article()
        article["available_at"] = "2026-09-30T08:00:00+09:00"
        try:
            packet = self._snapshot([article])
        except ValueError:
            return
        self.assertEqual(packet["news"], [])

    def test_effective_clock_selects_latest_visible_article_version(self) -> None:
        """取得順が逆でも、後に有効化された可視版を旧版へ巻き戻さない。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None はeffective時計を含む利用可能順で最新版を選んだこと。
        """
        old = self._article()
        old["fetched_at"] = "2026-09-30T06:30:00+09:00"
        new = {**old, "revision_id": "版2", "fetched_at": "2026-09-30T06:15:00+09:00",
               "effective_at": "2026-09-30T07:30:00+09:00"}
        packet = self._snapshot([new, old])
        self.assertEqual(len(packet["news"]), 1)
        self.assertEqual(packet["news"][0]["revision_id"], "版2")
        self.assertEqual(timestamp(packet["news"][0]["available_at"]), timestamp(new["effective_at"]))

    def test_day_reference_is_rejected_for_morning_day_forecast(self) -> None:
        """07:45のP0を前日デイへすり替え、ナイトの変動を再予測しない。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None は直前逆種セッション以外の基準が拒否されたこと。
        """
        data = material_fixture()
        data["references"]["NK225"].update(session="2026-09-29_Day", time="2026-09-29T15:40:00+09:00")
        with self.assertRaises(ValueError):
            build_material_ledger(**data)

    def test_early_session_price_is_not_a_terminal_proxy(self) -> None:
        """前ナイトの早い価格を、末尾終値代理として受け入れない。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None は公式終点の10分より前の価格が拒否されたこと。
        """
        data = material_fixture()
        data["references"]["NK225"]["time"] = "2026-09-29T17:01:00+09:00"
        with self.assertRaises(ValueError):
            build_material_ledger(**data)

    def test_later_retrieved_external_observation_is_rejected(self) -> None:
        """観測対象時計が過去でも、締切後に取得した外部資料を使わない。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None は外部資料の明示的な未来取得時計が拒否されたこと。
        """
        data = material_fixture()
        data["materials"][0]["external_observations"] = [{
            "name": "合成外部指数", "unit": "合成変化率", "value": 1,
            "source": "合成資料", "observed_at": "2026-09-30T07:30:00+09:00",
            "retrieved_at": "2026-09-30T08:00:00+09:00"}]
        with self.assertRaises(ValueError):
            build_material_ledger(**copy.deepcopy(data))

    def test_bound_source_rejects_article_and_p0_replacement(self) -> None:
        """点数が正常でも、根拠本文と先物基準の後からの置換を独立に拒否する。

        引数: self (MaterialContractAuditTests) は試験。
        返り値: None は元入力自己ハッシュ・P0・本文・時計の置換検知完了。
        """
        original = material_fixture()
        packet = {"as_of": original["as_of"], "session": "2026-09-30_Day",
                  "news": copy.deepcopy(original["articles"]),
                  "instruments": {symbol: {"reference": copy.deepcopy(reference)}
                                  for symbol, reference in original["references"].items()}}
        packet["packet_hash"] = digest(packet)
        ledger = build_material_ledger(**original, source_packet=packet)
        self.assertEqual(ledger["source_packet_hash"], digest(packet))
        for change in ("自己ハッシュ", "P0価格", "記事本文", "元時計"):
            with self.subTest(置換=change):
                data, source = copy.deepcopy(original), copy.deepcopy(packet)
                if change == "自己ハッシュ":
                    source["packet_hash"] = "置換した自己ハッシュ"
                elif change == "P0価格":
                    data["references"]["NK225"]["price"] += 1
                elif change == "記事本文":
                    data["articles"][0]["body"] = "元入力に存在しない合成本文"
                else:
                    data["articles"][0]["fetched_at"] = "2026-09-29T21:02:00+09:00"
                with self.assertRaises(ValueError):
                    build_material_ledger(**data, source_packet=source)


if __name__ == "__main__":
    unittest.main()
