"""材料点の未知伝播・先物基準・来歴・人間別回答を、実価格を使わず検査する。"""

import copy
import hashlib
import math
import unittest

from fundamentals.common import digest
from research_support.material_points import HORIZONS, POINT_KEYS, build_material_ledger, render_material_markdown


def material_fixture() -> dict:
    """未来価格を含まない、両先物と二つの逆符号材料の合成入力を作る。

    引数: なし。
    返り値: dict はbuild_material_ledgerへそのまま渡せる契約例。
    """
    references = {symbol: {"symbol": symbol, "contract": "202612", "price": price,
        "time": "2026-09-30T05:55:00+09:00", "session": "2026-09-29_Night", "proxy": True,
        "source": "合成先物足.csv"} for symbol, price in (("NK225", 50000), ("TOPIX", 3500))}
    articles = [{"id": str(index), "revision_id": "1", "canonical_url": "https://example.invalid/" + str(index),
        "headline": "合成記事", "body": "合成本文", "published_at": "2026-09-29T21:00:00+09:00",
        "fetched_at": "2026-09-29T21:01:00+09:00", "effective_at": "2026-09-29T21:00:00+09:00"} for index in (1, 2)]
    materials = []
    for index, impact, before in ((1, 3, 2), (2, -2, -1)):
        points = {"impact_pt": impact, "priced_in_before_reference_pt": before, "priced_in_after_reference_pt": 0,
            "evidence": {key: "合成記事と基準までの観測を踏まえた主観注釈。閉場中の追加約定観測なし。" for key in POINT_KEYS},
            "transmission": "合成の利益期待を通じた指数先物への影響仮説。",
            "observations": {"before_reference": [], "after_reference": []}}
        materials.append({"material_id": "材料" + str(index), "story_key": "事象" + str(index), "title": "合成材料" + str(index),
            "evidence": [{"id": str(index), "revision_id": "1"}], "status": "有効", "status_reason": "締切時点で反証なし。",
            "expires_at": None, "invalidation": "内容の否定または効果の消失。", "offsets": [],
            "news_facts": {"prior_expectation": None, "actual": "合成の新事実", "surprise": None,
                "first_public_at": None, "retrieved_at": "2026-09-29T21:01:00+09:00", "next_events": []},
            "external_observations": [],
            "impacts": {symbol: {horizon: copy.deepcopy(points) if horizon == "today" else None for horizon in HORIZONS}
                        for symbol in references}})
    materials[0]["offsets"] = [{"material_id": "材料2", "reason": "逆方向の利益期待が相殺。符号付き合計以外の追加控除をしない。"}]
    forecasts = {symbol: {horizon: {"direction": "上がる" if symbol == "NK225" else "下がる", "author_kind": "LLM",
        "author": "合成判定者", "reason": "合成の別判断。", "material_ids": ["材料1", "材料2"],
        "point_rationale": "残余＋1ptと−1ptは拮抗するが、当該先物への伝達順を別判断した。"} if horizon == "today" else None
        for horizon in HORIZONS} for symbol in references}
    return {"as_of": "2026-09-30T07:45:00+09:00", "references": references, "articles": articles,
            "materials": materials, "forecasts": forecasts}


def material_packet(data: dict) -> dict:
    """契約試験用の自己ハッシュ付き切出原本を合成する。

    引数: data (dict) は合成の材料入力。
    返り値: dict は原値束縛を検査できる切出入力。
    """
    packet = {"as_of": data["as_of"], "session": data["as_of"][:10] + ("_Day" if data["as_of"][11:13] == "07" else "_Night"),
              "news": copy.deepcopy(data["articles"]), "instruments": {symbol: {"reference": {
                  key: reference[key] for key in ("price", "time", "session", "contract")}}
                  for symbol, reference in data["references"].items()}}
    packet["packet_hash"] = digest(packet)
    return packet


class MaterialPointsTests(unittest.TestCase):
    """点を真の価格効果へ読み替えず、型と時間と演算の契約を検査する。"""

    def setUp(self) -> None:
        """各試験で独立した合成入力を作る。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data = material_fixture()

    def _build(self) -> dict:
        """試験中の入力から機械台帳を作る。

        引数: self (MaterialPointsTests) はテスト。
        返り値: dict は検査済み台帳。
        """
        return build_material_ledger(**self.data)

    def _point(self, index: int = 0, symbol: str = "NK225") -> dict:
        """試験で変更する期間別の点入力を取得する。

        引数: self (MaterialPointsTests) はテスト、index (int) は材料位置、symbol (str) は先物。
        返り値: dict は当日点への参照。
        """
        return self.data["materials"][index]["impacts"][symbol]["today"]

    def test_signed_points_and_offsets_do_not_double_subtract(self) -> None:
        """逆符号点を保ち、相殺記録のために合計から再控除しない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        ledger = self._build()
        group = ledger["aggregates"]["NK225"]["today"]
        self.assertEqual(group["totals"]["residual_pt"], 0)
        self.assertEqual(group["known_positive_residual_pt"], 1)
        self.assertEqual(group["known_negative_residual_pt"], -1)
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["reference_remaining_pt"], 1)
        self.assertEqual(ledger["ledger_hash"], digest({key: value for key, value in ledger.items() if key != "ledger_hash"}))

    def test_reference_and_asof_residual_are_different(self) -> None:
        """P0後に既反映した増分をP0基準寄与から二重に控除しない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self._point()["priced_in_after_reference_pt"] = 0.4
        point = self._build()["materials"][0]["impacts"]["NK225"]["today"]
        self.assertEqual(point["reference_remaining_pt"], 1)
        self.assertEqual(point["residual_pt"], 0.6)
        self.assertEqual(point["priced_in_total_pt"], 2.4)

    def test_unknown_pricing_keeps_total_unknown_and_known_subtotal_separate(self) -> None:
        """織込み不明をゼロへせず、既知小計と合計不明を別欄へ出す。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self._point()["priced_in_before_reference_pt"] = None
        ledger = self._build()
        point = ledger["materials"][0]["impacts"]["NK225"]["today"]
        self.assertIsNone(point["reference_remaining_pt"])
        self.assertIsNone(point["residual_pt"])
        group = ledger["aggregates"]["NK225"]["today"]
        self.assertIsNone(group["totals"]["residual_pt"])
        self.assertEqual(group["known_subtotals"]["residual_pt"], -1)
        self.assertEqual(group["unknown_material_ids"]["residual_pt"], ["材料1"])

    def test_unknown_after_preserves_reference_remaining(self) -> None:
        """P0後増分だけ不明ならP0起点残余を消さず、asof残余は不明にする。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self._point()["priced_in_after_reference_pt"] = None
        point = self._build()["materials"][0]["impacts"]["NK225"]["today"]
        self.assertEqual(point["reference_remaining_pt"], 1)
        self.assertIsNone(point["residual_pt"])

    def test_overpricing_and_reverse_signed_pricing_are_allowed(self) -> None:
        """織込みを割合へ制限せず、行き過ぎと逆符号の残余を保持する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self._point()["priced_in_before_reference_pt"] = 6
        self._point()["priced_in_after_reference_pt"] = -1
        point = self._build()["materials"][0]["impacts"]["NK225"]["today"]
        self.assertEqual(point["reference_remaining_pt"], -3)
        self.assertEqual(point["residual_pt"], -2)

    def test_future_article_rejected_before_numeric_points(self) -> None:
        """使わない記事であっても未来版を先に拒否し、点数解析に進まない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        for key in ("published_at", "fetched_at", "effective_at"):
            with self.subTest(clock=key):
                self.data = material_fixture()
                self.data["articles"][1][key] = "2026-09-30T07:46:00+09:00"
                self._point()["impact_pt"] = "数値解析を禁止する番兵"
                with self.assertRaisesRegex(ValueError, "締切後"):
                    self._build()

    def test_duplicate_article_revision_url_and_story_rejected(self) -> None:
        """再報道を同じ事象へ束ね、記事・改訂・URL・材料の二重計上を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        for field in ("id", "canonical_url"):
            self.data = material_fixture()
            self.data["articles"][1][field] = self.data["articles"][0][field]
            with self.assertRaises(ValueError):
                self._build()
        self.data = material_fixture()
        self.data["materials"][1]["story_key"] = "事象1"
        with self.assertRaisesRegex(ValueError, "再報道"):
            self._build()

    def test_unavailable_revision_and_absent_evidence_rejected(self) -> None:
        """根拠のない点や、切出されていない記事版を採用しない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["materials"][0]["evidence"][0]["revision_id"] = "未来版"
        with self.assertRaises(ValueError):
            self._build()
        self.data = material_fixture()
        self._point()["evidence"]["impact_pt"] = ""
        with self.assertRaises(ValueError):
            self._build()

    def test_no_point_to_class_conversion_or_frequency_fallback(self) -> None:
        """材料点が正でも人間の逆方向判断を保持し、根拠材料なしの回答は拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["forecasts"]["NK225"]["today"]["direction"] = "下がる"
        self.assertEqual(self._build()["forecasts"]["NK225"]["today"]["direction"], "下がる")
        self.data["forecasts"]["NK225"]["today"]["material_ids"] = []
        with self.assertRaisesRegex(ValueError, "頻度代替"):
            self._build()

    def test_unanswered_and_no_material_do_not_become_flat(self) -> None:
        """有効材料がない回を頻度やもみ合いで埋めず、無回答として残す。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["materials"] = []
        for symbol in ("NK225", "TOPIX"):
            row = self.data["forecasts"][symbol]["today"]
            row.update(direction=None, material_ids=[], point_rationale="有効材料がなく、回答を算出できない。")
        ledger = self._build()
        self.assertIsNone(ledger["forecasts"]["NK225"]["today"]["direction"])
        self.assertIsNone(ledger["aggregates"]["NK225"]["today"]["totals"]["residual_pt"])

    def test_inactive_material_not_zeroed_or_used_as_forecast_basis(self) -> None:
        """失効材料は原点を残して合計から除き、予測の使用材料に指定できない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["materials"][0]["status"] = "反証"
        for symbol in ("NK225", "TOPIX"):
            self.data["forecasts"][symbol]["today"]["material_ids"] = ["材料2"]
        ledger = self._build()
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["impact_pt"], 3)
        self.assertEqual(ledger["aggregates"]["NK225"]["today"]["totals"]["impact_pt"], -2)
        self.data["forecasts"]["NK225"]["today"]["material_ids"] = ["材料1"]
        with self.assertRaises(ValueError):
            self._build()

    def test_expiry_and_offsets_require_consistent_lifecycle(self) -> None:
        """失効時計の過ぎた有効材料や、自分自身への相殺指定を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["materials"][0]["expires_at"] = self.data["as_of"]
        with self.assertRaises(ValueError):
            self._build()
        self.data = material_fixture()
        self.data["materials"][0]["offsets"][0]["material_id"] = "材料1"
        with self.assertRaises(ValueError):
            self._build()

    def test_symbol_contract_and_before_after_observation_separation(self) -> None:
        """同先物・同限月とP0前後の境界を価格観測で検査する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        observation = {"symbol": "NK225", "contract": "202612", "start": "2026-09-29T22:00:00+09:00",
            "end": "2026-09-30T05:55:00+09:00", "start_price": 49000, "end_price": 50000, "source": "合成価格原本"}
        self._point()["observations"]["before_reference"] = [observation]
        self._build()
        for key, value in (("symbol", "TOPIX"), ("contract", "202703"), ("end", "2026-09-30T06:00:00+09:00")):
            with self.subTest(field=key):
                candidate = copy.deepcopy(observation)
                candidate[key] = value
                self._point()["observations"]["before_reference"] = [candidate]
                with self.assertRaises(ValueError):
                    self._build()
        self._point()["observations"]["before_reference"] = []
        self._point()["observations"]["after_reference"] = [observation]
        with self.assertRaises(ValueError):
            self._build()

    def test_future_external_reaction_and_future_known_schedule_rejected(self) -> None:
        """外部反応は点へ自動換算せず、締切後の実測や後知恵予定を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        row = {"name": "合成NASDAQ", "observed_at": "2026-09-30T07:40:00+09:00", "value": 1.2,
               "unit": "観測された変化率", "source": "合成外部原本"}
        self.data["materials"][0]["external_observations"] = [row]
        ledger = self._build()
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["priced_in_after_reference_pt"], 0)
        row["observed_at"] = "2026-09-30T07:46:00+09:00"
        with self.assertRaises(ValueError):
            self._build()
        self.data = material_fixture()
        self.data["materials"][0]["news_facts"]["next_events"] = [{"description": "合成予定", "scheduled_at": "2026-10-01T10:00:00+09:00",
            "known_at": "2026-09-30T07:46:00+09:00", "evidence": "締切後に発表された予定"}]
        with self.assertRaises(ValueError):
            self._build()

    def test_long_horizons_unknown_and_independent_symbol_points(self) -> None:
        """長期点を当日点で埋めず、NK225の点をTOPIXへ流用しない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self._point(symbol="TOPIX")["impact_pt"] = -4
        ledger = self._build()
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["impact_pt"], 3)
        self.assertEqual(ledger["materials"][0]["impacts"]["TOPIX"]["today"]["impact_pt"], -4)
        self.assertIsNone(ledger["materials"][0]["impacts"]["NK225"]["week"])
        self.assertIsNone(ledger["aggregates"]["NK225"]["week"]["totals"]["impact_pt"])

    def test_finite_scale_and_explicit_unknown_required(self) -> None:
        """不正値と尺度外を拒否し、数値欠損はNoneとして指定させる。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        for value in (math.nan, math.inf, True, "3", 5.01):
            self.data = material_fixture()
            self._point()["impact_pt"] = value
            with self.assertRaises(ValueError):
                self._build()
        self.data = material_fixture()
        del self._point()["priced_in_after_reference_pt"]
        with self.assertRaises(ValueError):
            self._build()

    def test_unknown_contract_requires_explicit_quality_and_is_not_verified_match(self) -> None:
        """原CSVの限月不明を許すが、None同士を同限月確認済みにしない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["references"]["NK225"]["contract"] = None
        with self.assertRaisesRegex(ValueError, "未確認"):
            self._build()
        self.data["references"]["NK225"]["contract_identity_status"] = "未確認"
        ledger = self._build()
        self.assertIsNone(ledger["references"]["NK225"]["contract"])
        self.assertEqual(ledger["quality"]["contract_identity_status"]["NK225"], "未確認")
        self.assertEqual(ledger["quality"]["contract_identity_status"]["TOPIX"], "確認済み")
        self.assertIn("限月不明", render_material_markdown(ledger))
        self.assertIn("品質不備", render_material_markdown(ledger))

    def test_reference_kind_and_night_same_day_are_required(self) -> None:
        """朝はナイト、夕方は当日デイとし、前日デイへの後退を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["references"]["NK225"].update(session="2026-09-29_Day", time="2026-09-29T15:40:00+09:00")
        with self.assertRaises(ValueError):
            self._build()
        self.data = material_fixture()
        self.data["as_of"] = "2026-09-30T16:00:00+09:00"
        for reference in self.data["references"].values():
            reference.update(session="2026-09-30_Day", time="2026-09-30T15:40:00+09:00")
        self._build()
        self.data["references"]["NK225"].update(session="2026-09-29_Day", time="2026-09-29T15:40:00+09:00")
        with self.assertRaises(ValueError):
            self._build()

    def test_terminal_proxy_ten_minute_boundary_and_exact_close(self) -> None:
        """代理の末尾10分境界を含め、場中の任意価格や偽の公式終値を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.data["references"]["NK225"]["time"] = "2026-09-30T05:50:00+09:00"
        self._build()
        for clock in ("2026-09-30T05:49:59+09:00", "2026-09-29T17:01:00+09:00"):
            self.data["references"]["NK225"]["time"] = clock
            with self.assertRaises(ValueError):
                self._build()
        self.data["references"]["NK225"].update(time="2026-09-30T05:55:00+09:00", proxy=False)
        with self.assertRaises(ValueError):
            self._build()
        self.data["references"]["NK225"]["time"] = "2026-09-30T06:00:00+09:00"
        self._build()

    def test_all_external_availability_clocks_are_guarded(self) -> None:
        """外部資料は観測時計だけでなく、存在する取得・有効・更新時計も検査する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        for key in ("retrieved_at", "fetched_at", "available_at", "effective_at", "updated_at", "published_at", "received_at", "known_at"):
            with self.subTest(clock=key):
                row = {"name": "合成外部株", "unit": "合成変化率", "source": "合成資料", "value": 1,
                       "observed_at": "2026-09-30T07:30:00+09:00", key: self.data["as_of"]}
                self.data["materials"][0]["external_observations"] = [row]
                self._build()
                row[key] = "2026-09-30T07:45:01+09:00"
                with self.assertRaises(ValueError):
                    self._build()

    def test_source_packet_binding_and_unbound_quality(self) -> None:
        """未突合を明示し、切出入力を渡した場合だけP0原値と記事版を束縛する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        self.assertEqual(self._build()["quality"]["reference_snapshot_binding"], "P0原値との突合未実行")
        packet = material_packet(self.data)
        bound = build_material_ledger(**self.data, source_packet=packet)
        self.assertEqual(bound["source_packet_hash"], digest(packet))
        self.assertIn("照合済み", bound["quality"]["reference_snapshot_binding"])
        self.data["references"]["NK225"]["price"] += 10
        with self.assertRaisesRegex(ValueError, "原値"):
            build_material_ledger(**self.data, source_packet=packet)

    def test_source_packet_hash_cutoff_and_article_content_guard(self) -> None:
        """自己hash破損、別締切、同記事版の本文改変をそれぞれ拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        packet = material_packet(self.data)
        packet["as_of"] = "2026-09-30T07:46:00+09:00"
        with self.assertRaisesRegex(ValueError, "ハッシュ"):
            build_material_ledger(**self.data, source_packet=packet)
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        with self.assertRaisesRegex(ValueError, "締切"):
            build_material_ledger(**self.data, source_packet=packet)
        packet = material_packet(self.data)
        self.data["articles"][0]["body"] = "原本と違う本文"
        with self.assertRaisesRegex(ValueError, "本文"):
            build_material_ledger(**self.data, source_packet=packet)

    def test_article_clock_binding_compares_instant_without_changing_it(self) -> None:
        """UTCとJSTの等価表記は許すが、原時計の削除や改変は許さない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        packet = material_packet(self.data)
        self.data["articles"][0]["fetched_at"] = "2026-09-29T12:01:00.000+00:00"
        build_material_ledger(**self.data, source_packet=packet)
        self.data["articles"][0]["fetched_at"] = "2026-09-29T12:02:00+00:00"
        with self.assertRaisesRegex(ValueError, "元時計"):
            build_material_ledger(**self.data, source_packet=packet)

    def test_attached_body_requires_utf8_hash_and_explicit_raw_provenance(self) -> None:
        """別取得本文はUTF8ハッシュと原本来歴を必須にし、原本ファイル未照合を明示する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        packet = material_packet(self.data)
        for article in packet["news"]:
            del article["body"]
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        with self.assertRaises(ValueError):
            build_material_ledger(**self.data, source_packet=packet)
        for article in self.data["articles"]:
            article.update(body_source="合成本文取得", body_raw_path="合成原本.json", body_raw_sha256="a" * 64,
                           body_scope="合成の全文", body_sha256=hashlib.sha256(article["body"].encode("utf-8")).hexdigest())
        bound = build_material_ledger(**self.data, source_packet=packet)
        self.assertIn("別途照合", bound["source_binding"]["body_notice"])
        self.assertEqual(bound["articles"][0]["body_raw_sha256"], "a" * 64)
        self.data["articles"][0]["body_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            build_material_ledger(**self.data, source_packet=packet)

    def test_raw_attached_body_accepts_structured_scope_with_valid_character_bounds(self) -> None:
        """別取得本文の構造化区間を受理し、文字数不一致やboolの添字を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        packet = material_packet(self.data)
        for article in packet["news"]:
            del article["body"]
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        for article in self.data["articles"]:
            article.update(body_source="合成の別取得本文", body_raw_path="合成原本.json", body_raw_sha256="a" * 64,
                body_scope={"start": 2, "end_exclusive": 2 + len(article["body"]), "notice": "原本の主文区間"},
                body_sha256=hashlib.sha256(article["body"].encode("utf-8")).hexdigest())
        ledger = build_material_ledger(**self.data, source_packet=packet)
        self.assertEqual(ledger["articles"][0]["body_scope"], self.data["articles"][0]["body_scope"])
        for scope in ({"start": True, "end_exclusive": 5, "notice": "不正添字"},
                      {"start": 2, "end_exclusive": 100, "notice": "長さ不一致"}):
            with self.subTest(scope=scope):
                self.data["articles"][0]["body_scope"] = scope
                with self.assertRaises(ValueError):
                    build_material_ledger(**self.data, source_packet=packet)

    def test_excerpt_body_without_raw_requires_exact_original_scope_and_utf8_hash(self) -> None:
        """抜粋由来はrawなしを許すが、元抜粋の区間・本文SHA・元抜粋保持を必須とする。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        for article in self.data["articles"]:
            article.update(body_excerpt="案内" + article["body"] + "関連記事", body_source="切出済み抜粋の主文",
                body_raw_path=None, body_raw_sha256=None,
                body_scope={"start": 2, "end_exclusive": 2 + len(article["body"]), "notice": "案内と関連記事を除外"},
                body_sha256=hashlib.sha256(article["body"].encode("utf-8")).hexdigest())
        packet = material_packet(self.data)
        for article in packet["news"]:
            for key in ("body", "body_source", "body_raw_path", "body_raw_sha256", "body_scope", "body_sha256"):
                del article[key]
        packet["packet_hash"] = digest({key: value for key, value in packet.items() if key != "packet_hash"})
        ledger = build_material_ledger(**self.data, source_packet=packet)
        self.assertIsNone(ledger["articles"][0]["body_raw_path"])
        self.assertIsNone(ledger["articles"][0]["body_raw_sha256"])
        original = copy.deepcopy(self.data["articles"][0])
        self.data["articles"][0]["body"] = "偽の本文"
        self.data["articles"][0]["body_sha256"] = hashlib.sha256("偽の本文".encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "元抜粋"):
            build_material_ledger(**self.data, source_packet=packet)
        self.data["articles"][0] = copy.deepcopy(original)
        self.data["articles"][0]["body_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            build_material_ledger(**self.data, source_packet=packet)
        self.data["articles"][0] = copy.deepcopy(original)
        del self.data["articles"][0]["body_excerpt"]
        with self.assertRaises(ValueError):
            build_material_ledger(**self.data, source_packet=packet)

    def test_markdown_has_short_material_reasons_and_keeps_json_full_text(self) -> None:
        """材料記事版・点根拠・伝達・反証を短く表示し、JSONの全文は保持する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        reason = "長い根拠" * 100
        self._point()["evidence"]["impact_pt"] = reason
        ledger = self._build()
        text = render_material_markdown(ledger)
        self.assertIn("根拠記事ID/版：1/1", text)
        self.assertIn("B0：", text)
        self.assertIn("ΔB：", text)
        self.assertIn("伝達：", text)
        self.assertIn("反証・失効：", text)
        self.assertIn("各120文字以内", text)
        self.assertNotIn(reason, text)
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["evidence"]["impact_pt"], reason)

    def test_unknown_observation_contract_propagates_quality_without_fabricating_identity(self) -> None:
        """基準限月が判明していても、観測側不明を同限月へ補完しない。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        observation = {"symbol": "NK225", "contract": None, "start": "2026-09-29T22:00:00+09:00",
            "end": "2026-09-30T05:55:00+09:00", "start_price": 49000, "end_price": 50000, "source": "合成価格原本"}
        self._point()["observations"]["before_reference"] = [observation]
        with self.assertRaisesRegex(ValueError, "未確認"):
            self._build()
        observation["contract_identity_status"] = "未確認"
        ledger = self._build()
        self.assertEqual(ledger["quality"]["contract_identity_status"]["NK225"], "未確認")
        saved = ledger["materials"][0]["impacts"]["NK225"]["today"]["observations"]["before_reference"][0]
        self.assertIsNone(saved["contract"])
        self.assertEqual(saved["contract_identity_status"], "未確認")

    def test_longer_horizon_uses_its_own_points_and_future_reference_is_rejected(self) -> None:
        """週の別注釈は当日点を変えず、未満了セッションのP0を拒否する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        week = copy.deepcopy(self._point())
        week["impact_pt"] = 4
        self.data["materials"][0]["impacts"]["NK225"]["week"] = week
        self.data["forecasts"]["NK225"]["week"] = copy.deepcopy(self.data["forecasts"]["NK225"]["today"])
        self.data["forecasts"]["NK225"]["week"]["material_ids"] = ["材料1"]
        ledger = self._build()
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["week"]["residual_pt"], 2)
        self.assertEqual(ledger["materials"][0]["impacts"]["NK225"]["today"]["residual_pt"], 1)
        self.data["references"]["NK225"]["session"] = "2026-09-30_Day"
        self.data["references"]["NK225"]["time"] = "2026-09-30T15:40:00+09:00"
        with self.assertRaises(ValueError):
            self._build()

    def test_inputs_unchanged_and_markdown_bound_to_points(self) -> None:
        """台帳生成で元注釈を変更せず、表の固定hashと改変検知を確認する。

        引数: self (MaterialPointsTests) はテスト。
        返り値: None。
        """
        original = copy.deepcopy(self.data)
        ledger = self._build()
        self.assertEqual(original, self.data)
        markdown = render_material_markdown(ledger)
        self.assertIn("P0起点 T−B0", markdown)
        self.assertIn("asof残余 T−B0−ΔB", markdown)
        self.assertIn("未校正", markdown)
        self.assertIn("当日点から自動補完していない", markdown)
        self.assertNotIn("上昇確率", markdown)
        self.assertEqual(ledger["forecasts"]["NK225"]["today"]["point_basis_hash"],
            digest([{row["material_id"]: row["impacts"]["NK225"]["today"]} for row in ledger["materials"]]))
        ledger["materials"][0]["impacts"]["NK225"]["today"]["impact_pt"] = 4
        with self.assertRaises(ValueError):
            render_material_markdown(ledger)


if __name__ == "__main__":
    unittest.main()
