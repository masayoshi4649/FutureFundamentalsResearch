"""ニュースの時点整合、版選択、書き出し形式の統合を検証する。"""

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fundamentals.news import load_export, select_articles, snapshot_payload


def _article(**changes: object) -> dict:
    """検証用の有効な記事データを生成する。

    引数:
        changes (object): 上書きする記事項目の名前付き引数。
    返り値:
        dict: 公表済みかつ取得済みのニュース記事。
    """
    record = {
        "id": "記事一",
        "revision_id": "初版",
        "canonical_url": "https://example.com/news/1",
        "headline": "国内景気に関する発表",
        "body": "予測時点に公表された本文。",
        "published_at": "2026-10-01T08:00:00Z",
        "fetched_at": "2026-10-01T08:05:00Z",
    }
    record.update(changes)
    return record


# ----------------------------------------


class NewsSelectionTests(unittest.TestCase):
    """利用可能だった記事の選択を検証する。"""

    def setUp(self) -> None:
        """予測基準日時を用意する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        self.as_of = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)

    def test_each_future_timestamp_excludes_article(self) -> None:
        """公表・取得・更新のいずれかが未来なら記事を除外する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        for field in ("published_at", "fetched_at", "updated_at"):
            with self.subTest(field=field):
                selected, excluded = select_articles(
                    [_article(**{field: "2026-10-01T09:00:00.000001Z"})], self.as_of,
                )
                self.assertEqual(selected, [])
                self.assertEqual(len(excluded), 1)
                self.assertEqual(excluded[0]["code"], "future_timestamp")
                self.assertIn(field, excluded[0]["reason"])

    def test_later_revision_does_not_erase_available_history(self) -> None:
        """未来の改訂本文を排除し、基準日時を進めたときだけ改訂版を選ぶ。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        initial = _article()
        revised = _article(
            revision_id="改訂版", updated_at="2026-10-01T10:00:00Z",
            fetched_at="2026-10-01T10:02:00Z", body="後で判明した結果を含む本文。",
        )
        selected, excluded = select_articles([revised, initial], self.as_of)
        self.assertEqual([article["revision_id"] for article in selected], ["初版"])
        self.assertEqual(selected[0]["body"], initial["body"])
        self.assertEqual(excluded[0]["revision_id"], "改訂版")
        selected, excluded = select_articles([revised, initial], self.as_of + timedelta(hours=2))
        self.assertEqual([article["revision_id"] for article in selected], ["改訂版"])
        self.assertEqual(excluded[0]["code"], "superseded_revision")

    def test_missing_publication_uses_fetch_time_and_accepts_boundary(self) -> None:
        """公表日時欠損時は取得日時を使い、基準日時と同時刻を許可する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        record = _article(fetched_at="2026-10-01T18:00:00+09:00")
        del record["published_at"]
        selected, excluded = select_articles([record], self.as_of)
        self.assertEqual(excluded, [])
        self.assertEqual(selected[0]["available_at"], "2026-10-01T09:00:00Z")
        self.assertNotIn("published_at", selected[0])

    def test_available_at_is_maximum_of_all_timestamps(self) -> None:
        """取得より後の更新時刻も利用可能時刻の計算に含める。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        record = _article(updated_at="2026-10-01T08:30:00Z")
        selected, excluded = select_articles([record], self.as_of)
        self.assertEqual(excluded, [])
        self.assertEqual(selected[0]["available_at"], "2026-10-01T08:30:00Z")

    def test_missing_fields_and_unzoned_times_are_visible(self) -> None:
        """必須項目欠損とタイムゾーン欠損を理由付きで記録する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        for field in ("id", "revision_id", "fetched_at", "canonical_url", "headline"):
            with self.subTest(missing=field):
                record = _article()
                del record[field]
                selected, excluded = select_articles([record], self.as_of)
                self.assertEqual(selected, [])
                self.assertIn(field, excluded[0]["reason"])
        for field in ("published_at", "fetched_at", "updated_at"):
            with self.subTest(unzoned=field):
                selected, excluded = select_articles(
                    [_article(**{field: "2026-10-01T08:00:00"})], self.as_of,
                )
                self.assertEqual(selected, [])
                self.assertEqual(excluded[0]["code"], "invalid_timestamp")
                self.assertIn("タイムゾーン", excluded[0]["reason"])

    def test_revisions_and_duplicate_urls_are_deduplicated(self) -> None:
        """同一記事の旧版と別記事識別子で登録された同一 URL を除く。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        records = [
            _article(),
            _article(revision_id="第二版", fetched_at="2026-10-01T08:10:00Z"),
            _article(id="重複記事", fetched_at="2026-10-01T08:15:00Z"),
            _article(id="別記事", canonical_url="https://example.com/news/2"),
        ]
        selected, excluded = select_articles(records, self.as_of)
        self.assertEqual({article["id"] for article in selected}, {"重複記事", "別記事"})
        self.assertEqual({item["code"] for item in excluded}, {"superseded_revision", "duplicate_url"})

    def test_input_is_not_modified_and_duplicate_revision_is_reported(self) -> None:
        """日時正規化で入力を書き換えず、同一版の重複も記録する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        record = _article(fetched_at=datetime(2026, 10, 1, 8, 5, tzinfo=timezone.utc))
        records = [record, copy.deepcopy(record)]
        original = copy.deepcopy(records)
        selected, excluded = select_articles(records, self.as_of)
        self.assertEqual(records, original)
        self.assertEqual(len(selected), 1)
        self.assertEqual(excluded[0]["code"], "duplicate_revision")

    def test_invalid_input_and_invalid_json_values_are_explicit(self) -> None:
        """不正レコードと保存不能な値を除外し、不正な基準日時は拒否する。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        selected, excluded = select_articles([None, _article(score=float("nan"))], self.as_of)
        self.assertEqual(selected, [])
        self.assertEqual([item["code"] for item in excluded], ["invalid_record", "invalid_json_value"])
        with self.assertRaisesRegex(ValueError, "タイムゾーン"):
            select_articles([], self.as_of.replace(tzinfo=None))

    def test_snapshot_hash_is_reproducible_and_commits_to_content(self) -> None:
        """証跡のハッシュを再計算でき、本文の差がハッシュに反映される。

        引数:
            self (NewsSelectionTests): このテストケース。
        返り値:
            None: なし。
        """
        record = _article()
        payload = snapshot_payload([record], self.as_of)
        expected_payload = {key: value for key, value in payload.items() if key != "canonical_hash"}
        expected = hashlib.sha256(json.dumps(
            expected_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(payload["canonical_hash"], expected)
        reordered = dict(reversed(list(record.items())))
        self.assertEqual(payload, snapshot_payload([reordered], self.as_of))
        self.assertNotEqual(payload["canonical_hash"], snapshot_payload(
            [_article(body="異なる本文。")], self.as_of,
        )["canonical_hash"])


# ----------------------------------------


class NewsExportTests(unittest.TestCase):
    """取得ツールの応答形式からの書き出し読込を検証する。"""

    def test_archive_directory_preserves_all_revisions_without_recursing(self) -> None:
        """版別原本を結合し、別用途のファイルや下位フォルダを読み込まない。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        first = _article()
        revised = _article(revision_id="第二版", body="改訂された本文。")
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            (archive / "初版.json").write_text(
                json.dumps({"articles": [first]}, ensure_ascii=False), encoding="utf-8",
            )
            (archive / "改訂.jsonl").write_text(
                json.dumps(revised, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            (archive / "説明.txt").write_text("記事ではない説明文。", encoding="utf-8")
            nested = archive / "別用途"
            nested.mkdir()
            (nested / "読込対象外.json").write_text("不正な JSON", encoding="utf-8")
            self.assertEqual(load_export(archive), [first, revised])

    def test_corrupt_archive_member_is_not_silently_skipped(self) -> None:
        """版別原本の一部が壊れた場合、完全に読めたようには扱わない。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            (archive / "正常.json").write_text(json.dumps({"articles": [_article()]}), encoding="utf-8")
            (archive / "破損.jsonl").write_text("不正な JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "有効な JSON"):
                load_export(archive)

    def test_supported_json_envelopes_preserve_revisions(self) -> None:
        """各応答形式を記事配列に統一し、同一記事の別版を保持する。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        records = [_article(), _article(revision_id="第二版")]
        envelopes = [
            records,
            {"articles": records},
            {"structuredContent": {"articles": records}, "content": [{"type": "text", "text": "説明文"}]},
            {"content": [{"type": "text", "text": json.dumps({"articles": records})}]},
            {"result": {"data": {"articles": records}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ニュース.json"
            for envelope in envelopes:
                with self.subTest(envelope=envelope):
                    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
                    self.assertEqual(load_export(path), records)
            path.write_text(json.dumps({"article": records[0]}, ensure_ascii=False), encoding="utf-8-sig")
            self.assertEqual(load_export(path), [records[0]])

    def test_jsonl_supports_envelopes_and_blank_lines(self) -> None:
        """空行を含む JSONL を読み、版を失わずに結合する。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        first, second = _article(), _article(revision_id="第二版")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ニュース.jsonl"
            path.write_text(
                json.dumps(first, ensure_ascii=False) + "\n\n" + json.dumps({"article": second}, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(load_export(path), [first, second])

    def test_malformed_export_fails_with_japanese_location(self) -> None:
        """不正な JSON、記事配列、本文を日本語の位置付きエラーにする。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        invalid_sources = [
            ("{}\n壊れた行", "2 行目"),
            (json.dumps({"articles": {}}), "配列"),
            (json.dumps({"content": [{"type": "text", "text": "説明文"}]}), "本文"),
            (json.dumps([1]), "記事辞書"),
            ("", "空"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ニュース.json"
            for source, message in invalid_sources:
                with self.subTest(source=source):
                    path.write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_export(path)

    def test_missing_article_fields_are_retained_for_audit(self) -> None:
        """欠損記事を読込時に消さず、選択時の除外理由として残す。

        引数:
            self (NewsExportTests): このテストケース。
        返り値:
            None: なし。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ニュース.json"
            path.write_text('{"articles": [{}]}', encoding="utf-8")
            records = load_export(path)
            self.assertEqual(records, [{}])
            selected, excluded = select_articles(records, datetime(2026, 10, 1, tzinfo=timezone.utc))
            self.assertEqual(selected, [])
            self.assertEqual(excluded[0]["code"], "missing_required_field")
