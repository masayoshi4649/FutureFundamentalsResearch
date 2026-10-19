"""ニュースの版履歴を保ち、予測時点で利用できた記事を選択する。

記事本文の更新履歴が保存されていない場合、過去の本文は復元できない。
また、日時による選別だけでは、予測を行う言語モデル自体が保持する未来の
知識の漏洩は防げない。予測入力の隔離と予測後の結果参照を別途管理する。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_datetime(value: Any, field: str) -> datetime:
    """タイムゾーンを持つ日時を解析し、協定世界時に統一する。

    引数:
        value (datetime | str): 日時、または ISO 8601 形式の日時文字列。
        field (str): エラーメッセージに含める項目名。
    返り値:
        datetime: タイムゾーンを持つ協定世界時の日時。
    """
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} にはタイムゾーンが必要です。")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} はタイムゾーン付きの有効な日時で指定してください。"
        ) from exc


def _iso(value: datetime) -> str:
    """協定世界時の日時を一定の文字列表現に変換する。

    引数:
        value (datetime): タイムゾーンを持つ日時。
    返り値:
        str: 協定世界時を末尾の Z で示した ISO 8601 文字列。
    """
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_text(value: Any) -> str:
    """ハッシュ生成に使用する一定形式の JSON を生成する。

    引数:
        value (Any): JSON に変換可能な値。
    返り値:
        str: キー順、空白、文字コード表現を統一した JSON 文字列。
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rejection(index: int, record: Any, code: str, reason: str) -> dict:
    """入力位置と記事識別情報を含む除外理由を生成する。

    引数:
        index (int): 入力配列内の位置。
        record (Any): 検査対象の入力値。
        code (str): 機械処理用の除外理由識別子。
        reason (str): 日本語の除外理由。
    返り値:
        dict: 入力位置、除外理由、利用可能な識別情報。
    """
    result = {"record_index": index, "code": code, "reason": reason}
    if isinstance(record, dict):
        for field in ("id", "revision_id", "canonical_url"):
            value = record.get(field)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                result[field] = value
    return result


# ----------------------------------------


def select_articles(records: list[dict], as_of: datetime) -> tuple[list[dict], list[dict]]:
    """予測時点で取得済みの有効な記事から、重複を除いた最新版を選択する。

    引数:
        records (list[dict]): NewsStore の検索・記事取得結果の全版の記事。
            id、revision_id、fetched_at、canonical_url、headline を必須とする。
        as_of (datetime): タイムゾーン付きの予測基準日時。
    返り値:
        tuple[list[dict], list[dict]]: 選択記事と、入力位置・識別情報・理由を
            含む除外一覧。記事の日時は協定世界時の ISO 8601 文字列に統一する。
            available_at は公表・取得・更新日時の最大値とする。

    published_at が欠損している場合は fetched_at を公表時点の代用とする。
    存在する公表・取得・更新日時のすべてが予測基準日時以下であることを
    要求する。同じ id の複数版は available_at が最新のものを選択し、
    同時刻の場合は更新日時、revision_id、記事 JSON の順で決定する。
    異なる id でも同じ canonical_url の記事は最新の一件にまとめる。
    入力値は変更せず、除外した旧版についても識別情報と理由を返す。

    内容更新が版履歴として保存されていない場合、過去の本文は復元できない。
    日時フィルタだけでは言語モデルの学習済み知識による未来の漏洩は防げない。
    """
    if not isinstance(records, list):
        raise TypeError("記事データは配列で指定してください。")
    if not isinstance(as_of, datetime):
        raise ValueError("予測基準日時はタイムゾーン付きの datetime で指定してください。")
    cutoff = _parse_datetime(as_of, "予測基準日時")
    candidates = []
    excluded = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            excluded.append(_rejection(index, record, "invalid_record", "記事が辞書ではありません。"))
            continue
        invalid_fields = []
        for field in ("id", "revision_id"):
            value = record.get(field)
            if (
                not isinstance(value, (str, int))
                or isinstance(value, bool)
                or (isinstance(value, str) and not value.strip())
            ):
                invalid_fields.append(field)
        for field in ("canonical_url", "headline"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                invalid_fields.append(field)
        if record.get("fetched_at") is None:
            invalid_fields.append("fetched_at")
        if invalid_fields:
            excluded.append(_rejection(
                index, record, "missing_required_field",
                "必須項目が欠損しているか、型が不正です: " + ", ".join(invalid_fields),
            ))
            continue

        # ----------------------------------------

        times = {}
        try:
            for field in ("published_at", "fetched_at", "updated_at"):
                if record.get(field) is not None:
                    times[field] = _parse_datetime(record[field], field)
        except ValueError as exc:
            excluded.append(_rejection(index, record, "invalid_timestamp", str(exc)))
            continue
        future_fields = [field for field, value in times.items() if value > cutoff]
        if future_fields:
            excluded.append(_rejection(
                index, record, "future_timestamp",
                "予測基準日時より後の日時を含みます: " + ", ".join(future_fields),
            ))
            continue
        available_at = max(times.values())
        article = dict(record)
        article["canonical_url"] = article["canonical_url"].strip()
        article.update({field: _iso(value) for field, value in times.items()})
        article["available_at"] = _iso(available_at)
        try:
            canonical_article = _json_text(article)
        except (TypeError, ValueError):
            excluded.append(_rejection(
                index, record, "invalid_json_value",
                "記事に JSON として保存できない値が含まれます。",
            ))
            continue
        effective_update = times.get("updated_at", times.get("published_at", times["fetched_at"]))
        order = (available_at, effective_update, str(article["revision_id"]), canonical_article)
        candidates.append((order, index, article))

    # ----------------------------------------

    by_id = {}
    for order, index, article in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        article_id = str(article["id"]).strip()
        if article_id in by_id:
            selected = by_id[article_id][2]
            same_revision = str(selected["revision_id"]) == str(article["revision_id"])
            excluded.append(_rejection(
                index, article,
                "duplicate_revision" if same_revision else "superseded_revision",
                "同じ記事版の重複です。" if same_revision else "予測時点で利用可能な、より新しい記事版があります。",
            ))
        else:
            by_id[article_id] = (order, index, article)

    selected = []
    seen_urls = set()
    for order, index, article in sorted(by_id.values(), key=lambda candidate: candidate[0], reverse=True):
        url = article["canonical_url"]
        if url in seen_urls:
            excluded.append(_rejection(
                index, article, "duplicate_url", "同じ canonical_url の記事がすでに選択されています。",
            ))
        else:
            selected.append(article)
            seen_urls.add(url)
    selected.sort(key=lambda article: (
        datetime.fromisoformat(article["available_at"]), str(article["id"]), str(article["revision_id"]),
    ))
    excluded.sort(key=lambda rejection: rejection["record_index"])
    return selected, excluded


def snapshot_payload(records: list[dict], as_of: datetime) -> dict:
    """予測入力の選択結果と除外理由を、検証可能なハッシュ付きで生成する。

    引数:
        records (list[dict]): 取得した全版の記事データ。
        as_of (datetime): タイムゾーン付きの予測基準日時。
    返り値:
        dict: as_of、articles、excluded、canonical_hash を持つスナップショット。
            ハッシュは canonical_hash 自体を除いた UTF-8 の一定形式 JSON の
            SHA-256。配列の順序と除外元の入力位置も証跡に含める。

    このハッシュは入力の同一性を検査するものであり、記事の真正性や完全性を
    保証しない。保存されていない本文の旧版は復元できず、日時による選別だけ
    では言語モデルの学習済み知識からの未来の漏洩を防ぐこともできない。
    """
    articles, excluded = select_articles(records, as_of)
    payload = {"as_of": _iso(as_of), "articles": articles, "excluded": excluded}
    payload["canonical_hash"] = hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()
    return payload


# ----------------------------------------


def _unpack_export(value: Any, location: str) -> list[dict]:
    """記事配列または MCP 応答の入れ子から記事辞書を取り出す。

    引数:
        value (Any): JSON として読み込んだ記事または応答。
        location (str): エラー位置を示す日本語の説明。
    返り値:
        list[dict]: 版を統合せずに保持した記事辞書の一覧。
    """
    if isinstance(value, list):
        articles = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{location} の {index + 1} 件目は記事辞書ではありません。")
            articles.extend(_unpack_export(item, f"{location} の {index + 1} 件目"))
        return articles
    if not isinstance(value, dict):
        raise ValueError(f"{location} は記事辞書または記事配列である必要があります。")
    if "structuredContent" in value and value["structuredContent"] is not None:
        return _unpack_export(value["structuredContent"], location + " の構造化データ")
    if "articles" in value:
        if not isinstance(value["articles"], list):
            raise ValueError(f"{location} の articles は配列である必要があります。")
        return _unpack_export(value["articles"], location + " の articles")
    if "article" in value:
        if not isinstance(value["article"], dict):
            raise ValueError(f"{location} の article は辞書である必要があります。")
        return [value["article"]]
    if "content" in value and "id" not in value:
        if not isinstance(value["content"], list):
            raise ValueError(f"{location} の content は配列である必要があります。")
        articles = []
        has_text = False
        for index, block in enumerate(value["content"]):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            has_text = True
            try:
                parsed = json.loads(block.get("text", ""))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{location} の {index + 1} 個目の本文は有効な JSON ではありません。") from exc
            articles.extend(_unpack_export(parsed, location + " の本文"))
        if not has_text:
            raise ValueError(f"{location} に JSON 形式のテキスト本文がありません。")
        return articles
    for wrapper in ("result", "data"):
        if wrapper in value and "id" not in value:
            return _unpack_export(value[wrapper], location + " の " + wrapper)
    return [value]


def load_export(path: Path) -> list[dict]:
    """書き出したニュースの JSON または JSONL を統一した記事配列で読み込む。

    引数:
        path (pathlib.Path): UTF-8 のニュース書き出しファイルまたは版別原本フォルダ。
    返り値:
        list[dict]: 全記事版を保持した記事辞書の一覧。MCP の structuredContent、
            content 内の JSON 本文、articles 配列、article 辞書、直接の記事配列、
            JSONL に対応する。構造化データがあれば本文より優先する。

    不正な JSON、未対応の構造、読込失敗は日本語の ValueError として通知する。
    記事の必須項目や日時の不備は、この段階で黙って除去せず選択時に記録する。
    """
    if Path(path).is_dir():
        records = []
        for source_path in sorted(Path(path).glob("*.json")) + sorted(Path(path).glob("*.jsonl")):
            records.extend(load_export(source_path))
        return records
    try:
        source = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, TypeError) as exc:
        raise ValueError(f"ニュースファイルを UTF-8 で読み込めません: {path}") from exc
    if not source.strip():
        raise ValueError("ニュースファイルが空です。記事がない場合は空配列を指定してください。")
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError:
        articles = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed_line = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ニュースファイルの {line_number} 行目は有効な JSON ではありません。") from exc
            articles.extend(_unpack_export(parsed_line, f"ニュースファイルの {line_number} 行目"))
        return articles
    return _unpack_export(parsed, "ニュースファイル")
