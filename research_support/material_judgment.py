"""明示された材料判断を、既存入口の判断JSONと機械的な出典記録へ組み立てる。

点数・方向・根拠・利用区間は呼出側が決める。本文を読んだという自己申告やactorは作らない。
実行時に読むファイルはarticle_usesで明示された取得原本だけで、価格ファイルは開かない。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from fundamentals.common import digest, timestamp
from research_support.material_points import HORIZONS, POINT_KEYS, SYMBOLS, build_material_ledger


CLOCKS = ("published_at", "fetched_at", "effective_at", "updated_at", "available_at", "retrieved_at")


def point_judgment(*, impact_pt: float | None, priced_in_before_reference_pt: float | None,
                   priced_in_after_reference_pt: float | None, impact_reason: str, before_reason: str,
                   after_reason: str, transmission: str, before_observations: list[dict],
                   after_observations: list[dict]) -> dict:
    """明示された三点・三根拠・前後の観測を既存の材料点形式へ写す。

    引数: impact_pt/priced_in_before_reference_pt/priced_in_after_reference_pt (float | None) は主観点または不明、
        impact_reason/before_reason/after_reason/transmission (str) は根拠と伝達経路、
        before_observations/after_observations (list[dict]) は基準前後の明示観測。
    返り値: dict は未校正の点入力。方向・点数・観測を補完せず、台帳の検査は組立時に行う。
    """
    return {"impact_pt": impact_pt, "priced_in_before_reference_pt": priced_in_before_reference_pt,
            "priced_in_after_reference_pt": priced_in_after_reference_pt,
            "evidence": dict(zip(POINT_KEYS, (impact_reason, before_reason, after_reason))),
            "transmission": transmission,
            "observations": {"before_reference": copy.deepcopy(before_observations),
                             "after_reference": copy.deepcopy(after_observations)}}


def _identity(row: dict) -> tuple[str, str]:
    """記事指定の識別子と版を照合用文字列へそろえる。

    引数: row (dict) は記事または明示的な利用指定。
    返り値: tuple[str, str] は記事IDと版。欠損・空文字・真偽値は拒否する。
    """
    if not isinstance(row, dict):
        raise ValueError("記事と利用指定は辞書が必要です。")
    result = []
    for key in ("id", "revision_id"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError("記事IDとrevision_idを明示してください。")
        result.append(str(value))
    return tuple(result)


def _scope(value: object, text: str) -> dict:
    """利用する非空の本文文字区間を検査し、全文の自動選択を避ける。

    引数: value (object) は始点・排他的終点・説明、text (str) は取得原本または既知抜粋。
    返り値: dict は検査済み区間。Python文字列の文字位置で指定する。
    """
    if not isinstance(value, dict) or set(value) != {"start", "end_exclusive", "notice"}:
        raise ValueError("scopeにはstart・end_exclusive・noticeを明示してください。")
    start, end, notice = value["start"], value["end_exclusive"], value["notice"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or not isinstance(notice, str) or not notice.strip()):
        raise ValueError("scopeは本文内の非空区間と説明が必要です。")
    return copy.deepcopy(value)


def _article(source: dict, use: dict, cutoff: str) -> tuple[dict, dict]:
    """指定版のメタデータを保持し、明示区間だけを本文・原本SHAへ束縛する。

    引数: source (dict) はpacketの元記事、use (dict) はID・版・modeと本文指定、cutoff (str) は締切。
    返り値: tuple[dict, dict] は判断用記事とファイル・区間を照合した事実記録。読了を意味しない。
    """
    mode = use.get("mode")
    if mode == "none":
        raise ValueError("本文なしの記事は選択せず別記してください。選択記事にはraw/excerptと明示scopeが必要です。")
    if mode not in ("raw", "excerpt"):
        raise ValueError("記事利用はmode=raw/excerptを明示してください。")
    allowed = {"id", "revision_id", "mode"}
    allowed.add("scope")
    if mode == "raw":
        allowed.update(("raw_path", "raw_sha256"))
    if set(use) - allowed:
        raise ValueError("記事利用指定に未対応の項目があります。")
    for key in CLOCKS:
        if source.get(key) is not None and timestamp(source[key]) > timestamp(cutoff):
            raise ValueError("締切後の記事時計は使えません。")
    article = copy.deepcopy(source)
    record = {"id": source["id"], "revision_id": source["revision_id"], "mode": mode,
              "canonical_url": source["canonical_url"], "packet_article_hash": digest(source),
              "clocks": {key: source[key] for key in CLOCKS if source.get(key) is not None},
              "body_raw_path": None, "body_raw_sha256": None, "body_scope": None, "body_sha256": None}
    raw_path, raw_hash, raw_clocks = None, None, None
    if mode == "raw":
        value = use.get("raw_path")
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("rawにはraw_pathが必要です。")
        path = Path(value).resolve()
        content = path.read_bytes()
        raw_hash = hashlib.sha256(content).hexdigest()
        if "raw_sha256" in use and use["raw_sha256"] != raw_hash:
            raise ValueError("取得原本の指定SHA256が一致しません。")
        raw = json.loads(content.decode("utf-8-sig"))["structuredContent"]["article"]
        if (_identity(raw) != _identity(source) or raw["revision_id"] != source["revision_id"]):
            raise ValueError("取得原本の記事ID・版がpacketと一致しません。")
        if raw.get("body_offset", 0) != 0:
            raise ValueError("body_offsetが0でない取得原本は主文区間を照合できません。")
        for key in ("published_at", "fetched_at"):
            if raw.get(key) is None:
                raise ValueError("取得原本には公表・取得時計が必要です。")
        raw_clocks = {key: raw[key] for key in CLOCKS if raw.get(key) is not None}
        for key, value in raw_clocks.items():
            if timestamp(value) > timestamp(cutoff):
                raise ValueError("締切後の取得本文時計は使えません。")
            if source.get(key) is not None and timestamp(value) != timestamp(source[key]):
                raise ValueError("取得原本の記事時計がpacketと一致しません。")
        if raw.get("canonical_url") is not None and raw["canonical_url"] != source["canonical_url"]:
            raise ValueError("取得原本の記事URLがpacketと一致しません。")
        text = raw.get("body_text")
        raw_path, body_source = str(path), "明示指定の取得原本本文"
    else:
        text, body_source = source.get("body_excerpt"), "packetの既知抜粋"
    if not isinstance(text, str):
        raise ValueError("指定した出典に本文文字列がありません。")
    scope = _scope(use.get("scope"), text)
    body = text[scope["start"]:scope["end_exclusive"]]
    if "body" in source and body != source["body"]:
        raise ValueError("packetに収録済みの本文は変更できません。")
    article.update(body=body, body_source=body_source, body_raw_path=raw_path, body_raw_sha256=raw_hash,
                   body_scope=scope, body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest())
    record.update({key: copy.deepcopy(article[key]) for key in
                   ("body_raw_path", "body_raw_sha256", "body_scope", "body_sha256")})
    if raw_clocks is not None:
        record["raw_clocks"] = raw_clocks
    return article, record


def _periods(rows: dict) -> dict:
    """両先物の当日欄を必須にし、未作成の長期欄だけをNoneへそろえる。

    引数: rows (dict) は先物別・期間別の点または回答。
    返り値: dict は4期間を持つ複製。未知の当日点や片方の先物を補完しない。
    """
    if not isinstance(rows, dict) or set(rows) != set(SYMBOLS):
        raise ValueError("NK225とTOPIXを別々に明示してください。")
    result = {}
    for symbol in SYMBOLS:
        row = rows[symbol]
        if not isinstance(row, dict) or "today" not in row or set(row) - set(HORIZONS):
            raise ValueError("当日欄は必須です。期間はtoday/week/month/quarterだけを指定してください。")
        result[symbol] = {horizon: copy.deepcopy(row.get(horizon)) for horizon in HORIZONS}
    return result


def assemble_material_judgment(packet: dict, *, article_uses: list[dict], reference_details: dict,
                               materials: list[dict], forecasts: dict) -> dict:
    """明示判断を既存material_reportの判断JSONと出典の事実記録へ組み立てる。

    引数: packet (dict) は自己ハッシュ付き時点入力、article_uses (list[dict]) は記事ID・版と明示本文指定、
        reference_details (dict) は両先物のproxy (bool) とsource (str)、materials (list[dict]) は呼出側の材料判断、
        forecasts (dict) は呼出側の先物別回答。当日欄は必須で、未作成の長期だけ省略できる。
    返り値: dict はjudgment、source_records、noticeを持つ。actorや読了申告は作らない。
        本文なしは選択記事へ含めず別記する。選択記事にはraw/excerptと明示scopeが必要。
    """
    if not isinstance(packet, dict) or packet.get("packet_hash") != digest({
            key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("packetの自己ハッシュが一致しません。")
    if not isinstance(reference_details, dict) or set(reference_details) != set(SYMBOLS):
        raise ValueError("両先物のreference_detailsが必要です。")
    references = {}
    for symbol in SYMBOLS:
        source = packet.get("instruments", {}).get(symbol, {}).get("reference")
        details = reference_details[symbol]
        if not isinstance(source, dict):
            raise ValueError("P0不足の先物は数値を補完できません。")
        if (not isinstance(details, dict) or set(details) != {"proxy", "source"}
                or type(details["proxy"]) is not bool
                or not isinstance(details["source"], str) or not details["source"].strip()):
            raise ValueError("reference_detailsにはproxy真偽値と出典sourceだけを明示してください。")
        references[symbol] = {key: copy.deepcopy(source[key]) for key in ("price", "time", "session", "contract")}
        references[symbol].update(symbol=symbol, **copy.deepcopy(details))
        references[symbol]["contract_identity_status"] = "未確認" if source["contract"] is None else "確認済み"
    original = packet.get("news")
    if not isinstance(original, list) or not isinstance(article_uses, list):
        raise ValueError("packet記事とarticle_usesは配列が必要です。")
    visible = {_identity(row): row for row in original}
    if len(visible) != len(original):
        raise ValueError("packetの記事ID・版が重複しています。")
    articles, records, used = [], [], set()
    for use in article_uses:
        identity = _identity(use)
        if identity not in visible or identity in used:
            raise ValueError("packetにない記事版または重複した利用指定です。")
        used.add(identity)
        article, record = _article(visible[identity], use, packet["as_of"])
        articles.append(article)
        records.append(record)

    # ----------------------------------------

    if not isinstance(materials, list):
        raise ValueError("materialsは明示判断の配列が必要です。")
    judgments = copy.deepcopy(materials)
    for material in judgments:
        material["impacts"] = _periods(material["impacts"])
    judgment = {"as_of": packet["as_of"], "references": references, "articles": articles,
                "materials": judgments, "forecasts": _periods(forecasts)}
    build_material_ledger(**judgment, source_packet=packet)
    facts = {"packet_hash": packet["packet_hash"], "source_packet_hash": digest(packet),
             "as_of": packet["as_of"], "judgment_hash": digest(judgment),
             "references": copy.deepcopy(references), "articles": records,
             "notice": "ファイル・記事版・時計・明示区間の機械照合記録。読了や結果未閲覧の自己申告ではない。"}
    facts["record_hash"] = digest(facts)
    return {"judgment": judgment, "source_records": facts,
            "notice": "material_reportの判断入力へ保存可能。actorは別途明示入力が必要。本文なしの記事は選択せず別記する。"}
