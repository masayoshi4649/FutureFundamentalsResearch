"""切出済みニュースの主観的な材料点と既反映量を検査し、予測の根拠を別に固定する。

点数は未校正の注釈であり、価格変化率・確率・因果効果ではない。
記事の意味上の重複、注釈者の未来知識、根拠の正しさは日時検査だけでは保証できない。
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from datetime import timedelta
from decimal import Decimal

from fundamentals.common import digest, timestamp
from fundamentals.engine import session_plan
from research_support.five_class import CLASSES


SYMBOLS = ("NK225", "TOPIX")
HORIZONS = ("today", "week", "month", "quarter")
HORIZON_NAMES = {"today": "当日", "week": "1週間", "month": "1か月", "quarter": "3か月"}
POINT_KEYS = ("impact_pt", "priced_in_before_reference_pt", "priced_in_after_reference_pt")
COMPUTED_KEYS = ("priced_in_total_pt", "reference_remaining_pt", "residual_pt")
VERSION = "材料点台帳１"
POINT_NOTICE = "未校正の主観点。％リターン・確率・勝率へ変換しない。点数の符号から五分類を自動決定しない。"


def _text(value: object, name: str) -> str:
    """必須の根拠文または識別子を検査する。

    引数: value (object) は候補、name (str) は項目名。
    返り値: str は空白だけではない文字列。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + "には空でない文字列が必要です。")
    return value


def _number(value: object, name: str, unknown: bool = True) -> Decimal | None:
    """有限な点数を十進数で検査し、不明を数値ゼロへ補完しない。

    引数: value (object) は数値またはNone、name (str) は項目名、unknown (bool) はNoneを許す指定。
    返り値: Decimal | None は検査済みの点数。
    """
    if value is None and unknown:
        return None
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(name + "は有限数値または明示的な不明Noneが必要です。")
    return Decimal(str(value))


def _float(value: Decimal | None) -> float | None:
    """計算した十進数をJSON数値に変換し、桁あふれを拒否する。

    引数: value (Decimal | None) は計算済みの点数。
    返り値: float | None は有限数値または不明。
    """
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("点数の演算結果が有限範囲を超えています。")
    return result


def _identity(record: dict) -> tuple[str, str]:
    """根拠記事の識別子と指定版を同じ形式へそろえる。

    引数: record (dict) は記事または記事版への参照。
    返り値: tuple[str, str] は記事識別子と改訂識別子。
    """
    if not isinstance(record, dict):
        raise ValueError("記事または記事参照は辞書が必要です。")
    values = []
    for key in ("id", "revision_id"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError("記事にはidとrevision_idが必要です。")
        values.append(str(value))
    return tuple(values)


def _articles(articles: list[dict], cutoff: str) -> tuple[list[dict], set[tuple[str, str]]]:
    """全記事の公表・取得・有効時計を先に検査し、旧版や同一URLの再投入を拒否する。

    引数: articles (list[dict]) は切出済み記事、cutoff (str) は情報締切。
    返り値: tuple[list[dict], set[tuple[str, str]]] は来歴と利用できる記事版の集合。
    """
    if not isinstance(articles, list):
        raise ValueError("記事は配列で指定してください。")
    deadline, identities, urls, ids, rows = timestamp(cutoff), set(), set(), set(), []
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("各記事は辞書で指定してください。")
        identity, url = _identity(article), _text(article.get("canonical_url"), "記事URL")
        clocks = {key: timestamp(article[key]) for key in ("published_at", "fetched_at", "effective_at")}
        for key in ("updated_at", "available_at"):
            if article.get(key) is not None:
                clocks[key] = timestamp(article[key])
        if any(clock > deadline for clock in clocks.values()):
            raise ValueError("情報締切後の公表・取得・有効版の記事は使えません。")
        if clocks["effective_at"] < clocks["published_at"]:
            raise ValueError("記事版の有効時計を公表時計より前へ繰り上げられません。")
        if identity in identities or identity[0] in ids or url in urls:
            raise ValueError("同じ記事・複数改訂・同一URLの再投入はできません。")
        identities.add(identity)
        ids.add(identity[0])
        urls.add(url)
        rows.append({"id": identity[0], "revision_id": identity[1], "canonical_url": url,
                     **{key: value.isoformat() for key, value in clocks.items()}, "source_hash": digest(article)})
        rows[-1].update({key: copy.deepcopy(article[key]) for key in (
            "body_source", "body_raw_path", "body_raw_sha256", "body_scope", "body_sha256") if key in article})
    return rows, identities


def _references(references: dict, cutoff: str) -> dict:
    """両先物の基準価格・限月・基準時計を検査する。

    引数: references (dict) は先物別P0、cutoff (str) は情報締切。
    返り値: dict は基準値を変えずに複製した来歴。休日暦から直前取引日を推測せず、原値との突合は別に行う。
    """
    if not isinstance(references, dict) or set(references) != set(SYMBOLS):
        raise ValueError("NK225とTOPIXの同じ先物基準が必要です。")
    result, plan = {}, session_plan(timestamp(cutoff))
    for symbol in SYMBOLS:
        row = references[symbol]
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise ValueError("基準価格の先物と辞書キーが一致しません。")
        if row.get("contract") is None:
            if row.get("contract_identity_status") != "未確認":
                raise ValueError("限月不明にはcontract=Noneとcontract_identity_status=未確認の明示が必要です。")
            identity_status = "未確認"
        else:
            _text(row["contract"], "基準の限月")
            if row.get("contract_identity_status", "確認済み") != "確認済み":
                raise ValueError("判明した限月の確認状態が不整合です。")
            identity_status = "確認済み"
        _text(row.get("source"), "基準の価格原本")
        price = _number(row.get("price"), "基準価格", False)
        if price <= 0 or type(row.get("proxy")) is not bool:
            raise ValueError("正の基準価格と終値代理かどうかの明示が必要です。")
        session = row.get("session")
        if not isinstance(session, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}_(?:Day|Night)", session):
            raise ValueError("基準セッションの形式が不正です。")
        date, kind = session.split("_")
        expected_kind = "Night" if plan["kind"] == "Day" else "Day"
        if kind != expected_kind or (plan["kind"] == "Night" and date != timestamp(cutoff).date().isoformat()):
            raise ValueError("朝のP0は直前ナイト、夜間のP0は当日デイである必要があります。")
        start = timestamp(date + ("T08:45:00+09:00" if kind == "Day" else "T17:00:00+09:00"))
        end = timestamp(date + ("T15:45:00+09:00" if kind == "Day" else "T06:00:00+09:00"))
        if kind == "Night":
            end += timedelta(days=1)
        observed = timestamp(row["time"])
        if not start <= observed <= end <= timestamp(cutoff):
            raise ValueError("基準時計が指定セッションまたは情報締切と整合しません。")
        if end - observed > timedelta(minutes=10):
            raise ValueError("基準の終値代理は公式終点から10分以内の観測が必要です。")
        if timestamp(cutoff) - end > timedelta(days=4):
            raise ValueError("基準が4日より古く、直前セッションを確認できません。")
        if not row["proxy"] and observed != end:
            raise ValueError("終値代理ではない価格は公式終点の時計が必要です。")
        result[symbol] = {key: copy.deepcopy(row[key]) for key in ("symbol", "contract", "price", "time", "session", "proxy", "source")}
        result[symbol]["contract_identity_status"] = identity_status
    return result


def _body_attachment(article: dict, source: dict) -> None:
    """追加本文をUTF-8ハッシュと取得原本来歴、または元抜粋の厳密区間で検査する。

    引数: article/source (dict) は本文付き記事と切出済み元記事。
    返り値: None。原本ファイルそのもののSHA・本文区間照合は保存担当が別に行う。
    """
    body = _text(article["body"], "追加取得した本文")
    _text(article.get("body_source"), "追加本文のbody_source")
    scope = article.get("body_scope")
    if isinstance(scope, dict):
        start, end = scope.get("start"), scope.get("end_exclusive")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or end - start != len(body):
            raise ValueError("本文の構造化範囲には非負整数の始点・終点と本文文字数の一致が必要です。")
        _text(scope.get("notice"), "本文範囲の説明")
    else:
        _text(scope, "追加本文のbody_scope")
    if not isinstance(article.get("body_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", article["body_sha256"]):
        raise ValueError("追加本文には本文のSHA256来歴が必要です。")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != article["body_sha256"]:
        raise ValueError("追加本文のUTF-8ハッシュが一致しません。")
    if "body_raw_path" not in article or "body_raw_sha256" not in article:
        raise ValueError("本文原本のパスとSHAは未使用でも明示的なNoneが必要です。")
    raw_path, raw_hash = article["body_raw_path"], article["body_raw_sha256"]
    if raw_path is None and raw_hash is None:
        excerpt = source.get("body_excerpt")
        if (not isinstance(scope, dict) or not isinstance(excerpt, str)
                or scope["end_exclusive"] > len(excerpt) or excerpt[scope["start"]:scope["end_exclusive"]] != body):
            raise ValueError("原本取得なしの本文は、元抜粋の明示区間と完全一致する必要があります。")
    else:
        _text(raw_path, "追加本文のbody_raw_path")
        if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
            raise ValueError("別取得した本文には取得原本のSHA256来歴が必要です。")


def _bind_packet(packet: dict | None, cutoff: str, references: dict, articles: list[dict]) -> dict:
    """切出入力が渡された場合だけ、P0原値と指定記事版を自己ハッシュ付き原本へ束縛する。

    引数: packet (dict | None) は時点切出、cutoff (str) は締切、references (dict) は検査済みP0、
        articles (list[dict]) は使用記事版。
    返り値: dict は束縛先と照合状態。未提供時はP0原値との突合未実行を明示する。
    """
    if packet is None:
        return {"source_packet_hash": None, "packet_hash": None, "status": "P0原値との突合未実行"}
    if not isinstance(packet, dict) or packet.get("packet_hash") != digest({key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("切出入力の自己ハッシュが一致しません。")
    plan = session_plan(timestamp(cutoff))
    if timestamp(packet["as_of"]) != timestamp(cutoff) or packet.get("session") != plan["session"]:
        raise ValueError("材料台帳と切出入力の締切または対象場が一致しません。")
    instruments = packet.get("instruments")
    if not isinstance(instruments, dict) or set(instruments) != set(SYMBOLS):
        raise ValueError("切出入力には両先物の基準が必要です。")
    for symbol in SYMBOLS:
        if not isinstance(instruments[symbol], dict):
            raise ValueError("切出入力の各先物は辞書が必要です。")
        source, reference = instruments[symbol].get("reference"), references[symbol]
        if (not isinstance(source, dict) or not {"price", "time", "session", "contract"} <= set(source)
                or any(source[key] != reference[key] for key in ("price", "session", "contract"))
                or timestamp(source["time"]) != timestamp(reference["time"])):
            raise ValueError("P0の原値・時刻・セッション・限月が切出入力と一致しません。")
    original = packet.get("news")
    if not isinstance(original, list) or len({_identity(row) for row in original}) != len(original):
        raise ValueError("切出入力の記事版が重複または欠損しています。")
    visible = {_identity(row): row for row in original}
    clocks = ("published_at", "fetched_at", "effective_at", "updated_at", "available_at", "retrieved_at")
    for article in articles:
        identity = _identity(article)
        if identity not in visible:
            raise ValueError("切出入力に存在しない記事版を材料根拠へ追加できません。")
        source = visible[identity]
        for key in ("canonical_url", "headline", "body_excerpt", "body"):
            if key in source and article.get(key) != source[key]:
                raise ValueError("根拠記事の本文・見出し・URLが切出入力と一致しません。")
            if key in article and key not in source and key != "body":
                raise ValueError("切出入力にない本文・見出し・URLを追加できません。")
        if "body" in article and "body" not in source:
            _body_attachment(article, source)
        for key in clocks:
            if source.get(key) is not None:
                if article.get(key) is None or timestamp(article[key]) != timestamp(source[key]):
                    raise ValueError("根拠記事の元時計を変更・削除できません。")
            elif article.get(key) is not None:
                raise ValueError("切出入力にない記事時計を補完できません。")
    return {"source_packet_hash": digest(packet), "packet_hash": packet["packet_hash"], "status": "P0原値・使用記事版の切出入力照合済み",
            "body_notice": "追加本文のUTF-8ハッシュを照合。抜粋由来は元抜粋の区間一致も必須。別取得の原本ファイルSHA・本文区間は別途照合が必要。"}


def _observation(row: dict, reference: dict, cutoff: str, before: bool) -> dict:
    """既反映点の同先物観測を基準前と基準後に分離して検査する。

    引数: row/reference (dict) は価格観測と基準、cutoff (str) は締切、before (bool) はP0以前の指定。
    返り値: dict は同先物観測と限月照合の状態。限月不明同士を同限月確認済みにしない。
    """
    if not isinstance(row, dict) or row.get("symbol") != reference["symbol"] or "contract" not in row:
        raise ValueError("既反映の観測は基準と同じ先物と限月情報の明示が必要です。")
    contract, base_contract = row["contract"], reference["contract"]
    if contract is not None:
        _text(contract, "観測の限月")
    if contract is not None and base_contract is not None and contract != base_contract:
        raise ValueError("既反映の観測の判明した限月が基準と一致しません。")
    identity_status = "未確認" if contract is None or base_contract is None else "確認済み"
    if identity_status == "未確認" and row.get("contract_identity_status") != "未確認":
        raise ValueError("観測の限月照合が未確認である旨の明示が必要です。")
    start, end, base = timestamp(row["start"]), timestamp(row["end"]), timestamp(reference["time"])
    if not start <= end <= timestamp(cutoff) or (before and end > base) or (not before and start < base):
        raise ValueError("P0以前とP0後asofまでの観測をまたいで使えません。")
    for key in ("start_price", "end_price"):
        if _number(row.get(key), "観測価格", False) <= 0:
            raise ValueError("観測価格は正の数値が必要です。")
    _text(row.get("source"), "価格観測の原本")
    result = copy.deepcopy(row)
    result["contract_identity_status"] = identity_status
    return result


def _context(material: dict, cutoff: str) -> tuple[dict, list[dict]]:
    """期待・実績・初報・外部反応・後続予定を、点数計算へ混ぜずに保存する。

    引数: material (dict) は材料、cutoff (str) は情報締切。
    返り値: tuple[dict, list[dict]] は明示的な不明を含む事実欄と外部観測。
    """
    facts = copy.deepcopy(material.get("news_facts", {}))
    allowed = {"prior_expectation", "actual", "surprise", "first_public_at", "retrieved_at", "next_events"}
    if not isinstance(facts, dict) or set(facts) - allowed:
        raise ValueError("ニュース事実欄の項目が不正です。")
    for key in allowed:
        facts.setdefault(key, [] if key == "next_events" else None)
    for key in ("first_public_at", "retrieved_at"):
        if facts[key] is not None and timestamp(facts[key]) > timestamp(cutoff):
            raise ValueError("締切後に判明した材料時計は使えません。")
    if not isinstance(facts["next_events"], list):
        raise ValueError("後続予定は配列が必要です。")
    for planned in facts["next_events"]:
        _text(planned.get("description"), "後続予定")
        _text(planned.get("evidence"), "後続予定の事前根拠")
        timestamp(planned["scheduled_at"])
        if timestamp(planned["known_at"]) > timestamp(cutoff):
            raise ValueError("後続予定そのものを締切までに知り得た根拠が必要です。")
    external = copy.deepcopy(material.get("external_observations", []))
    if not isinstance(external, list):
        raise ValueError("外部観測は配列が必要です。")
    for row in external:
        for key in ("name", "unit", "source"):
            _text(row.get(key), "外部観測の" + key)
        clocks = [timestamp(row["observed_at"])]
        clocks.extend(timestamp(row[key]) for key in ("retrieved_at", "fetched_at", "available_at", "effective_at",
            "updated_at", "published_at", "received_at", "known_at") if row.get(key) is not None)
        if any(clock > timestamp(cutoff) for clock in clocks):
            raise ValueError("締切後の外部観測は使えません。")
        if row.get("value") is not None and isinstance(row["value"], (int, float)):
            _number(row["value"], "外部観測")
    return facts, external


def _points(row: dict, reference: dict, cutoff: str) -> dict:
    """影響T・P0までのB0・P0後の増分ΔBから、二種類の残余を別計算する。

    引数: row (dict) は期間別の点と根拠、reference (dict) は同先物基準、cutoff (str) は締切。
    返り値: dict はT−B0とT−B0−ΔBを分離した点。未知入力を含む差はNone。
    """
    if not isinstance(row, dict) or not set(POINT_KEYS) <= set(row):
        raise ValueError("期間別の影響点・P0以前既反映点・P0後増分点が必要です。")
    evidence = row.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(POINT_KEYS):
        raise ValueError("三つの点数には、それぞれ根拠または不明理由が必要です。")
    for key in POINT_KEYS:
        _text(evidence[key], key + "の根拠")
    _text(row.get("transmission"), "同先物への影響経路")
    observations = row.get("observations")
    if not isinstance(observations, dict) or set(observations) != {"before_reference", "after_reference"}:
        raise ValueError("基準前と基準後の観測欄を明示してください。")
    observed = {}
    for key in ("before_reference", "after_reference"):
        if not isinstance(observations[key], list):
            raise ValueError("価格観測は配列が必要です。")
        observed[key] = [_observation(item, reference, cutoff, key == "before_reference") for item in observations[key]]
    total, before, after = [_number(row[key], key) for key in POINT_KEYS]
    if total is not None and abs(total) > 5:
        raise ValueError("材料影響点Tの共通尺度は−5から＋5です。")
    remaining = total - before if total is not None and before is not None else None
    residual = remaining - after if remaining is not None and after is not None else None
    priced = before + after if before is not None and after is not None else None
    result = {key: _float(value) for key, value in zip(POINT_KEYS, (total, before, after))}
    result.update(priced_in_total_pt=_float(priced), reference_remaining_pt=_float(remaining),
                  residual_pt=_float(residual), evidence=copy.deepcopy(evidence),
                  transmission=row["transmission"], observations=observed)
    result["point_hash"] = digest(result)
    return result


def _aggregate(materials: list[dict], symbol: str, horizon: str) -> dict:
    """有効材料だけを合計し、既知部分小計と全材料を含む合計を区別する。

    引数: materials (list[dict]) は点数台帳、symbol/horizon (str) は先物と期間。
    返り値: dict は未知を残した合計、既知部分、正負の残余小計。
    """
    active = [row for row in materials if row["status"] == "有効"]
    rows = [(row["material_id"], row["impacts"][symbol][horizon]) for row in active]
    totals, subtotals, unknown = {}, {}, {}
    for key in POINT_KEYS + COMPUTED_KEYS:
        known = [_number(row[key], key) for _, row in rows if row is not None and row[key] is not None]
        unknown[key] = [identity for identity, row in rows if row is None or row[key] is None]
        subtotal = _float(sum(known, Decimal(0))) if known else None
        subtotals[key], totals[key] = subtotal, subtotal if rows and not unknown[key] else None
    residuals = [_number(row["residual_pt"], "残余") for _, row in rows if row is not None and row["residual_pt"] is not None]
    return {"active_material_n": len(rows), "totals": totals, "known_subtotals": subtotals,
            "unknown_material_ids": unknown,
            "known_positive_residual_pt": _float(sum((value for value in residuals if value > 0), Decimal(0))) if residuals else None,
            "known_negative_residual_pt": _float(sum((value for value in residuals if value < 0), Decimal(0))) if residuals else None,
            "status": "有効材料なし" if not rows else "未校正・一部不明" if any(unknown.values()) else "未校正・全点既知"}


def _forecast(row: dict | None, materials: list[dict], symbol: str, horizon: str) -> dict | None:
    """五分類を人間・LLMの別判断として検査し、使用した材料点へ束縛する。

    引数: row (dict | None) は別判断、materials (list[dict]) は点数、symbol/horizon (str) は先物と期間。
    返り値: dict | None は根拠付き回答または未算出。符号変換と頻度代替は行わない。
    """
    if row is None:
        if horizon == "today":
            raise ValueError("当日の回答欄は、無回答理由も含めて明示してください。")
        return None
    if not isinstance(row, dict) or row.get("direction") not in (*CLASSES, None):
        raise ValueError("回答は指定五分類または明示的な無回答Noneが必要です。")
    if row.get("author_kind") not in ("人間", "LLM"):
        raise ValueError("五分類は人間またはLLMによる根拠付きの別判断が必要です。")
    for key in ("author", "reason", "point_rationale"):
        _text(row.get(key), "回答の" + key)
    ids = row.get("material_ids")
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("使用材料は重複のない材料識別子配列が必要です。")
    by_id = {item["material_id"]: item for item in materials}
    if any(identity not in by_id or by_id[identity]["status"] != "有効"
           or by_id[identity]["impacts"][symbol][horizon] is None for identity in ids):
        raise ValueError("回答根拠に存在しない・失効した・期間未算出の材料は使えません。")
    if row["direction"] is not None and not ids:
        raise ValueError("材料点を使わない自動回答・頻度代替はできません。")
    result = {key: copy.deepcopy(row[key]) for key in ("direction", "author_kind", "author", "reason", "material_ids", "point_rationale")}
    result["point_basis_hash"] = digest([{identity: by_id[identity]["impacts"][symbol][horizon]} for identity in ids])
    return result


def _contract_quality(references: dict, materials: list[dict]) -> dict:
    """基準または根拠観測の限月不明を、材料台帳の品質欄へ集約する。

    引数: references (dict) は基準、materials (list[dict]) は検査済み材料点。
    返り値: dict は先物別の確認状態。不明同士の一致は確認済みとしない。
    """
    result = {}
    for symbol in SYMBOLS:
        uncertain = references[symbol]["contract_identity_status"] == "未確認"
        for material in materials:
            for point in material["impacts"][symbol].values():
                if point is not None:
                    uncertain = uncertain or any(row["contract_identity_status"] == "未確認"
                        for observations in point["observations"].values() for row in observations)
        result[symbol] = "未確認" if uncertain else "確認済み"
    return result


def build_material_ledger(as_of: str, references: dict, articles: list[dict],
                          materials: list[dict], forecasts: dict, source_packet: dict | None = None) -> dict:
    """ニュース整理・織込み点検・別判断を一つの時点固定台帳へまとめる。

    引数: as_of (str) は時差付き締切、references (dict) は両先物P0、articles (list[dict]) は切出済み記事、
        materials (list[dict]) は根拠付き材料点、forecasts (dict) は先物・期間別の人間またはLLM回答、
        source_packet (dict | None) はP0原値と使用記事版を束縛する切出入力。
    返り値: dict は機械台帳。価格結果での点数推定や頻度fallbackはせず、未算出を保持する。
    """
    cutoff = timestamp(as_of).isoformat()
    article_rows, visible = _articles(articles, cutoff)
    bases = _references(references, cutoff)
    source_binding = _bind_packet(source_packet, cutoff, bases, articles)
    if not isinstance(materials, list):
        raise ValueError("材料は配列で指定してください。")
    checked, identities, stories = [], set(), set()
    for material in materials:
        if not isinstance(material, dict):
            raise ValueError("材料は辞書で指定してください。")
        identity = _text(material.get("material_id"), "材料識別子")
        story = _text(material.get("story_key"), "同一材料を束ねる識別子")
        if identity in identities or story in stories:
            raise ValueError("同じ材料・再報道を別材料として重複計上できません。")
        identities.add(identity)
        stories.add(story)
        _text(material.get("title"), "材料名")
        evidence = material.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(_identity(item) not in visible for item in evidence):
            raise ValueError("材料には締切までに利用できる指定記事版の根拠が必要です。")
        if len({_identity(item) for item in evidence}) != len(evidence):
            raise ValueError("材料内の根拠記事版が重複しています。")
        if material.get("status") not in ("有効", "失効", "反証"):
            raise ValueError("材料の有効・失効・反証状態を明示してください。")
        for key in ("status_reason", "invalidation"):
            _text(material.get(key), "材料の" + key)
        if "expires_at" not in material:
            raise ValueError("材料失効時計は不明でもNoneを明示してください。")
        if (material["expires_at"] is not None and timestamp(material["expires_at"]) <= timestamp(cutoff)
                and material["status"] == "有効"):
            raise ValueError("既に失効した材料を有効として残せません。")
        facts, external = _context(material, cutoff)
        impacts = material.get("impacts")
        if not isinstance(impacts, dict) or set(impacts) != set(SYMBOLS):
            raise ValueError("材料影響はNK225・TOPIXを別々に指定してください。")
        points = {}
        for symbol in SYMBOLS:
            if not isinstance(impacts[symbol], dict) or set(impacts[symbol]) != set(HORIZONS):
                raise ValueError("4期間を別々に指定し、未算出期間はNoneを明示してください。")
            points[symbol] = {}
            for horizon in HORIZONS:
                row = impacts[symbol][horizon]
                if row is None and horizon == "today":
                    raise ValueError("当日の材料点は未知3点を含めて明示してください。")
                points[symbol][horizon] = None if row is None else _points(row, bases[symbol], cutoff)
        checked.append({key: copy.deepcopy(material[key]) for key in (
            "material_id", "story_key", "title", "evidence", "status", "status_reason", "expires_at", "invalidation", "offsets")})
        checked[-1].update(news_facts=facts, external_observations=external, impacts=points)
    for row in checked:
        offsets = row["offsets"]
        if not isinstance(offsets, list):
            raise ValueError("相殺材料は空配列または識別子と理由の配列が必要です。")
        seen = set()
        for offset in offsets:
            if (offset.get("material_id") not in identities or offset["material_id"] == row["material_id"]
                    or offset["material_id"] in seen):
                raise ValueError("相殺先は存在する別材料を重複なく指定してください。")
            seen.add(offset["material_id"])
            _text(offset.get("reason"), "相殺の経路と根拠")

    # ----------------------------------------

    if not isinstance(forecasts, dict) or set(forecasts) != set(SYMBOLS):
        raise ValueError("回答は両先物を別々に指定してください。")
    answers, totals = {}, {}
    for symbol in SYMBOLS:
        if not isinstance(forecasts[symbol], dict) or set(forecasts[symbol]) != set(HORIZONS):
            raise ValueError("4期間の回答を別々に指定してください。")
        answers[symbol] = {horizon: _forecast(forecasts[symbol][horizon], checked, symbol, horizon) for horizon in HORIZONS}
        totals[symbol] = {horizon: _aggregate(checked, symbol, horizon) for horizon in HORIZONS}
    result = {"version": VERSION, "as_of": cutoff, "point_notice": POINT_NOTICE,
              "scale": {"impact_pt_min": -5, "impact_pt_max": 5, "already_priced_signed": True,
                        "reference_remaining": "T−B0", "residual_after_asof": "T−B0−ΔB", "confidence_multiplier": False},
              "references": bases, "articles": article_rows, "materials": checked, "aggregates": totals, "forecasts": answers,
              "source_binding": source_binding, "source_packet_hash": source_binding["source_packet_hash"],
              "quality": {"contract_identity_status": _contract_quality(bases, checked),
                          "reference_snapshot_binding": source_binding["status"],
                          "pricing_assessment_basis": "asofで利用できる根拠からP0以前の既反映を再構成。P0当時の注釈保存を意味しない。"},
              "limitation": "点数は根拠付き主観注釈。意味上の重複や根拠の因果性は機械検査では確定しない。外部反応を国内先物既反映へ自動換算しない。"}
    result["ledger_hash"] = digest(result)
    return result


def _cell(value: object) -> str:
    """表に含める文字列の改行と区切り文字を逃がす。

    引数: value (object) は表の値。
    返り値: str は一行のMarkdown表セル。
    """
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _point_text(value: float | None) -> str:
    """点数と不明を短く表示する。

    引数: value (float | None) は未校正点。
    返り値: str は符号付き点数または不明。
    """
    return "不明" if value is None else f"{value:+.3g}"


def _short_reason(value: object, limit: int = 120) -> str:
    """材料根拠を一行へまとめ、長文は表示上だけ省略する。

    引数: value (object) は全文、limit (int) は省略記号を含めた表示上限。
    返り値: str は短い根拠。元のJSON文字列は変更しない。
    """
    text = _cell(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def render_material_markdown(ledger: dict) -> str:
    """機械台帳から、点・織込み・残余・別回答の短いMarkdown表を作る。

    引数: ledger (dict) はbuild_material_ledgerの自己ハッシュ付き返り値。
    返り値: str は箇条書きと材料表。未校正点を％や確率で表示しない。
    """
    if ledger.get("version") != VERSION or ledger.get("ledger_hash") != digest({key: value for key, value in ledger.items() if key != "ledger_hash"}):
        raise ValueError("材料点台帳の版または自己ハッシュが一致しません。")
    lines = ["# 材料点と織込み", "", "- 情報締切：" + ledger["as_of"], "- " + POINT_NOTICE,
             "- T：影響点、B0：P0以前既反映、ΔB：P0後asofまで同先物の増分。", ""]
    for symbol, reference in ledger["references"].items():
        lines.append(f"- {symbol} {reference['contract'] or '限月不明'}：P0={reference['price']}（{reference['time']}、{'終値代理' if reference['proxy'] else '終値'}、限月{reference['contract_identity_status']}）。")
    lines.append("- B0はasof時点の根拠による再構成。限月未確認の価格を同限月実証済みとは扱わない。")
    lines.append("- 原値束縛：" + ledger["quality"]["reference_snapshot_binding"] + "。")
    if any(value == "未確認" for value in ledger["quality"]["contract_identity_status"].values()):
        lines.append("- 品質不備：基準または材料根拠の価格に限月未確認があり、同限月での反応は実証できない。")
    lines += ["", "| 先物 | 期間 | 材料 | T | B0 | ΔB | 既反映合計 B0＋ΔB | P0起点 T−B0 | asof残余 T−B0−ΔB | 状態 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for material in ledger["materials"]:
        for symbol in SYMBOLS:
            for horizon in HORIZONS:
                row = material["impacts"][symbol][horizon]
                if row is not None:
                    cells = [symbol, HORIZON_NAMES[horizon], material["title"],
                             *[_point_text(row[key]) for key in POINT_KEYS],
                             _point_text(row["priced_in_total_pt"]),
                             _point_text(row["reference_remaining_pt"]), _point_text(row["residual_pt"]), material["status"]]
                    lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines += ["", "- 失効・反証材料は点数をゼロへ書換えず、合計から区別。相殺関係だけで追加減算しない。"]
    lines += ["- 材料根拠は各120文字以内の省略表示。全文・記事版来歴・観測値はJSON台帳へ保存。"]
    for material in ledger["materials"]:
        evidence = "、".join(str(item["id"]) + "/" + str(item["revision_id"]) for item in material["evidence"])
        lines.append("- " + _short_reason(material["material_id"] + "：" + material["title"]) + "。根拠記事ID/版：" + _cell(evidence) + "。")
        lines.append("  - 反証・失効：" + _short_reason(material["invalidation"]) + "／" + _short_reason(material["status_reason"]))
        for symbol in SYMBOLS:
            for horizon in HORIZONS:
                point = material["impacts"][symbol][horizon]
                if point is not None:
                    lines.append("  - " + symbol + "／" + HORIZON_NAMES[horizon] + "：")
                    for label, key in zip(("T", "B0", "ΔB"), POINT_KEYS):
                        lines.append("    - " + label + "：" + _short_reason(point["evidence"][key]))
                    lines.append("    - 伝達：" + _short_reason(point["transmission"]))
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            aggregate, answer = ledger["aggregates"][symbol][horizon], ledger["forecasts"][symbol][horizon]
            if horizon == "today" or answer is not None:
                text = "未算出" if answer is None else answer["direction"] or "無回答"
                lines.append(f"- {symbol}／{HORIZON_NAMES[horizon]}：{text}。P0起点合計{_point_text(aggregate['totals']['reference_remaining_pt'])}pt、asof残余合計{_point_text(aggregate['totals']['residual_pt'])}pt（{aggregate['status']}）。")
                if answer is not None:
                    lines.append("  - 根拠：" + _short_reason(answer["point_rationale"]))
    lines += ["- 週・月・3か月の未算出点は当日点から自動補完していない。", "- 台帳固定ハッシュ：" + ledger["ledger_hash"] + "。"]
    return "\n".join(lines) + "\n"
