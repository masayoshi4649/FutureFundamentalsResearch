"""締切内のニュースから、明記されたＮＹ引けの数値と証跡を取り出す。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime

from .common import digest, timestamp


SERIES = ("dow", "nasdaq", "usdjpy", "wti")
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
DIGEST_PATTERN = re.compile(r"(?:(?:\d{4}年)?\d{1,2}月)?\d{1,2}日の米国市場ダイジェスト")
STOCK_PATTERNS = {
    "dow": re.compile(rf"(?:ダウ平均|NYダウ)(?:は)?\s*({NUMBER})\s*ドル(高|安)の\s*({NUMBER})\s*ドル"),
    "nasdaq": re.compile(rf"ナスダック(?:総合(?:指数)?)?は\s*({NUMBER})\s*(?:ポイント|pt)(高|安)の\s*({NUMBER})\s*(?:ポイント|pt)?(?=[、,。で\s]|$)"),
}
WTI_PATTERN = re.compile(
    rf"WTI先物(?P<contract>\d{{1,2}}月限)は[、,\s]*前営業日比\s*"
    rf"(?P<change>[+-](?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)\s*ドル\s*"
    rf"\(\s*(?P<pct>[+-](?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)\s*%\s*\)の\s*"
    rf"(?P<close>{NUMBER})\s*ドルで通常取引を終了した"
)
FX_PATTERN = re.compile(rf"(?:(\d{{1,3}})円(\d{{1,2}})銭|({NUMBER})円)で引け(?:た|る)")


def _normalise(value: str) -> str:
    """表記を正規化し、数値解析に用いる符号を統一する。

    引数: value (str) は記事の文字列。
    返り値: str は全角数字・括弧・百分率・マイナスを正規化した文字列。
    """
    return unicodedata.normalize("NFKC", value).replace("−", "-").replace("‐", "-")


def _number(value: str) -> float:
    """有限の数値文字列を浮動小数へ変換する。

    引数: value (str) は桁区切りを含み得る数値。
    返り値: float は有限値。不正値では ValueError を送出する。
    """
    result = float(value.replace(",", ""))
    if not math.isfinite(result):
        raise ValueError("有限の数値が必要です。")
    return result


def _evidence(article: dict, excerpt: str) -> dict:
    """解釈に使った原文と記事の時計を保存する。

    引数: article (dict) は日時検証後の記事、excerpt (str) は未正規化の原文。
    返り値: dict は識別子、原文抜粋、公表・取得・更新・利用可能日時。
    """
    return {
        "id": article["id"], "revision_id": article.get("revision_id"),
        "excerpt": excerpt.strip()[:320],
        "published_at": article["published_at"], "fetched_at": article["fetched_at"],
        "updated_at": article.get("updated_at"), "available_at": article["available_at"],
    }


def _dated_market(article: dict, series: str) -> str | None:
    """記事に明記された月日だけから市場日を求める。

    引数: article (dict) は締切内の記事、series (str) は対象系列。
    返り値: str | None は年月日。月日を確認できない記事は None。

    配信日から前営業日を推測しない。年のない月日は配信年へ対応し、
    一月配信の前年十二月だけ年跨ぎを認める。日だけの記述は補完しない。
    """
    headline = _normalise(article["headline"])
    body = _normalise(article["body_text"])
    if series == "wti":
        patterns = [r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日のNY原油先物"]
    else:
        patterns = [
            r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日の(?:米国市場|米国株式市場|ニューヨーク外為市場)",
        ]
        if DIGEST_PATTERN.match(headline):
            patterns.append(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日のNY原油先物")
    published = timestamp(article["published_at"]).date()
    dates = set()
    for pattern in patterns:
        for year_text, month_text, day_text in re.findall(pattern, headline + "\n" + body):
            year = int(year_text) if year_text else published.year
            month, day = int(month_text), int(day_text)
            if not year_text and published.month == 1 and month == 12:
                year -= 1
            try:
                stated = date(year, month, day)
            except ValueError:
                continue
            if stated <= published:
                dates.add(stated.isoformat())
    return next(iter(dates)) if len(dates) == 1 else None


def _fact(series: str, close: float, change: float | None, reported_pct: float | None,
          contract: str | None, article: dict, excerpt: str) -> dict:
    """明記された引け値から一つの観測候補を組み立てる。

    引数: series (str) は系列、close (float) は引け値、change (float | None) は
        符号付き前日比、reported_pct (float | None) は記事の百分率、
        contract (str | None) は記事の限月、article (dict) は記事、excerpt (str) は原文。
    返り値: dict は観測候補。正の前日値を計算できない場合は ValueError。
    """
    if close <= 0 or not math.isfinite(close):
        raise ValueError("引け値には正の有限値が必要です。")
    previous = close - change if change is not None else None
    if previous is not None and previous <= 0:
        raise ValueError("前日値を正数で計算できません。")
    pct = change / previous * 100 if previous is not None else None
    flags = []
    if pct is not None and reported_pct is not None and abs(pct - reported_pct) > 0.015:
        flags.append("reported_change_pct_mismatch")
    return {
        "series": series, "market_date": _dated_market(article, series),
        "observed_at": None, "close": close, "change_points": change,
        "change_pct": pct, "reported_change_pct": reported_pct,
        "implied_previous_close": previous, "contract": contract,
        "causality": "未識別", "evidence": [_evidence(article, excerpt)], "flags": flags,
    }


def _extract_article(article: dict) -> tuple[list[dict], list[dict]]:
    """一記事の明示引け表現だけを解析する。

    引数: article (dict) は日時検証と版選択を済ませた記事。
    返り値: tuple[list[dict], list[dict]] は観測候補と数値の曖昧さの記録。
    """
    headline = _normalise(article["headline"])
    body = article["body_text"]
    digest_article = bool(DIGEST_PATTERN.match(headline))
    stock_article = digest_article or headline.startswith("NY株式")
    fx_article = digest_article or headline.startswith("NY為替")
    oil_article = digest_article or headline.startswith("NY原油")
    result, flags = [], []
    sentences = re.findall(r"[^。\n]+(?:。|$)", body)
    for original in sentences:
        text = _normalise(original)
        if stock_article and re.search(r"取引を終了した|で引けた", text):
            for series, pattern in STOCK_PATTERNS.items():
                for match in pattern.finditer(text):
                    try:
                        raw_change, direction, raw_close = match.groups()
                        change = _number(raw_change)
                        if (raw_change.startswith("-") and direction == "高") or (raw_change.startswith("+") and direction == "安"):
                            raise ValueError("前日比の符号と高安の表記が矛盾しています。")
                        change = abs(change) * (1 if direction == "高" else -1)
                        result.append(_fact(series, _number(raw_close), change, None, None, article, original))
                    except ValueError:
                        flags.append({"id": article["id"], "series": series, "code": "ambiguous_numeric_value"})

        # ----------------------------------------
        # 為替はＮＹ記事内のドル円に限り、範囲や高安から引け値を推測しない。
        if fx_article and ("ドル・円は" in text or "ドル円は" in text):
            segment = re.split(r"ドル・円は|ドル円は", text, maxsplit=1)[1]
            if re.search(r"ユーロ|ポンド|スイス", segment):
                segment = re.split(r"ユーロ|ポンド|スイス", segment, maxsplit=1)[0]
            matches = list(FX_PATTERN.finditer(segment))
            if len(matches) == 1:
                try:
                    yen, sen, decimal = matches[0].groups()
                    close = int(yen) + int(sen) / 100 if yen is not None else _number(decimal)
                    result.append(_fact("usdjpy", close, None, None, None, article, original))
                except ValueError:
                    flags.append({"id": article["id"], "series": "usdjpy", "code": "ambiguous_numeric_value"})

        # ----------------------------------------
        # ＷＴＩは通常取引終了・前営業日比・記事内百分率を同じ文で要求する。
        if oil_article:
            for match in WTI_PATTERN.finditer(text):
                try:
                    result.append(_fact(
                        "wti", _number(match["close"]), _number(match["change"]),
                        _number(match["pct"]), match["contract"], article, original,
                    ))
                except ValueError:
                    flags.append({"id": article["id"], "series": "wti", "code": "ambiguous_numeric_value"})
    by_series = defaultdict(set)
    for fact in result:
        by_series[fact["series"]].add(_value_key(fact))
    ambiguous = {series for series, values in by_series.items() if len(values) > 1}
    for series in sorted(ambiguous):
        flags.append({"id": article["id"], "series": series, "code": "ambiguous_article_values"})
    return [fact for fact in result if fact["series"] not in ambiguous], flags


# ----------------------------------------
# 記事の時計検証と観測の重複統合


def _eligible_articles(articles: list[dict], cutoff: datetime) -> tuple[list[dict], list[dict]]:
    """本文に触れる前に締切と記事の版を検査する。

    引数: articles (list[dict]) は原本記事、cutoff (datetime) は時差付きの締切。
    返り値: tuple[list[dict], list[dict]] は利用可能な版と除外理由。
    """
    candidates, excluded = [], []
    for index, record in enumerate(articles):
        if not isinstance(record, dict):
            excluded.append({"record_index": index, "code": "invalid_record"})
            continue
        article_id = record.get("id")
        if isinstance(article_id, bool) or not isinstance(article_id, (int, str)) or not str(article_id).strip():
            excluded.append({"record_index": index, "code": "missing_article_id"})
            continue
        times = {}
        try:
            for field in ("published_at", "fetched_at", "updated_at"):
                value = record.get(field)
                if value is None and field != "updated_at":
                    raise ValueError("公表・取得日時は必須です。")
                if value is not None:
                    times[field] = timestamp(value)
        except (ValueError, TypeError, AttributeError, OverflowError):
            excluded.append({"record_index": index, "code": "invalid_or_missing_timestamp"})
            continue
        if any(value > cutoff for value in times.values()):
            excluded.append({"record_index": index, "code": "future_timestamp"})
            continue
        if not isinstance(record.get("body_text"), str) or not isinstance(record.get("headline"), str):
            excluded.append({"record_index": index, "code": "missing_text"})
            continue
        article = dict(record)
        article.update({field: value.isoformat() for field, value in times.items()})
        article["available_at"] = max(times.values()).isoformat()
        tie_fields = {key: article.get(key) for key in
                      ("id", "revision_id", "headline", "body_text", "published_at", "fetched_at", "updated_at")}
        order = (max(times.values()), times.get("updated_at", times["published_at"]),
                 str(record.get("revision_id", "")), digest(tie_fields))
        candidates.append((order, index, article))
    selected, seen = [], set()
    for order, index, article in sorted(candidates, key=lambda item: item[0], reverse=True):
        article_id = str(article["id"])
        if article_id in seen:
            excluded.append({"record_index": index, "code": "duplicate_or_superseded_article"})
        else:
            selected.append(article)
            seen.add(article_id)
    return selected, sorted(excluded, key=lambda item: item["record_index"])


def _value_key(fact: dict) -> tuple:
    """観測値の同一性を比較するキーを作る。

    引数: fact (dict) は観測候補。
    返り値: tuple は系列・引け・前日比・記事百分率・限月。
    """
    return tuple(fact[key] for key in ("series", "close", "change_points", "reported_change_pct", "contract"))


def _merge_observations(candidates: list[dict]) -> dict[str, list[dict]]:
    """同じ観測の転載を統合し、締切内の明示日へ根拠を束ねる。

    引数: candidates (list[dict]) は締切内だけから作成した観測候補。
    返り値: dict[str, list[dict]] は系列ごとの重複しない観測。
    """
    explicit = defaultdict(set)
    for fact in candidates:
        if fact["market_date"] is not None:
            explicit[_value_key(fact)].add(fact["market_date"])
    grouped = {}
    for candidate in candidates:
        fact = dict(candidate)
        dates = explicit[_value_key(fact)]
        if fact["market_date"] is None and len(dates) == 1:
            fact["market_date"] = next(iter(dates))
        elif fact["market_date"] is None and len(dates) > 1:
            fact["flags"] = list(fact["flags"]) + ["ambiguous_market_date"]
        key = (_value_key(fact), fact["market_date"])
        if key not in grouped:
            grouped[key] = {**fact, "evidence": [], "flags": set(fact["flags"])}
        saved = grouped[key]
        saved["flags"].update(fact["flags"])
        saved["evidence"].extend(fact["evidence"])
    result = {series: [] for series in SERIES}
    for fact in grouped.values():
        evidence = {digest(item): item for item in fact["evidence"]}
        fact["evidence"] = sorted(evidence.values(), key=lambda item: (item["available_at"], item["published_at"], str(item["id"])))
        fact["source_ids"] = sorted({item["id"] for item in fact["evidence"]}, key=str)
        fact["published_at"] = min(item["published_at"] for item in fact["evidence"])
        fact["available_at"] = min(item["available_at"] for item in fact["evidence"])
        fact["latest_evidence_at"] = max(item["available_at"] for item in fact["evidence"])
        fact["observation_key"] = fact["series"] + ":" + digest({"values": _value_key(fact), "market_date": fact["market_date"]})[:20]
        fact["flags"] = sorted(fact["flags"])
        result[fact["series"]].append(fact)
    for facts in result.values():
        facts.sort(key=lambda fact: (fact["latest_evidence_at"], fact["published_at"], fact["observation_key"]))
    return result


def _check_consistency(observations: dict[str, list[dict]]) -> list[dict]:
    """明示日の競合と、隣接暦日の前日値の不整合を記録する。

    引数: observations (dict[str, list[dict]]) は統合した観測。
    返り値: list[dict] は原本の数値を変更せずに記録した不整合。

    休日を推測しないため、前日値の比較は異なる明示日が一暦日差で、
    同じ限月の場合に限定する。比較対象のない欠測値は補完しない。
    """
    flags = []
    for series, facts in observations.items():
        by_date = defaultdict(list)
        for fact in facts:
            if fact["market_date"] is not None:
                by_date[fact["market_date"]].append(fact)
            for code in fact["flags"]:
                flags.append({"series": series, "observation_key": fact["observation_key"], "code": code})
        for market_date, variants in by_date.items():
            if len(variants) > 1:
                for fact in variants:
                    fact["flags"].append("same_market_date_conflict")
                flags.append({"series": series, "market_date": market_date, "code": "same_market_date_conflict"})
        dated = sorted((values[0] for values in by_date.values() if len(values) == 1), key=lambda fact: fact["market_date"])
        for previous, current in zip(dated, dated[1:]):
            gap = (date.fromisoformat(current["market_date"]) - date.fromisoformat(previous["market_date"])).days
            if gap != 1 or current["contract"] != previous["contract"] or current["implied_previous_close"] is None:
                continue
            difference = current["implied_previous_close"] - previous["close"]
            current["previous_observation_comparison"] = {
                "previous_market_date": previous["market_date"], "previous_observed_close": previous["close"],
                "implied_previous_close": current["implied_previous_close"], "difference": difference,
            }
            if abs(difference) > 0.02:
                code = "previous_observed_close_mismatch"
                current["flags"].append(code)
                flags.append({"series": series, "market_date": current["market_date"], "observation_key": current["observation_key"], "code": code,
                              "previous_market_date": previous["market_date"], "difference": difference})
    return flags


# ----------------------------------------
# 公開ＡＰＩ


def extract_market_facts(articles: list[dict], as_of: str | datetime) -> dict:
    """締切内の記事からＮＹ引けの数値と重複しない観測証跡を抽出する。

    引数:
        articles (list[dict]): ニュース原本。id、published_at、fetched_at、
            headline、body_text を必須とし、updated_at は存在時に検査する。
        as_of (str | datetime): 時差付きの情報締切。
    返り値:
        dict: as_of、series（各系列の最新観測または None）、observations
            （全観測）、coverage（抽出件数）、excluded、flags を持つ辞書。

    本文解析より先に公表・取得・更新の時計を検査する。百分率は
    change / (close - change) * 100。原油の記事百分率も別に残す。
    明示されない引け値・マクロ実績・市場予想・発生時計を補完しない。
    配信時計をショック発生時計に転用せず、先物への因果は未識別とする。
    """
    if not isinstance(articles, list):
        raise TypeError("記事は配列で指定してください。")
    cutoff = timestamp(as_of)
    eligible, excluded = _eligible_articles(articles, cutoff)
    candidates, flags, parsed_ids = [], [], set()
    for article in eligible:
        facts, article_flags = _extract_article(article)
        candidates.extend(facts)
        flags.extend(article_flags)
        if facts:
            parsed_ids.add(str(article["id"]))
    observations = _merge_observations(candidates)
    flags.extend(_check_consistency(observations))
    latest = {}
    for series, facts in observations.items():
        selected = facts[-1] if facts else None
        latest[series] = None if selected and "same_market_date_conflict" in selected["flags"] else selected
    count = sum(len(facts) for facts in observations.values())
    return {
        "as_of": cutoff.isoformat(), "series": latest, "observations": observations,
        "coverage": {
            "input_articles": len(articles), "eligible_articles": len(eligible),
            "articles_with_facts": len(parsed_ids), "unique_observations": count,
            "duplicate_observations": len(candidates) - count, "excluded_articles": len(excluded),
        },
        "excluded": excluded, "flags": flags,
        "causality": "未識別", "observed_at": None,
    }
