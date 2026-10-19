"""臨時材料の認定証跡を検査し、公表前までの別評価を作る。"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timedelta

from fundamentals.common import timestamp
from fundamentals.engine import candle_hash
from fundamentals.prices import Candle, SYMBOLS
from research_support.five_class import (
    CLASSES, EvaluationContract, _decimal, _number, _priority_classes, _session_times,
    _validate_bar, evaluate_priority_path, validate_contract,
)


EVENT_FIELDS = frozenset({
    "event_id", "assessment_as_of", "first_public_at", "article_ids", "source_revision_hash",
    "decision", "schedule_status", "affected_symbols", "novelty_evidence", "schedule_evidence",
    "impact_evidence", "timing_evidence", "judge_id",
})
_EVIDENCE_FIELDS = ("novelty_evidence", "schedule_evidence", "impact_evidence", "timing_evidence")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")


def _evidence_present(value: object) -> bool:
    """根拠に空白以外の記述または構造化した資料があるかを確認する。

    引数: value (object) は文字列または辞書・リストの根拠。
    返り値: bool は非空文字列を含む証跡の有無。内容の因果認定は行わない。
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and any(_evidence_present(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and any(_evidence_present(item) for item in value)
    return False


def _validate_event(event: dict) -> tuple[datetime, datetime | None]:
    """材料判定の必須証跡と予定区分を検査する。

    引数: event (dict) は一つの締切に対する材料判定。
    返り値: tuple[datetime, datetime | None] は判定対象締切と確認できた初回公表時刻。
        初報時計が未確認の保留はNoneを保持し、記事公開上限を材料時計へ代用しない。
        採用の大きさ・因果を価格や予測の当否から独自に認定しない。
    """
    if not isinstance(event, dict) or not EVENT_FIELDS <= set(event):
        raise ValueError("材料判定には全必須項目が必要です。")
    for field in ("event_id", "judge_id"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError("材料と判定者の識別子は非空文字列が必要です。")
    if event["decision"] not in ("採用", "棄却", "保留"):
        raise ValueError("材料の判定は採用・棄却・保留を指定してください。")
    if not isinstance(event["assessment_as_of"], (str, datetime)):
        raise ValueError("情報締切には実在する時差付き日時が必要です。")
    assessed = timestamp(event["assessment_as_of"])
    public = event["first_public_at"]
    if public is None:
        if event["decision"] != "保留":
            raise ValueError("初回公表時計が未確認の材料は保留だけに使用できます。")
        published = None
    elif not isinstance(public, (str, datetime)):
        raise ValueError("初回公表には実在する時差付き日時または保留のnullが必要です。")
    else:
        published = timestamp(public)
    timing = event["timing_evidence"]
    lower, upper = None, None
    if isinstance(timing, dict):
        for key in ("material_public_lower_bound", "article_public_upper_bound"):
            value = timing.get(key)
            if value is not None and not isinstance(value, (str, datetime)):
                raise ValueError("公表時計の上下限には時差付き日時が必要です。")
        if timing.get("material_public_lower_bound") is not None:
            lower = timestamp(timing["material_public_lower_bound"])
        if timing.get("article_public_upper_bound") is not None:
            upper = timestamp(timing["article_public_upper_bound"])
    if published is None and upper is None:
        raise ValueError("初報時計が未確認の保留には記事公開上限の時計が必要です。")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("材料公表の下限は記事公開上限以下が必要です。")
    if published is not None and ((lower is not None and published < lower) or (upper is not None and published > upper)):
        raise ValueError("確認済み初報時計が根拠の上下限と矛盾しています。")
    if event["schedule_status"] not in ("臨時・予見不能", "予定済み", "不明"):
        raise ValueError("材料の予定区分が不正です。")
    symbols = event["affected_symbols"]
    if (not isinstance(symbols, list) or not symbols or any(symbol not in SYMBOLS for symbol in symbols)
            or len(set(symbols)) != len(symbols)):
        raise ValueError("材料の適用先は重複のない対応先物のリストが必要です。")
    articles = event["article_ids"]
    if (not isinstance(articles, list) or any(type(item) not in (str, int) or not str(item).strip() for item in articles)
            or len(set(map(str, articles))) != len(articles)):
        raise ValueError("材料の出典記事は重複のない識別子のリストが必要です。")
    if event["decision"] == "採用":
        if event["schedule_status"] != "臨時・予見不能":
            raise ValueError("予定済みまたは予定不明の材料は打切りへ採用できません。")
        if not articles or not isinstance(event["source_revision_hash"], str) or not _HASH.fullmatch(event["source_revision_hash"]):
            raise ValueError("採用材料には記事出典と版を固定するSHA-256が必要です。")
        if any(not _evidence_present(event[field]) for field in _EVIDENCE_FIELDS):
            raise ValueError("採用材料には新規性・予定・影響・初回公表時刻の根拠が必要です。")
    return assessed, published


def _publication_bounds(event: dict) -> tuple[datetime | None, datetime]:
    """確認済み初報時計か、未確認時計の根拠付き上下限を返す。

    引数: event (dict) は検証済みの材料判定。
    返り値: tuple[datetime | None, datetime] は公表下限と上限。下限Noneは不明を表す。
        確認済み時計は両端が同じであり、記事公開上限から架空の初報時計を生成しない。
    """
    _, published = _validate_event(event)
    if published is not None:
        return published, published
    evidence = event["timing_evidence"]
    lower = evidence.get("material_public_lower_bound")
    return None if lower is None else timestamp(lower), timestamp(evidence["article_public_upper_bound"])


def _pending_applies(event: dict, cutoff: datetime, symbol: str, limit: datetime) -> bool:
    """保留材料の公表が評価窓内だった可能性を、確認済み時計または上下限から調べる。

    引数: event (dict) は材料判定、cutoff (datetime) は元予測締切、symbol (str) は対象先物、
        limit (datetime) は採用材料時計または公式終了。
    返り値: bool は締切後・評価窓終了前に公表された可能性。
        上限が終了以後でも場中公表の可能性を捨てない。上限が締切以前なら既知として適用しない。
    """
    assessed, published = _validate_event(event)
    if (assessed != cutoff or event["decision"] != "保留" or event["schedule_status"] == "予定済み"
            or symbol not in event["affected_symbols"]):
        return False
    if published is not None:
        return cutoff < published < limit
    lower, upper = _publication_bounds(event)
    return upper > cutoff and (lower is None or lower < limit)


def select_event(events: list[dict], as_of: datetime, symbol: str, end: datetime) -> dict | None:
    """当該締切・先物に認定済みの最初の臨時材料を固定順で選ぶ。

    引数: events (list[dict]) は結果側だけが使用する材料台帳、as_of (datetime) は情報締切、
        symbol (str) は先物、end (datetime) は公式終了予定。
    返り値: dict | None は公表時計・材料識別子の順に選んだ採用材料の複製。
        締切以前、終了以降、別締切、別先物、保留・棄却の材料は選ばない。
        予測回答・実績当否・後の価格変動を選択条件に使わない。
    """
    cutoff, full_end = timestamp(as_of), timestamp(end)
    if symbol not in SYMBOLS or cutoff >= full_end:
        raise ValueError("対応先物と、公式終了より前の情報締切が必要です。")
    if not isinstance(events, list):
        raise ValueError("材料台帳はリストが必要です。")
    candidates, identities = [], set()
    for event in events:
        assessed, published = _validate_event(event)
        identity = (assessed, event["event_id"])
        if identity in identities:
            raise ValueError("同じ締切・材料の判定を重複して登録できません。")
        identities.add(identity)
        if (assessed == cutoff and published is not None and event["decision"] == "採用" and symbol in event["affected_symbols"]
                and cutoff < published < full_end):
            candidates.append((published, event["event_id"], event))
    return None if not candidates else copy.deepcopy(min(candidates, key=lambda item: item[:2])[2])


# ----------------------------------------
# 価格を解釈する前に時間だけで対象窓を限定する。


def _visible_bars(candles: list[Candle], end: datetime) -> tuple[list[Candle], int]:
    """窓を跨ぐ足・窓終了後の足をOHLCや確定指定より先に取り除く。

    引数: candles (list[Candle]) は原価格足、end (datetime) は安全な名目評価終点。
    返り値: tuple[list[Candle], int] は窓内に完結する足と除外数。
        足を補間せず、跨ぐ足から公表前の架空価格を作らない。
    """
    if not isinstance(candles, list) or any(not isinstance(bar, Candle) for bar in candles):
        raise ValueError("対象足はCandleのリストが必要です。")
    visible, excluded = [], 0
    for bar in candles:
        if any(not isinstance(value, datetime) or value.utcoffset() is None for value in (bar.time, bar.end)):
            raise ValueError("対象足はタイムゾーン付き日時が必要です。")
        if bar.time >= end or bar.end > end:
            excluded += 1
            continue
        visible.append(bar)
    return visible, excluded


def _pending_events(events: list[dict], cutoff: datetime, symbol: str, limit: datetime) -> list[dict]:
    """確定した打切りより先に判断未了の臨時候補が残るかを調べる。

    引数: events (list[dict]) は検証済み台帳、cutoff (datetime) は情報締切、symbol (str) は先物、
        limit (datetime) は採用材料の公表または公式終了。
    返り値: list[dict] は公表時計または記事上限時計・材料識別子順の未決候補。
        この固定順は、初報時計不明の材料の実際の公表順を意味しない。予定済みは含めない。
    """
    candidates = [event for event in events if _pending_applies(event, cutoff, symbol, limit)]
    return copy.deepcopy(sorted(candidates, key=lambda event: (_publication_bounds(event)[1], event["event_id"])))


def _prefix(reference: Candle, candles: list[Candle], session: str, start: datetime,
            end: datetime, rules: EvaluationContract, template: dict) -> dict:
    """材料前の価格窓だけの品質と優先分類を作る。

    引数: reference (Candle) は前先物終値、candles (list[Candle]) は窓内の足、session (str) は対象場、
        start/end (datetime) は対象場開始と名目窓終了、rules (EvaluationContract) は固定規約、
        template (dict) は既存評価で検証済みの基準・規約。
    返り値: dict は材料前の確定・未確定評価。公式終了予定は評価窓へ別保存する。
    """
    bars, incomplete = [], 0
    contracts = {reference.contract} if reference.contract is not None else set()
    for bar in sorted(candles, key=lambda item: item.time):
        if bar.symbol != reference.symbol or bar.session != session or bar.time < start:
            raise ValueError("窓内の銘柄・セッション・開始時刻が不整合です。")
        if type(bar.closed) is not bool:
            raise ValueError("窓内の確定指定は真偽値が必要です。")
        if bar.contract is not None:
            contracts.add(bar.contract)
        if not bar.closed:
            incomplete += 1
            continue
        _validate_bar(bar, "材料前の対象足", rules.interval_minutes)
        bars.append(bar)
    if len(contracts) > 1:
        raise ValueError("基準と材料前の対象足の限月が一致しません。")
    if any(previous.end > current.time for previous, current in zip(bars, bars[1:])):
        raise ValueError("材料前の足に重複または時間の重なりがあります。")
    step = timedelta(minutes=rules.interval_minutes)
    if any((bar.time - start).total_seconds() % step.total_seconds() for bar in bars):
        raise ValueError("材料前の足が契約の時間格子と一致しません。")
    elapsed_seconds, step_seconds = (end - start).total_seconds(), step.total_seconds()
    expected = int((elapsed_seconds + step_seconds - 1) // step_seconds)
    missing = expected - len(bars)
    unobserved = missing > 0 or expected == 0
    lag = None if not bars else (end - bars[-1].end).total_seconds() / 60
    endpoint_available = bool(bars) and lag <= rules.endpoint_max_lag_minutes
    base = _decimal(reference.close, "基準終値")
    last = _decimal(bars[-1].close, "材料前の対象終値") if endpoint_available else None
    high = max((_decimal(bar.high, "高値") for bar in bars), default=None)
    low = min((_decimal(bar.low, "安値") for bar in bars), default=None)
    terminal = None if last is None else (last - base) / base * 100
    high_pct, low_pct = (None, None) if high is None else ((high - base) / base * 100, (low - base) / base * 100)
    range_pct = None if high is None else (high - low) / base * 100
    excursion = None if high is None else max(abs(high_pct), abs(low_pct))
    target = True if excursion is not None and excursion >= rules.movement_threshold_pct else None if unobserved else False

    # ----------------------------------------

    flat_terminal = None if terminal is None else abs(terminal) <= rules.flat_terminal_threshold_pct
    flat_certain = (range_pct is not None and range_pct <= rules.range_threshold_pct
                    and flat_terminal is True and not unobserved)
    flat_possible = (range_pct is None or (range_pct <= rules.range_threshold_pct and flat_terminal is not False)) and not flat_certain
    candidates = _priority_classes(base, bars, start, end, rules)
    if expected == 0:
        candidates = set(CLASSES) - {"もみ合い"}
    if flat_certain:
        definite, uncertain, flat_status = {"もみ合い"}, set(), "成立"
    elif flat_possible:
        definite, uncertain, flat_status = set(), candidates | {"もみ合い"}, "未確定"
    elif len(candidates) == 1:
        definite, uncertain, flat_status = candidates, set(), "不成立"
    else:
        definite, uncertain, flat_status = set(), candidates, "不成立"
    result = copy.deepcopy(template)
    result.update({
        "target_status": "対象" if target is True else "対象外" if target is False else "未確定",
        "correct_candidates": [label for label in CLASSES if label in definite],
        "uncertain_candidates": [label for label in CLASSES if label in uncertain],
        "metrics": {"reference_price": float(base), "quote_price": _number(last),
                    "quote_time": bars[-1].end.isoformat() if endpoint_available else None,
                    "actual_pct": _number(terminal), "high_pct": _number(high_pct), "low_pct": _number(low_pct),
                    "session_range_pct": _number(range_pct), "observed_max_abs_pct": _number(excursion)},
        "flat_status": flat_status,
        "path_class_candidates_before_flat": [label for label in CLASSES if label in candidates],
        "path_hash": candle_hash(bars),
    })
    result["quality"].update({
        "observed_rows": len(bars), "expected_rows": expected, "missing_rows": missing,
        "unclosed_rows": incomplete, "incomplete_volume_rows": sum(not bar.volume_complete for bar in bars),
        "endpoint_available": endpoint_available, "endpoint_lag_minutes": lag,
        "has_unobserved_path": unobserved, "session_range_is_lower_bound": unobserved,
        "contract_identity_available": all(bar.contract is not None for bar in [reference, *bars]),
        "endpoint_status": "最終観測終値の代用" if endpoint_available else "終点観測不足",
        "partial_interval_seconds": elapsed_seconds % step_seconds,
    })
    return result


def evaluate_event_window(reference: Candle, candles: list[Candle], session: str, as_of: datetime,
                          events: list[dict], contract: dict | EvaluationContract) -> dict:
    """認定した臨時材料の公表前までを、全セッション評価と別に評価する。

    引数: reference (Candle) は同じ先物の前終値、candles (list[Candle]) は原価格足、
        session (str) は対象場、as_of (datetime) は情報締切、events (list[dict]) は結果専用の材料台帳、
        contract (dict | EvaluationContract) は事前固定した五分類規約。
    返り値: dict は既存評価と共通の項目と、切下げ時計・実観測時計・選択材料を持つ評価窓。
        材料なしは既存優先分類と同じ値。開始前失効はscoreable=Falseとして予定回数に残す。
        実時計で材料公表・名目評価終点へ到達済みかの検査は呼出側の責務。
        この結果や後発材料台帳を当回の予測入力へ渡してはならない。
    """
    rules, cutoff = validate_contract(contract), timestamp(as_of)
    start, full_end = _session_times(session)
    template = evaluate_priority_path(reference, [], session, full_end, rules)
    if cutoff >= start or reference.end > cutoff:
        raise ValueError("開始前の情報締切と、その時点で確定済みの基準足が必要です。")
    event = select_event(events, cutoff, reference.symbol, full_end)
    public = None if event is None else timestamp(event["first_public_at"])
    pending = _pending_events(events, cutoff, reference.symbol, public if public is not None else full_end)
    nominal = full_end if public is None else public.replace(second=0, microsecond=0)
    expired = public is not None and public <= start
    visible, excluded = _visible_bars(candles, nominal)
    if expired:
        result = template
        result["correct_candidates"], result["uncertain_candidates"] = [], []
        result["quality"].update({"observed_rows": 0, "expected_rows": 0, "missing_rows": 0,
                                  "has_unobserved_path": False, "session_range_is_lower_bound": False,
                                  "endpoint_status": "開始前失効"})
        result["flat_status"] = "未評価"
        result["path_class_candidates_before_flat"] = []
    elif event is None:
        result = evaluate_priority_path(reference, visible, session, full_end, rules)
    else:
        result = _prefix(reference, visible, session, start, nominal, rules, template)
    result["target_end"] = nominal.isoformat()
    quote = result["metrics"]["quote_time"]
    quote_time = None if quote is None else timestamp(quote)
    result["evaluation_window"] = {
        "version": "臨時材料前１", "status": "開始前失効" if expired else "全セッション" if event is None else "材料前まで",
        "as_of": cutoff.isoformat(), "scheduled_full_end": full_end.isoformat(),
        "source_event_time": None if public is None else public.isoformat(),
        "nominal_window_end": nominal.isoformat(), "effective_quote_time": quote,
        "event_to_nominal_seconds": None if public is None else (public - nominal).total_seconds(),
        "event_to_quote_seconds": None if public is None or quote_time is None else (public - quote_time).total_seconds(),
        "quote_to_nominal_seconds": None if quote_time is None else (nominal - quote_time).total_seconds(),
        "scoreable": not expired, "planned_count": True, "selected_event": event,
        "pending_events": pending, "adjudication_unknown": bool(pending),
        "excluded_after_window_rows": excluded,
        "maturity_required_at": (public if public is not None else full_end).isoformat(),
        "label_available_at": (public if public is not None else full_end).isoformat(),
        "actual_clock_maturity_check": "材料公表と評価終点の実時計での満了は呼出側で検査する。",
        "selection_policy": "当該締切の認定済み臨時材料のうち初回公表時刻・材料IDの順で最初を選ぶ。",
        "rounding_policy": "公表時計を分単位で切り下げ、終了がその時計以内の確定足だけを使用する。",
    }
    if pending and not expired:
        result["provisional_evaluation"] = {key: copy.deepcopy(value) for key, value in result.items()
                                             if key != "evaluation_window"}
        result["target_status"] = "未確定"
        result["correct_candidates"] = []
        result["uncertain_candidates"] = list(CLASSES)
        result["flat_status"] = "材料認定未了"
    return result
