"""固定済み契約に従い、先物５分類の確定・未確定と採点上下限を分ける。"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation

from fundamentals.common import JST, timestamp
from fundamentals.engine import candle_hash, scheduled_end
from fundamentals.prices import Candle, SYMBOLS


CLASSES = ("上がる", "下がる", "上がってから下がる", "下がってから上がる", "もみ合い")
_PERCENT_FIELDS = ("movement_threshold_pct", "direction_threshold_pct", "initial_excursion_pct",
                   "reversal_threshold_pct", "range_threshold_pct", "flat_terminal_threshold_pct")
_SESSION = re.compile(r"\d{4}-\d{2}-\d{2}_(?:Day|Night)\Z")


@dataclass(frozen=True)
class EvaluationContract:
    """事前固定が必要な全閾値と観測終値の代用許容を明示する契約。"""

    target_basis: str
    movement_threshold_pct: Decimal
    direction_threshold_pct: Decimal
    initial_excursion_pct: Decimal
    reversal_threshold_pct: Decimal
    reversal_requires_reference_cross: bool
    range_threshold_pct: Decimal
    flat_terminal_threshold_pct: Decimal
    interval_minutes: int
    reference_max_lag_minutes: int
    endpoint_max_lag_minutes: int

    def to_dict(self) -> dict:
        """契約を境界値の丸めを伴わない保存用辞書へ変換する。

        引数: self (EvaluationContract) は検証済み契約。
        返り値: dict は百分率を十進文字列で記したJSON保存可能な契約。
        """
        return {field.name: str(getattr(self, field.name)) if field.name in _PERCENT_FIELDS
                else getattr(self, field.name) for field in fields(self)}


def _decimal(value: object, name: str) -> Decimal:
    """有限の数値を文字列表現から十進数へ変換する。

    引数: value (object) は数値または十進文字列、name (str) は項目名。
    返り値: Decimal は浮動小数点演算を介さない値。
    """
    if isinstance(value, bool):
        raise ValueError(f"{name}に真偽値は指定できません。")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name}には有限の数値が必要です。") from None
    if not result.is_finite():
        raise ValueError(f"{name}には有限の数値が必要です。")
    return result


def validate_contract(contract: dict | EvaluationContract) -> EvaluationContract:
    """省略や未知の項目を認めず、５分類契約を検証する。

    引数: contract (dict | EvaluationContract) は予測前に固定する全設定。
    返り値: EvaluationContract は数値を十進数に正規化した契約。
    """
    names = {field.name for field in fields(EvaluationContract)}
    if isinstance(contract, EvaluationContract):
        values = {name: getattr(contract, name) for name in names}
    elif isinstance(contract, dict) and set(contract) == names:
        values = dict(contract)
    else:
        raise ValueError("契約は全必須項目を過不足なく指定してください。")
    if values["target_basis"] not in ("終点", "場中到達"):
        raise ValueError("対象基準は終点または場中到達を指定してください。")
    if type(values["reversal_requires_reference_cross"]) is not bool:
        raise ValueError("終点の基準跨ぎ指定は真偽値が必要です。")
    for name in _PERCENT_FIELDS:
        values[name] = _decimal(values[name], name)
        if values[name] < 0 or (name not in ("flat_terminal_threshold_pct", "direction_threshold_pct") and values[name] == 0):
            raise ValueError("閾値は正数、方向・もみ合い終点閾値は０以上を指定してください。")
    for name in ("interval_minutes", "reference_max_lag_minutes", "endpoint_max_lag_minutes"):
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError("足間隔と許容遅延は０以上の整数が必要です。")
    if values["interval_minutes"] not in (1, 5):
        raise ValueError("足間隔は１分または５分が必要です。")
    return EvaluationContract(**values)


# ----------------------------------------
# 時間・銘柄・足の検証と、足内順序の候補追跡


def _session_times(session: str) -> tuple[datetime, datetime]:
    """固定セッション名の公式開始・終了予定を求める。

    引数: session (str) は日付とDayまたはNightを含む名称。
    返り値: tuple[datetime, datetime] は日本時間の開始と終了。
    """
    if not isinstance(session, str) or not _SESSION.fullmatch(session):
        raise ValueError("セッション名の形式が不正です。")
    day = datetime.strptime(session[:10], "%Y-%m-%d").date()
    return datetime.combine(day, time(8, 45) if session.endswith("_Day") else time(17), JST), scheduled_end(session)


def _validate_bar(bar: Candle, name: str, interval: int | None = None) -> None:
    """確定価格足の日時・OHLCと必要な足間隔を検証する。

    引数: bar (Candle) は対象足、name (str) は項目名、interval (int | None) は必要な分数。
    返り値: None。不正な価格・時間・確定状態は例外で拒否する。
    """
    if not isinstance(bar, Candle) or bar.symbol not in SYMBOLS:
        raise ValueError(f"{name}は対応する先物のCandleが必要です。")
    if any(not isinstance(value, datetime) or value.utcoffset() is None for value in (bar.time, bar.end)):
        raise ValueError(f"{name}はタイムゾーン付き日時が必要です。")
    if bar.end <= bar.time or (interval is not None and bar.end - bar.time != timedelta(minutes=interval)):
        raise ValueError(f"{name}の足間隔が契約と不整合です。")
    if type(bar.closed) is not bool or type(bar.volume_complete) is not bool:
        raise ValueError(f"{name}の確定・出来高完備指定は真偽値が必要です。")
    values = [_decimal(getattr(bar, field), name + field) for field in ("open", "high", "low", "close")]
    opened, high, low, closed = values
    if low <= 0 or not low <= min(opened, closed) <= max(opened, closed) <= high:
        raise ValueError(f"{name}のOHLCが不整合です。")


def _reversal_states(bars: list[Candle], reference: Decimal, contract: EvaluationContract) -> set[tuple[bool, bool]]:
    """OHLCの高安順序を両方追い、確実な折返しと可能な折返しを分ける。

    引数: bars (list[Candle]) は確定足、reference (Decimal) は前先物終値、
        contract (EvaluationContract) は到達・反転閾値の契約。
    返り値: set[tuple[bool, bool]] は上昇後下落・下落後上昇の成立候補。
    """
    initial = reference * contract.initial_excursion_pct / 100
    reverse = reference * contract.reversal_threshold_pct / 100
    peak = trough = reference
    states = {(False, False)}
    for bar in bars:
        opened, high, low, closed = (_decimal(getattr(bar, field), field) for field in ("open", "high", "low", "close"))
        updated = set()
        for down_done, up_done in states:
            for order in ((opened, high, low, closed), (opened, low, high, closed)):
                local_peak, local_trough = peak, trough
                local_down, local_up = down_done, up_done
                for price in order:
                    if local_peak >= reference + initial and local_peak - price >= reverse:
                        down_done_now = True
                    else:
                        down_done_now = False
                    if local_trough <= reference - initial and price - local_trough >= reverse:
                        up_done_now = True
                    else:
                        up_done_now = False
                    local_down = local_down or down_done_now
                    local_up = local_up or up_done_now
                    local_peak, local_trough = max(local_peak, price), min(local_trough, price)
                updated.add((local_down, local_up))
        states = updated
        peak, trough = max(peak, high), min(trough, low)
    return states


def _number(value: Decimal | None) -> float | None:
    """表示用の百分率をJSONで保存できる有限数値へ変換する。

    引数: value (Decimal | None) は算出値または未確定。
    返り値: float | None は表示用数値。判定には変換前の十進数を用いる。
    """
    return None if value is None else float(value)


def evaluate_path(reference: Candle, candles: list[Candle], session: str, target_end: datetime,
                  contract: dict | EvaluationContract) -> dict:
    """同じ先物の前終値を基準に、対象判定と５分類の確定・未確定集合を作る。

    引数: reference (Candle) は前セッション最終観測、candles (list[Candle]) は対象場だけの足、
        session (str) は固定対象、target_end (datetime) は公式終了予定、
        contract (dict | EvaluationContract) は予測前に固定する全設定。
    返り値: dict は対象状態、確定正解集合、未確定候補集合、百分率、品質と価格ハッシュ。
        公式終値がない場合、許容遅延内の確定足終値を代理と明記する。実時計の満了検査は呼出側で行う。
        通常方向閾値が０なら終点の正負だけを用い、変化０を上がる・下がるの両方へ含めない。
    """
    rules = validate_contract(contract)
    start, end = _session_times(session)
    if not isinstance(target_end, datetime) or target_end.utcoffset() is None:
        raise ValueError("対象終点はタイムゾーン付き日時が必要です。")
    if timestamp(target_end) != end:
        raise ValueError("対象終点がセッション公式終了予定と不整合です。")
    _validate_bar(reference, "基準足")
    ref_start, ref_end = _session_times(reference.session)
    expected = "Night" if session.endswith("_Day") else "Day"
    if not reference.closed or not reference.session.endswith("_" + expected):
        raise ValueError("確定した直前の逆種セッションが基準に必要です。")
    if session.endswith("_Night") and reference.session[:10] != session[:10]:
        raise ValueError("ナイト基準は当日デイの同じ先物終値が必要です。")
    if (ref_end > start or start - ref_end > timedelta(days=4) or reference.time < ref_start
            or reference.end > ref_end or ref_end - reference.end > timedelta(minutes=rules.reference_max_lag_minutes)):
        raise ValueError("基準セッション・基準時刻または許容遅延が不整合です。")

    # ----------------------------------------

    if not isinstance(candles, list) or any(not isinstance(bar, Candle) for bar in candles):
        raise ValueError("対象足はCandleのリストが必要です。")
    if any(not isinstance(bar.time, datetime) or not isinstance(bar.end, datetime)
           or bar.time.utcoffset() is None or bar.end.utcoffset() is None for bar in candles):
        raise ValueError("対象足はタイムゾーン付き日時が必要です。")
    ordered = sorted(candles, key=lambda bar: bar.time)
    bars, incomplete = [], 0
    contracts = {reference.contract} if reference.contract is not None else set()
    for bar in ordered:
        if not isinstance(bar, Candle) or bar.symbol != reference.symbol or bar.session != session:
            raise ValueError("対象足の銘柄またはセッションが基準・対象と不整合です。")
        if bar.time.utcoffset() is None or bar.end.utcoffset() is None or bar.time < start or bar.end > end:
            raise ValueError("対象セッション外または未来の足を混ぜられません。")
        if type(bar.closed) is not bool:
            raise ValueError("足の確定指定は真偽値が必要です。")
        if bar.contract is not None:
            contracts.add(bar.contract)
        if not bar.closed:
            incomplete += 1
            continue
        _validate_bar(bar, "対象足", rules.interval_minutes)
        bars.append(bar)
    if len(contracts) > 1:
        raise ValueError("基準と対象足の限月が一致しません。")
    if any(previous.end > current.time for previous, current in zip(bars, bars[1:])):
        raise ValueError("足の重複または時間の重なりがあります。")
    step = timedelta(minutes=rules.interval_minutes)
    if any((bar.time - start).total_seconds() % step.total_seconds() for bar in bars):
        raise ValueError("足の開始が契約の時間格子と一致しません。")
    expected_rows = int((end - start) / step)
    missing_rows = expected_rows - len(bars)
    endpoint_lag = None if not bars else (end - bars[-1].end).total_seconds() / 60
    endpoint_available = bool(bars) and endpoint_lag <= rules.endpoint_max_lag_minutes
    has_unobserved_path = missing_rows > 0
    base = _decimal(reference.close, "基準終値")
    last = _decimal(bars[-1].close, "対象終値") if endpoint_available else None
    high = max((_decimal(bar.high, "高値") for bar in bars), default=None)
    low = min((_decimal(bar.low, "安値") for bar in bars), default=None)
    terminal_pct = None if last is None else (last - base) / base * 100
    high_pct = None if high is None else (high - base) / base * 100
    low_pct = None if low is None else (low - base) / base * 100
    range_pct = None if high is None else (high - low) / base * 100
    observed_excursion = None if high is None else max(abs(high_pct), abs(low_pct))

    # ----------------------------------------

    if rules.target_basis == "終点":
        target = None if terminal_pct is None else abs(terminal_pct) >= rules.movement_threshold_pct
    elif observed_excursion is not None and observed_excursion >= rules.movement_threshold_pct:
        target = True
    elif has_unobserved_path:
        target = None
    else:
        target = False
    definite, possible = set(), set()
    for label, sign in (("上がる", 1), ("下がる", -1)):
        if terminal_pct is None:
            possible.add(label)
        elif (terminal_pct * sign > 0 if rules.direction_threshold_pct == 0
              else terminal_pct * sign >= rules.direction_threshold_pct):
            definite.add(label)
    flat_terminal = None if terminal_pct is None else abs(terminal_pct) <= rules.flat_terminal_threshold_pct
    if range_pct is None or (range_pct <= rules.range_threshold_pct and flat_terminal is not False):
        if range_pct is not None and not has_unobserved_path and flat_terminal is True:
            definite.add("もみ合い")
        else:
            possible.add("もみ合い")
    states = _reversal_states(bars, base, rules)
    for index, label in enumerate(("上がってから下がる", "下がってから上がる")):
        end_cross = None if last is None else last <= base if index == 0 else last >= base
        if rules.reversal_requires_reference_cross and end_cross is False:
            continue
        certain = all(state[index] for state in states)
        plausible = any(state[index] for state in states) or has_unobserved_path
        if certain and (not rules.reversal_requires_reference_cross or end_cross is True):
            definite.add(label)
        elif plausible:
            possible.add(label)
    possible -= definite
    return {
        "schema_version": 1, "session": session, "symbol": reference.symbol,
        "target_end": end.isoformat(), "target_status": "対象" if target is True else "対象外" if target is False else "未確定",
        "correct_candidates": [label for label in CLASSES if label in definite],
        "uncertain_candidates": [label for label in CLASSES if label in possible],
        "metrics": {"reference_price": float(base), "quote_price": _number(last),
                    "quote_time": bars[-1].end.isoformat() if endpoint_available else None,
                    "actual_pct": _number(terminal_pct), "high_pct": _number(high_pct), "low_pct": _number(low_pct),
                    "session_range_pct": _number(range_pct), "observed_max_abs_pct": _number(observed_excursion)},
        "quality": {"observed_rows": len(bars), "expected_rows": expected_rows, "missing_rows": missing_rows,
                    "unclosed_rows": incomplete, "incomplete_volume_rows": sum(not bar.volume_complete for bar in bars),
                    "endpoint_available": endpoint_available, "endpoint_lag_minutes": endpoint_lag,
                    "reference_lag_minutes": (ref_end - reference.end).total_seconds() / 60,
                    "has_unobserved_path": has_unobserved_path, "contract_identity_available": all(bar.contract is not None for bar in [reference, *bars]),
                    "session_range_is_lower_bound": has_unobserved_path,
                    "endpoint_status": "最終観測終値の代用" if endpoint_available else "終点観測不足",
                    "intrabar_order": "高安順序の両候補を比較。足内往復の存在は断定しない。",
                    "limitations": ["足の不足と無約定は識別していない。", "直前の営業場であることは呼出側の履歴選定で検証する。",
                                    "価格収集遅延・訂正前の版と公式終値の一致は、この入力だけでは確認できない。"]},
        "reference_hash": candle_hash([reference]), "path_hash": candle_hash(bars), "contract": rules.to_dict(),
    }


def score_prediction(prediction: str | None, evaluation: dict) -> dict:
    """確定正解・可能正解を分け、無回答や品質不明を成績分母から消さない。

    引数: prediction (str | None) は５種類の回答または無回答、evaluation (dict) は経路評価。
    返り値: dict は採点状態、正解数の下限・上限、分母採用、対象状態。
        対象不明も保守的な分母へ残し、正解下限は対象・正解の両方が確定した場合だけ１。
    """
    if prediction is not None and prediction not in CLASSES:
        raise ValueError("予測回答は指定の５種類または無回答Noneが必要です。")
    target = evaluation["target_status"]
    if target not in ("対象", "対象外", "未確定"):
        raise ValueError("対象状態が不正です。")
    correct, uncertain = set(evaluation["correct_candidates"]), set(evaluation["uncertain_candidates"])
    if not correct <= set(CLASSES) or not uncertain <= set(CLASSES) or correct & uncertain:
        raise ValueError("確定正解集合と未確定集合が不整合です。")
    lower = target == "対象" and prediction in correct
    upper = target != "対象外" and prediction is not None and prediction in correct | uncertain
    if target == "対象外":
        status = "対象外"
    elif prediction is None or not upper:
        status = "不正解"
    elif lower:
        status = "正解"
    else:
        status = "未確定"
    return {"prediction": prediction, "status": status, "target_status": target,
            "correct_lower": lower, "correct_upper": upper, "denominator": target != "対象外",
            "unanswered": prediction is None}


# ----------------------------------------
# もみ合いを優先し、最初の到達方向と最初の反転で一つに分類する別方式


def _priority_price(state: tuple[int, bool, Decimal], price: Decimal, upper: Decimal, lower: Decimal,
                    reverse: Decimal) -> tuple[int, bool, Decimal]:
    """最初の基準到達方向を固定した経路状態へ、一つの価格を反映する。

    引数: state (tuple[int, bool, Decimal]) は初回方向・反転済み・到達後極値、price (Decimal) は次価格、
        upper/lower (Decimal) は初回到達境界、reverse (Decimal) は反転に必要な価格差。
    返り値: tuple[int, bool, Decimal] は反映後状態。反転済みの方向は後から変更しない。
    """
    direction, reversed_already, extreme = state
    if direction == 0:
        if price >= upper:
            return 1, False, price
        if price <= lower:
            return -1, False, price
        return state
    if reversed_already:
        return state
    if direction == 1:
        extreme = max(extreme, price)
        if extreme - price >= reverse:
            return direction, True, Decimal(0)
    else:
        extreme = min(extreme, price)
        if price - extreme >= reverse:
            return direction, True, Decimal(0)
    return direction, False, extreme


def _priority_gap(states: set[tuple[int, bool, Decimal]], upper: Decimal,
                  lower: Decimal) -> set[tuple[int, bool, Decimal]]:
    """未観測の時間に可能な初回到達・反転を追加し、不成立を断定しない。

    引数: states (set[tuple[int, bool, Decimal]]) は観測済み候補、upper/lower (Decimal) は初回到達境界。
    返り値: set[tuple[int, bool, Decimal]] は未観測経路を含む候補。未反転では最小限の到達極値を保つ。
        未観測中の極値拡大は反転済み候補で表現し、後続の反転可能性を取り落とさない。
    """
    result = set(states)
    for direction, reversed_already, _ in states:
        if direction == 0:
            result.update(((1, False, upper), (-1, False, lower), (1, True, Decimal(0)), (-1, True, Decimal(0))))
        elif not reversed_already:
            result.add((direction, True, Decimal(0)))
    return result


def _priority_classes(reference: Decimal, bars: list[Candle], start: datetime, end: datetime,
                       contract: EvaluationContract) -> set[str]:
    """足内高安の両順序と未観測時間を追跡し、初回方向に対応する分類候補を返す。

    引数: reference (Decimal) は先物前終値、bars (list[Candle]) は時間順確定足、
        start/end (datetime) は対象場の開始終了、contract (EvaluationContract) は事前固定契約。
    返り値: set[str] はもみ合いを優先する前の経路分類候補。到達なし状態は分類しない。
    """
    initial = reference * contract.initial_excursion_pct / 100
    reverse = reference * contract.reversal_threshold_pct / 100
    upper, lower = reference + initial, reference - initial
    states, cursor = {(0, False, Decimal(0))}, start
    for bar in bars:
        if bar.time > cursor:
            states = _priority_gap(states, upper, lower)
        opened, high, low, closed = (_decimal(getattr(bar, field), field) for field in ("open", "high", "low", "close"))
        next_states = set()
        for state in states:
            for order in ((opened, high, low, closed), (opened, low, high, closed)):
                updated = state
                for price in order:
                    updated = _priority_price(updated, price, upper, lower, reverse)
                next_states.add(updated)
        states, cursor = next_states, bar.end
    if cursor < end:
        states = _priority_gap(states, upper, lower)
    names = {(1, False): "上がる", (-1, False): "下がる",
             (1, True): "上がってから下がる", (-1, True): "下がってから上がる"}
    return {names[(direction, reverse_done)] for direction, reverse_done, _ in states if direction != 0}


def evaluate_priority_path(reference: Candle, candles: list[Candle], session: str, target_end: datetime,
                           contract: dict | EvaluationContract) -> dict:
    """もみ合い最優先と初回到達方向を固定した５分類を、未確定も含め一つへ絞る。

    引数: reference (Candle) は同じ先物の前セッション終値、candles (list[Candle]) は対象場の足、
        session (str) は対象場、target_end (datetime) は公式終了予定、
        contract (dict | EvaluationContract) は全必須設定を持つ事前固定契約。
    返り値: dict は一意に確定した正解１種類、または未確定の候補集合と基準・経路の品質。
        もみ合いを優先し、不成立確定後は最初の±到達方向、その方向の最初の反転で分類する。
        反転後の再反転で分類を変更しない。通常方向に終点閾値を用いず、互換項目direction_threshold_pctは参照しない。
    """
    rules = validate_contract(contract)
    if (rules.target_basis != "場中到達" or rules.movement_threshold_pct != rules.initial_excursion_pct
            or rules.reversal_requires_reference_cross):
        raise ValueError("優先分類は場中到達、対象幅と初回到達幅の一致、終点跨ぎ不要の契約が必要です。")
    result = evaluate_path(reference, candles, session, target_end, rules)
    start, end = _session_times(session)
    bars = sorted((bar for bar in candles if bar.closed), key=lambda bar: bar.time)
    candidates = _priority_classes(_decimal(reference.close, "基準終値"), bars, start, end, rules)
    flat_certain = "もみ合い" in result["correct_candidates"]
    flat_possible = "もみ合い" in result["uncertain_candidates"]
    if flat_certain:
        definite, uncertain, flat_status = {"もみ合い"}, set(), "成立"
    elif flat_possible:
        definite, uncertain, flat_status = set(), candidates | {"もみ合い"}, "未確定"
    elif len(candidates) == 1:
        definite, uncertain, flat_status = candidates, set(), "不成立"
    else:
        definite, uncertain, flat_status = set(), candidates, "不成立"
    result["correct_candidates"] = [label for label in CLASSES if label in definite]
    result["uncertain_candidates"] = [label for label in CLASSES if label in uncertain]
    result["classification_policy"] = "もみ合い優先・初回到達方向と最初の反転"
    result["flat_status"] = flat_status
    result["path_class_candidates_before_flat"] = [label for label in CLASSES if label in candidates]
    return result
