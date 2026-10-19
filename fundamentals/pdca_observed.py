"""確定済み先物足から、次セッション前の既反応量と欠測を保存する。"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

from .common import JST, timestamp
from .engine import candle_hash, reference_quote, scheduled_end, session_plan
from .prices import Candle, load_candles, session_groups


def _bounds(session: str) -> tuple[datetime, datetime]:
    """セッション名から既定の立会開始と終了予定を取得する。

    引数: session (str) は日付と Day または Night の識別子。
    返り値: tuple[datetime, datetime] は日本時間の開始と終了予定。
    """
    day = datetime.strptime(session[:10], "%Y-%m-%d").date()
    start = time(17) if session.endswith("_Night") else time(8, 45)
    return datetime.combine(day, start, JST), scheduled_end(session)


def _quote(candle: Candle) -> dict:
    """１分足の最後の観測値を、公式終値の代用として記録する。

    引数: candle (Candle) は終了済みの１分足。
    返り値: dict は価格、足の開始・終了、セッションと末尾未観測分。
    """
    return {
        "price": candle.close,
        "time": candle.end.isoformat(),
        "bar_start": candle.time.isoformat(),
        "session": candle.session,
        "source": "candle_1min",
        "contract": candle.contract,
        "is_official_close": False,
        "tail_missing_minutes": int((scheduled_end(candle.session) - candle.end).total_seconds() / 60),
        "description": "同じ先物の最終観測値。クロージング・オークションを含む公式終値の代用。",
    }


def _late_window(rows: list[Candle], end: datetime) -> dict:
    """公式終了前60分の観測点間変化と、窓内の欠測を記録する。

    引数: rows (list[Candle]) は当該セッションの確定１分足、end (datetime) は終了予定。
    返り値: dict は窓の予定・実観測時刻、足数、完全性と観測点間変化率。
    """
    start = end - timedelta(minutes=60)
    selected = [row for row in rows if start <= row.time and row.end <= end]
    return {
        "nominal_start": start.isoformat(),
        "nominal_end": end.isoformat(),
        "observed_start": selected[0].time.isoformat() if selected else None,
        "observed_end": selected[-1].end.isoformat() if selected else None,
        "observed_candles": len(selected),
        "expected_candles": 60,
        "missing_interval_slots": 60 - len(selected),
        "complete": len(selected) == 60,
        "start_price": selected[0].open if selected else None,
        "end_price": selected[-1].close if selected else None,
        "change_pct": (selected[-1].close / selected[0].open - 1) * 100 if selected else None,
        "description": "窓内で最初に観測した足の始値から最後の足の終値への変化。欠測時は完全な60分の変化ではない。",
    }


def _session_state(group: list[Candle], visible: list[Candle]) -> dict:
    """一つ前の同先物基準と、終了済みセッションの既反応量を計算する。

    引数: group (list[Candle]) は同一セッションの確定足、visible (list[Candle]) は締切内の価格。
    返り値: dict は騰落率、高安、戻り、終値位置、欠測と価格証跡。
    """
    session = group[0].session
    start, end = _bounds(session)
    rows = [row for row in group if start <= row.time and row.end <= end]
    if not rows:
        raise ValueError("セッションの立会時間内に確定１分足がありません。")
    historical_as_of = start - timedelta(minutes=1)
    before = [row for row in visible if row.end <= historical_as_of]
    try:
        reference = reference_quote(before, historical_as_of, session_plan(historical_as_of))
        reference_reason = None
    except ValueError as error:
        reference, reference_reason = None, str(error)

    # ----------------------------------------

    high, low, last = max(row.high for row in rows), min(row.low for row in rows), rows[-1]
    expected = int((end - start).total_seconds() / 60)
    internal_missing = sum(int((current.time - previous.end).total_seconds() / 60)
                           for previous, current in zip(rows, rows[1:]))
    return {
        "session": session,
        "nominal_start": start.isoformat(),
        "nominal_end": end.isoformat(),
        "reference": _quote(reference) if reference else None,
        "reference_reason": reference_reason,
        "last": _quote(last),
        "return_pct": (last.close / reference.close - 1) * 100 if reference else None,
        "high": high,
        "low": low,
        "low_to_last_pct": (last.close / low - 1) * 100,
        "close_location": (last.close - low) / (high - low) if high > low else None,
        "range_pct": (high - low) / reference.close * 100 if reference else None,
        "range_denominator": "前セッション基準" if reference else "直前基準欠損",
        "observed_candles": len(rows),
        "expected_candles": expected,
        "missing_interval_slots": expected - len(rows),
        "internal_missing_minutes": internal_missing,
        "leading_missing_minutes": int((rows[0].time - start).total_seconds() / 60),
        "tail_missing_minutes": int((end - last.end).total_seconds() / 60),
        "complete": len(rows) == expected,
        "incomplete_volume_candles": sum(not row.volume_complete for row in rows),
        "late_window": _late_window(rows, end),
        "price_hash": candle_hash(rows),
    }


# ----------------------------------------
# 情報締切内の入力からだけ生成する公開インターフェース


def observed_state(prices_root: Path, symbol: str, as_of: str | datetime) -> dict:
    """予測締切までの直前２セッションの既反応量と先物基準を取得する。

    公式終了予定を過ぎたセッションだけを使用する。基準は既存の選択契約に
    従い、末尾10分以内の１分足最終観測値に固定する。欠損を５分足や古い
    同種セッションで補完せず、価格変動を材料の因果的な織込み率とみなさない。

    引数: prices_root (Path) は読取専用価格ルート、symbol (str) は NK225 または TOPIX、
        as_of (str | datetime) は時差付きの開始前予測締切。
    返り値: dict は基準、直前２観測セッション、欠測、利用可否と制約。
        未収録や末尾不足は基準不足と null で返し、不正な入力形式は例外。
    """
    cutoff = timestamp(as_of)
    plan = session_plan(cutoff)
    visible = load_candles(Path(prices_root), symbol, interval=1, cutoff=cutoff)
    groups = [group for group in session_groups(visible) if scheduled_end(group[0].session) <= cutoff]
    try:
        reference = reference_quote(visible, cutoff, plan)
        reason = None
    except ValueError as error:
        reference, reason = None, str(error)
    states = [_session_state(group, visible) for group in groups[-2:]]
    return {
        "schema_version": "1",
        "as_of": cutoff.isoformat(),
        "symbol": symbol,
        "target_session": plan["session"],
        "reference": _quote(reference) if reference else None,
        "status": "利用可能" if reference else "基準不足",
        "reason": reason,
        "previous_session": states[-1] if states else None,
        "preceding_session": states[-2] if len(states) >= 2 else None,
        "limitations": [
            "１分足だけを使用し、基準・最終値は公式終値の代用。引け板寄せ価格は補完しない。",
            "CSVには受信・保存時刻と訂正履歴がなく、当時の実際の取得可能性は証明できない。",
            "限月情報がないため、同一限月の継続性は未確認。",
            "欠測は予定１分枠に足がない状態。無約定との区別や取引所休日の照合は未実施。",
            "直前２観測セッションを保存し、欠落したセッションや価格を推測で補わない。",
            "既反応量は価格の観測事実であり、特定材料による因果的な織込み率ではない。",
        ],
    }
