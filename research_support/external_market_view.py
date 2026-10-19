"""締切と固定遅延に従って、外部市場の比較観測だけを組み立てる。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from typing import Any, Mapping


NOTICES = (
    "全履歴は後日収集であり、訂正履歴と初配信時計を持たない。厳密な当時受信の実証ではない。",
    "固定遅延は開発上の仮定であり、配信遅延の上限や当時の可知性を保証しない。",
    "表示対象は足終端と固定遅延が各アンカー以前の観測だけであり、欠測や鮮度超過を先行補完しない。",
    "採用足はアンカーより固定遅延以上古い観測であり、アンカー時計そのものの正確な同時価格ではない。",
    "外部系列間の比較観測だけを返す。国内先物の基準価格、材料点、方向予測、因果判定は生成しない。",
    "騰落率は表示単位における観測値の相対変化である。利回り観測を株価と同一視せず、ベーシスポイントへ換算しない。",
)
_OHLC = ("Open", "High", "Low", "Close")
_ROW_KEYS = frozenset(("Datetime", *_OHLC, "Volume", "Adj Close", "Dividends", "Stock Splits", "Capital Gains"))
_METADATA_KEYS = frozenset(("library", "library_version", "source_name", "source_url", "terms_url", "unofficial_client"))


@dataclass(frozen=True)
class InstrumentSpec:
    """呼出側が指定する系列の意味と表示単位。"""

    label: str
    unit: str
    rising_meaning: str
    series_notice: str


def _clock(value: Any, label: str) -> datetime:
    """
    機能: タイムゾーン付きの時計を検証し、協定世界時へ統一する。
    引数: value (str) はISO形式の時計、label (str) はエラー対象名。
    返り値: datetime 型のタイムゾーン付き時計。
    """
    if not isinstance(value, str):
        raise ValueError(f"{label}はタイムゾーン付きISO時計が必要です。")
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label}はタイムゾーン付きISO時計が必要です。") from error


def _digest(value: Any) -> str:
    """
    機能: JSON応答の内容を変更せず、正規化したSHA256を求める。
    引数: value (JSON互換値) は識別対象の元応答。
    返り値: str 型のSHA256十六進表記。
    """
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("元応答は有限数だけを含むJSON互換値が必要です。") from error
    return sha256(encoded).hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    """
    機能: 真偽値を除く有限数と、必要に応じて正値を検証する。
    引数: value (int | float) は数値、label (str) は対象名、positive (bool) は正値制約。
    返り値: float 型の検証済み数値。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}は有限数が必要です。")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label}は有限数が必要です。") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label}の値が不正です。")
    return result


def _bars(rows: Any) -> list[tuple[datetime, dict[str, Any]]]:
    """
    機能: 足を検証し、完全欠測と同一重複を除いた観測候補を作る。
    引数: rows (list[dict]) はDatetimeを足始端とする元データ。
    返り値: list[tuple[datetime, dict]] 型の時刻順の足。全OHLCがnullの行は候補にしない。
    """
    if not isinstance(rows, list):
        raise ValueError("dataは行の配列が必要です。")
    seen: dict[datetime, dict[str, Any]] = {}
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - _ROW_KEYS or not {"Datetime", *_OHLC} <= set(row):
            raise ValueError("足の列が許可された構成ではありません。")
        start = _clock(row["Datetime"], "Datetime")
        empty = all(row[key] is None for key in _OHLC)
        normalized: dict[str, Any] = {"Datetime": start.isoformat()}
        for key, value in row.items():
            if key == "Datetime":
                continue
            if empty and value is None:
                normalized[key] = None
                continue
            normalized[key] = _number(value, key, positive=key in _OHLC or key == "Adj Close")
            if key == "Volume" and normalized[key] < 0:
                raise ValueError("Volumeは非負値が必要です。")
        if not empty and not (normalized["Low"] <= normalized["Open"] <= normalized["High"] and normalized["Low"] <= normalized["Close"] <= normalized["High"]):
            raise ValueError("OHLCの大小関係が矛盾しています。")
        if start in seen:
            if seen[start] != normalized:
                raise ValueError("同一始端の足が一致しません。")
            continue
        seen[start] = normalized
        if not empty:
            result.append((start, normalized))
    return sorted(result, key=lambda item: item[0])


def _source(response: Mapping[str, Any]) -> dict[str, Any]:
    """
    機能: 収集時計と許可された出所情報だけを抽出する。
    引数: response (Mapping[str, Any]) は構造化された元応答。
    返り値: dict[str, Any] 型の出所。取得範囲と元行は含まない。
    """
    source: dict[str, Any] = {}
    for key in ("version", "provider", "dataset"):
        value = response.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{key}は応答識別用の文字列または整数が必要です。")
        source[key] = value
    source["collected_at"] = _clock(response.get("collected_at"), "collected_at").isoformat()
    metadata = response.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadataは辞書が必要です。")
    copied = {}
    for key in sorted(_METADATA_KEYS & metadata.keys()):
        value = metadata[key]
        if key == "unofficial_client":
            if not isinstance(value, bool):
                raise ValueError("unofficial_clientは真偽値が必要です。")
        elif not isinstance(value, str):
            raise ValueError(f"metadata.{key}は文字列が必要です。")
        copied[key] = value
    source["metadata"] = copied
    return source


def build_external_view(
    responses: Mapping[str, Any],
    *,
    as_of: str,
    anchors: Mapping[str, str],
    instrument_specs: Mapping[str, InstrumentSpec],
    interval_minutes: int = 5,
    display_lag_minutes: int = 30,
    max_staleness_minutes: int = 60,
) -> dict[str, Any]:
    """
    機能: 足終端・固定遅延・鮮度を検査し、指定アンカー間の外部比較観測を作る純粋関数。
    引数: responses (Mapping[str, dict]) はticker別応答。structuredContentラッパーも受理する。
          as_of (str) は上限時計、anchors (Mapping[str, str]) は比較名と時計。
          instrument_specs (Mapping[str, InstrumentSpec]) は系列の意味と単位。
          interval_minutes (int) は足幅、display_lag_minutes (int) は開発仮定の固定遅延、
          max_staleness_minutes (int) は足終端からアンカーまでの鮮度上限。
    返り値: dict[str, Any] 型の出所・SHA256・採用足・全アンカー対の比較。元応答を変更しない。
    """
    for name, value in (("interval_minutes", interval_minutes), ("display_lag_minutes", display_lag_minutes), ("max_staleness_minutes", max_staleness_minutes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "interval_minutes" else 0):
            raise ValueError(f"{name}の分数が不正です。")
    cutoff = _clock(as_of, "as_of")
    if not isinstance(anchors, Mapping) or not anchors:
        raise ValueError("anchorsには比較時計が必要です。")
    clocks = {}
    for name, value in anchors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("アンカー名は空でない文字列が必要です。")
        clocks[name] = _clock(value, "アンカー")
        if clocks[name] > cutoff:
            raise ValueError("アンカーはas_of以下である必要があります。")
    if not isinstance(responses, Mapping) or not responses or set(responses) != set(instrument_specs):
        raise ValueError("応答と系列仕様のtickerが一致する必要があります。")
    for ticker, spec in instrument_specs.items():
        if not isinstance(ticker, str) or not ticker or not isinstance(spec, InstrumentSpec) or any(not isinstance(value, str) or not value for value in asdict(spec).values()):
            raise ValueError("tickerと系列仕様の各項目には空でない文字列が必要です。")

    # ----------------------------------------
    # 元応答の識別と、各アンカーまでに利用できる足の選択。
    result: dict[str, Any] = {
        "schema_version": 1,
        "as_of": cutoff.isoformat(),
        "anchors": {name: value.isoformat() for name, value in clocks.items()},
        "assumptions": {"interval_minutes": interval_minutes, "display_lag_minutes": display_lag_minutes, "max_staleness_minutes": max_staleness_minutes, "timestamp_meaning": "足始端"},
        "raw_responses_sha256": _digest(responses),
        "notices": list(NOTICES),
        "instruments": {},
    }
    try:
        width = timedelta(minutes=interval_minutes)
        lag = timedelta(minutes=display_lag_minutes)
    except OverflowError as error:
        raise ValueError("足幅または固定遅延が日時の表現範囲を超えています。") from error
    ordered_names = sorted(clocks, key=lambda name: (clocks[name], name))
    for ticker in sorted(responses):
        raw = responses[ticker]
        if not isinstance(raw, dict):
            raise ValueError("各応答は辞書が必要です。")
        response = raw.get("structuredContent", raw)
        if not isinstance(response, dict):
            raise ValueError("structuredContentは辞書が必要です。")
        source = _source(response)
        bars = _bars(response.get("data"))
        selected = {}
        for name, clock in clocks.items():
            try:
                latest_start = clock - width - lag
            except OverflowError as error:
                raise ValueError("足幅または固定遅延がアンカー時計の表現範囲を超えています。") from error
            eligible = [(start, row) for start, row in bars if start <= latest_start]
            if not eligible:
                selected[name] = {"status": "欠測", "anchor_at": clock.isoformat(), "bar": None, "age_minutes": None}
                continue
            start, row = eligible[-1]
            end = start + width
            age = (clock - end).total_seconds() / 60
            selected[name] = {
                "status": "利用可" if age <= max_staleness_minutes else "鮮度超過",
                "anchor_at": clock.isoformat(),
                "age_minutes": age,
                "bar": {"start": start.isoformat(), "end": end.isoformat(), "assumed_visible_at": (end + lag).isoformat(), "close": row["Close"]},
            }

        # ----------------------------------------
        # 明示されたアンカー間の単純な観測差。欠測と鮮度超過は計算しない。
        comparisons = []
        for first, second in combinations(ordered_names, 2):
            left, right = selected[first], selected[second]
            comparison = {"from": first, "to": second, "anchor_elapsed_minutes": (clocks[second] - clocks[first]).total_seconds() / 60, "bar_elapsed_minutes": None, "absolute_change": None, "relative_change_pct": None, "unit": instrument_specs[ticker].unit}
            if left["status"] != "利用可" or right["status"] != "利用可":
                comparison["status"] = "欠測" if "欠測" in (left["status"], right["status"]) else "鮮度超過"
            else:
                left_bar, right_bar = left["bar"], right["bar"]
                comparison["bar_elapsed_minutes"] = (_clock(right_bar["end"], "足終端") - _clock(left_bar["end"], "足終端")).total_seconds() / 60
                difference = right_bar["close"] - left_bar["close"]
                comparison["absolute_change"] = difference
                comparison["relative_change_pct"] = difference / left_bar["close"] * 100
                if not math.isfinite(comparison["relative_change_pct"]):
                    raise ValueError("相対変化率が有限数の表現範囲を超えています。")
                comparison["status"] = "同一足" if left_bar["start"] == right_bar["start"] else ("同値" if difference == 0 else "比較可")
            comparisons.append(comparison)
        result["instruments"][ticker] = {"raw_response_sha256": _digest(raw), "source": source, "meaning": asdict(instrument_specs[ticker]), "observations": selected, "comparisons": comparisons}
    return result
