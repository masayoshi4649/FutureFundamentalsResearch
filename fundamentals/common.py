"""日時、証跡保存、経験分布の共通処理。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))


def timestamp(value: str | datetime) -> datetime:
    """時差付き日時を解析する。

    引数: value (str | datetime) は時差付き日時。
    返り値: datetime は日本時間へ変換した日時。
    """
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("日時には時差が必要です。")
    return parsed.astimezone(JST)


def now() -> str:
    """現在の記録時刻を取得する。

    引数: なし。
    返り値: str は時差付きの日本時間。
    """
    return datetime.now(JST).isoformat()


def read_json(path: Path) -> Any:
    """日本語の構造化ファイルを読み込む。

    引数: path (Path) は入力ファイル。
    返り値: Any は解析した値。
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value: Any) -> str:
    """構造化データの変更検知用ハッシュを計算する。

    引数: value (Any) は構造化可能な値。
    返り値: str は標準化した内容の SHA-256。
    """
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any, exclusive: bool = True) -> None:
    """証跡を日本語で保存する。

    引数: path (Path) は保存先、value (Any) は保存値、exclusive (bool) は上書き禁止の指定。
    返り値: None。既存ファイルへは標準で書き込まない。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def quantile(values: list[float], probability: float) -> float | None:
    """線形補間した経験分位点を求める。

    引数: values (list[float]) は有限の標本、probability (float) は０から１の分位。
    返り値: float | None は分位点。空標本では None。
    """
    if not 0 <= probability <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("分位点には有限値と０から１の確率が必要です。")
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def finite(value: Any, lower: float, upper: float, label: str) -> float:
    """数値の範囲と有限性を検証する。

    引数: value (Any) は値、lower/upper (float) は両端、label (str) は項目名。
    返り値: float は検証済み数値。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} は数値が必要です。")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{label} は {lower} 以上 {upper} 以下の有限値が必要です。")
    return result
