"""先物の保存済み価格を読み取り、予測時点で確定した足だけを取り出す。"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


JST = timezone(timedelta(hours=9))
SYMBOLS = ("NK225", "TOPIX")
REQUIRED_COLUMNS = {
    "time", "open", "high", "low", "close", "volume", "volume_complete", "closed"
}
SESSION_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_(?:Day|Night)\Z")


@dataclass(frozen=True)
class Candle:
    """開始・終了時刻と出典を持つ確定前後の価格足を表す。"""

    time: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    session: str
    symbol: str
    contract: str | None = None
    closed: bool = True
    volume_complete: bool = True


# ----------------------------------------
# CSV の検証と読み取り


def _aware_time(value: datetime, name: str) -> datetime:
    """
    タイムゾーン付き日時を検証し、日本時間へ変換する。

    引数:
        value (datetime): 検証する日時。
        name (str): エラーに表示する項目名。
    返り値:
        datetime: 日本時間の日時。
    """
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} にはタイムゾーン付き日時が必要です。")
    return value.astimezone(JST)


def _boolean(value: str | None, name: str) -> bool:
    """
    CSV の真偽値を厳密に読み取る。

    引数:
        value (str | None): true または false の文字列。
        name (str): 対象の列名。
    返り値:
        bool: 読み取った真偽値。
    """
    if value not in ("true", "false"):
        raise ValueError(f"{name} は true または false である必要があります。")
    return value == "true"


def _data_directory(root: Path, symbol: str, interval: int) -> Path:
    """
    銘柄と足間隔を検証して CSV フォルダを求める。

    引数:
        root (Path): FutureAlgorithmResearch のルート。
        symbol (str): NK225 または TOPIX。
        interval (int): 足の分数。1 または 5。
    返り値:
        Path: 読み取り対象フォルダ。
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"未対応の銘柄です: {symbol}")
    if type(interval) is not int or interval not in (1, 5):
        raise ValueError("足間隔は 1 分または 5 分を指定してください。")
    directory = Path(root) / f"candle_{interval}min" / symbol
    if not directory.is_dir():
        raise FileNotFoundError(f"価格フォルダがありません: {directory}")
    return directory


def _session_start(path: Path) -> datetime:
    """
    ファイル名が表すセッションの開始時刻を求める。

    引数:
        path (Path): YYYY-MM-DD_Day または Night を名前に持つファイル。
    返り値:
        datetime: 日中は 08:45、夜間は 17:00 とした日本時間の開始時刻。
    """
    if not SESSION_PATTERN.fullmatch(path.stem):
        raise ValueError(f"セッション名が不正です: {path.name}")
    session_date = datetime.strptime(path.stem[:10], "%Y-%m-%d").replace(tzinfo=JST)
    if path.stem.endswith("_Day"):
        return session_date.replace(hour=8, minute=45)
    return session_date.replace(hour=17)


def _read_file(
    path: Path,
    symbol: str,
    interval: int,
    cutoff: datetime | None = None,
) -> list[Candle]:
    """
    1 セッションの CSV を読み取り、時刻・価格・フラグを検証する。

    引数:
        path (Path): CSV ファイル。
        symbol (str): 対象銘柄。
        interval (int): 足の分数。
        cutoff (datetime | None): 終了時刻の上限。省略時は全行を検証する。
    返り値:
        list[Candle]: 上限までの足。未確定足も含む。
    """
    _session_start(path)
    candles = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if "time" not in columns or (cutoff is None and not REQUIRED_COLUMNS.issubset(columns)):
            raise ValueError(f"必須列が不足しています: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                started = _aware_time(datetime.fromisoformat(row["time"]), "time")
                ended = started + timedelta(minutes=interval)
                if cutoff is not None and ended > cutoff:
                    continue
                if not REQUIRED_COLUMNS.issubset(columns):
                    raise ValueError("必須列が不足しています。")
                values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("価格と出来高には有限値が必要です。")
                if min(values[key] for key in ("open", "high", "low", "close")) <= 0:
                    raise ValueError("価格は正の数である必要があります。")
                if values["volume"] < 0:
                    raise ValueError("出来高を負数にはできません。")
                if not (
                    values["low"] <= min(values["open"], values["close"])
                    <= max(values["open"], values["close"]) <= values["high"]
                ):
                    raise ValueError("始値・終値が高値と安値の範囲外です。")
                if started.second or started.microsecond or started.minute % interval:
                    raise ValueError("足開始時刻が足間隔に一致していません。")
                if candles and started < candles[-1].end:
                    raise ValueError("足の時刻が重複または逆転しています。")
                candle = Candle(
                    time=started,
                    end=ended,
                    **values,
                    session=path.stem,
                    symbol=symbol,
                    contract=None,
                    closed=_boolean(row["closed"], "closed"),
                    volume_complete=_boolean(row["volume_complete"], "volume_complete"),
                )
                candles.append(candle)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return candles


def load_candles(
    root: Path,
    symbol: str,
    interval: int = 1,
    cutoff: datetime | None = None,
) -> list[Candle]:
    """
    保存済み足を読み取り、指定時刻までに終了した確定足を返す。

    CSV の time は足の開始時刻と扱う。確定フラグが true でも終了時刻が
    cutoff を超える足は含めない。CSV に限月がないため contract は None。
    保存・受信時刻がないので、過去の収集遅延や後日訂正は検証できない。
    上限より後の足の数値・フラグは検証せず、ファイル名上の開始時刻が
    上限以降のセッションも開かない。開始は日中 08:45・夜間 17:00 とする。
    対象ファイル内の日時が不正な行は前後を判別できないため例外とする。

    引数:
        root (Path): FutureAlgorithmResearch のルート。
        symbol (str): NK225 または TOPIX。
        interval (int): 足の分数。1 または 5。
        cutoff (datetime | None): 利用可能時刻の上限。タイムゾーン必須。
    返り値:
        list[Candle]: 終了時刻が上限以下の確定足。上限省略時は全確定足。
    """
    directory = _data_directory(root, symbol, interval)
    limit = _aware_time(cutoff, "cutoff") if cutoff is not None else None
    candles = []
    for path in sorted(directory.glob("*.csv")):
        if limit is not None and _session_start(path) >= limit:
            continue
        for candle in _read_file(path, symbol, interval, limit):
            if candle.closed and (limit is None or candle.end <= limit):
                candles.append(candle)
    candles.sort(key=lambda candle: candle.time)
    for previous, current in zip(candles, candles[1:]):
        if current.time < previous.end:
            raise ValueError(f"セッションをまたいだ足の重複があります: {current.time.isoformat()}")
    return candles


def session_groups(candles: list[Candle]) -> list[list[Candle]]:
    """
    足を銘柄とセッションでまとめ、時系列に並べる。

    引数:
        candles (list[Candle]): 対象の足。入力順は任意。
    返り値:
        list[list[Candle]]: 同じ銘柄・セッションに属する足の時系列リスト。
    """
    groups: dict[tuple[str, str], list[Candle]] = {}
    for candle in sorted(candles, key=lambda item: (item.time, item.symbol)):
        groups.setdefault((candle.symbol, candle.session), []).append(candle)
    return list(groups.values())


# ----------------------------------------
# データ範囲の点検と生板の限定読み取り


def inventory(root: Path) -> dict:
    """
    対象価格の件数・期間・同一セッション内の欠測候補を集計する。

    足間の空白だけを欠測候補とし、閉場や週末を欠損扱いしない。
    先頭末尾の欠測や休業日の正否は、取引所カレンダーなしには確定できない。

    引数:
        root (Path): FutureAlgorithmResearch のルート。
    返り値:
        dict: 銘柄別・足別の件数、範囲、欠測候補、生板容量と制約。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"価格ルートがありません: {root}")
    result = {
        "root": str(root.resolve()),
        "symbols": {},
        "limitations": [
            "CSV に限月・受信時刻・保存時刻はなく、過去の収集遅延や訂正履歴を復元できません。",
            "生板の CurrentPriceTime は受信時刻ではなく、古い時刻や null を含みます。",
            "欠測候補は同一セッション内の足の空白です。無約定による空白との区別は未確認です。",
            "取引所カレンダーとの照合を行わず、セッション先頭・末尾や日付全体の欠落は判定しません。",
        ],
    }
    for symbol in SYMBOLS:
        symbol_result = {}
        for interval in (1, 5):
            directory = root / f"candle_{interval}min" / symbol
            files = sorted(directory.glob("*.csv"))
            all_rows = []
            empty_files = []
            for path in files:
                rows = _read_file(path, symbol, interval)
                all_rows.extend(rows)
                if not rows:
                    empty_files.append(path.name)
            all_rows.sort(key=lambda candle: candle.time)
            missing_slots = 0
            gap_count = 0
            for group in session_groups(all_rows):
                for previous, current in zip(group, group[1:]):
                    if current.time > previous.end:
                        gap_count += 1
                        missing_slots += int((current.time - previous.end).total_seconds() // (60 * interval))
            closed_rows = [candle for candle in all_rows if candle.closed]
            symbol_result[f"{interval}min"] = {
                "files": len(files),
                "rows": len(all_rows),
                "closed_rows": len(closed_rows),
                "first_time": all_rows[0].time.isoformat() if all_rows else None,
                "last_time": all_rows[-1].time.isoformat() if all_rows else None,
                "last_end": all_rows[-1].end.isoformat() if all_rows else None,
                "incomplete_volume_rows": sum(not candle.volume_complete for candle in all_rows),
                "missing_interval_slots": missing_slots,
                "gap_count": gap_count,
                "sessions": [path.stem for path in files],
                "files_without_rows": empty_files,
            }
        book_files = sorted((root / "book" / symbol).glob("*.jsonl"))
        symbol_result["book"] = {
            "files": len(book_files),
            "bytes": sum(path.stat().st_size for path in book_files),
            "first_session": book_files[0].stem if book_files else None,
            "last_session": book_files[-1].stem if book_files else None,
        }
        result["symbols"][symbol] = symbol_result
    return result


def read_book_sample(path: Path, limit: int = 1) -> list[dict]:
    """
    生板 JSONL の先頭から指定件数だけを読み取る。

    収集時刻がないため、CurrentPriceTime による過去時点の利用可否判定には
    使わない。これは形式確認専用であり予測用の板特徴量を生成しない。

    引数:
        path (Path): 生板 JSONL ファイル。
        limit (int): 最大読み取り件数。0 以上。
    返り値:
        list[dict]: ファイル先頭の生板オブジェクト。
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("読み取り件数は 0 以上の整数にしてください。")
    sample = []
    if limit == 0:
        return sample
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: 生板は JSON オブジェクトである必要があります。")
            sample.append(record)
            if len(sample) >= limit:
                break
    return sample
