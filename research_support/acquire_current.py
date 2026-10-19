"""満了後の当日１分足を読取APIから取得し、原本とは別の価格ルートを作る。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from fundamentals.common import now, timestamp, write_json
from fundamentals.engine import scheduled_end
from fundamentals.prices import SYMBOLS, SESSION_PATTERN, load_candles
from fundamentals.replay import _file_hash


BASE_URL = "http://10.10.100.1:8080/api/v1/history"
COLUMNS = ("time", "open", "high", "low", "close", "volume", "volume_complete", "closed")


def validate_response(response: dict, symbol: str, session: str, observed_at: str) -> list[dict]:
    """APIのセッション・取得時計と足の確定状態を検証してCSV列へ写す。

    引数: response (dict) はdata/snapshotを持つ応答、symbol (str) は先物、session (str)
    は対象セッション、observed_at (str) は取得完了の実時計。
    返り値: list[dict] は確定済みかつ名目終点以前の足。未来行の価格は解釈しない。
    """
    if symbol not in SYMBOLS or not SESSION_PATTERN.fullmatch(session):
        raise ValueError("対象先物またはセッション形式が不正です。")
    end, observed = scheduled_end(session), timestamp(observed_at)
    start = timestamp(session[:10] + ("T08:45:00+09:00" if session.endswith("_Day") else "T17:00:00+09:00"))
    if observed < end:
        raise ValueError("公式終了予定が来るまで結果の取得はできません。")
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise ValueError("API応答にはdata配列が必要です。")
    snapshot = response.get("snapshot", {})
    if snapshot.get("sessionId") != session or snapshot.get("code") != symbol:
        raise ValueError("APIが返したセッションまたは先物が異なります。")
    remote_time = timestamp(snapshot["asOf"])
    if not end <= remote_time <= observed:
        raise ValueError("収集側のスナップショット時計が満了前または未来です。")
    if timestamp(snapshot["session"]["endsAt"]) != end:
        raise ValueError("収集側と評価側の終了予定が一致しません。")
    rows = []
    for row in response["data"]:
        started = timestamp(row["time"])
        if started < start:
            raise ValueError("当日API応答に別セッションの足が含まれています。")
        if started + timedelta(minutes=1) > end:
            continue
        if type(row.get("closed")) is not bool or type(row.get("volumeComplete")) is not bool:
            raise ValueError("APIの確定・出来高フラグは真偽値が必要です。")
        if not row["closed"]:
            continue
        rows.append({**{key: row[key] for key in COLUMNS[:6]},
                     "volume_complete": "true" if row["volumeComplete"] else "false", "closed": "true"})
    if not rows:
        raise ValueError("満了セッションの確定足がありません。")
    return rows


def acquire(source: Path, output: Path, session: str, base_url: str = BASE_URL) -> dict:
    """満了後だけAPIを読み、過去原本のコピーと当日足を別ルートへ保存する。

    引数: source (Path) は読取専用の元価格、output (Path) は未使用の保存先、
    session (str) は対象セッション、base_url (str) は親リポジトリ記載の読取API。
    返り値: dict は取得完了時刻・スナップショット・原本とコピーのハッシュ。
    """
    if not SESSION_PATTERN.fullmatch(session):
        raise ValueError("対象は日付_Dayまたは日付_Nightで指定してください。")
    end = scheduled_end(session)
    if timestamp(now()) < end:
        raise ValueError("満了前の結果取得を拒否します。")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("既存の保存先は上書きできません。")
    if output == source or source in output.parents:
        raise ValueError("元価格の外に新しい保存先を用意してください。")
    downloaded = {}
    for symbol in SYMBOLS:
        query = urlencode({"to": end.isoformat()})
        url = base_url.rstrip("/") + f"/{symbol}/candles/1min?" + query
        with urlopen(url, timeout=15) as response:
            raw = response.read(8_000_000)
        received_at = now()
        payload = json.loads(raw.decode("utf-8-sig"))
        rows = validate_response(payload, symbol, session, received_at)
        downloaded[symbol] = {"url": url, "received_at": received_at, "payload": payload, "rows": rows}

    # ----------------------------------------

    output.mkdir(parents=True)
    copied, acquired = [], {}
    for symbol in SYMBOLS:
        directory = output / "candle_1min" / symbol
        directory.mkdir(parents=True)
        originals = sorted((source / "candle_1min" / symbol).glob("*.csv"))
        if not originals:
            raise ValueError("元の先物１分足がありません。")
        for path in originals:
            if path.stem == session:
                continue
            destination = directory / path.name
            with path.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            original_hash = _file_hash(path)
            if original_hash != _file_hash(destination):
                raise ValueError("コピー中に過去原本が変わりました。")
            copied.append({"source": str(path), "copy": str(destination), "file_hash": original_hash})
        item = downloaded[symbol]
        raw_path = output / (symbol + "_API原本.json")
        write_json(raw_path, {key: item[key] for key in ("url", "received_at", "payload")})
        csv_path = directory / (session + ".csv")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(item["rows"])
        candles = load_candles(output, symbol, cutoff=timestamp(item["received_at"]))
        current = [candle for candle in candles if candle.session == session]
        if not current or not timedelta(0) <= end - current[-1].end <= timedelta(minutes=10):
            raise ValueError("結果価格の代理が公式終了前10分以内にありません。")
        acquired[symbol] = {"raw_hash": _file_hash(raw_path), "csv_hash": _file_hash(csv_path),
                            "snapshot": item["payload"]["snapshot"], "received_at": item["received_at"],
                            "rows": len(current), "last_end": current[-1].end.isoformat(),
                            "is_official_close": False}
    manifest = {"created_at": now(), "session": session, "source": str(source), "copied": copied,
                "acquired": acquired, "limitations": ["親リポジトリの原本は変更していない。",
                 "APIの１分足最終観測値。公式引け板寄せ価格を取得したとはみなさない。",
                 "今回の取得時計は保存するが、過去CSVの受信履歴は復元できない。"]}
    write_json(output / "取得証跡.json", manifest)
    return manifest


# ----------------------------------------
# セッション切替後の確定保存ファイル


def acquire_archived(source: Path, output: Path, session: str, archive: Path) -> dict:
    """満了後に確定保存されたCSVを検査し、過去原本と別の価格ルートへ写す。

    引数: source (Path) は読取専用の過去価格、output (Path) は未使用保存先、
    session (str) は対象場、archive (Path) は親リポジトリ記載の読取共有ルート。
    返り値: dict は読取前後のサイズ・更新時計、原本ハッシュ、取得時計とコピー証跡。
    """
    if not SESSION_PATTERN.fullmatch(session) or timestamp(now()) < scheduled_end(session):
        raise ValueError("正しいセッションの公式終了予定後だけ取得できます。")
    source, output, archive = Path(source).resolve(), Path(output).resolve(), Path(archive).resolve()
    if output.exists():
        raise FileExistsError("保存済みの追加価格は上書きできません。")
    if output == source or source in output.parents or output == archive or archive in output.parents:
        raise ValueError("元価格と共有元の外へ保存してください。")
    downloaded = {}
    for symbol in SYMBOLS:
        original = archive / "candle_1min" / symbol / (session + ".csv")
        before = original.stat()
        body = original.read_bytes()
        after = original.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(body) != after.st_size:
            raise ValueError("書込み中の保存ファイルは取得できません。")
        downloaded[symbol] = {"path": original, "body": body, "size": after.st_size,
                              "mtime_ns": after.st_mtime_ns, "received_at": now()}
    output.mkdir(parents=True)
    copied, acquired = [], {}
    for symbol in SYMBOLS:
        directory = output / "candle_1min" / symbol
        directory.mkdir(parents=True)
        originals = sorted((source / "candle_1min" / symbol).glob("*.csv"))
        if not originals:
            raise ValueError("コピーする過去１分足がありません。")
        for original in originals:
            if original.stem == session:
                continue
            destination = directory / original.name
            with original.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            original_hash = _file_hash(original)
            if original_hash != _file_hash(destination):
                raise ValueError("コピー中に過去原本が変わりました。")
            copied.append({"source": str(original), "copy": str(destination), "file_hash": original_hash})
        item = downloaded[symbol]
        destination = directory / (session + ".csv")
        with destination.open("xb") as stream:
            stream.write(item["body"])
        raw = output / (symbol + "_保存原本.csv")
        with raw.open("xb") as stream:
            stream.write(item["body"])
        snapshot = item["path"].stat()
        source_hash = _file_hash(item["path"])
        if ((snapshot.st_size, snapshot.st_mtime_ns) != (item["size"], item["mtime_ns"])
                or source_hash != _file_hash(raw)):
            raise ValueError("取得中に共有の保存原本が変わりました。")
        candles = load_candles(output, symbol, cutoff=timestamp(item["received_at"]))
        current = [candle for candle in candles if candle.session == session]
        if not current or not timedelta(0) <= scheduled_end(session) - current[-1].end <= timedelta(minutes=10):
            raise ValueError("結果価格は終了予定前10分以内の代理が必要です。")
        acquired[symbol] = {"raw_hash": _file_hash(raw), "csv_hash": _file_hash(destination),
                            "snapshot": {"source": str(item["path"]), "size": item["size"],
                                         "mtime_ns": item["mtime_ns"], "source_hash": source_hash},
                            "received_at": item["received_at"], "rows": len(current),
                            "last_end": current[-1].end.isoformat(), "is_official_close": False}
    manifest = {"created_at": now(), "method": "満了後の確定保存CSV", "session": session,
                "source": str(source), "archive": str(archive), "copied": copied, "acquired": acquired,
                "limitations": ["元価格と共有元は読取のみ。", "１分足最終観測値は公式終値の代理。",
                                "読取中の安定性は確認したが、過去の受信・訂正履歴は復元できない。"]}
    write_json(output / "取得証跡.json", manifest)
    return manifest


def main() -> None:
    """満了後の価格取得だけをコマンド行から実行する。

    引数: なし。元価格・未使用保存先・対象セッションを指定する。
    返り値: None。取得件数と最終足時刻を表示する。予測や採点は別操作。
    """
    parser = argparse.ArgumentParser(description="読取APIから満了後の価格を別フォルダへ取得")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--archive", type=Path, help="API切替後の確定CSV共有ルート。指定時はHTTPを使わない")
    args = parser.parse_args()
    value = (acquire_archived(args.source, args.out, args.session, args.archive) if args.archive else
             acquire(args.source, args.out, args.session))
    print({symbol: {key: item[key] for key in ("rows", "last_end")} for symbol, item in value["acquired"].items()})


if __name__ == "__main__":
    main()
