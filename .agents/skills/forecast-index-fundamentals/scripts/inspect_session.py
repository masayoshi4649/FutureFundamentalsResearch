"""確定１分足から現在の価格反応と事前に指定した反証条件を点検する。"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path


MINUTE = timedelta(minutes=1)


def parse_time(value):
    """
    機能: タイムゾーンを必須として日時を解釈する。
    引数: value (str): RFC3339形式の日時。
    返り値: datetime: タイムゾーン付き日時。
    """
    if not isinstance(value, str):
        raise ValueError("日時はタイムゾーン付き文字列が必要です。")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
        raise ValueError("日時はRFC3339形式とタイムゾーンが必要です。")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("日時を解釈できません。") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("日時にはタイムゾーンが必要です。")
    return parsed


def positive_number(value, label):
    """
    機能: 価格が有限の正数であることを検証する。
    引数: value (object): 入力値。label (str): エラー時の項目名。
    返り値: float: 検証した価格。
    """
    if isinstance(value, bool):
        raise ValueError(f"{label}は有限の正数が必要です。")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}は有限の正数が必要です。") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label}は有限の正数が必要です。")
    return result


def boolean_value(value, label, required=True):
    """
    機能: 真偽値とCSVの真偽文字列を検証する。
    引数: value (object): 入力値。label (str): 項目名。required (bool): 必須指定。
    返り値: bool | None: 真偽値、任意項目が空なら未確認。
    """
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    if not required and (value is None or value == ""):
        return None
    raise ValueError(f"{label}はtrueまたはfalseが必要です。")


def read_input(path):
    """
    機能: 元入力を変更せず、収録JSONまたは１分足CSVを読む。
    引数: path (Path): 入力ファイル。
    返り値: tuple[list[dict], dict]: 足一覧と出典情報。
    """
    raw = path.read_bytes()
    content = raw.decode("utf-8-sig")
    if content.lstrip().startswith(("{", "[")):
        document = json.loads(content)
        if not isinstance(document, dict):
            raise ValueError("JSONの最上位はdata配列を持つオブジェクトが必要です。")
        rows = document.get("data")
        if not isinstance(rows, list):
            raise ValueError("JSONには足配列dataが必要です。")
        source_format = "json"
    else:
        reader = csv.DictReader(io.StringIO(content))
        required = {"time", "open", "high", "low", "close", "closed"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSVにtime・OHLC・closedの必須列がありません。")
        rows = list(reader)
        source_format = "csv"
    return rows, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": source_format,
        "snapshot_usage": "snapshotは計算に使用しない。確定判定はas_ofとclosedによる。",
    }


def normalize_rows(rows):
    """
    機能: 時刻・OHLC・確定状態を検証し、同一時刻の重複を拒否する。
    引数: rows (list[dict]): 元入力の１分足。
    返り値: list[dict]: 時刻順の正規化した足。
    """
    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("各足はオブジェクトが必要です。")
        start = parse_time(row.get("time"))
        if start.second or start.microsecond:
            raise ValueError("足の開始時刻は分境界である必要があります。")
        if start in seen:
            raise ValueError("同一開始時刻の足が重複しています。")
        seen.add(start)
        prices = {key: positive_number(row.get(key), key)
                  for key in ("open", "high", "low", "close")}
        if not (prices["low"] <= min(prices["open"], prices["close"])
                <= max(prices["open"], prices["close"]) <= prices["high"]):
            raise ValueError("OHLCの大小関係が不正です。")
        normalized.append({
            "start": start,
            "end": start + MINUTE,
            **prices,
            "closed": boolean_value(row.get("closed"), "closed"),
            "volume_complete": boolean_value(
                row.get("volumeComplete", row.get("volume_complete")),
                "volumeComplete", required=False),
        })
    return sorted(normalized, key=lambda row: row["start"])


def price_summary(rows, p0, reference_price):
    """
    機能: 観測できた足の始終値・高安・変化率を集計する。
    引数: rows (list[dict]): 確定足。p0 (float): 前セッション基準。reference_price (float | None): 比較価格。
    返り値: dict | None: 価格集計、足がない場合は未測定。
    """
    if not rows:
        return None
    first, last = rows[0], rows[-1]
    high = max(row["high"] for row in rows)
    low = min(row["low"] for row in rows)
    close = last["close"]
    return {
        "first_bar_start": first["start"].isoformat(),
        "latest_bar_start": last["start"].isoformat(),
        "latest_bar_end": last["end"].isoformat(),
        "observed_bar_count": len(rows),
        "first_observed_open": first["open"],
        "high": high,
        "low": low,
        "latest_close": close,
        "close_change_pct_p0": (close / p0 - 1) * 100,
        "high_change_pct_p0": (high / p0 - 1) * 100,
        "low_change_pct_p0": (low / p0 - 1) * 100,
        "change_pct_first_observed_open": (close / first["open"] - 1) * 100,
        "change_from_reference_pct": (
            (close / reference_price - 1) * 100 if reference_price else None),
        "recovery_from_low_pct_p0": (close - low) / p0 * 100,
        "pullback_from_high_pct_p0": (close - high) / p0 * 100,
        "close_position_pct_observed_range": (
            (close - low) / (high - low) * 100 if high > low else None),
    }


def complete_five_minute_bars(rows, session_start, complete_until):
    """
    機能: セッション開始を基準に５本すべて揃う確定５分足だけを作る。
    引数: rows (list[dict]): 確定１分足。session_start (datetime): セッション開始。complete_until (datetime): 確定済み境界。
    返り値: tuple[list[dict], list[str]]: 確定５分足と欠落した完成予定足の開始時刻。
    """
    by_time = {row["start"]: row for row in rows}
    bars, incomplete = [], []
    cursor = session_start
    while cursor + 5 * MINUTE <= complete_until:
        members = [by_time.get(cursor + offset * MINUTE) for offset in range(5)]
        if all(member is not None for member in members):
            bars.append({
                "start": cursor,
                "end": cursor + 5 * MINUTE,
                "open": members[0]["open"],
                "high": max(member["high"] for member in members),
                "low": min(member["low"] for member in members),
                "close": members[-1]["close"],
            })
        else:
            incomplete.append(cursor.isoformat())
        cursor += 5 * MINUTE
    return bars, incomplete


def serialize_bar(bar):
    """
    機能: 根拠足の時刻をJSON用文字列へ変換する。
    引数: bar (dict): 確定５分足。
    返り値: dict: JSONへ保存できる根拠足。
    """
    return {key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in bar.items()}


def evaluate_rules(bars, rules, complete_until):
    """
    機能: 事前条件の厳密超過・未満と欠測のない連続成立を点検する。
    引数: bars (list[dict]): 完全な５分足。rules (list[dict]): 事前ルール。complete_until (datetime): 最新完成境界。
    返り値: list[dict]: 初成立・直近成立・現在成立の判定と根拠足。
    """
    if not isinstance(rules, list):
        raise ValueError("rulesはJSON配列が必要です。")
    results, seen = [], set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("各ルールはオブジェクトが必要です。")
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
            raise ValueError("ルールidは重複しない空でない文字列が必要です。")
        seen.add(rule_id)
        operator = rule.get("operator")
        if operator not in ("above", "below"):
            raise ValueError("operatorはaboveまたはbelowが必要です。")
        if rule.get("bar_minutes", 5) != 5:
            raise ValueError("bar_minutesは5のみ対応しています。")
        count = rule.get("consecutive_bars", 2)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("consecutive_barsは正の整数が必要です。")
        threshold = positive_number(rule.get("threshold"), "threshold")
        run, matches = [], []
        for bar in bars:
            matches_threshold = (bar["close"] > threshold if operator == "above"
                                 else bar["close"] < threshold)
            if not matches_threshold:
                run = []
                continue
            if run and run[-1]["end"] != bar["start"]:
                run = []
            run.append(bar)
            if len(run) >= count:
                matches.append({
                    "established_at": bar["end"].isoformat(),
                    "evidence_bars": [serialize_bar(item) for item in run[-count:]],
                })
        latest = matches[-1] if matches else None
        latest_boundary = complete_until - timedelta(
            minutes=int((complete_until - bars[0]["start"]) / MINUTE) % 5
        ) if bars else None
        results.append({
            "id": rule_id, "operator": operator, "threshold": threshold,
            "consecutive_bars": count, "bar_minutes": 5,
            "first_match": matches[0] if matches else None,
            "latest_match": latest,
            "matching_window_count": len(matches),
            "holds_at_latest_completed_five_minute_boundary": bool(
                latest and latest["established_at"] == latest_boundary.isoformat()),
            "judgment_basis": "５分足終値。aboveは厳密超過、belowは厳密未満。欠測を跨ぐ連続は不成立。",
        })
    return results


def inspect(rows, as_of, session_start, p0, reference_price=None, rules=None):
    """
    機能: 指定情報時点における価格反応と任意の反証条件を集計する。
    引数: rows (list[dict]): 元足。as_of (str): 情報時点。session_start (str): 開始時刻。p0 (float): 前セッション基準。reference_price (float | None): 比較価格。rules (list[dict] | None): 事前条件。
    返り値: dict: 欠測と限界を明示した点検結果。
    """
    cutoff, start = parse_time(as_of), parse_time(session_start)
    if cutoff < start or start.second or start.microsecond:
        raise ValueError("session-startはas-of以前の分境界が必要です。")
    p0 = positive_number(p0, "p0")
    reference = (positive_number(reference_price, "reference-price")
                 if reference_price is not None else None)
    normalized = normalize_rows(rows)
    included = []
    excluded = {"before_session": 0, "not_finished_as_of": 0, "not_closed": 0}
    for row in normalized:
        if row["start"] < start:
            excluded["before_session"] += 1
        elif row["end"] > cutoff:
            excluded["not_finished_as_of"] += 1
        elif not row["closed"]:
            excluded["not_closed"] += 1
        else:
            included.append(row)

    # ----------------------------------------
    expected_count = int((cutoff - start) / MINUTE)
    complete_until = start + expected_count * MINUTE
    by_time = {row["start"]: row for row in included}
    missing = [(start + minute * MINUTE).isoformat()
               for minute in range(expected_count) if start + minute * MINUTE not in by_time]
    windows = []
    for minutes in (5, 15, 30):
        window_start = max(start, complete_until - minutes * MINUTE)
        window_rows = [row for row in included if row["start"] >= window_start]
        expected = int((complete_until - window_start) / MINUTE)
        windows.append({
            "minutes": minutes, "start": window_start.isoformat(),
            "end": complete_until.isoformat(), "expected_bar_count": expected,
            "missing_bar_count": expected - len(window_rows),
            "fully_observed": expected > 0 and expected == len(window_rows),
            "full_requested_window_elapsed": complete_until - start >= minutes * MINUTE,
            "prices": price_summary(window_rows, p0, reference),
        })
    five_bars, incomplete_five = complete_five_minute_bars(included, start, complete_until)
    summary = price_summary(included, p0, reference)
    return {
        "scope": "確定足の価格反応と指定条件だけを点検。材料の織込み率・因果・予想は算出しない。",
        "as_of": cutoff.isoformat(), "session_start": start.isoformat(),
        "p0": p0, "reference_price": reference,
        "p0_note": "同じ先物の前セッション終値を指定する。公式終値か最終観測代理かは利用者が別途記録する。",
        "units": {
            "prices": "入力と同じ価格単位",
            "pct_p0": "百分率。分母はp0。回復は(直近終値−観測安値)/p0、押戻しは(直近終値−観測高値)/p0。",
            "pct_first_observed_open": "百分率。分母は窓内の最初の観測足の始値。",
            "pct_reference": "百分率。分母はreference_price。未指定なら未測定。",
            "pct_observed_range": "百分率。(直近終値−観測安値)/(観測高値−観測安値)。値幅ゼロなら未測定。織込み率ではない。",
        },
        "session_open": by_time[start]["open"] if start in by_time else None,
        "prices": summary, "recent_windows": windows,
        "quality": {
            "input_bar_count": len(normalized), "included_bar_count": len(included),
            "excluded": excluded, "expected_completed_bar_count": expected_count,
            "missing_completed_bar_starts": missing, "missing_completed_bar_count": len(missing),
            "complete_five_minute_bar_count": len(five_bars),
            "incomplete_five_minute_bar_starts": incomplete_five,
            "volume_incomplete_bar_count": sum(row["volume_complete"] is False for row in included),
            "volume_unknown_bar_count": sum(row["volume_complete"] is None for row in included),
            "latest_close_age_seconds": (cutoff - included[-1]["end"]).total_seconds() if included else None,
            "limitations": [
                "欠測区間の高値・安値は未確認。観測高安をセッション全体の確定高安とみなさない。",
                "volumeCompleteは価格の確定条件に代用しない。出来高そのものは集計しない。",
                "snapshotや取得時刻だけでは、過去時点での取得可能性や収録後の訂正有無を証明できない。",
            ],
        },
        "rules": evaluate_rules(five_bars, [] if rules is None else rules, complete_until),
        "rules_note": "規則の事前登録は利用者が原本で証明する。本スクリプトの成立判定だけでは事前性を証明しない。",
    }


def main(argv=None):
    """
    機能: CLI入力を読み取り、既存ファイルを上書きせず点検JSONを保存する。
    引数: argv (list[str] | None): 引数一覧。省略時はプロセス引数。
    返り値: int: 成功時０。入力不正時は引数エラーで終了する。
    """
    parser = argparse.ArgumentParser(description="確定１分足の価格反応と事前反証条件を点検します。")
    parser.add_argument("--input", required=True, type=Path, help="収録JSONまたはcandle_1min CSV")
    parser.add_argument("--as-of", required=True, help="情報時点。タイムゾーン必須")
    parser.add_argument("--session-start", required=True, help="セッション開始。タイムゾーン必須")
    parser.add_argument("--p0", required=True, type=float, help="同じ先物の前セッション基準価格")
    parser.add_argument("--reference-price", type=float, help="朝の現価格など、追加の比較価格")
    parser.add_argument("--rules", type=Path, help="事前反証条件のJSON配列")
    parser.add_argument("--output", required=True, type=Path, help="新規出力JSON。既存ファイルは禁止")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("出力ファイルは既に存在します。新しいパスを指定してください。")
        rows, source = read_input(args.input)
        rules = []
        rules_source = None
        if args.rules:
            rule_bytes = args.rules.read_bytes()
            rules = json.loads(rule_bytes.decode("utf-8-sig"))
            rules_source = {"path": str(args.rules.resolve()),
                            "sha256": hashlib.sha256(rule_bytes).hexdigest()}
        result = inspect(rows, args.as_of, args.session_start, args.p0,
                         args.reference_price, rules)
        result["source"] = source
        result["rules_source"] = rules_source
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"価格点検を保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
