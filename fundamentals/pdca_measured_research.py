"""実測した既反応と固定した方向分類を順に再生する開発研究。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import digest, finite, now, read_json, timestamp, write_json
from .pdca_measured_model import predict_measured
from .pdca_news_facts import _eligible_articles, extract_market_facts
from .pdca_observed import observed_state
from .pdca_research import validate_annotation
from .replay import _append_audit, _audit, _file_hash, _range_text
from .news import load_export


SYMBOLS = ("NK225", "TOPIX")
MODES = {"observed": "07_実測先物の方向分類", "raw_news": "08_生の材料点と実測先物",
         "market_news": "09_記事中の市場数値と実測先物"}


def _direction(value: float) -> str:
    """固定した実績方向を返す。

    引数: value (float) は同じ先物の前場基準の騰落率％。
    返り値: str は上昇・下落・横ばい。
    """
    return "上昇" if value > 0.05 else "下落" if value < -0.05 else "横ばい"


def _fx_change(facts: dict) -> float | None:
    """明示された異なる市場日のドル円引けだけを比較する。

    引数: facts (dict) は締切内の記事から抽出した市場観測。
    返り値: float | None は引け変化率％。日付・前値・一意性不足はNone。
    """
    latest = facts["series"].get("usdjpy")
    if not latest or not latest.get("market_date"):
        return None
    older = [item for item in facts["observations"].get("usdjpy", [])
             if item.get("market_date") and item["market_date"] < latest["market_date"]]
    if not older:
        return None
    previous_day = max(item["market_date"] for item in older)
    prices = {item["close"] for item in older if item["market_date"] == previous_day}
    if len(prices) != 1:
        return None
    previous = prices.pop()
    return (latest["close"] / previous - 1) * 100 if previous > 0 else None


def visible_input(packet: dict, annotation: dict, prices: Path,
                  news_articles: list[dict] | None = None) -> dict:
    """可視の価格・記事・固定注釈だけから全方式の共通入力を作る。

    引数: packet (dict) は締切付き切出入力、annotation (dict) は固定注釈、
    prices (Path) は原価格ルート、news_articles (list[dict] | None) は数値観測用の
    版付き記事原本。省略時は切出記事。渡した原本も本文の解釈前に締切選別する。
    返り値: dict は実測状態・記事観測・各先物の13候補特徴。実績は含まない。
    """
    validate_annotation(annotation, packet)
    states = {symbol: observed_state(prices, symbol, packet["as_of"]) for symbol in SYMBOLS}
    eligible_news, _ = _eligible_articles(packet["news"] if news_articles is None else news_articles,
                                          timestamp(packet["as_of"]))
    facts = extract_market_facts(eligible_news, packet["as_of"])
    instruments = {}
    for symbol in SYMBOLS:
        state = states[symbol]
        if state["status"] != "利用可能" or state["target_session"] != packet["session"]:
            raise ValueError("実測状態と固定入力の対象セッションが一致しません。")
        reference = packet["instruments"][symbol]["reference"]
        actual_reference = state["reference"]
        if (actual_reference["session"] != reference["session"]
                or actual_reference["price"] != reference["price"]
                or timestamp(actual_reference["time"]) != timestamp(reference["time"])):
            raise ValueError("実測特徴の基準は固定した１分足基準と同じである必要があります。")
        previous = state["previous_session"]
        other = states[SYMBOLS[1] if symbol == SYMBOLS[0] else SYMBOLS[0]]["previous_session"]
        relative = None
        if (other and other["session"] == previous["session"]
                and other["return_pct"] is not None and previous["return_pct"] is not None):
            relative = previous["return_pct"] - other["return_pct"]
        features = {"previous_return_pct": previous["return_pct"],
                    "close_location_centered": None if previous["close_location"] is None else previous["close_location"] - 0.5,
                    "range_pct": previous["range_pct"],
                    "late_change_pct": previous["late_window"]["change_pct"],
                    "relative_return_pct": relative, "is_night": float(packet["kind"] == "Night")}
        features.update({key: annotation["instruments"][symbol]["channels"][key]["score"]
                         for key in ("growth", "discount", "fx", "cost_risk")})
        for series, key in (("nasdaq", "us_nasdaq_return_pct"), ("dow", "us_dow_return_pct")):
            item = facts["series"].get(series)
            inconsistent = item and any(flag in item["flags"] for flag in
                                       ("same_market_date_conflict", "reported_change_pct_mismatch", "previous_observed_close_mismatch"))
            recent = item and (timestamp(packet["as_of"]) - timestamp(item["available_at"])).total_seconds() <= 48 * 3600
            features[key] = item.get("change_pct") if item and recent and not inconsistent else None
        fx = facts["series"].get("usdjpy")
        recent_fx = fx and (timestamp(packet["as_of"]) - timestamp(fx["available_at"])).total_seconds() <= 48 * 3600
        features["usdjpy_change_pct"] = _fx_change(facts) if recent_fx else None
        instruments[symbol] = {"symbol": symbol, "kind": packet["kind"],
                               "as_of": packet["as_of"], "features": features}
    return {"as_of": packet["as_of"], "session": packet["session"], "states": states,
            "market_facts": facts, "instruments": instruments,
            "channel_annotations": annotation["instruments"],
            "source_packet_hash": digest(packet), "annotation_hash": digest(annotation),
            "annotation_status": "生の材料点は未校正の注釈判断。実測値とは区別する",
            "causal_priced_in_fraction": None}


# ----------------------------------------
# 原本の固定と満了した初期教師


def _initial_labels(path: Path, cutoff: str) -> dict:
    """初期教師の満了時刻を実績値の解釈より先に確認する。

    引数: path (Path) は既存教師、cutoff (str) は最初の評価締切。
    返り値: dict は終点・先物別の実績行。重複・未来・不正数値は例外。
    """
    raw = read_json(path)
    if raw.get("mode") != "回想・開発再生" or not isinstance(raw.get("rows"), list):
        raise ValueError("初期教師は回想専用のrowsが必要です。")
    for row in raw["rows"]:
        if timestamp(row["label_end"]) >= timestamp(cutoff):
            raise ValueError("初期教師は最初の締切より前に満了している必要があります。")
    selected = {}
    for row in raw["rows"]:
        key = (timestamp(row["label_end"]).isoformat(), row["symbol"])
        if row["symbol"] not in SYMBOLS or key in selected:
            raise ValueError("教師の対象先物が不正、または終点が重複しています。")
        selected[key] = {"symbol": row["symbol"], "label_end": key[0],
                         "actual_pct": finite(row["actual_pct"], -math.inf, math.inf, "教師実績")}
    return selected


def _bundle(inputs: Path, root: Path, packets: list[Path], initial: list[Path], news_paths: tuple[Path, ...] = ()) -> dict:
    """計画・可視入力・注釈・初期教師・実装を結果開示前に固定する。

    引数: inputs (Path) は既存入力、root (Path) は新研究ルート、packets (list[Path])
    は評価入力、initial (list[Path]) は初期学習入力、news_paths (tuple[Path, ...]) は記事原本。
    返り値: dict は絶対パス別バイトハッシュ。結果原本はまだ読まない。
    """
    files = [root / "実験計画.md", inputs / "training" / "labels.json"]
    files.extend(news_paths)
    files.extend(sorted(Path(__file__).parent.glob("*.py")))
    for path in packets + initial:
        files.extend([path, path.parent.parent / "annotations" / path.name])
    return {str(path.resolve()): _file_hash(path) for path in files}


def _verify(files: dict) -> None:
    """固定原本の変更を次の開示前に拒否する。

    引数: files (dict) は絶対パスと固定ハッシュ。
    返り値: None。不一致は例外。
    """
    if any(not Path(path).is_file() or _file_hash(Path(path)) != expected for path, expected in files.items()):
        raise ValueError("固定した入力・計画・初期教師または実装が変更されています。")


def _source_labels(source: Path, packet: dict) -> tuple[list[dict], str]:
    """当回予測固定後に原実績を読み、基準と終点を照合する。

    引数: source (Path) は原回フォルダ、packet (dict) は可視入力。
    返り値: tuple[list[dict], str] は両先物の照合済み当日実績と原結果ハッシュ。
    """
    if _audit(source)[-1]["phase"] != "review":
        raise ValueError("原実績の振り返りが固定されていません。")
    original = read_json(source / "予測.json")
    if timestamp(original["as_of"]) != timestamp(packet["as_of"]) or original["session"] != packet["session"]:
        raise ValueError("原実績の対象時点が異なります。")
    result = read_json(source / "result.json")
    if result["forecast_hash"] != original["forecast_hash"]:
        raise ValueError("原結果が原予測に結び付いていません。")
    labels = [item for item in result["labels"] if item["horizon"] == "today"]
    if len(labels) != 2 or {item["symbol"] for item in labels} != set(SYMBOLS):
        raise ValueError("当日実績は各先物１行だけ必要です。")
    for item in labels:
        reference = packet["instruments"][item["symbol"]]["reference"]
        original_reference = original["instruments"][item["symbol"]]["reference"]
        if (timestamp(item["label_end"]) != timestamp(packet["targets"]["today"])
                or reference["price"] != item["reference_price"]
                or reference["price"] != original_reference["price"]
                or reference["session"] != original_reference["session"]
                or timestamp(reference["time"]) != timestamp(original_reference["time"])):
            raise ValueError("原実績と前セッション基準または終点が一致しません。")
        quote_age = (timestamp(item["label_end"]) - timestamp(item["quote_time"])).total_seconds()
        if not 0 <= quote_age <= 600:
            raise ValueError("結果価格は公式終点以前10分以内の代理が必要です。")
        delta = (item["quote_price"] / item["reference_price"] - 1) * 100
        if not math.isclose(delta, item["actual_pct"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("原実績の価格と百分率が整合しません。")
    return labels, digest(result)


# ----------------------------------------
# 予測固定、結果開示、短い振り返り


def _prediction_md(packet: dict, current: dict, predictions: dict, mode: str) -> str:
    """予測前に見えた材料と実測状態を短いMarkdownへ整形する。

    引数: packet (dict) は可視入力、current (dict) は実測入力、predictions (dict)
    は両先物予測、mode (str) は固定方式。
    返り値: str は箇条書き主体の予測文。結果情報は受け取らない。
    """
    lines = [f"# 予測：{packet['session']}", "", f"- 情報締切：{packet['as_of']}。方式：{mode}。",
             "- 既知期間の開発再生。クラススコアは確率ではない。", ""]
    for symbol in SYMBOLS:
        item = predictions[symbol]
        previous = current["states"][symbol]["previous_session"]
        reference = packet["instruments"][symbol]["reference"]
        late = previous["late_window"]
        location = "欠測" if previous["close_location"] is None else f"{previous['close_location']:.3f}"
        ret = "欠測" if previous["return_pct"] is None else f"{previous['return_pct']:+.3f}%"
        scores = "、".join(f"{key}未算出" if value is None else f"{key}{value:+.3f}"
                         for key, value in item["class_scores"].items())
        lines += [f"- {symbol}：{item['direction']}。基準{reference['price']}（{reference['time']}、公式終値代理）。",
                  f"  - 直前先物{ret}、レンジ内終値位置{location}。",
                  f"  - 末尾窓：{late['observed_candles']}/{late['expected_candles']}足。{'完備' if late['complete'] else '未完備'}。補間なし。",
                  f"  - 過去教師{item['training_n']}件。方向スコア：{scores}。",
                  f"  - 経験範囲：{_range_text(packet['instruments'][symbol]['horizons']['today']['distribution'])}。"]
        if mode == "raw_news":
            for key in ("growth", "discount", "fx", "cost_risk"):
                note = current["channel_annotations"][symbol]["channels"][key]
                lines.append(f"  - {key}：{note['reason']}（未校正の生の点数{note['score']:+.2f}）。")
        if mode == "market_news":
            values = current["instruments"][symbol]["features"]
            text = "、".join(f"{key}は欠測" if values[key] is None else f"{key}{values[key]:+.3f}%"
                             for key in ("us_nasdaq_return_pct", "us_dow_return_pct", "usdjpy_change_pct"))
            lines.append(f"  - 記事の市場観測：{text}。異なる市場窓で因果未識別。")
        if item["out_of_range_features"]:
            lines.append("  - 学習域外：" + "、".join(item["out_of_range_features"]) + "。")
    return "\n".join(lines) + "\n"


def run_measured(root: Path, inputs: Path, source: Path, prices: Path, mode: str,
                 news_paths: tuple[Path, ...] = ()) -> dict:
    """固定した一方式を、初期教師と18回の逐次更新で開発再生する。

    引数: root (Path) は計画と新出力、inputs (Path) は既存注釈・可視入力・教師、
    source (Path) は固定原実績、prices (Path) は原価格、mode (str) は固定３方式の一つ、
    news_paths (tuple[Path, ...]) は数値観測用の版付き記事原本。省略時は切出記事。
    返り値: dict は全件成績と基準比較。既存出力は上書きしない。
    """
    if mode not in MODES:
        raise ValueError("方式はobserved、raw_news、market_newsを指定してください。")
    output = root / MODES[mode]
    if output.exists():
        raise FileExistsError("既存の改善循環は上書きできません。")
    packets = sorted((inputs / "masked").glob("*.json"))
    if not packets:
        raise ValueError("評価用の可視入力がありません。")
    labels = _initial_labels(inputs / "training" / "labels.json", read_json(packets[0])["as_of"])
    initial = [path for path in sorted((inputs / "training" / "masked").glob("*.json"))
               if any(timestamp(read_json(path)["targets"]["today"]) == timestamp(key[0]) for key in labels)]
    files = _bundle(inputs, root, packets, initial, news_paths)
    news_articles = [item for path in news_paths for item in load_export(path)] if news_paths else None
    manifest = {"mode": mode, "created_at": now(), "independent_validation": False,
                "files": files, "evaluation_rounds": [path.stem for path in packets],
                "initial_rounds": [path.stem for path in initial], "initial_rows": len(labels)}
    manifest["manifest_hash"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    snapshot = {path: {"content": Path(path).read_text(encoding="utf-8-sig"), "file_hash": expected}
                for path, expected in files.items() if Path(path).suffix in (".py", ".md")}
    write_json(output / "実装控え.json", {"files": snapshot, "manifest_hash": manifest["manifest_hash"]})
    fixed_files = {**files, str((output / "manifest.json").resolve()): _file_hash(output / "manifest.json"),
                   str((output / "実装控え.json").resolve()): _file_hash(output / "実装控え.json")}
    history, used = [], set()
    for path in initial:
        packet = read_json(path)
        current = visible_input(packet, read_json(path.parent.parent / "annotations" / path.name), prices, news_articles)
        for symbol in SYMBOLS:
            key = (timestamp(packet["targets"]["today"]).isoformat(), symbol)
            if key not in labels or key in used:
                raise ValueError("初期入力と教師が一対一に対応していません。")
            history.append({**current["instruments"][symbol], **labels[key]})
            used.add(key)
        write_json(output / "初期入力" / path.name, current)
    if used != set(labels):
        raise ValueError("初期教師の入力が不足しています。")
    write_json(output / "初期教師.json", {"rows": history, "purpose": "学習専用・評価分母へ含めない"})

    # ----------------------------------------

    rows, previous_end, previous_cutoff, previous_destination = [], None, None, None
    for path in packets:
        _verify(fixed_files)
        if previous_destination and _audit(previous_destination)[-1]["phase"] != "review":
            raise ValueError("前回の振り返り固定前に次回へ進めません。")
        packet = read_json(path)
        as_of = timestamp(packet["as_of"])
        end = timestamp(packet["targets"]["today"])
        if end <= as_of or (previous_cutoff and as_of <= previous_cutoff) or (previous_end and previous_end >= as_of):
            raise ValueError("対象終点・締切・前回満了時刻の順序が不正です。")
        annotation = read_json(inputs / "annotations" / path.name)
        current = visible_input(packet, annotation, prices, news_articles)
        destination = output / path.stem
        write_json(destination / "入力.json", current)
        _append_audit(destination, "prepare", as_of, [destination / "入力.json"])
        predictions = {symbol: predict_measured(current["instruments"][symbol], history, packet["as_of"], mode)
                       for symbol in SYMBOLS}
        prediction = {"as_of": packet["as_of"], "session": packet["session"], "issued_at": now(),
                      "mode": "回想・開発再生", "method": mode, "instruments": predictions,
                      "manifest_hash": manifest["manifest_hash"], "input_hash": digest(current),
                      "manifest_file_hash": _file_hash(output / "manifest.json"),
                      "implementation_copy_hash": _file_hash(output / "実装控え.json")}
        prediction["forecast_hash"] = digest(prediction)
        write_json(destination / "予測.json", prediction)
        (destination / "予測.md").write_text(_prediction_md(packet, current, predictions, mode), encoding="utf-8")
        _append_audit(destination, "seal", as_of, [destination / "予測.json", destination / "予測.md"])
        _verify(fixed_files)

        # ----------------------------------------

        actual, original_hash = _source_labels(source / path.stem, packet)
        result_labels, feedback = [], []
        review = [f"# 振り返り：{packet['session']}", "", "- 全件評価。後発材料の有無で外れを除外しない。", ""]
        for label in actual:
            symbol = label["symbol"]
            direction = predictions[symbol]["direction"]
            actual_direction = _direction(label["actual_pct"])
            hit = direction == actual_direction
            selected = {key: label[key] for key in ("symbol", "label_end", "reference_price", "quote_price",
                                                    "quote_time", "actual_pct", "low_pct", "high_pct", "path_hash")}
            selected.update(direction=direction, actual_direction=actual_direction, direction_hit=hit)
            result_labels.append(selected)
            feature = current["instruments"][symbol]["features"]
            rows.append({"round": path.stem, "kind": packet["kind"], **selected,
                         "up_hit": actual_direction == "上昇", "down_hit": actual_direction == "下落",
                         "following_hit": feature["previous_return_pct"] is not None and actual_direction == _direction(feature["previous_return_pct"]),
                         "fitted_n": predictions[symbol]["fitted_n"]})
            feedback.append({**current["instruments"][symbol], "actual_pct": label["actual_pct"], "label_end": label["label_end"]})
            diagnosis = "実測特徴と方向が整合したが因果は未識別。" if hit else "既反応量と次場方向の関係を捉えきれず。原因は未識別。"
            review += [f"- {symbol}：予測{direction}→実績{label['actual_pct']:+.3f}%（{'一致' if hit else '外れ'}）。",
                       f"  - 診断：{diagnosis}", "  - 次回：この満了結果を追加。規則は変更せず過去だけで再推定。"]
        result = {"forecast_hash": prediction["forecast_hash"], "source_result_hash": original_hash, "labels": result_labels}
        write_json(destination / "result.json", result)
        _append_audit(destination, "reveal", end, [destination / "result.json"])
        write_json(destination / "review.json", {"forecast_hash": prediction["forecast_hash"],
                   "result_hash": digest(result), "recorded_at": now(), "next_training": feedback})
        (destination / "振り返り.md").write_text("\n".join(review) + "\n", encoding="utf-8")
        _append_audit(destination, "review", end, [destination / "review.json", destination / "振り返り.md"])
        history.extend(feedback)
        previous_end, previous_cutoff = end, as_of
        previous_destination = destination

    # ----------------------------------------

    _verify(fixed_files)
    groups = {}
    for row in rows:
        for key in (row["symbol"], row["symbol"] + "／" + row["kind"]):
            group = groups.setdefault(key, {"n": 0, "hits": 0, "issued": 0, "held": 0,
                                            "up_hits": 0, "down_hits": 0, "following_hits": 0})
            group["n"] += 1
            group["hits"] += int(row["direction_hit"])
            group["held"] += int(row["direction"] == "保留")
            group["issued"] += int(row["direction"] != "保留")
            for field in ("up", "down", "following"):
                group[field + "_hits"] += int(row[field + "_hit"])
    for group in groups.values():
        group.update(accuracy=group["hits"] / group["n"],
                     issuance=group["issued"] / group["n"],
                     issued_accuracy=group["hits"] / group["issued"] if group["issued"] else None)
    summary = {"mode": "回想・開発再生", "method": mode, "groups": groups, "rows": rows,
               "independent_validation": False, "operational_acceptance": False, "accuracy_target": 0.8,
               "target_reached": all(groups[symbol]["accuracy"] >= 0.8 for symbol in SYMBOLS),
               "manifest_hash": manifest["manifest_hash"], "initial_rows": len(labels)}
    write_json(output / "summary.json", summary)
    write_json(output / "学習状態.json", {"mode": "回想・開発再生", "rows": history,
               "history_time_contract": "満了時刻だけを制約。当時の受信・評価記録時計は未検証"})
    lines = [f"# 改善循環：{MODES[mode]}", "", "- 既知18時点の開発再生。独立検証ではない。初期24教師は評価分母へ含めない。", "",
             "| 対象 | 一致／全件 | 一致率 | 発行／全予定 | 保留 | 常時上昇 | 常時下落 | 直前追随 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, group in groups.items():
        lines.append(f"| {key} | {group['hits']}/{group['n']} | {group['accuracy']:.1%} | {group['issued']}/{group['n']} | {group['held']} | {group['up_hits']}/{group['n']} | {group['down_hits']}/{group['n']} | {group['following_hits']}/{group['n']} |")
    lines += ["", "- 判定：" + ("開発期間の数値目標到達。独立合格にはしない。" if summary["target_reached"] else "80％未達・不合格。"), ""]
    lines.extend(f"- {path.stem}：[予測]({path.stem}/予測.md)／[振り返り]({path.stem}/振り返り.md)。" for path in packets)
    (output / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """実測特徴の開発再生をコマンド行から実行する。

    引数: なし。コマンド行から計画ルート・既存入力・原実績・価格・方式を指定する。
    返り値: None。両先物の全件成績を表示する。
    """
    parser = argparse.ArgumentParser(description="実測した既反応と方向分類の開発再生")
    for name, help_text in (("root", "計画と新規出力"), ("inputs", "既存入力・注釈・初期教師"),
                            ("source", "固定原実績"), ("prices", "原価格ルート")):
        parser.add_argument("--" + name, type=Path, required=True, help=help_text)
    parser.add_argument("--mode", choices=tuple(MODES), required=True, help="固定した比較方式")
    parser.add_argument("--news", type=Path, nargs="*", default=[], help="数値観測用の版付き記事原本")
    args = parser.parse_args()
    result = run_measured(args.root, args.inputs, args.source, args.prices, args.mode, tuple(args.news))
    print({key: result["groups"][key] for key in SYMBOLS})


if __name__ == "__main__":
    main()
