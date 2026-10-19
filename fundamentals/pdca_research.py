"""固定注釈から予測・結果開示・数値更新を順番に行う開発再生。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import digest, now, read_json, timestamp, write_json
from .pdca import CHANNELS, SYMBOLS, _history_rows, predict_learned, predict_rule
from .replay import _append_audit, _audit, _file_hash, _range_text


def validate_annotation(annotation: dict, packet: dict) -> None:
    """注釈の締切と記事根拠を可視入力に照合する。

    引数: annotation (dict) は固定注釈、packet (dict) は結果を含まない切出入力。
    返り値: None。不一致や根拠なしの非ゼロ採点は例外。
    """
    if timestamp(annotation["as_of"]) != timestamp(packet["as_of"]):
        raise ValueError("材料注釈の情報締切が一致しません。")
    if digest({key: value for key, value in packet.items() if key != "packet_hash"}) != packet["packet_hash"]:
        raise ValueError("切出入力のハッシュが一致しません。")
    visible = {str(item["id"]) for item in packet["news"]}
    for item in packet["news"]:
        if not item.get("available_at"):
            raise ValueError("記事には利用可能時刻available_atが必要です。")
        for key in ("available_at", "published_at", "fetched_at", "updated_at"):
            if item.get(key) and timestamp(item[key]) > timestamp(packet["as_of"]):
                raise ValueError("未来の記事が切出入力に含まれています。")
    if set(annotation["instruments"]) != set(SYMBOLS):
        raise ValueError("注釈には両先物が必要です。")
    for instrument in annotation["instruments"].values():
        if set(instrument["channels"]) != set(CHANNELS):
            raise ValueError("４つの材料経路が必要です。")
        for channel in instrument["channels"].values():
            if set(channel) != {"score", "confidence", "novelty", "priced_in", "evidence", "reason"}:
                raise ValueError("材料経路の項目が一致しません。")
            evidence = channel["evidence"]
            if not isinstance(evidence, list) or len(evidence) > 3 or any(str(item) not in visible for item in evidence):
                raise ValueError("可視記事にない根拠、または過多の根拠があります。")
            if not evidence and (channel["score"] != 0 or channel["confidence"] != 0):
                raise ValueError("記事根拠なしの材料点数は使えません。")
            if not isinstance(channel["reason"], str) or not channel["reason"].strip():
                raise ValueError("材料注釈の理由が必要です。")


def _direction(value: float) -> str:
    """先物の実績変化を固定した中立帯で分類する。

    引数: value (float) は前セッション基準の実績百分率。
    返り値: str は上昇、下落、横ばい。
    """
    return "上昇" if value > 0.05 else "下落" if value < -0.05 else "横ばい"


def _verify_manifest(root: Path, output: Path, manifest: dict) -> None:
    """固定した計画・手順・入力・注釈と実装の変更を結果開示前に拒否する。

    引数: root (Path) は入力ルート、output (Path) は循環出力、manifest (dict) は開始時の固定一覧。
    返り値: None。保存一覧の自己ハッシュまたは原本との不一致は例外。
    """
    saved = read_json(output / "manifest.json")
    if saved != manifest or digest({key: value for key, value in saved.items() if key != "manifest_hash"}) != saved["manifest_hash"]:
        raise ValueError("固定一覧のハッシュが一致しません。")
    files = [(root / "実験計画.md", manifest["plan_hash"]),
             (root / "注釈手順.md", manifest["instructions_hash"]),
             (Path(__file__).with_name("pdca.py"), manifest["implementation_hash"]),
             (Path(__file__), manifest["controller_hash"]),
             (output / "実装控え.json", manifest["implementation_copy_hash"])]
    if "cycle_plan_hash" in manifest:
        files.append((root / "循環03_計画.md", manifest["cycle_plan_hash"]))
    if "scaled_plan_hash" in manifest:
        files.append((root / "循環06_計画.md", manifest["scaled_plan_hash"]))
    if "initial_history_path" in manifest:
        files.append((Path(manifest["initial_history_path"]), manifest["initial_history_file_hash"]))
    for item in manifest["inputs"]:
        files.extend([(root / "masked" / (item["round"] + ".json"), item["packet_hash"]),
                      (root / "annotations" / (item["round"] + ".json"), item["annotation_hash"])])
    if any(not path.is_file() or _file_hash(path) != expected for path, expected in files):
        raise ValueError("固定した計画・手順・入力・注釈または実装が変更されています。")


def _load_initial_history(path: Path, first_packet: dict) -> tuple[list[dict], dict]:
    """最初の締切より前に満了した初期教師だけを数値検証前に選別する。

    引数: path (Path) は回想専用教師JSON、first_packet (dict) は最初の可視入力。
    返り値: tuple[list[dict], dict] は正規化した同先物履歴と原本・内容ハッシュおよび件数。
    """
    raw_bytes = path.read_bytes()
    original = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(original, dict) or original.get("mode") != "回想・開発再生" or not isinstance(original.get("rows"), list):
        raise ValueError("初期教師には回想・開発再生のmodeとrows配列が必要です。")
    cutoff = timestamp(first_packet["as_of"])
    for row in original["rows"]:
        if not isinstance(row, dict) or timestamp(row["label_end"]) >= cutoff:
            raise ValueError("初期教師は最初の情報締切より厳密に前の満了実績だけを使えます。")

    # ----------------------------------------

    normalized = []
    counts = {}
    for symbol in SYMBOLS:
        selected = _history_rows(original["rows"], first_packet, symbol)
        normalized.extend(selected)
        counts[symbol] = len(selected)
    normalized.sort(key=lambda row: (timestamp(row["label_end"]), row["symbol"]))
    return normalized, {"initial_history_path": str(path.resolve()),
                        "initial_history_file_hash": hashlib.sha256(raw_bytes).hexdigest(),
                        "initial_history_normalized_hash": digest(normalized),
                        "initial_history_n": len(normalized), "initial_history_counts": counts,
                        "initial_history_endpoint_n": len({row["label_end"] for row in normalized})}


def run_cycle(root: Path, source: Path, identity: str, method: str,
              initial_history_path: Path | None = None) -> dict:
    """一つの改善方式を、各回の予測固定後に結果を開示して再生する。

    引数: root (Path) は固定入力と注釈のルート、source (Path) は原実績、
    identity (str) は新規循環名、method (str) はrule、ridge、analog、ridge_scaled、initial_history_path
    (Path | None) は最初の締切より前だけの学習専用教師。評価分母へは加えない。
    返り値: dict は指数別成績、基準比較、証跡。既存循環は上書きしない。
    """
    if method not in ("rule", "ridge", "analog", "ridge_scaled") or Path(identity).name != identity:
        raise ValueError("方式または循環名が不正です。")
    packets = sorted((root / "masked").glob("*.json"))
    if not packets or any(not (root / "annotations" / path.name).exists() for path in packets):
        raise ValueError("全回の固定注釈が必要です。")
    output = root / identity
    if output.exists():
        raise FileExistsError("改善循環は上書きできません。")
    initial_history, initial_metadata = [], {}
    if initial_history_path is not None:
        initial_history_path = Path(initial_history_path)
        initial_history, initial_metadata = _load_initial_history(initial_history_path, read_json(packets[0]))
    copied_files = {"fundamentals/pdca.py": Path(__file__).with_name("pdca.py"),
                    "fundamentals/pdca_research.py": Path(__file__),
                    "実験計画.md": root / "実験計画.md", "注釈手順.md": root / "注釈手順.md"}
    if method == "analog":
        copied_files["循環03_計画.md"] = root / "循環03_計画.md"
    if method == "ridge_scaled":
        copied_files["循環06_計画.md"] = root / "循環06_計画.md"
    if initial_history_path is not None:
        copied_files["初期教師.json"] = initial_history_path
    implementation_copy = {"created_at": now(), "files": {
        name: {"content": path.read_bytes().decode("utf-8"), "file_hash": _file_hash(path)}
        for name, path in copied_files.items()}}
    implementation_copy["snapshot_hash"] = digest(implementation_copy)
    write_json(output / "実装控え.json", implementation_copy)
    manifest = {"created_at": now(), "method": method, "mode": "回想・開発再生", "accuracy_target": 0.8,
                "independent_validation": False, "source": str(source),
                "implementation_hash": _file_hash(Path(__file__).with_name("pdca.py")),
                "controller_hash": _file_hash(Path(__file__)),
                "plan_hash": _file_hash(root / "実験計画.md"),
                "implementation_copy": "実装控え.json", "implementation_copy_hash": _file_hash(output / "実装控え.json"),
                "instructions_hash": _file_hash(root / "注釈手順.md"),
                "inputs": [{"round": path.stem, "packet_hash": _file_hash(path),
                            "annotation_hash": _file_hash(root / "annotations" / path.name)} for path in packets]}
    if method == "analog":
        manifest["cycle_plan_hash"] = _file_hash(root / "循環03_計画.md")
    if method == "ridge_scaled":
        manifest["scaled_plan_hash"] = _file_hash(root / "循環06_計画.md")
    manifest.update(initial_metadata)
    manifest["manifest_hash"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    history, rows, previous_as_of, previous_end = list(initial_history), [], None, None
    for packet_path in packets:
        _verify_manifest(root, output, manifest)
        packet = read_json(packet_path)
        annotation = read_json(root / "annotations" / packet_path.name)
        validate_annotation(annotation, packet)
        path = output / packet_path.stem
        as_of = timestamp(packet["as_of"])
        if previous_as_of is not None and as_of <= previous_as_of:
            raise ValueError("各回の情報締切は厳密な昇順が必要です。")
        if previous_end is not None and previous_end > as_of:
            raise ValueError("前回の満了実績が次回の情報締切より後です。")
        if timestamp(packet["targets"]["today"]) <= as_of:
            raise ValueError("当日予測の評価終点は情報締切より後が必要です。")
        write_json(path / "入力.json", {"packet": packet, "annotation": annotation})
        _append_audit(path, "prepare", as_of, [path / "入力.json"])
        predictions = {symbol: predict_rule(annotation, packet, symbol) if method == "rule" else
                       predict_learned(annotation, packet, symbol, history, mode=method) for symbol in SYMBOLS}
        prediction = {"as_of": packet["as_of"], "session": packet["session"], "issued_at": now(), "method": method,
                      "mode": "回想・開発再生", "packet_hash": digest(packet), "annotation_hash": digest(annotation),
                      "manifest_hash": manifest["manifest_hash"],
                      "instruments": predictions}
        if initial_history_path is not None:
            prediction.update({key: initial_metadata[key] for key in
                               ("initial_history_n", "initial_history_counts", "initial_history_normalized_hash")})
        prediction["forecast_hash"] = digest(prediction)
        write_json(path / "予測.json", prediction)
        lines = [f"# 予測：{packet['session']}", "", f"- 情報締切：{packet['as_of']}。方式：{method}。",
                 "- 開発再生。点数は確率ではなく、今回の再使用期間だけで合格にしない。", ""]
        for symbol, item in predictions.items():
            reference = packet["instruments"][symbol]["reference"]
            learned_text = f"類例照合{item['analogs_n']}件" if method == "analog" else f"実際の係数学習{item['fitted_n']}件"
            lines += [f"- {symbol}：{item['direction']}。基準{reference['price']}（{reference['time']}、公式終値代用）。",
                      f"  - 推定：{item['status']}。{learned_text}。"]
            lines += [f"  - {channel}：{annotation['instruments'][symbol]['channels'][channel]['reason']}（実効点{item['features'][channel]:+.4f}）。" for channel in CHANNELS]
            estimate = "未校正の点数で百分率は未発行" if item["prediction_pct"] is None else f"探索的な終点推定{item['prediction_pct']:+.3f}%"
            if method == "analog":
                estimate = "少数類例の方向照合のみ。百分率・校正済み確率は未発行"
            lines += [f"  - 数値：{estimate}。", f"  - 経験範囲：{_range_text(packet['instruments'][symbol]['horizons']['today']['distribution'])}。"]
            if method == "analog" and item["analogs_n"]:
                shares = "、".join(f"{key}{value:.1%}" if value is not None else f"{key}算出不能"
                                   for key, value in item["direction_weight_shares"].items())
                ends = "、".join(timestamp(analog["label_end"]).strftime("%m-%d %H:%M") for analog in item["analogs"])
                lines += [f"  - 方向別の参考重み比（未校正）：{shares}。",
                          f"  - 参照した評価終点（JST）：{ends}。"]
            if method == "ridge_scaled" and item["scaling"]["out_of_range_features"]:
                outside = "、".join(item["scaling"]["out_of_range_features"])
                lines.append(f"  - 学習域外の特徴：{outside}。未校正の外挿。")
        lines += [f"- 固定hash：{prediction['forecast_hash']}。", ""]
        (path / "予測.md").write_text("\n".join(lines), encoding="utf-8")
        _append_audit(path, "seal", as_of, [path / "予測.json", path / "予測.md"])
        _verify_manifest(root, output, manifest)

        # ----------------------------------------

        original_path = source / packet_path.stem
        if _audit(original_path)[-1]["phase"] != "review":
            raise ValueError("原実績が固定済みではありません。")
        original_prediction = read_json(original_path / "予測.json")
        if timestamp(original_prediction["as_of"]) != as_of or original_prediction["session"] != packet["session"]:
            raise ValueError("原予測と情報締切・対象セッションが一致しません。")
        original = read_json(original_path / "result.json")
        if original["forecast_hash"] != original_prediction["forecast_hash"]:
            raise ValueError("原実績が原予測の固定ハッシュと一致しません。")
        today_labels = [label for label in original["labels"] if label["horizon"] == "today"]
        if len(today_labels) != len(SYMBOLS) or {label["symbol"] for label in today_labels} != set(SYMBOLS):
            raise ValueError("原実績には両先物の当日ラベルが各１件だけ必要です。")
        labels = []
        for label in today_labels:
            symbol = label["symbol"]
            if timestamp(label["label_end"]) != timestamp(packet["targets"]["today"]):
                raise ValueError("原実績の評価終点が今回の当日予測と一致しません。")
            if packet["instruments"][symbol]["reference"]["session"] != original_prediction["instruments"][symbol]["reference"]["session"]:
                raise ValueError("元予測と基準先物セッションが一致しません。")
            if packet["instruments"][symbol]["reference"]["price"] != label["reference_price"]:
                raise ValueError("元実績と基準先物価格が一致しません。")
            item = {key: label[key] for key in ("symbol", "label_end", "reference_price", "quote_price", "quote_time", "actual_pct", "low_pct", "high_pct", "path_hash")}
            item.update(direction=predictions[symbol]["direction"], actual_direction=_direction(label["actual_pct"]))
            item["direction_hit"] = item["direction"] == item["actual_direction"]
            labels.append(item)
        result = {"forecast_hash": prediction["forecast_hash"], "source_result_hash": digest(original),
                  "manifest_hash": manifest["manifest_hash"], "labels": labels}
        write_json(path / "result.json", result)
        observed = max(timestamp(item["label_end"]) for item in labels)
        _append_audit(path, "reveal", observed, [path / "result.json"])
        review_lines = [f"# 振り返り：{packet['session']}", "", "- 開発再生。原因の断定や未知材料による失敗の除外はしない。", ""]
        for label in labels:
            symbol = label["symbol"]
            base = predict_rule(annotation, packet, symbol)["direction"]
            diagnosis = "材料の符号と実績が整合。因果は未識別。" if label["direction_hit"] else "材料点数・織込み・反応係数のいずれかが不足。因果は未識別。"
            if method in ("ridge", "ridge_scaled") and predictions[symbol]["fitted_n"]:
                diagnosis += f"固定規則は{base}、更新後は{label['direction']}。"
            if method == "analog" and predictions[symbol]["fitted_n"]:
                diagnosis += f"固定規則は{base}、類例照合後は{label['direction']}。少数類例の因果は未識別。"
            next_action = "この満了実績を追加し、次の締切で類例照合。" if method == "analog" else "この満了実績を追加し、次の締切で係数を再計算。"
            review_lines += [f"- {symbol}：予測{label['direction']}→{label['actual_pct']:+.3f}%（{'一致' if label['direction_hit'] else '外れ'}）。",
                             f"  - 診断：{diagnosis}", f"  - 次回：{next_action}"]
            history.append({"symbol": symbol, "features": predictions[symbol]["features"], "actual_pct": label["actual_pct"], "label_end": label["label_end"]})
            previous = packet["instruments"][symbol]["recent_sessions"][-1]["return_pct"]
            rows.append({"round": packet_path.stem, "symbol": symbol, "kind": packet["kind"], **label,
                         "fitted_n": predictions[symbol]["fitted_n"], "up_hit": label["actual_direction"] == "上昇",
                         "down_hit": label["actual_direction"] == "下落", "following_hit": label["actual_direction"] == _direction(previous)})
        write_json(path / "review.json", {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result), "recorded_at": now()})
        (path / "振り返り.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
        _append_audit(path, "review", observed, [path / "review.json", path / "振り返り.md"])
        previous_as_of, previous_end = as_of, observed

    # ----------------------------------------

    _verify_manifest(root, output, manifest)
    groups = {}
    for row in rows:
        for key in (row["symbol"], row["symbol"] + "／" + row["kind"]):
            group = groups.setdefault(key, {"n": 0, "hits": 0, "fitted_predictions": 0, "up_hits": 0, "down_hits": 0, "following_hits": 0})
            group["n"] += 1
            group["hits"] += int(row["direction_hit"])
            group["fitted_predictions"] += int(row["fitted_n"] > 0)
            for field in ("up", "down", "following"):
                group[field + "_hits"] += int(row[field + "_hit"])
    for group in groups.values():
        group["accuracy"] = group["hits"] / group["n"]
        group["target_reached"] = group["accuracy"] >= 0.8
    summary = {"mode": "回想・開発再生", "method": method, "generated_at": now(), "accuracy_target": 0.8,
               "independent_validation": False, "groups": groups, "rows": rows,
               "manifest_hash": manifest["manifest_hash"],
               "target_reached": all(groups[symbol]["target_reached"] for symbol in SYMBOLS),
               "operational_acceptance": False}
    if initial_history_path is not None:
        summary.update({key: initial_metadata[key] for key in
                        ("initial_history_n", "initial_history_counts", "initial_history_endpoint_n", "initial_history_normalized_hash")})
    write_json(output / "summary.json", summary)
    model_column = "類例照合後の発行" if method == "analog" else "係数学習後の発行"
    lines = [f"# 改善循環：{identity}", "", f"- 合格目標：各先物80％以上。全{len(packets)}回で方向を発行。保留０。",
             "- 既知期間の開発再生。独立検証による合格とは区別。", "",
             f"| 対象 | 一致／全件 | 一致率 | {model_column} | 常時上昇 | 常時下落 | 直前追随 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, group in groups.items():
        lines.append(f"| {key} | {group['hits']}/{group['n']} | {group['accuracy']:.1%} | {group['fitted_predictions']} | {group['up_hits']}/{group['n']} | {group['down_hits']}/{group['n']} | {group['following_hits']}/{group['n']} |")
    lines += ["", "- 数値目標：" + ("到達。ただし未使用期間による確認が必要。" if summary["target_reached"] else "未達・不合格。"), ""]
    if initial_history_path is not None:
        counts = "、".join(f"{symbol}{count}件" for symbol, count in initial_metadata["initial_history_counts"].items())
        lines += [f"- 学習専用の初期教師：{counts}。評価分母へ含めない。", ""]
    for path in packets:
        lines.append(f"- {path.stem}：[予測]({path.stem}/予測.md)／[振り返り]({path.stem}/振り返り.md)。")
    (output / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """改善循環をコマンド行から実行する。

    引数: なし。コマンド行に入力ルート、原実績、循環名、方式を指定する。
    返り値: None。指数別の成績を表示する。
    """
    parser = argparse.ArgumentParser(description="固定注釈によるPDCA開発再生")
    parser.add_argument("--root", type=Path, required=True, help="固定入力・注釈のルート")
    parser.add_argument("--source", type=Path, required=True, help="原実績のルート")
    parser.add_argument("--cycle", required=True, help="新規の循環名")
    parser.add_argument("--method", choices=("rule", "ridge", "analog", "ridge_scaled"), required=True, help="今回の方式")
    parser.add_argument("--initial-history", type=Path, help="最初の締切より前に満了した学習専用教師JSON")
    args = parser.parse_args()
    result = run_cycle(args.root, args.source, args.cycle, args.method, initial_history_path=args.initial_history)
    print({key: group for key, group in result["groups"].items() if key in SYMBOLS})


if __name__ == "__main__":
    main()
