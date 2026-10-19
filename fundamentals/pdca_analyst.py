"""各締切で独立した予測担当を使い、全文と実測反応の直接判断を順次保存する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import digest, now, read_json, timestamp, write_json
from .pdca_measured_research import SYMBOLS, _direction, _source_labels
from .pdca_research import validate_annotation
from .replay import _append_audit, _audit, _file_hash, _range_text


def _verify(root: Path) -> dict:
    """計画、可視原本、実装と開始時の一覧が変わっていないことを確認する。

    引数: root (Path) は実験ルート。
    返り値: dict は検証済みの固定一覧。不整合は例外。
    """
    manifest = read_json(root / "manifest.json")
    if digest({key: value for key, value in manifest.items() if key != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("実験の固定一覧が変更されています。")
    for name, expected in manifest["files"].items():
        if not Path(name).is_file() or _file_hash(Path(name)) != expected:
            raise ValueError("計画、入力、教師または実装が変更されています。")
    return manifest


def _past_rows(rows: list[dict], cutoff: str) -> None:
    """教師全件の時刻を先に検査し、未来と重複を拒否する。

    引数: rows (list[dict]) は学習候補、cutoff (str) は今回締切。
    返り値: None。未来実績を数値として解釈しない。
    """
    for row in rows:
        if timestamp(row["label_end"]) >= timestamp(cutoff):
            raise ValueError("教師は締切より前に満了した行だけを使用できます。")
        if timestamp(row["as_of"]) >= timestamp(row["label_end"]):
            raise ValueError("教師の情報締切は満了より前が必要です。")
    seen = set()
    for row in rows:
        key = (row["symbol"], timestamp(row["label_end"]).isoformat())
        if row["symbol"] not in SYMBOLS or key in seen:
            raise ValueError("教師の先物が不正、または満了実績が重複しています。")
        seen.add(key)


def prepare_analyst(root: Path, inputs: Path, measured: Path, index: int) -> dict:
    """当回だけの全文、先物状態、満了教師と前回振り返りを切り出す。

    引数: root (Path) は新実験、inputs (Path) は既存masked/annotations、
        measured (Path) は実測方式の入力と初期教師、index (int) は１始まりの回番号。
    返り値: dict は予測担当へ渡す入力。原結果は読み取らない。
    """
    root, inputs, measured = Path(root), Path(inputs), Path(measured)
    packets = sorted((inputs / "masked").glob("*.json"))
    if type(index) is not int or not 1 <= index <= len(packets):
        raise ValueError("回番号は可視入力の範囲内の整数が必要です。")
    if not (root / "manifest.json").exists():
        if index != 1:
            raise ValueError("最初の回から実験を固定してください。")
        paths = [root / "実験計画.md", root / "予測担当の手順.md", measured / "初期教師.json"]
        paths.extend(sorted(Path(__file__).parent.glob("*.py")))
        for path in packets:
            paths.extend([path, inputs / "annotations" / path.name, measured / path.stem / "入力.json"])
        manifest = {"created_at": now(), "mode": "回想・既知期間の開発再生", "independent_validation": False,
                    "rounds": [path.stem for path in packets], "inputs": str(inputs.resolve()),
                    "measured": str(measured.resolve()), "files": {str(path.resolve()): _file_hash(path) for path in paths}}
        manifest["manifest_hash"] = digest(manifest)
        write_json(root / "manifest.json", manifest)
    manifest = _verify(root)
    if manifest["inputs"] != str(inputs.resolve()) or manifest["measured"] != str(measured.resolve()):
        raise ValueError("開始時と異なる原本は使用できません。")
    expected = manifest["rounds"]
    if [path.stem for path in packets] != expected:
        raise ValueError("可視入力の日程が変更されています。")
    path, inherited, history = packets[index - 1], [], read_json(measured / "初期教師.json")["rows"]
    packet = read_json(path)
    annotation = read_json(inputs / "annotations" / path.name)
    validate_annotation(annotation, packet)
    destination = root / path.stem
    if destination.exists():
        raise FileExistsError("各回入力と予測は上書きできません。")
    for number, identity in enumerate(expected):
        previous = root / identity
        if number >= index - 1:
            if previous.exists():
                raise ValueError("今回以降の予測が既に存在します。")
            continue
        if _audit(previous)[-1]["phase"] != "review":
            raise ValueError("前回の振り返りを固定してから次へ進んでください。")
        review = read_json(previous / "review.json")
        if timestamp(review["available_at"]) >= timestamp(packet["as_of"]):
            raise ValueError("未来の振り返りは入力にできません。")
        inherited.append(review["feedback"])
        history.extend(review["next_training"])
    _past_rows(history, packet["as_of"])

    # ----------------------------------------

    current = read_json(measured / path.stem / "入力.json")
    if current["as_of"] != packet["as_of"] or current["session"] != packet["session"]:
        raise ValueError("実測入力の時点または対象が違います。")
    for symbol in SYMBOLS:
        actual = current["states"][symbol]["reference"]
        reference = packet["instruments"][symbol]["reference"]
        if any(actual[key] != reference[key] for key in ("price", "time", "session")):
            raise ValueError("先物の１分足基準が固定入力と違います。")
    facts = current["market_facts"]
    value = {"as_of": packet["as_of"], "session": packet["session"], "kind": packet["kind"],
             "targets": packet["targets"], "instruments": packet["instruments"], "news": packet["news"],
             "observed_states": current["states"],
             "market_facts": {key: facts[key] for key in ("series", "observations", "flags", "causality") if key in facts},
             "known_history": history, "own_past_reviews": inherited,
             "limitations": ["既知期間の回想。別担当の入力分離は未使用期間の独立検証を意味しない。",
                             "基準は同じ先物の直前セッション最終観測値で、公式終値の代用。",
                             "材料と価格の因果、長期、確率は未実証。価格経験範囲はニュース条件付き予測区間ではない。"]}
    value["input_hash"] = digest(value)
    write_json(destination / "masked" / "入力.json", value)
    _append_audit(destination, "prepare", timestamp(packet["as_of"]), [destination / "masked" / "入力.json"])
    return value


def seal_analyst(round_path: Path, actor: dict) -> dict:
    """新規の独立担当と予測原本を検証し、結果を読む前に固定する。

    引数: round_path (Path) は当回、actor (dict) は担当ID・分離方法・作成完了証跡。
    返り値: dict は固定した両先物予測。原結果は読み取らない。
    """
    _verify(round_path.parent)
    if _audit(round_path)[-1]["phase"] != "prepare":
        raise ValueError("予測固定は準備直後の一度だけです。")
    path = round_path / "masked" / "入力.json"
    packet = read_json(path)
    if actor.get("fork_turns") != "none" or actor.get("completed") is not True or actor.get("input_file_hash") != _file_hash(path):
        raise ValueError("独立担当の入力・作成完了証跡が不正です。")
    if not actor.get("agent_id") or not actor.get("task_name"):
        raise ValueError("予測担当の識別子が必要です。")
    for previous in round_path.parent.glob("*/actor.json"):
        if read_json(previous)["agent_id"] == actor["agent_id"]:
            raise ValueError("予測担当を別締切へ再利用できません。")
    raw = read_json(round_path / "予測入力.json")
    if set(raw) != {"instruments"} or set(raw["instruments"]) != set(SYMBOLS):
        raise ValueError("予測は両先物のinstrumentsだけが必要です。")
    visible = {str(item["id"]) for item in packet["news"]}
    for item in raw["instruments"].values():
        if set(item) != {"direction", "evidence_ids", "already_reacted", "additional_reaction", "invalidation"}:
            raise ValueError("予測の必須項目が一致しません。")
        if item["direction"] not in ("上昇", "下落", "横ばい"):
            raise ValueError("方向は上昇・下落・横ばいから選択してください。")
        ids = item["evidence_ids"]
        if not isinstance(ids, list) or len(ids) > 3 or len(set(map(str, ids))) != len(ids) or any(str(key) not in visible for key in ids):
            raise ValueError("根拠は当回可視記事から重複なく３件以内です。")
        for key, maximum in (("already_reacted", 140), ("additional_reaction", 140), ("invalidation", 100)):
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > maximum or "\n" in item[key]:
                raise ValueError("予測理由は指定字数以内の一行が必要です。")
    write_json(round_path / "actor.json", actor)
    prediction = {"as_of": packet["as_of"], "session": packet["session"], "issued_at": now(),
                  "input_hash": packet["input_hash"], "actor_hash": digest(actor),
                  "mode": "回想・既知期間の開発再生", **raw}
    prediction["forecast_hash"] = digest(prediction)
    write_json(round_path / "予測.json", prediction)
    lines = [f"# 予測：{packet['session']}", "", f"- 情報締切：{packet['as_of']}。別担当による直接判断、既知期間の回想。",
             "- 基準は同じ先物の前セッション最終観測値。公式終値の代用。", ""]
    for symbol in SYMBOLS:
        item, reference = raw["instruments"][symbol], packet["instruments"][symbol]["reference"]
        lines += [f"- {symbol}：{item['direction']}。基準{reference['price']}（{reference['time']}）。",
                  "  - 既反応：" + item["already_reacted"], "  - 追加反応：" + item["additional_reaction"],
                  "  - 反証：" + item["invalidation"], "  - 根拠記事：" + "、".join(map(str, item["evidence_ids"])) + "。",
                  "  - 当日経験範囲：" + _range_text(packet["instruments"][symbol]["horizons"]["today"]["distribution"]) + "。"]
    lines += ["", "- １週間・１か月・３か月と確率は今回未実証。経験範囲はニュース予測の校正区間ではない。",
              f"- 予測ハッシュ：{prediction['forecast_hash']}"]
    (round_path / "予測.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(round_path, "seal", timestamp(packet["as_of"]),
                  [round_path / name for name in ("actor.json", "予測入力.json", "予測.json", "予測.md")])
    return prediction


def review_analyst(round_path: Path, source: Path) -> dict:
    """固定証跡を照合してから初めて当回原結果を読み、採点と次回教師を保存する。

    引数: round_path (Path) は当回、source (Path) は原実績の全回ルート。
    返り値: dict は両先物の振り返り。原結果による予測修正は行わない。
    """
    _verify(round_path.parent)
    if _audit(round_path)[-1]["phase"] != "seal":
        raise ValueError("結果の読取は予測固定後の一度だけです。")
    packet, prediction = read_json(round_path / "masked" / "入力.json"), read_json(round_path / "予測.json")
    labels, original_hash = _source_labels(Path(source) / round_path.name, packet)
    result = {"forecast_hash": prediction["forecast_hash"], "source_result_hash": original_hash,
              "read_after_seal_at": now(), "labels": []}
    teachers, feedback = [], {"session": packet["session"], "instruments": {}}
    lines = [f"# 振り返り：{packet['session']}", "", "- 全件評価。後発材料を理由に外れを除外しない。", ""]
    for label in labels:
        symbol = label["symbol"]
        direction, actual = prediction["instruments"][symbol]["direction"], _direction(label["actual_pct"])
        row = {**label, "direction": direction, "actual_direction": actual, "direction_hit": direction == actual}
        result["labels"].append(row)
        previous = packet["observed_states"][symbol]["previous_session"]
        teachers.append({"symbol": symbol, "as_of": packet["as_of"], "kind": packet["kind"],
                         "label_end": label["label_end"], "actual_pct": label["actual_pct"],
                         "features": {"previous_return_pct": previous["return_pct"],
                                      "close_location_centered": None if previous["close_location"] is None else previous["close_location"] - 0.5,
                                      "range_pct": previous["range_pct"], "late_change_pct": previous["late_window"]["change_pct"]}})
        feedback["instruments"][symbol] = {**prediction["instruments"][symbol], "actual_pct": label["actual_pct"],
                                          "actual_direction": actual, "direction_hit": direction == actual,
                                          "next_action": "既反応と次場の追加反応を分け、同様の材料・価格状態との整合を照合する。単独因果は未識別。"}
        lines += [f"- {symbol}：予測{direction}→実績{label['actual_pct']:+.3f}%（{'一致' if direction == actual else '外れ'}）。",
                  "  - 診断：" + ("方向は整合。単独材料の因果と再現性は未識別。" if direction == actual else "既反応後の追加方向を捉えられず。原因は未識別。"),
                  "  - 次回：満了結果を渡し、同じ材料の再加点・反発断定を点検。"]
    write_json(round_path / "result.json", result)
    end = timestamp(packet["targets"]["today"])
    _append_audit(round_path, "reveal", end, [round_path / "result.json"])
    review = {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
              "available_at": end.isoformat(), "recorded_at": now(), "feedback": feedback, "next_training": teachers}
    write_json(round_path / "review.json", review)
    (round_path / "振り返り.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(round_path, "review", end, [round_path / "review.json", round_path / "振り返り.md"])
    return review


def summarize_analyst(root: Path) -> dict:
    """完了した全予定回を再採点し、発行率・基準比較と不採用判定を保存する。

    引数: root (Path) は実験ルート。
    返り値: dict は全件成績。既知期間を独立検証とは扱わない。
    """
    manifest = _verify(root)
    groups, rows = {}, []
    for identity in manifest["rounds"]:
        path = root / identity
        if _audit(path)[-1]["phase"] != "review":
            raise ValueError("全予定回の振り返り完了後に集計してください。")
        packet, result = read_json(path / "masked" / "入力.json"), read_json(path / "result.json")
        for label in result["labels"]:
            symbol, actual = label["symbol"], _direction(label["actual_pct"])
            following = packet["observed_states"][symbol]["previous_session"]["return_pct"]
            row = {"round": identity, "symbol": symbol, "kind": packet["kind"], "actual_pct": label["actual_pct"],
                   "direction": label["direction"], "actual_direction": actual, "hit": label["direction"] == actual}
            rows.append(row)
            for key in (symbol, symbol + "／" + packet["kind"]):
                group = groups.setdefault(key, {"n": 0, "hits": 0, "issued": 0, "held": 0,
                                                "up_hits": 0, "down_hits": 0, "following_hits": 0})
                group["n"] += 1
                group["issued"] += 1
                group["hits"] += int(row["hit"])
                group["up_hits"] += int(actual == "上昇")
                group["down_hits"] += int(actual == "下落")
                group["following_hits"] += int(following is not None and actual == _direction(following))
    for group in groups.values():
        group["accuracy"] = group["hits"] / group["n"]
    summary = {"mode": "回想・既知期間の開発再生", "independent_validation": False, "operational_acceptance": False,
               "accuracy_target": 0.8, "target_reached": all(groups[s]["accuracy"] >= 0.8 for s in SYMBOLS),
               "groups": groups, "rows": rows, "rounds": len(manifest["rounds"]), "manifest_hash": manifest["manifest_hash"]}
    write_json(root / "summary.json", summary)
    lines = ["# 循環10：全文と実測反応による直接判断", "", "- 各締切の予測担当を新規作成。全件の固定→開示→振り返りを順次実行。",
             "- 既知18時点の開発再生。担当分離は未使用期間の独立検証を意味しない。初期24教師は分母に含めない。", "",
             "| 対象 | 一致／全件 | 一致率 | 発行／全件 | 保留 | 常時上昇 | 常時下落 | 直前追随 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, value in groups.items():
        lines.append(f"| {key} | {value['hits']}/{value['n']} | {value['accuracy']:.1%} | {value['issued']}/{value['n']} | 0 | {value['up_hits']}/{value['n']} | {value['down_hits']}/{value['n']} | {value['following_hits']}/{value['n']} |")
    lines += ["", "- 判定：" + ("開発目標到達。独立未使用期間では未確認。" if summary["target_reached"] else "80％未達・不採用。"),
              "- １週間・１か月・３か月と確率は未実証。後発材料で失敗を除外しない。", ""]
    lines += [f"- {identity}：[予測]({identity}/予測.md)／[振り返り]({identity}/振り返り.md)。" for identity in manifest["rounds"]]
    (root / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnosis = ["# 診断", "", "- 四つの材料点への圧縮をやめ、各締切の全文・先物既反応・満了履歴を直接判断へ戻した。",
                 "- 一致・不一致は同じ先物の前セッションから対象終点までで判定。横ばい境界は±0.05%。",
                 "- 各回の因果は未識別。新規担当による判断でも、既知期間の再利用による開発選択は残る。",
                 "- 採用条件は両先物80％以上と未使用期間での確認。発行100％・保留０を維持。"]
    (root / "診断.md").write_text("\n".join(diagnosis) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """独立担当の間に実行する準備・固定・開示・集計を呼び出す。

    引数: なし。操作、ルート、必要な原本と回番号をコマンド行で指定する。
    返り値: None。操作結果の小さな要約を表示する。
    """
    parser = argparse.ArgumentParser(description="締切ごとに担当を分離する全文直接判断")
    parser.add_argument("action", choices=("prepare", "seal", "review", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--round")
    parser.add_argument("--actor", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare_analyst(args.root, args.inputs, args.measured, args.index)
        print({"session": value["session"], "news": len(value["news"]), "teachers": len(value["known_history"]), "input_hash": value["input_hash"]})
    elif args.action == "seal":
        value = seal_analyst(args.root / args.round, read_json(args.actor))
        print({symbol: item["direction"] for symbol, item in value["instruments"].items()})
    elif args.action == "review":
        value = review_analyst(args.root / args.round, args.source)
        print({symbol: item["direction_hit"] for symbol, item in value["feedback"]["instruments"].items()})
    else:
        print(summarize_analyst(args.root)["groups"])


if __name__ == "__main__":
    main()
