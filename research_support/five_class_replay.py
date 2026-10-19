"""五分類の規約を事前固定し、予測・結果開示・個別考察を分ける回想手順。"""

from __future__ import annotations

import argparse
import copy
import json
from decimal import Decimal
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.engine import reference_quote, session_plan
from fundamentals.pdca_analyst import _verify, prepare_analyst
from fundamentals.prices import SYMBOLS, load_candles
from fundamentals.replay import _append_audit, _audit, _file_hash
from research_support import five_class as evaluation_module
from research_support.five_class import score_prediction, validate_contract


ANSWERS = ("上がる", "下がる", "上がってから下がる", "下がってから上がる", "もみ合い")
PROTOCOL_FILE = "五分類手順の固定.json"
PREPARE_FILE = "五分類の準備固定.json"
OVERLAP_POLICY = "もみ合い優先・初回到達方向"


def _contract(root: Path) -> dict:
    """確認済みの規約だけを受け付け、利用者指定の数値境界を確認する。

    引数: root (Path) は新実験のルート。
    返り値: dict は確認日時・版・評価パラメーターを持つ規約。
    """
    value = read_json(root / "五分類規約.json")
    if value.get("confirmed") is not True or not value.get("version") or not value.get("confirmed_at"):
        raise ValueError("五分類規約を確認してから予測実験を開始してください。")
    if timestamp(value["confirmed_at"]) > timestamp(now()):
        raise ValueError("規約の確認日時が未来です。")
    if value.get("overlap_policy") != OVERLAP_POLICY:
        raise ValueError("分類優先規約は『もみ合い優先・初回到達方向』の確認済み方式が必要です。")
    criteria = value["criteria"]
    rules = validate_contract(criteria)
    for key, expected in (("movement_threshold_pct", 0.5), ("initial_excursion_pct", 0.5),
                          ("reversal_threshold_pct", 0.5), ("range_threshold_pct", 1),
                          ("flat_terminal_threshold_pct", 0.5)):
        if getattr(rules, key) != Decimal(str(expected)):
            raise ValueError("利用者指定の対象0.5％・高安幅1％・終点0.5％と一致しません。")
    if rules.target_basis != "場中到達" or rules.reversal_requires_reference_cross is not False:
        raise ValueError("確認済みの場中到達・基準跨ぎ不要の規約と一致しません。")
    return value


def _protocol(root: Path, create: bool = False) -> dict:
    """規約・静的指示・別版実装を初回準備前に固定し、以降の変更を拒否する。

    引数: root (Path) は新実験、create (bool) は最初の固定作成を許可する指定。
    返り値: dict はハッシュ検査済みの固定情報。
    """
    _contract(root)
    instructions = (root / "予測担当の手順.md").read_text(encoding="utf-8-sig")
    if any(answer not in instructions for answer in ANSWERS):
        raise ValueError("予測担当の静的指示に五つの回答を明記してください。")
    path = root / PROTOCOL_FILE
    if not path.exists():
        if not create or (root / "manifest.json").exists():
            raise ValueError("五分類手順は初回入力の作成より前に固定してください。")
        files = [root / "五分類規約.json", root / "予測担当の手順.md", Path(__file__),
                 Path(__file__).with_name("five_class.py")]
        value = {"version": "五分類１", "created_at": now(),
                 "files": {str(item.resolve()): _file_hash(item) for item in files}}
        value["protocol_hash"] = digest(value)
        write_json(path, value)
    value = read_json(path)
    if digest({key: item for key, item in value.items() if key != "protocol_hash"}) != value["protocol_hash"]:
        raise ValueError("五分類手順の自己ハッシュが不一致です。")
    for name, expected in value["files"].items():
        if _file_hash(Path(name)) != expected:
            raise ValueError("固定した五分類規約・静的指示または実装が変更されています。")
    return value


def _bound(round_path: Path) -> dict:
    """準備監査と規約・実装の束縛を、予測前から照合する。

    引数: round_path (Path) は当回のディレクトリ。
    返り値: dict は照合済みの手順固定情報。
    """
    _verify(round_path.parent)
    records = _audit(round_path)
    if PREPARE_FILE not in records[0]["files"]:
        raise ValueError("五分類の準備固定が監査へ結び付いていません。")
    prepared = read_json(round_path / PREPARE_FILE)
    if (prepared["protocol_file_hash"] != _file_hash(round_path.parent / PROTOCOL_FILE)
            or prepared["base_manifest_file_hash"] != _file_hash(round_path.parent / "manifest.json")):
        raise ValueError("準備後に五分類手順の固定情報が変更されています。")
    return _protocol(round_path.parent)


def prepare_five(root: Path, inputs: Path, measured: Path, index: int) -> dict:
    """旧採点を使わず、当回の可視入力へ事前固定した五分類規約を追加する。

    引数: root (Path) は新実験、inputs (Path) は可視原本、measured (Path) は実測原本、
        index (int) は１から始まる回番号。
    返り値: dict は当回の独立担当へ渡す入力。
    """
    root = Path(root)
    protocol = _protocol(root, create=index == 1)
    if (root / "manifest.json").exists():
        for identity in _verify(root)["rounds"][:index - 1]:
            _bound(root / identity)
    value = prepare_analyst(root, Path(inputs), Path(measured), index)
    destination = root / _verify(root)["rounds"][index - 1]
    records = _audit(destination)
    value["evaluation_contract"] = _contract(root)
    value["history_label_scope"] = "初期教師のactual_pctは終点実績のみ。five_class_evaluationがある行はその候補・品質を参照し、未確定を捏造しない。"
    value["input_hash"] = digest({key: item for key, item in value.items() if key != "input_hash"})
    write_json(destination / "masked" / "入力.json", value, exclusive=False)
    write_json(destination / PREPARE_FILE, {"protocol_hash": protocol["protocol_hash"],
               "protocol_file_hash": _file_hash(root / PROTOCOL_FILE),
               "base_manifest_file_hash": _file_hash(root / "manifest.json")})

    # ----------------------------------------
    # 子へ渡す前の準備記録に、新規約と最終入力をまとめて束縛する。

    if len(records) != 1 or records[0]["phase"] != "prepare":
        raise ValueError("準備記録の更新は担当へ入力を渡す前の一度だけです。")
    record = records[0]
    record["files"]["masked/入力.json"] = _file_hash(destination / "masked" / "入力.json")
    record["files"][PREPARE_FILE] = _file_hash(destination / PREPARE_FILE)
    record["prepare_started_at"], record["actual_at"] = record["actual_at"], now()
    record["entry_hash"] = digest({key: item for key, item in record.items() if key != "entry_hash"})
    (destination / "audit.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _bound(destination)
    return value


def _validate_items(items: dict, visible: set[str]) -> None:
    """五分類または無回答と、当回記事に結び付いた短い理由を検証する。

    引数: items (dict) は両先物の予測、visible (set[str]) は可視記事ID。
    返り値: None。不正な回答や参照を例外にする。
    """
    if not isinstance(items, dict) or set(items) != set(SYMBOLS):
        raise ValueError("両先物の予測が必要です。")
    for item in items.values():
        if set(item) != {"direction", "evidence_ids", "already_reacted", "additional_reaction", "invalidation"}:
            raise ValueError("予測の必須項目が一致しません。")
        if item["direction"] not in (*ANSWERS, None):
            raise ValueError("五つの回答か、無回答を示すnullが必要です。")
        ids = item["evidence_ids"]
        if (not isinstance(ids, list) or len(ids) > 3 or len(set(map(str, ids))) != len(ids)
                or any(str(key) not in visible for key in ids)):
            raise ValueError("根拠は当回の記事から重複なく３件以内です。")
        for key, maximum in (("already_reacted", 140), ("additional_reaction", 140), ("invalidation", 100)):
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > maximum or "\n" in item[key]:
                raise ValueError("予測理由は指定字数内の一行で保存してください。")


def seal_five(round_path: Path, actor: dict) -> dict:
    """新規担当の五分類予測を結果読取前に固定し、回答なしも明示して残す。

    引数: round_path (Path) は当回、actor (dict) は独立担当の識別子と完了証跡。
    返り値: dict は固定した予測。
    """
    protocol = _bound(round_path)
    if _audit(round_path)[-1]["phase"] != "prepare":
        raise ValueError("予測固定は準備直後の一度だけです。")
    input_path = round_path / "masked" / "入力.json"
    if (actor.get("fork_turns") != "none" or actor.get("completed") is not True
            or actor.get("input_file_hash") != _file_hash(input_path)
            or not actor.get("agent_id") or not actor.get("task_name")):
        raise ValueError("独立した当回担当と入力の完了証跡が必要です。")
    if any(read_json(path)["agent_id"] == actor["agent_id"] for path in round_path.parent.glob("*/actor.json")):
        raise ValueError("同じ担当を別締切の予測へ再利用できません。")
    packet, raw = read_json(input_path), read_json(round_path / "予測入力.json")
    if set(raw) != {"instruments"}:
        raise ValueError("予測入力にはinstrumentsだけを保存してください。")
    _validate_items(raw["instruments"], {str(item["id"]) for item in packet["news"]})
    actor = {**actor, "protocol_hash": protocol["protocol_hash"]}
    write_json(round_path / "actor.json", actor)
    value = {"as_of": packet["as_of"], "session": packet["session"], "issued_at": now(),
             "mode": "回想・既知期間の開発再生", "input_hash": packet["input_hash"],
             "protocol_hash": protocol["protocol_hash"], "actor_hash": digest(actor), **raw}
    value["forecast_hash"] = digest(value)
    write_json(round_path / "予測.json", value)
    lines = [f"# 予測：{packet['session']}", "", f"- 情報締切：{packet['as_of']}。回想・五分類の開発実験。",
             "- 同じ先物の前セッション最終観測値が基準。現物前日比で採点しない。", ""]
    for symbol, item in raw["instruments"].items():
        reference = packet["instruments"][symbol]["reference"]
        lines += [f"- {symbol}：{item['direction'] or '無回答'}。基準{reference['price']}（{reference['time']}）。",
                  "  - 既反応：" + item["already_reacted"], "  - 次場：" + item["additional_reaction"],
                  "  - 反証：" + item["invalidation"]]
    lines += ["", "- １週間・１か月・３か月、確率、ニュース条件付きレンジは未実証。",
              f"- 固定ハッシュ：{value['forecast_hash']}。"]
    (round_path / "予測.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(round_path, "seal", timestamp(packet["as_of"]),
                  [round_path / name for name in ("actor.json", "予測入力.json", "予測.json", "予測.md")])
    return value


def reveal_five(round_path: Path, prices: Path) -> dict:
    """満了かつ予測固定後に当回の先物経路を読み、新規約だけで評価する。

    引数: round_path (Path) は当回、prices (Path) は読み取り専用の価格ルート。
    返り値: dict は旧採点を混ぜない五分類の結果。
    """
    protocol = _bound(round_path)
    if _audit(round_path)[-1]["phase"] != "seal":
        raise ValueError("結果開示は予測固定後の一度だけです。")
    packet, prediction = read_json(round_path / "masked" / "入力.json"), read_json(round_path / "予測.json")
    target = timestamp(packet["targets"]["today"])
    if target > timestamp(now()):
        raise ValueError("対象終点が実時計で未満了のため価格結果を読み取れません。")
    evaluator = getattr(evaluation_module, "evaluate_priority_path", None)
    if not callable(evaluator):
        raise ValueError("確認済み規約の優先分類器が未実装のため価格結果を読み取れません。")
    labels = []
    interval = packet["evaluation_contract"]["criteria"]["interval_minutes"]
    for symbol in SYMBOLS:
        candles = load_candles(Path(prices), symbol, interval=1, cutoff=target)
        reference = reference_quote(candles, timestamp(packet["as_of"]), session_plan(timestamp(packet["as_of"])))
        fixed = packet["instruments"][symbol]["reference"]
        if (reference.close != fixed["price"] or reference.end != timestamp(fixed["time"])
                or reference.session != fixed["session"]):
            raise ValueError("結果取得時の先物基準が固定予測の基準と一致しません。")
        path_candles = candles if interval == 1 else load_candles(Path(prices), symbol, interval=interval, cutoff=target)
        path = [bar for bar in path_candles if bar.session == packet["session"]]
        evaluation = evaluator(reference, path, packet["session"], target,
                               packet["evaluation_contract"]["criteria"])
        answer = prediction["instruments"][symbol]["direction"]
        score = score_prediction(answer, evaluation)
        source = Path(prices) / f"candle_{interval}min" / symbol / (packet["session"] + ".csv")
        labels.append({"symbol": symbol, "label_end": target.isoformat(), "direction": answer,
                       "evaluation": evaluation, "score": score,
                       "source_file": str(source.resolve()),
                       "source_file_hash": _file_hash(source) if source.exists() else None})
    value = {"forecast_hash": prediction["forecast_hash"], "protocol_hash": protocol["protocol_hash"],
             "read_after_seal_at": now(), "labels": labels}
    write_json(round_path / "result.json", value)
    _append_audit(round_path, "reveal", target, [round_path / "result.json"])
    return value


def finalize_five(round_path: Path, reflection: dict, actor: dict) -> dict:
    """同じ当回担当の個別考察を固定し、次の締切へ渡せる履歴にする。

    引数: round_path (Path) は結果開示済み当回、reflection (dict) は短い個別考察、
        actor (dict) は考察担当の識別子と完了証跡。
    返り値: dict は予測・結果・考察の結び付いた次回用履歴。
    """
    protocol = _bound(round_path)
    if _audit(round_path)[-1]["phase"] != "reveal":
        raise ValueError("個別考察は結果開示後の一度だけ固定してください。")
    packet, prediction = read_json(round_path / "masked" / "入力.json"), read_json(round_path / "予測.json")
    result, original_actor = read_json(round_path / "result.json"), read_json(round_path / "actor.json")
    if (actor.get("completed") is not True or actor.get("agent_id") != original_actor["agent_id"]
            or reflection.get("forecast_hash") != prediction["forecast_hash"]
            or reflection.get("result_hash") != digest(result)):
        raise ValueError("考察の当回担当・予測・結果の結び付きが一致しません。")
    if set(reflection) != {"forecast_hash", "result_hash", "instruments"} or set(reflection["instruments"]) != set(SYMBOLS):
        raise ValueError("考察は指定形式で両先物を含めてください。")
    labels = {row["symbol"]: row for row in result["labels"]}
    visible = {str(item["id"]) for item in packet["news"]}
    teachers, feedback = [], {"session": packet["session"], "instruments": {}}
    lines = [f"# 振り返り：{packet['session']}", "", "- 原因は仮説。後発材料や経路不明を理由に予定回を削除しない。", ""]
    for symbol, item in reflection["instruments"].items():
        row = labels[symbol]
        if set(item) != {"predicted_direction", "score_status", "causality", "reason", "next_action", "evidence_ids"}:
            raise ValueError("個別考察の必須項目が一致しません。")
        if (item["predicted_direction"] != row["direction"] or item["score_status"] != row["score"]["status"]
                or item["causality"] not in ("仮説", "未識別")):
            raise ValueError("考察の固定予測・採点状態・因果状態が一致しません。")
        for key, maximum in (("reason", 200), ("next_action", 160)):
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > maximum or "\n" in item[key]:
                raise ValueError("当回の比較理由・次回対応を指定字数内の一行にしてください。")
        ids = item["evidence_ids"]
        if (not isinstance(ids, list) or len(ids) > 3 or len(set(map(str, ids))) != len(ids)
                or any(str(key) not in visible for key in ids)):
            raise ValueError("考察根拠は可視記事から３件以内です。")
        evaluation = row["evaluation"]
        metrics = evaluation["metrics"]
        actual = metrics.get("actual_pct")
        if actual is not None:
            teachers.append({"symbol": symbol, "as_of": packet["as_of"], "kind": packet["kind"],
                             "label_end": row["label_end"], "actual_pct": float(actual), "features": {},
                             "five_class_evaluation": copy.deepcopy(evaluation)})
        predicted = prediction["instruments"][symbol]
        feedback["instruments"][symbol] = {**{key: value for key, value in predicted.items() if key != "evidence_ids"},
            **{key: value for key, value in item.items() if key != "evidence_ids"},
            "prediction_evidence_ids": predicted["evidence_ids"], "reflection_evidence_ids": ids,
            "evaluation": copy.deepcopy(evaluation), "score": copy.deepcopy(row["score"])}
        actual_text = "終点未観測" if actual is None else f"終点{float(actual):+.3f}%"
        lines += [f"- {symbol}：{row['direction'] or '無回答'} → {row['score']['status']}（{actual_text}）。",
                  "  - 考察：" + item["reason"], "  - 次回：" + item["next_action"]]
    write_json(round_path / "考察入力.json", reflection)
    write_json(round_path / "考察担当.json", {**actor, "reflection_hash": digest(reflection)})
    value = {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
             "protocol_hash": protocol["protocol_hash"], "available_at": packet["targets"]["today"],
             "recorded_at": now(), "feedback": feedback, "next_training": teachers}
    write_json(round_path / "review.json", value)
    (round_path / "振り返り.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(round_path, "review", timestamp(packet["targets"]["today"]),
                  [round_path / name for name in ("review.json", "振り返り.md", "考察入力.json", "考察担当.json")])
    return value


def _new_group() -> dict:
    """本体と定数回答に共通する対象分母・正解数の空集計を作る。

    引数: なし。
    返り値: dict は対象状態・無回答・確実／可能正解を分けた集計。
    """
    return {"scheduled": 0, "target": 0, "target_unknown": 0, "outside": 0,
            "unanswered": 0, "target_unanswered": 0, "target_unknown_unanswered": 0,
            "outside_unanswered": 0, "denominator": 0, "correct_lower": 0, "correct_upper": 0,
            "known_correct_lower": 0, "known_correct_upper": 0, "unknown_possible_correct": 0}


def _add_score(group: dict, answer: str | None, evaluation: dict, score: dict) -> None:
    """同じ対象状態と採点契約で、本体または定数回答の１件を集計する。

    引数: group (dict) は集計、answer (str | None) は回答、evaluation (dict) は経路評価、score (dict) は採点。
    返り値: None。対象不明も採点の主分母へ残す。
    """
    target = evaluation["target_status"]
    group["scheduled"] += 1
    group["unanswered"] += int(answer is None)
    group[{"対象": "target", "対象外": "outside", "未確定": "target_unknown"}[target]] += 1
    if answer is None:
        group[{"対象": "target_unanswered", "対象外": "outside_unanswered", "未確定": "target_unknown_unanswered"}[target]] += 1
    for name in ("denominator", "correct_lower", "correct_upper"):
        group[name] += int(score[name])
    if target == "対象":
        group["known_correct_lower"] += int(score["correct_lower"])
        group["known_correct_upper"] += int(score["correct_upper"])
    elif target == "未確定":
        group["unknown_possible_correct"] += int(score["correct_upper"])


def _finish_rates(group: dict) -> None:
    """対象不明の採否による真の精度境界と、分母固定の参考比率を分ける。

    引数: group (dict) は対象・不明・対象外を集計済みの値。
    返り値: None。主下限、楽観上限、確定対象のみの精度を追加する。
    """
    denominator = group["denominator"]
    group["accuracy_lower"] = group["correct_lower"] / denominator if denominator else None
    group["fixed_denominator_possible_rate"] = group["correct_upper"] / denominator if denominator else None
    group["known_target_accuracy_lower"] = group["known_correct_lower"] / group["target"] if group["target"] else None
    group["known_target_accuracy_upper"] = group["known_correct_upper"] / group["target"] if group["target"] else None
    optimistic_n = group["target"] + group["unknown_possible_correct"]
    group["upper_denominator"] = optimistic_n
    group["accuracy_upper"] = (group["known_correct_upper"] + group["unknown_possible_correct"]) / optimistic_n if optimistic_n else (0.0 if denominator else None)


def summarize_five(root: Path) -> dict:
    """予定全回を残し、対象・対象不明・対象外と精度の下限上限を集計する。

    引数: root (Path) は全回の考察が完了した新実験。
    返り値: dict は先物別・日中夜間別の開発成績。
    """
    manifest = _verify(root)
    groups, rows = {}, []
    for identity in manifest["rounds"]:
        path = root / identity
        _bound(path)
        if _audit(path)[-1]["phase"] != "review":
            raise ValueError("予定全回の考察が終わってから集計してください。")
        packet, result = read_json(path / "masked" / "入力.json"), read_json(path / "result.json")
        for label in result["labels"]:
            score = score_prediction(label["direction"], label["evaluation"])
            if score != label["score"]:
                raise ValueError("保存した当回採点が新規約による再計算と一致しません。")
            constants = {answer: score_prediction(answer, label["evaluation"]) for answer in ANSWERS}
            row = {"round": identity, "kind": packet["kind"], **label, "constant_scores": constants}
            rows.append(row)
            for key in (label["symbol"], label["symbol"] + "／" + packet["kind"]):
                group = groups.setdefault(key, {**_new_group(), "baselines": {answer: _new_group() for answer in ANSWERS}})
                _add_score(group, label["direction"], label["evaluation"], score)
                for answer in ANSWERS:
                    _add_score(group["baselines"][answer], answer, label["evaluation"], constants[answer])
    for group in groups.values():
        _finish_rates(group)
        for baseline in group["baselines"].values():
            _finish_rates(baseline)
    value = {"mode": "回想・既知期間の開発再生", "independent_validation": False,
             "operational_acceptance": False, "accuracy_target": 0.8, "groups": groups, "rows": rows,
             "baseline_spec": "５回答それぞれを全予定回で発行。全方式を同じ対象評価と主分母で比較し、事後選択しない。",
             "target_reached": all(groups[s]["accuracy_lower"] is not None and groups[s]["accuracy_lower"] >= 0.8 for s in SYMBOLS)}
    write_json(root / "summary.json", value)
    lines = ["# 五分類の開発結果", "", "- 同じ既知期間の再生。独立検証・運用採用は未達。",
             "- 下限は対象不明を全て外れとして分母へ残す。上限は正解し得る不明例だけ対象、残りは対象外という楽観条件。", "",
             "| 対象 | 全予定 | 確定対象 | 対象不明 | 対象外 | 無回答 | 下限正解／分母 | 下限精度 | 楽観上限 | 確定対象のみの確実正解 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, group in groups.items():
        rate = "算出不能" if group["accuracy_lower"] is None else f"{group['accuracy_lower']:.1%}"
        upper = "算出不能" if group["accuracy_upper"] is None else f"{group['accuracy_upper']:.1%}"
        lines.append(f"| {name} | {group['scheduled']} | {group['target']} | {group['target_unknown']} | {group['outside']} | {group['unanswered']} | {group['correct_lower']}/{group['denominator']} | {rate} | {upper} | {group['known_correct_lower']}/{group['target']} |")
    lines += ["", "- 対象不明しかなく真の対象が０件の場合、対象精度は定義できない。上限は対象が１件以上ある場合の条件付き境界。",
              "- 分母を固定した可能正解率と、確定対象だけの成績はsummary.jsonへ別保存。"]
    lines += ["", "- 定数回答の比較：全予定回に同じ回答を出す５方式を固定順で全て掲載。値は同じ主分母での正解下限。", "",
              "| 対象 | " + " | ".join(ANSWERS) + " |", "| --- | " + " | ".join("---" for _ in ANSWERS) + " |"]
    for name, group in groups.items():
        cells = []
        for answer in ANSWERS:
            baseline = group["baselines"][answer]
            rate = "算出不能" if baseline["accuracy_lower"] is None else f"{baseline['accuracy_lower']:.1%}"
            cells.append(f"{baseline['correct_lower']}/{baseline['denominator']}（{rate}）")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += ["", "- 開発目標：" + ("下限で80％到達。未使用期間の確認が必要。" if value["target_reached"] else "80％未達・未完成。")]
    (root / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def main() -> None:
    """準備・固定・結果開示・考察確定・集計を別々のコマンドで実行する。

    引数: なし。対象操作・パスをコマンド行から受け取る。
    返り値: None。操作結果の短い要約を表示する。
    """
    parser = argparse.ArgumentParser(description="規約を先に固定する五分類の回想検証")
    parser.add_argument("action", choices=("prepare", "seal", "reveal", "review", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--round")
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--reflection", type=Path)
    args = parser.parse_args()
    path = args.root / args.round if args.round else args.root
    if args.action == "prepare":
        value = prepare_five(args.root, args.inputs, args.measured, args.index)
        print({"session": value["session"], "input_hash": value["input_hash"]})
    elif args.action == "seal":
        value = seal_five(path, read_json(args.actor))
        print({key: item["direction"] for key, item in value["instruments"].items()})
    elif args.action == "reveal":
        print({row["symbol"]: row["score"]["status"] for row in reveal_five(path, args.prices)["labels"]})
    elif args.action == "review":
        finalize_five(path, read_json(args.reflection), read_json(args.actor))
        print("個別考察を固定しました。")
    else:
        print(summarize_five(args.root)["groups"])


if __name__ == "__main__":
    main()
