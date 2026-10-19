"""未使用時点向けに境界と個別考察を固定する、旧実験を変更しない別版手順。"""

from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.pdca_analyst import _verify, prepare_analyst, seal_analyst
from fundamentals.pdca_measured_research import SYMBOLS, _direction, _source_labels
from fundamentals.replay import _append_audit, _audit, _file_hash


BOUNDARY_SENTENCE = "横ばいは-0.05％以上+0.05％以下（境界を含む）。"
PROTOCOL_FILE = "個別考察手順の固定.json"
PREPARE_FILE = "準備固定.json"


def _protocol(root: Path, create: bool = False) -> dict:
    """横ばい境界を事前確認し、別版実装と静的指示を固定・再照合する。

    引数: root (Path) は新しい実験、create (bool) は初回の固定作成を許可する指定。
    返り値: dict は自己ハッシュを照合した手順固定情報。
    """
    instructions = root / "予測担当の手順.md"
    normal = unicodedata.normalize("NFKC", instructions.read_text(encoding="utf-8-sig"))
    if unicodedata.normalize("NFKC", BOUNDARY_SENTENCE) not in normal:
        raise ValueError("静的指示に横ばい境界±0.05％と境界を含む定義を明記してください。")
    path = root / PROTOCOL_FILE
    if not path.exists():
        if not create or (root / "manifest.json").exists():
            raise ValueError("別版手順は初回準備より前に固定する必要があります。")
        value = {"version": "2", "boundary_pct": 0.05, "required_sentence": BOUNDARY_SENTENCE,
                 "wrapper_path": str(Path(__file__).resolve()), "wrapper_hash": _file_hash(Path(__file__)),
                 "instructions_path": str(instructions.resolve()), "instructions_hash": _file_hash(instructions)}
        value["protocol_hash"] = digest(value)
        write_json(path, value)
    value = read_json(path)
    if digest({key: item for key, item in value.items() if key != "protocol_hash"}) != value["protocol_hash"]:
        raise ValueError("別版手順の固定情報が変更されています。")
    if (value["wrapper_path"] != str(Path(__file__).resolve())
            or value["wrapper_hash"] != _file_hash(Path(__file__))
            or value["instructions_path"] != str(instructions.resolve())
            or value["instructions_hash"] != _file_hash(instructions)):
        raise ValueError("別版実装または静的指示が変更されています。")
    return value


def _bound_protocol(round_path: Path) -> dict:
    """当回の担当固定に別版実装と指示が結び付いていることを照合する。

    引数: round_path (Path) は予測固定済みの当回。
    返り値: dict は照合した別版手順。
    """
    _verify(round_path.parent)
    _verify_prepare(round_path)
    value = _protocol(round_path.parent)
    actor = read_json(round_path / "actor.json")
    if (actor.get("protocol_hash") != value["protocol_hash"]
            or actor.get("protocol_file_hash") != _file_hash(round_path.parent / PROTOCOL_FILE)):
        raise ValueError("担当固定と別版手順の結び付きが一致しません。")
    return value


def _verify_prepare(round_path: Path) -> None:
    """準備時点で監査へ束縛した手順・実装・指示のハッシュを照合する。

    引数: round_path (Path) は当回。
    返り値: None。準備後から初回予測固定までの同時変更も拒否する。
    """
    records = _audit(round_path)
    if PREPARE_FILE not in records[0]["files"]:
        raise ValueError("別版手順が準備監査へ固定されていません。")
    prepared = read_json(round_path / PREPARE_FILE)
    root = round_path.parent
    if (prepared["protocol_file_hash"] != _file_hash(root / PROTOCOL_FILE)
            or prepared["wrapper_hash"] != _file_hash(Path(__file__))
            or prepared["instructions_hash"] != _file_hash(root / "予測担当の手順.md")
            or prepared["base_manifest_file_hash"] != _file_hash(root / "manifest.json")):
        raise ValueError("準備固定後に別版手順・実装・指示または一覧が変更されています。")


def prepare_v2(root: Path, inputs: Path, measured: Path, index: int) -> dict:
    """境界を検査した新実験だけに、旧版の可視入力切出しを適用する。

    引数: root (Path) は新実験、inputs (Path) は可視原本、measured (Path) は実測原本、
        index (int) は１始まりの回番号。
    返り値: dict は当回入力。既存の循環10は更新しない。
    """
    root = Path(root)
    protocol = _protocol(root, create=index == 1)
    if (root / "manifest.json").exists():
        manifest = _verify(root)
        for identity in manifest["rounds"][:index - 1]:
            _bound_protocol(root / identity)
    value = prepare_analyst(root, Path(inputs), Path(measured), index)
    manifest = _verify(root)
    destination = root / manifest["rounds"][index - 1]
    fixed = {"protocol_hash": protocol["protocol_hash"], "protocol_file_hash": _file_hash(root / PROTOCOL_FILE),
             "wrapper_hash": protocol["wrapper_hash"], "instructions_hash": protocol["instructions_hash"],
             "base_manifest_file_hash": _file_hash(root / "manifest.json")}
    write_json(destination / PREPARE_FILE, fixed)

    # ----------------------------------------
    # 子へ入力を返す前に、まだ一段階だけの準備記録へ別版の固定一覧を束縛する。

    records = _audit(destination)
    if len(records) != 1 or records[0]["phase"] != "prepare":
        raise ValueError("別版手順の追加固定は入力を渡す前の準備段階に限ります。")
    record = records[0]
    record["files"][PREPARE_FILE] = _file_hash(destination / PREPARE_FILE)
    record["prepare_started_at"] = record["actual_at"]
    record["actual_at"] = now()
    record["entry_hash"] = digest({key: item for key, item in record.items() if key != "entry_hash"})
    (destination / "audit.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _verify_prepare(destination)
    return value


def seal_v2(round_path: Path, actor: dict) -> dict:
    """担当と予測に、別版実装・境界指示の固定ハッシュを追加して固定する。

    引数: round_path (Path) は当回、actor (dict) は独立予測担当の作成完了証跡。
    返り値: dict は旧版と互換の固定予測。
    """
    _verify_prepare(round_path)
    value = _protocol(round_path.parent)
    bound_actor = {**actor, "protocol_hash": value["protocol_hash"],
                   "protocol_file_hash": _file_hash(round_path.parent / PROTOCOL_FILE)}
    return seal_analyst(round_path, bound_actor)


def reveal_v2(round_path: Path, source: Path) -> dict:
    """当回予測固定後に結果だけを開示し、考察前のreviewを作らない。

    引数: round_path (Path) は固定済み当回、source (Path) は原結果ルート。
    返り値: dict は当回採点と隔離した原採点を含む結果。次回へはまだ進めない。
    """
    if _audit(round_path)[-1]["phase"] != "seal":
        raise ValueError("結果開示は予測固定後の一度だけです。")
    protocol = _bound_protocol(round_path)
    packet, prediction = read_json(round_path / "masked" / "入力.json"), read_json(round_path / "予測.json")
    if timestamp(packet["targets"]["today"]) > timestamp(now()):
        raise ValueError("対象終点が実時計で未満了のため原結果を読み取れません。")
    original, original_hash = _source_labels(Path(source) / round_path.name, packet)
    labels = []
    for item in original:
        symbol = item["symbol"]
        direction = prediction["instruments"][symbol]["direction"]
        actual = _direction(item["actual_pct"])
        row = {key: item[key] for key in ("symbol", "horizon", "label_end", "reference_price", "quote_price",
                                         "quote_time", "actual_pct", "low_pct", "high_pct", "path_hash") if key in item}
        row.update(direction=direction, actual_direction=actual, direction_hit=direction == actual,
                   metrics={"direction_hit": direction == actual}, source_metrics=copy.deepcopy(item.get("metrics")))
        labels.append(row)
    result = {"forecast_hash": prediction["forecast_hash"], "source_result_hash": original_hash,
              "protocol_hash": protocol["protocol_hash"], "read_after_seal_at": now(), "labels": labels}
    write_json(round_path / "result.json", result)
    _append_audit(round_path, "reveal", timestamp(packet["targets"]["today"]), [round_path / "result.json"])
    return result


def finalize_review_v2(round_path: Path, reflection_path: Path, reflection_actor: dict) -> dict:
    """当回固有の比較・理由・次回対応を照合してから、振り返りを確定する。

    引数: round_path (Path) は結果開示済み当回、reflection_path (Path) は担当が新規保存した考察JSON、
        reflection_actor (dict) は同じ当回担当の識別子と考察作成完了証跡。
    返り値: dict は次回へ渡せる個別考察と満了教師。
    """
    if _audit(round_path)[-1]["phase"] != "reveal":
        raise ValueError("個別考察の固定は結果開示後の一度だけです。")
    protocol = _bound_protocol(round_path)
    packet, prediction = read_json(round_path / "masked" / "入力.json"), read_json(round_path / "予測.json")
    result, actor = read_json(round_path / "result.json"), read_json(round_path / "actor.json")
    reflection = read_json(reflection_path)
    if (reflection_actor.get("agent_id") != actor["agent_id"] or reflection_actor.get("completed") is not True
            or reflection.get("forecast_hash") != prediction["forecast_hash"]
            or reflection.get("result_hash") != digest(result)):
        raise ValueError("当回担当、考察の作成完了、予測または結果の結び付きが違います。")
    if set(reflection) != {"forecast_hash", "result_hash", "instruments"} or set(reflection["instruments"]) != set(SYMBOLS):
        raise ValueError("考察は当回の両先物を含む指定形式が必要です。")
    by_symbol = {row["symbol"]: row for row in result["labels"]}
    visible = {str(item["id"]) for item in packet["news"]}
    for symbol, item in reflection["instruments"].items():
        if set(item) != {"predicted_direction", "actual_direction", "causality", "reason", "next_action", "evidence_ids"}:
            raise ValueError("考察の必須項目が一致しません。")
        if (item["predicted_direction"] != by_symbol[symbol]["direction"]
                or item["actual_direction"] != by_symbol[symbol]["actual_direction"]
                or item["causality"] not in ("未識別", "仮説")):
            raise ValueError("考察の予測・実績方向または因果の状態が一致しません。")
        for key, maximum in (("reason", 200), ("next_action", 160)):
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > maximum or "\n" in item[key]:
                raise ValueError("当回の比較理由と次回対応は指定字数内の一行が必要です。")
        ids = item["evidence_ids"]
        if not isinstance(ids, list) or len(ids) > 3 or len(set(map(str, ids))) != len(ids) or any(str(key) not in visible for key in ids):
            raise ValueError("考察根拠は当回の可視記事３件以内です。")

    # ----------------------------------------

    destination = round_path / "考察入力.json"
    if reflection_path.resolve() != destination.resolve():
        write_json(destination, reflection)
    proof = {**reflection_actor, "forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
             "reflection_file_hash": _file_hash(destination), "protocol_hash": protocol["protocol_hash"],
             "allowed_inputs": {str(path.resolve()): _file_hash(path) for path in (
                 round_path.parent / "予測担当の手順.md", round_path / "masked" / "入力.json",
                 round_path / "予測.json", round_path / "result.json")}}
    write_json(round_path / "考察担当.json", proof)
    teachers, feedback = [], {"session": packet["session"], "instruments": {}}
    lines = [f"# 振り返り：{packet['session']}", "", "- 結果開示後の当回担当による個別考察。原因は候補として扱い、外れを除外しない。", ""]
    for symbol in SYMBOLS:
        row, item = by_symbol[symbol], reflection["instruments"][symbol]
        previous = packet["observed_states"][symbol]["previous_session"]
        teachers.append({"symbol": symbol, "as_of": packet["as_of"], "kind": packet["kind"],
                         "label_end": row["label_end"], "actual_pct": row["actual_pct"],
                         "features": {"previous_return_pct": previous["return_pct"],
                                      "close_location_centered": None if previous["close_location"] is None else previous["close_location"] - 0.5,
                                      "range_pct": previous["range_pct"], "late_change_pct": previous["late_window"]["change_pct"]}})
        predicted = prediction["instruments"][symbol]
        feedback["instruments"][symbol] = {
            **{key: value for key, value in predicted.items() if key != "evidence_ids"},
            **{key: value for key, value in item.items() if key != "evidence_ids"},
            "prediction_evidence_ids": copy.deepcopy(predicted["evidence_ids"]),
            "reflection_evidence_ids": copy.deepcopy(item["evidence_ids"]),
            "actual_pct": row["actual_pct"], "direction_hit": row["direction_hit"]}
        lines += [f"- {symbol}：予測{row['direction']}→実績{row['actual_pct']:+.3f}%（{'一致' if row['direction_hit'] else '外れ'}）。",
                  "  - 考察：" + item["reason"] + "（因果：" + item["causality"] + "）。",
                  "  - 次回：" + item["next_action"]]
    review = {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
              "protocol_hash": protocol["protocol_hash"], "available_at": packet["targets"]["today"],
              "recorded_at": now(), "feedback": feedback, "next_training": teachers,
              "reflection_hash": digest(reflection)}
    write_json(round_path / "review.json", review)
    (round_path / "振り返り.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(round_path, "review", timestamp(packet["targets"]["today"]),
                  [round_path / name for name in ("review.json", "振り返り.md", "考察入力.json", "考察担当.json")])
    return review
