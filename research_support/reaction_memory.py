"""固定済みの予測・実績・振り返りを、時点別に再利用する反応事例台帳。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.replay import _audit, _file_hash


POINTS = ("impact_pt", "priced_in_before_reference_pt", "priced_in_after_reference_pt",
          "reference_remaining_pt", "residual_pt")
NOTICE = "単発の共同行動の観測。材料の因果効果・独立標本・将来の勝率を意味しない。"


def _check_hash(value: dict, field: str) -> None:
    """自己ハッシュを照合し、保存内容の置換を拒否する。

    引数: value (dict) は保存内容、field (str) は自己ハッシュの項目名。
    返り値: None。ハッシュ不一致はValueError。
    """
    if value.get(field) != digest({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"自己ハッシュが一致しません: {field}")


def _summary(label: dict, score: dict) -> dict:
    """分類候補・欠測・評価窓を確定し直さずに保存する。

    引数: label (dict) は固定した評価、score (dict) は当回採点。
    返り値: dict は観測と未確定を区別した記録。
    """
    evaluation = label["evaluation"]
    return {"label_end": label["label_end"], "session": evaluation["session"],
            "target_status": evaluation["target_status"],
            "correct_candidates": copy.deepcopy(evaluation.get("correct_candidates", [])),
            "uncertain_candidates": copy.deepcopy(evaluation.get("uncertain_candidates", [])),
            "metrics": copy.deepcopy(evaluation.get("metrics", {})),
            "quality": copy.deepcopy(evaluation.get("quality", {})),
            "evaluation_window": copy.deepcopy(evaluation.get("evaluation_window", {})),
            "score": copy.deepcopy(score)}


def _event_latest_clock(value: object) -> list:
    """認定根拠の明示された利用可能時計を集める。

    引数: value (object) は認定の入れ子構造。
    返り値: list[datetime] は時計の下限。記事公開上限を実初報へ置換しない。
    """
    clocks = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("first_public_at", "published_at", "fetched_at", "retrieved_at", "available_at", "effective_at",
                       "updated_at", "received_at", "known_at", "article_public_upper_bound") and item:
                clocks.append(timestamp(item))
            elif isinstance(item, (dict, list)):
                clocks.extend(_event_latest_clock(item))
    elif isinstance(value, list):
        for item in value:
            clocks.extend(_event_latest_clock(item))
    return clocks


def verify_memory(memory_root: Path) -> list[dict]:
    """追記台帳の連鎖と元ファイル・事例のハッシュを照合する。

    引数: memory_root (Path) は反応台帳の保存先。
    返り値: list[dict] は検証済みの事例。元資料を含む変更や孤立事例はValueError。
    """
    memory_root = Path(memory_root)
    audit_path = memory_root / "audit.jsonl"
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line] if audit_path.exists() else []
    previous, cases, names = None, [], set()
    for record in records:
        _check_hash(record, "entry_hash")
        if record["previous_hash"] != previous or record["case_id"] in names:
            raise ValueError("反応台帳の追記順序または識別子が不正です。")
        path = memory_root / "cases" / (record["case_id"] + ".json")
        if not path.is_file() or _file_hash(path) != record["case_file_sha256"]:
            raise ValueError("固定した反応事例が変更されています。")
        case = read_json(path)
        _check_hash(case, "case_hash")
        if case["case_id"] != record["case_id"] or case["case_hash"] != record["case_hash"]:
            raise ValueError("事例と監査記録が一致しません。")
        for source, expected in case["source_hashes"].items():
            if not Path(source).is_file() or _file_hash(Path(source)) != expected:
                raise ValueError(f"反応事例の元ファイルが変更されています: {source}")
        cases.append(case)
        names.add(case["case_id"])
        previous = record["entry_hash"]
    actual = {path.stem for path in (memory_root / "cases").glob("*.json")}
    if actual != names:
        raise ValueError("監査記録にない反応事例があります。")
    return cases


def append_reaction_case(round_path: Path, memory_root: Path, *,
                         evidence_available_at: str | None = None,
                         simulated_review_at: str | None = None) -> dict:
    """完了した当回の固定資料を照合し、新しい反応事例を一度だけ追記する。

    引数: round_path (Path) はreview段階まで完了した当回、memory_root (Path) は台帳先。
    evidence_available_at (str | None) は認定・考察の全根拠を利用可能となった再生時計。
    未確認ならNoneとし、後のsnapshotでは認定・考察を開示しない。
    simulated_review_at (str | None) は振り返りを完了した再生時計。省略時は元監査時計。
    返り値: dict は自己ハッシュ付き事例。材料因果や正しい反転方向を追加認定しない。
    """
    round_path, memory_root = Path(round_path).resolve(), Path(memory_root).resolve()
    verify_memory(memory_root)
    audit = _audit(round_path)
    audit_sha256 = _file_hash(round_path / "audit.jsonl")
    if len(audit) != 4 or audit[-1]["phase"] != "review":
        raise ValueError("予測固定・開示・振り返りまで完了した当回だけ追記できます。")
    required = {"予測.json": "seal", "result.json": "reveal", "振り返り入力.json": "review"}
    for name, phase in required.items():
        record = next(item for item in audit if item["phase"] == phase)
        if name not in record["files"]:
            raise ValueError(f"当回監査へ固定されていないファイルです: {name}")
    forecast, result, review = (read_json(round_path / name) for name in required)
    _check_hash(forecast, "ledger_hash")
    forecast_hash, result_hash = forecast["ledger_hash"], digest(result)
    if result.get("forecast_hash") != forecast_hash or review.get("forecast_hash") != forecast_hash:
        raise ValueError("予測と実績・振り返りの結び付きが一致しません。")
    if review.get("result_hash") != result_hash:
        raise ValueError("振り返りが同じ結果全体へ結び付いていません。")
    as_of = timestamp(forecast["as_of"])
    labels = result["labels"]
    if not labels or len({item["symbol"] for item in labels}) != len(labels):
        raise ValueError("結果には重複しない先物が必要です。")
    ends = [timestamp(item["full_reference"]["label_end"]) for item in labels]
    simulated_end = max(ends)
    review_at = timestamp(simulated_review_at or audit[-1]["simulated_at"])
    if simulated_end <= as_of or review_at < simulated_end or review_at < timestamp(audit[-1]["simulated_at"]):
        raise ValueError("対象場の満了前へ振り返り時計を繰り上げられません。")
    evidence_at = timestamp(evidence_available_at) if evidence_available_at is not None else None
    known_clocks = _event_latest_clock([item["primary"]["evaluation"].get("evaluation_window", {}) for item in labels])
    if evidence_at is not None and known_clocks and evidence_at < max(known_clocks):
        raise ValueError("認定根拠の記事時計より前を利用可能時計に指定できません。")

    # ----------------------------------------
    observations = []
    sessions = set()
    for item in labels:
        symbol = item["symbol"]
        prediction = forecast["forecasts"][symbol]["today"]
        if prediction["direction"] != item["answer"] or symbol not in review["reviews"]:
            raise ValueError("結果・振り返りの対象先物または固定回答が一致しません。")
        primary = _summary(item["primary"], item["score"])
        full = _summary(item["full_reference"], item["full_reference_score"])
        if timestamp(primary["label_end"]) > simulated_end:
            raise ValueError("主評価の終点が全場終点を越えています。")
        sessions.add(full["session"])
        materials = []
        for material in forecast["materials"]:
            points = material.get("impacts", {}).get(symbol, {}).get("today")
            if points is None:
                continue
            materials.append({"material_id": material["material_id"], "story_key": material["story_key"],
                              "title": material["title"], "status": material.get("status"),
                              "evidence": copy.deepcopy(material.get("evidence", [])),
                              "points": {key: points.get(key) for key in POINTS},
                              "channel": points.get("transmission", ""),
                              "visible_observation_windows": copy.deepcopy(points.get("observations", {})),
                              "pricing_basis": copy.deepcopy(points.get("evidence", {})),
                              "causal_status": "因果未識別"})
        observations.append({"symbol": symbol, "reference": copy.deepcopy(forecast["references"][symbol]),
                             "prediction": copy.deepcopy(prediction), "materials": materials,
                             "primary": primary, "full_reference": full,
                             "review": copy.deepcopy(review["reviews"][symbol]),
                             "causal_status": "因果未識別"})
    if len(sessions) != 1:
        raise ValueError("一事例は同一場の両先物に限ります。")
    case = {"version": "市場反応事例１", "case_id": forecast_hash,
            "session_key": next(iter(sessions)), "as_of": as_of.isoformat(),
            "simulated_end": simulated_end.isoformat(), "simulated_review_at": review_at.isoformat(),
            "recorded_at": now(), "source_review_recorded_at": audit[-1]["actual_at"],
            "evidence_available_at": evidence_at.isoformat() if evidence_at else None,
            "forecast_hash": forecast_hash, "result_hash": result_hash, "review_hash": digest(review),
            "source_hashes": {**{str((round_path / name).resolve()): expected
                                   for entry in audit for name, expected in entry["files"].items()},
                              str(round_path / "audit.jsonl"): audit_sha256},
            "observations": observations, "causal_status": "因果未識別", "notice": NOTICE}
    case["case_hash"] = digest(case)
    if _audit(round_path) != audit or _file_hash(round_path / "audit.jsonl") != audit_sha256:
        raise ValueError("反応事例の作成中に元の当回監査が変わっています。")
    path = memory_root / "cases" / (case["case_id"] + ".json")
    write_json(path, case)
    records_path = memory_root / "audit.jsonl"
    old_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line] if records_path.exists() else []
    record = {"recorded_at": now(), "case_id": case["case_id"], "case_hash": case["case_hash"],
              "case_file_sha256": _file_hash(path),
              "previous_hash": old_records[-1]["entry_hash"] if old_records else None}
    record["entry_hash"] = digest(record)
    with records_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    verify_memory(memory_root)
    return case


def _similarity(material: dict, keywords: list[str], story_keys: list[str], channels: list[str]) -> int:
    """指定条件への文字列一致を順位に使う。

    引数: material (dict) は材料、keywords/story_keys/channels (list[str]) は照合条件。
    返り値: int は一致項目数。値は確率・方向・独立標本数ではない。
    """
    text = " ".join(str(material.get(key, "")) for key in ("material_id", "story_key", "title", "channel")).casefold()
    return (sum(word.casefold() in text for word in keywords)
            + 3 * sum(word == material["story_key"] for word in story_keys)
            + sum(word.casefold() in material["channel"].casefold() for word in channels))


def snapshot_memory(memory_root: Path, as_of: str, out: Path, *, keywords: list[str] | None = None,
                    story_keys: list[str] | None = None, channels: list[str] | None = None,
                    limit: int = 6, mode: str = "回想") -> dict:
    """当回より前の満了・振り返り済み事例だけを短い時点入力へ保存する。

    引数: memory_root (Path) は台帳、as_of (str) は次回情報締切、out (Path) は新規保存先。
    keywords/story_keys/channels (list[str] | None) は近い材料の照合条件、limit (int) は最大場数。
    mode (str) は回想または運用。運用では実際の記録時計も締切以前に限る。
    返り値: dict は自己ハッシュ付きsnapshot。未来の認定・考察は本文へ含めない。
    """
    cutoff = timestamp(as_of)
    if mode not in ("回想", "運用") or type(limit) is not int or not 1 <= limit <= 30:
        raise ValueError("modeは回想・運用、limitは１〜30場を指定してください。")
    keywords, story_keys, channels = keywords or [], story_keys or [], channels or []
    if any(not isinstance(value, str) or not value.strip() for values in (keywords, story_keys, channels) for value in values):
        raise ValueError("照合語は空白以外の文字列が必要です。")
    cases = verify_memory(Path(memory_root))
    selected = []
    for case in cases:
        if (timestamp(case["as_of"]) >= cutoff or timestamp(case["simulated_end"]) > cutoff
                or timestamp(case["simulated_review_at"]) > cutoff):
            continue
        if mode == "運用" and max(timestamp(case[key]) for key in ("recorded_at", "source_review_recorded_at")) > cutoff:
            continue
        materials = [material for row in case["observations"] for material in row["materials"]]
        similarity = max((_similarity(material, keywords, story_keys, channels) for material in materials), default=0)
        if (keywords or story_keys or channels) and not similarity:
            continue
        selected.append((similarity, case))
    selected.sort(key=lambda item: (item[0], timestamp(item[1]["simulated_end"]), item[1]["case_id"]), reverse=True)

    # ----------------------------------------
    visible, seen_sessions = [], set()
    for similarity, case in selected:
        if case["session_key"] in seen_sessions:
            continue
        seen_sessions.add(case["session_key"])
        available = case["evidence_available_at"]
        review_visible = available is not None and timestamp(available) <= cutoff
        row = {key: copy.deepcopy(case[key]) for key in ("case_id", "case_hash", "session_key", "as_of", "simulated_end",
                                                         "simulated_review_at", "forecast_hash", "result_hash", "review_hash")}
        row.update({"similarity": similarity, "review_visible": review_visible,
                    "evidence_available_at": available if review_visible else None, "observations": []})
        for observation in case["observations"]:
            full = copy.deepcopy(observation["full_reference"])
            full.pop("evaluation_window", None)
            item = {key: copy.deepcopy(observation[key]) for key in ("symbol", "reference", "prediction", "materials")}
            item["full_reference"] = full
            item["causal_status"] = "因果未識別"
            if review_visible:
                item["primary"] = copy.deepcopy(observation["primary"])
                item["review"] = copy.deepcopy(observation["review"])
                item["learning_status"] = "観測と考察の候補・確定因果規則には使わない"
            else:
                item["primary"] = None
                item["review"] = None
                item["learning_status"] = "全場観測専用・認定と考察の根拠時計未確認または後発"
            row["observations"].append(item)
        visible.append(row)
        if len(visible) == limit:
            break
    value = {"version": "時点別市場反応１", "as_of": cutoff.isoformat(), "mode": mode,
             "created_at": now(), "query": {"keywords": keywords, "story_keys": story_keys, "channels": channels},
             "session_count": len(visible), "instrument_observation_count": sum(len(row["observations"]) for row in visible),
             "independent_sample_count": None, "cases": visible,
             "notice": NOTICE + " 同一場の２先物や複数材料を別の独立サンプルへ数えない。事例数から予測確率を算出しない。"}
    value["snapshot_hash"] = digest(value)
    out = Path(out)
    if out.exists():
        raise FileExistsError("snapshotは未使用の保存先へ固定してください。")
    out.mkdir(parents=True)
    write_json(out / "snapshot.json", value)
    (out / "要点.md").write_text(render_snapshot_markdown(value), encoding="utf-8")
    return value


def render_snapshot_markdown(snapshot: dict) -> str:
    """次回判断に必要な過去反応と未識別事項を短く表示する。

    引数: snapshot (dict) は時点別の反応事例。
    返り値: str は箇条書きMarkdown。未確定分類を単一の正解にしない。
    """
    lines = ["# 次回へ渡す市場反応", "", f"- 情報締切：{snapshot['as_of']}／{snapshot['mode']}。",
             f"- {snapshot['session_count']}場・{snapshot['instrument_observation_count']}先物観測。独立標本数は未測定。",
             "- 材料点は未校正。因果・確率・次の方向へ自動変換しない。"]
    for case in snapshot["cases"]:
        lines += [f"- {case['session_key']}：認定・考察は{'可視' if case['review_visible'] else '未開示（全場観測のみ）'}。"]
        for row in case["observations"]:
            full = row["full_reference"]
            metrics = full["metrics"]
            actual = metrics.get("actual_pct")
            actual_text = "不明" if actual is None else f"{actual:+.3f}％"
            titles = "、".join(material["title"] for material in row["materials"][:3])
            candidates = "／".join(full["correct_candidates"] or full["uncertain_candidates"]) or "未確定"
            lines.append(f"  - {row['symbol']}：{titles}。終点{actual_text}、参考{full['score']['status']}、分類候補={candidates}。")
            quality = full["quality"]
            lines.append(f"    - 欠測{quality.get('missing_rows', '不明')}本、限月{'確認済み' if quality.get('contract_identity_available') else '未確認'}。材料別因果は未識別。")
            if row["review"]:
                action = " ".join(row["review"].get("next_action", "").split())
                lines.append("    - 次回の確認候補：" + action[:200] + ("…" if len(action) > 200 else ""))
    lines += ["- 共通確認：同先物P0・直近の新事実・継続／反転の対立仮説を毎回点検し、一回の外れから逆方向へ固定しない。", ""]
    return "\n".join(lines)


def main() -> None:
    """追記・時点切出の最小CLIを実行する。

    引数: なし。コマンド引数を標準入力定義から読む。
    返り値: None。日本語の件数だけを標準出力へ出す。
    """
    parser = argparse.ArgumentParser(description="固定済みの市場反応を追記し、次回へ時点切出する")
    commands = parser.add_subparsers(dest="command", required=True)
    append = commands.add_parser("append", help="振り返り完了後の当回を一度だけ追加")
    append.add_argument("--round", type=Path, required=True)
    append.add_argument("--memory", type=Path, required=True)
    append.add_argument("--evidence-available-at")
    append.add_argument("--simulated-review-at")
    snapshot = commands.add_parser("snapshot", help="次回の情報締切までの事例を固定")
    snapshot.add_argument("--memory", type=Path, required=True)
    snapshot.add_argument("--as-of", required=True)
    snapshot.add_argument("--out", type=Path, required=True)
    snapshot.add_argument("--keyword", action="append", default=[])
    snapshot.add_argument("--story-key", action="append", default=[])
    snapshot.add_argument("--channel", action="append", default=[])
    snapshot.add_argument("--limit", type=int, default=6)
    snapshot.add_argument("--mode", choices=("回想", "運用"), default="回想")
    args = parser.parse_args()
    if args.command == "append":
        case = append_reaction_case(args.round, args.memory, evidence_available_at=args.evidence_available_at,
                                    simulated_review_at=args.simulated_review_at)
        print(f"反応事例を１場追加：{case['case_hash']}")
    else:
        result = snapshot_memory(args.memory, args.as_of, args.out, keywords=args.keyword,
                                 story_keys=args.story_key, channels=args.channel, limit=args.limit, mode=args.mode)
        print(f"時点入力を固定：{result['session_count']}場")


if __name__ == "__main__":
    main()
