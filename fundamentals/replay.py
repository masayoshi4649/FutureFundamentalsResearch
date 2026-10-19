"""切出入力、予測固定、結果開示、短い振り返りを順番に保存する回想研究。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from .common import digest, now, read_json, timestamp, write_json
from .engine import HORIZONS, evaluate, forecast, scheduled_end
from .news import select_articles
from .prices import SYMBOLS, load_candles, session_groups

DIRECTIONS = ("上昇", "下落", "横ばい", "保留")
LONG_HORIZONS = ("week", "month", "quarter")


def _file_hash(path: Path) -> str:
    """保存ファイルのバイト列を識別する。

    引数: path (Path) は対象ファイル。
    返り値: str は SHA-256。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(round_path: Path) -> list[dict]:
    """監査記録の連鎖と各段階で固定したファイルを照合する。

    引数: round_path (Path) は一回の保存先。
    返り値: list[dict] は検証済みの段階記録。
    """
    path = round_path / "audit.jsonl"
    if not path.exists():
        raise ValueError("準備段階の監査記録がありません。")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not 1 <= len(records) <= 4:
        raise ValueError("監査記録の段階数が不正です。")
    previous = None
    for index, record in enumerate(records):
        if record["previous_hash"] != previous or digest({k: v for k, v in record.items() if k != "entry_hash"}) != record["entry_hash"]:
            raise ValueError("監査記録のハッシュが一致しません。")
        if record["phase"] != ("prepare", "seal", "reveal", "review")[index]:
            raise ValueError("準備・固定・開示・振り返りの順序が不正です。")
        for relative, expected in record["files"].items():
            candidate = round_path / relative
            if not candidate.is_file() or _file_hash(candidate) != expected:
                raise ValueError(f"固定ファイルが変更されています: {relative}")
        previous = record["entry_hash"]
    return records


def _append_audit(round_path: Path, phase: str, simulated_at: datetime, files: list[Path]) -> None:
    """実際の記録時刻と再生時刻を区別して段階を追記する。

    引数: round_path (Path) は保存先、phase (str) は段階、simulated_at (datetime) は再生時刻、
    files (list[Path]) はこの段階で固定するファイル。
    返り値: None。
    """
    records = _audit(round_path) if (round_path / "audit.jsonl").exists() else []
    if len(records) >= 4 or phase != ("prepare", "seal", "reveal", "review")[len(records)]:
        raise ValueError("段階の順序が不正です。")
    value = {"phase": phase, "actual_at": now(), "simulated_at": timestamp(simulated_at).isoformat(),
             "previous_hash": records[-1]["entry_hash"] if records else None,
             "files": {path.relative_to(round_path).as_posix(): _file_hash(path) for path in files}}
    value["entry_hash"] = digest(value)
    with (round_path / "audit.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _text(value: object, name: str, maximum: int = 500) -> str:
    """短文を検証し、改行を一行へそろえる。

    引数: value (object) は入力、name (str) は項目名、maximum (int) は文字数上限。
    返り値: str は空でない短文。
    """
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} は１〜{maximum}文字の短文が必要です。")
    return " ".join(value.split())


def _strings(value: object, maximum: int, name: str) -> list[str]:
    """箇条書きの最大件数と各短文を検証する。

    引数: value (object) は入力、maximum (int) は最大件数、name (str) は項目名。
    返り値: list[str] は検証済み短文。
    """
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} は最大{maximum}件の配列が必要です。")
    return [_text(item, name) for item in value]


def _feedback(root: Path, rows: list[dict], as_of: datetime) -> list[dict]:
    """満了済みの振り返りだけを、保存済み証跡と照合して引き継ぐ。

    引数: root (Path) は反復ルート、rows (list[dict]) は引継ぎ候補、as_of (datetime) は締切。
    返り値: list[dict] は未来評価や差替えのない引継ぎ。
    """
    if not isinstance(rows, list):
        raise ValueError("引継ぎは配列が必要です。")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("引継ぎは辞書が必要です。")
        for field in ("simulated_available_at", "available_at", "label_end"):
            if field not in row or timestamp(row[field]) > as_of:
                raise ValueError("未来または利用時刻不明の振り返りは引き継げません。")
        source = root / _round_id(row.get("source_round"))
        records = _audit(source)
        if records[-1]["phase"] != "review" or row not in read_json(source / "feedback.json"):
            raise ValueError("引継ぎが固定済み振り返りと一致しません。")
    return copy.deepcopy(rows)


def _round_id(value: object) -> str:
    """保存先を反復ルート直下へ制限する。

    引数: value (object) は回の識別子。
    返り値: str はパス区切りや親参照を含まない識別子。
    """
    if not isinstance(value, str) or not re.fullmatch(r"[\w\-]+", value) or value in (".", ".."):
        raise ValueError("回の識別子は文字・数字・下線・ハイフンで指定してください。")
    return value


# ----------------------------------------
# 可視入力の切出しと、予測担当に渡さない機械計算


def prepare_round(root: Path, round_id: str, as_of: datetime, prices: Path,
                  articles: list[dict], feedback: list[dict]) -> dict:
    """過去の可知情報だけを予測用の別フォルダへ切り出す。

    引数: root (Path) は新規反復のルート、round_id (str) は回名、as_of (datetime) は締切、
    prices (Path) は読取専用価格、articles (list[dict]) は記事全版、feedback (list[dict]) は前回知見。
    返り値: dict は予測担当へ渡せる入力。未来記事の件数や識別子も含めない。
    """
    root, as_of = Path(root), timestamp(as_of)
    round_path = root / _round_id(round_id)
    if round_path.exists():
        raise FileExistsError("各回の保存先は上書きできません。")
    for audit_path in root.glob("*/audit.jsonl"):
        records = _audit(audit_path.parent)
        if records[-1]["phase"] != "review":
            raise ValueError("前回の振り返りを完了してから次回を準備してください。")
        if timestamp(records[0]["simulated_at"]) >= as_of or timestamp(records[-1]["simulated_at"]) > as_of:
            raise ValueError("次回締切より後の予測・振り返りが存在します。")
    inherited = _feedback(root, feedback, as_of)
    selected, _ = select_articles(articles, as_of)
    selected = [item for item in selected if timestamp(item["available_at"]) >= as_of - timedelta(hours=48)][-12:]
    visible_news = []
    for item in selected:
        news = {key: item[key] for key in ("id", "revision_id", "headline", "canonical_url", "available_at", "published_at", "fetched_at", "updated_at", "source_id", "source_name") if key in item}
        news["body_excerpt"] = str(item.get("body_text") or item.get("body_excerpt") or "")[:1200]
        visible_news.append(news)

    # ----------------------------------------

    machine = forecast(Path(prices), selected, [], as_of, round_path / "機械予測", history=None, mode="回想")
    instruments = {}
    for symbol in SYMBOLS:
        visible = load_candles(Path(prices), symbol, cutoff=as_of)
        groups = [group for group in session_groups(visible) if scheduled_end(group[0].session) <= as_of]
        recent = []
        for index in range(max(1, len(groups) - 6), len(groups)):
            group, previous = groups[index], groups[index - 1][-1].close
            recent.append({"session": group[0].session,
                           "return_pct": (group[-1].close / previous - 1) * 100,
                           "low_pct": (min(bar.low for bar in group) / previous - 1) * 100,
                           "high_pct": (max(bar.high for bar in group) / previous - 1) * 100})
        instrument = machine["instruments"][symbol]
        instruments[symbol] = {"reference": copy.deepcopy(instrument["reference"]), "recent_sessions": recent,
                               "horizons": {key: {"distribution": copy.deepcopy(value["distribution"])} for key, value in instrument["horizons"].items()}}
    packet = {key: machine[key] for key in ("as_of", "session", "kind", "targets")}
    packet.update(instruments=instruments, news=visible_news, feedback=inherited)
    baseline = {"packet_hash": digest(packet), "machine_forecast_hash": machine["forecast_hash"],
                "instruments": {symbol: {"reference": item["reference"], "horizons": item["horizons"]} for symbol, item in instruments.items()}}
    baseline["baseline_hash"] = digest(baseline)
    write_json(round_path / "masked" / "packet.json", packet)
    write_json(round_path / "baseline.json", baseline)
    lines = ["# 予測担当への入力", "", f"- 情報締切：{packet['as_of']}", f"- 対象：{packet['session']}",
             "- packet.jsonだけを読み、回の直下へ予測入力.jsonを作成する。", "- 未来記事や結果価格を検索・参照しない。",
             "- 記事は利用可能時刻が締切以前の直近48時間から新しい最大12件。本文抜粋は最大1200字。", "- 同じ先物の前セッション最終観測値を基準とする。公式終値の代用。",
             "- 数値範囲は機械計算した経験分布を固定使用。予測区間として未校正。", "- 回想研究。入力隔離はOSのアクセス制限やモデル内部の未来知識排除を証明しない。",
             "- 収集したニュース提供元・範囲だけを材料とし、網羅性を仮定しない。", "",
             "方向は上昇・下落・横ばい・保留。根拠最大3件、リスク最大2件、適用知見最大3件。", "長期はweek/month/quarterごとにdirectionとpointerを記す。", ""]
    (round_path / "masked" / "入力.md").write_text("\n".join(lines), encoding="utf-8")
    _append_audit(round_path, "prepare", as_of, [round_path / "masked" / "packet.json", round_path / "masked" / "入力.md", round_path / "baseline.json", round_path / "機械予測" / "forecast.json"])
    return packet


def _prediction_input(value: dict) -> dict:
    """予測担当の方向・根拠だけを受け付け、数値範囲の差替えを拒否する。

    引数: value (dict) は予測入力。
    返り値: dict は検証済みの短文予測。
    """
    if not isinstance(value, dict) or set(value) - {"instruments", "lessons_applied"} or set(value.get("instruments", {})) != set(SYMBOLS):
        raise ValueError("予測入力は両銘柄のinstrumentsとlessons_appliedだけを指定してください。")
    result = {"instruments": {}, "lessons_applied": _strings(value.get("lessons_applied", []), 3, "適用知見")}
    for symbol in SYMBOLS:
        item = value["instruments"][symbol]
        if not isinstance(item, dict) or set(item) != {"direction", "reasons", "risks", "long_term"} or item["direction"] not in DIRECTIONS:
            raise ValueError("銘柄予測の方向または項目が不正です。")
        if not isinstance(item["long_term"], dict) or set(item["long_term"]) != set(LONG_HORIZONS):
            raise ValueError("長期見通しはweek/month/quarterが必要です。")
        long_term = {}
        for horizon in LONG_HORIZONS:
            long = item["long_term"][horizon]
            if not isinstance(long, dict) or set(long) != {"direction", "pointer"} or long["direction"] not in DIRECTIONS:
                raise ValueError("長期見通しの方向または項目が不正です。")
            long_term[horizon] = {"direction": long["direction"], "pointer": _text(long["pointer"], "長期確認点")}
        result["instruments"][symbol] = {"direction": item["direction"], "reasons": _strings(item["reasons"], 3, "根拠"),
                                         "risks": _strings(item["risks"], 2, "リスク"), "long_term": long_term}
    return result


def _range_text(distribution: dict) -> str:
    """終点と経路の経験範囲を短い日本語にする。

    引数: distribution (dict) は固定済み経験分布。
    返り値: str は未校正表示付き範囲。
    """
    values = []
    for field, label in (("terminal_range_pct", "終点"), ("path_range_pct", "経路高安")):
        bounds = distribution[field]
        values.append(f"{label}{bounds[0]:+.2f}〜{bounds[1]:+.2f}%" if bounds else f"{label}算出不能（標本不足）")
    return "／".join(values) + f"、n={distribution['n']}、未校正"


def seal_prediction(round_path: Path) -> dict:
    """予測JSONと短文MDを、結果開示より前に上書き禁止で固定する。

    引数: round_path (Path) は準備済みの回。
    返り値: dict は方向・固定経験範囲・入力証跡を持つハッシュ付き予測。
    """
    round_path = Path(round_path)
    records = _audit(round_path)
    if records[-1]["phase"] != "prepare":
        raise ValueError("予測固定は準備直後に一度だけ実行できます。")
    candidates = [path for path in (round_path / "予測入力.json", round_path / "masked" / "予測入力.json") if path.exists()]
    if len(candidates) != 1:
        raise ValueError("予測入力.jsonを回の直下かmaskedのいずれか一方へ保存してください。")
    input_path = candidates[0]
    entry = _prediction_input(read_json(input_path))
    packet, baseline = read_json(round_path / "masked" / "packet.json"), read_json(round_path / "baseline.json")
    value = {key: packet[key] for key in ("as_of", "session", "kind", "targets")}
    value.update(mode="回想・入力隔離順次反復", issued_at=now(), simulated_issued_at=packet["as_of"],
                 packet_hash=digest(packet), baseline_hash=baseline["baseline_hash"],
                 machine_forecast_hash=baseline["machine_forecast_hash"], input_hash=digest(read_json(input_path)), **entry)
    actor_path = round_path / "actor.json"
    if actor_path.exists():
        actor = read_json(actor_path)
        if not isinstance(actor, dict) or actor.get("packet_hash") != digest(packet) or actor.get("fork_turns") != "none" or actor.get("allowed_input") != "masked/packet.json":
            raise ValueError("予測担当の文脈隔離記録が入力と一致しません。")
        value["actor"] = actor
    value["feedback_applications"] = [{"source_forecast_hash": row["forecast_hash"], "source_review_hash": row["review_hash"],
                                       "next_action": row["next_action"],
                                       "status": "採用" if row["next_action"] in entry["lessons_applied"] else "未採用"} for row in packet["feedback"]]
    for symbol, item in value["instruments"].items():
        item["reference"] = copy.deepcopy(baseline["instruments"][symbol]["reference"])
        item["horizons"] = {}
        for horizon, distribution in baseline["instruments"][symbol]["horizons"].items():
            direction = item["direction"] if horizon == "today" else item["long_term"][horizon]["direction"]
            item["horizons"][horizon] = {"direction": direction, "distribution": copy.deepcopy(distribution["distribution"])}
    value["forecast_hash"] = digest(value)
    lines = [f"# 予測：{value['session']}", "", f"- 情報締切：{value['as_of']}", f"- 実際の固定時刻：{value['issued_at']}",
             "- 回想研究。方向は定性的仮説、範囲は価格だけの経験分布で未校正。", "- 基準・終点は先物の最終観測値で、公式終値の代用。", ""]
    for symbol, item in value["instruments"].items():
        reference = item["reference"]
        lines += [f"- {symbol}：{item['direction']}。基準{reference['price']}（{reference['session']}、{reference['time']}）。"]
        lines += [f"  - 根拠：{reason}" for reason in item["reasons"]]
        lines += [f"  - リスク：{risk}" for risk in item["risks"]]
        for horizon, horizon_item in item["horizons"].items():
            pointer = "" if horizon == "today" else "。確認点：" + item["long_term"][horizon]["pointer"]
            lines.append(f"  - {HORIZONS[horizon]}：{horizon_item['direction']}。{_range_text(horizon_item['distribution'])}{pointer}")
    lines += [f"- 次回へ適用：{lesson}" for lesson in value["lessons_applied"]]
    lines += [f"- 予測ハッシュ：{value['forecast_hash']}", ""]
    if len(lines) > 60:
        raise ValueError("予測文書は60行以内にしてください。")
    write_json(round_path / "予測.json", value)
    with (round_path / "予測.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    frozen = [input_path, round_path / "予測.json", round_path / "予測.md"]
    if actor_path.exists():
        frozen.append(actor_path)
    _append_audit(round_path, "seal", timestamp(packet["as_of"]), frozen)
    return value


# ----------------------------------------
# 固定後の実績開示と、次回だけに渡す振り返り


def _reflections(reflection: dict) -> dict:
    """短い原因仮説を検証し、証拠不足を原因不明として残す。

    引数: reflection (dict) は銘柄別または共通の原因メモ。
    返り値: dict は銘柄別の分類・理由・次回対応・後発材料識別子。
    """
    if not isinstance(reflection, dict):
        raise ValueError("振り返りは辞書で指定してください。")
    result = {}
    for symbol in SYMBOLS:
        item = reflection.get(symbol, reflection)
        if not isinstance(item, dict):
            raise ValueError("銘柄別の振り返りは辞書で指定してください。")
        ids = item.get("later_material_ids", [])
        if not isinstance(ids, list) or any(isinstance(identity, bool) or not isinstance(identity, (str, int)) for identity in ids):
            raise ValueError("後発材料識別子は文字列または整数の配列です。")
        result[symbol] = {"classification": _text(item.get("classification", "原因不明"), "分類"),
                          "reason": _text(item.get("reason", "単独の因果関係を特定できる証拠は不足。"), "理由"),
                          "next_action": _text(item.get("next_action", "当初材料の反対仮説と追加反応を次回も確認する。"), "次回対応"),
                          "later_material_ids": ids}
    return result


def review_round(round_path: Path, prices: Path, post_articles: list[dict], observed_at: datetime,
                 reflection: dict | None = None) -> dict:
    """固定予測を実績へ照合し、結果開示後に短い振り返りを追記する。

    引数: round_path (Path) は固定済み予測、prices (Path) は読取専用価格、
    post_articles (list[dict]) は後発記事、observed_at (datetime) は再生上の結果取得時刻、
    reflection (dict | None) は原因メモ。Noneなら開示まで、辞書なら振り返りまで完了する。
    返り値: dict は満了済み銘柄別実績、保留期間、次回引継ぎと完了状態。

    review_hash は review_hash 自身と feedback を除いた振り返り本体を対象とする。
    feedback はこのハッシュを参照し、配列全体は監査記録のファイルハッシュで固定する。
    """
    round_path, observed_at = Path(round_path), timestamp(observed_at)
    records = _audit(round_path)
    if records[-1]["phase"] not in ("seal", "reveal"):
        raise ValueError("予測固定後かつ振り返り完了前だけ結果を開示できます。")
    value = read_json(round_path / "予測.json")
    if digest({key: item for key, item in value.items() if key != "forecast_hash"}) != value["forecast_hash"]:
        raise ValueError("予測ハッシュが一致しません。")
    if observed_at < timestamp(value["targets"]["today"]):
        raise ValueError("対象セッションの終了予定前には結果を開示できません。")
    if reflection is not None:
        reflections = _reflections(reflection)
    if records[-1]["phase"] == "seal":
        evaluation = evaluate(round_path / "機械予測", Path(prices), observed_at, post_articles)
        labels = []
        for source in evaluation["labels"]:
            label = copy.deepcopy(source)
            direction = value["instruments"][label["symbol"]]["horizons"][label["horizon"]]["direction"]
            actual_direction = label["metrics"]["actual_direction"]
            label["direction"] = direction
            label["metrics"] = {key: label["metrics"][key] for key in ("actual_direction", "terminal_covered", "path_covered", "interval_score")}
            label["metrics"]["direction_hit"] = None if direction == "保留" else direction == actual_direction
            labels.append(label)
        result = {"forecast_hash": value["forecast_hash"], "machine_forecast_hash": value["machine_forecast_hash"],
                  "machine_evaluation_hash": digest(evaluation), "observed_at": observed_at.isoformat(), "recorded_at": now(),
                  "labels": labels, "pending": evaluation["pending"],
                  "limitations": ["後発材料の有無によって全件成績から除外しない。", "未満了期間は未評価。長期の追跡完了や予測優位性を主張しない。"]}
        result["result_hash"] = digest(result)
        write_json(round_path / "result.json", result)
        evaluation_path = round_path / "機械予測" / ("evaluation-" + observed_at.strftime("%Y%m%dT%H%M%S%z") + ".json")
        _append_audit(round_path, "reveal", observed_at, [round_path / "result.json", evaluation_path])
    else:
        result = read_json(round_path / "result.json")
        if observed_at != timestamp(result["observed_at"]):
            raise ValueError("開示済み結果の観測時刻は変更できません。")
    if reflection is None:
        return {**result, "feedback": [], "pending_review": True}

    # ----------------------------------------

    for symbol, item in reflections.items():
        candidates = {str(candidate["id"]) for label in result["labels"] if label["symbol"] == symbol for candidate in label["later_material_candidates"]}
        if any(str(identity) not in candidates for identity in item["later_material_ids"]):
            raise ValueError("後発材料識別子が開示済み候補に含まれていません。")
    review = {"forecast_hash": value["forecast_hash"], "result_hash": result["result_hash"],
              "simulated_available_at": observed_at.isoformat(), "recorded_at": now(), "reflection": reflections}
    review["review_hash"] = digest(review)
    feedback = []
    for label in result["labels"]:
        item = reflections[label["symbol"]]
        feedback.append({"source_round": round_path.name, "forecast_hash": value["forecast_hash"], "result_hash": result["result_hash"],
                         "review_hash": review["review_hash"], "symbol": label["symbol"], "horizon": label["horizon"],
                         "direction": label["direction"], "actual_pct": label["actual_pct"],
                         "hit": label["metrics"]["direction_hit"], "coverage": label["metrics"]["terminal_covered"],
                         "next_action": item["next_action"], "action_status": "知見候補・次回採否判断",
                         "label_end": label["label_end"], "simulated_available_at": observed_at.isoformat(), "available_at": observed_at.isoformat()})
    lines = [f"# 振り返り：{value['session']}", "", f"- 結果取得の再生時刻：{observed_at.isoformat()}",
             "- 全件評価。後発材料で失敗を免除せず、単発の一致を普遍則にしない。", ""]
    for symbol, item in reflections.items():
        for label in result["labels"]:
            if label["symbol"] == symbol:
                hit = "保留" if label["metrics"]["direction_hit"] is None else "一致" if label["metrics"]["direction_hit"] else "不一致"
                cover = "未発行" if label["metrics"]["terminal_covered"] is None else "範囲内" if label["metrics"]["terminal_covered"] else "範囲外"
                lines.append(f"- {symbol}・{HORIZONS[label['horizon']]}：予測{label['direction']}→実績{label['actual_pct']:+.3f}%（{hit}、終点{cover}）。")
        lines += [f"  - 分類：{item['classification']}。理由：{item['reason']}", f"  - 次回：{item['next_action']}",
                  f"  - 後発材料候補：{'、'.join(map(str, item['later_material_ids'])) or '指定なし'}。因果は未認定。"]
    lines += [f"- 未評価：{len(result['pending'])}件（未満了・休場・終端観測不足）。", "- 長期の満了追跡・方向優位性の実証は未完了。", ""]
    review["feedback"] = feedback
    write_json(round_path / "review.json", review)
    write_json(round_path / "feedback.json", feedback)
    with (round_path / "振り返り.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    _append_audit(round_path, "review", observed_at, [round_path / "review.json", round_path / "feedback.json", round_path / "振り返り.md"])
    return {**result, "reflection": reflections, "feedback": feedback, "review_hash": review["review_hash"], "pending_review": False}


def summarize_replay(root: Path) -> dict:
    """全回を保留や横ばいも含め集計し、短い総括を新規保存する。

    引数: root (Path) は反復ルート。
    返り値: dict は銘柄・期間別の全件数、方向発行分母、固定範囲成績と各回リンク。
    """
    root = Path(root)
    groups, rounds = {}, []
    for audit_path in sorted(root.glob("*/audit.jsonl")):
        round_path = audit_path.parent
        records = _audit(round_path)
        issued = len(records) >= 2
        prediction = read_json(round_path / "予測.json") if issued else read_json(round_path / "masked" / "packet.json")
        result = read_json(round_path / "result.json") if (round_path / "result.json").exists() else {"labels": []}
        labels = {(item["symbol"], item["horizon"]): item for item in result["labels"]}
        rounds.append({"round": round_path.name, "as_of": prediction["as_of"], "phase": records[-1]["phase"], "issued": issued,
                       "prediction": f"{round_path.name}/予測.md" if issued else None,
                       "review": f"{round_path.name}/振り返り.md" if records[-1]["phase"] == "review" else None})
        for symbol, instrument in prediction["instruments"].items():
            for horizon, item in instrument["horizons"].items():
                group = groups.setdefault(f"{symbol}／{horizon}", {"prepared": 0, "unissued": 0, "forecasts": 0, "evaluated": 0, "abstained": 0, "direction_n": 0,
                                                                  "direction_hits": 0, "range_n": 0, "range_hits": 0, "range_width_total": 0.0})
                group["prepared"] += 1
                if not issued:
                    group["unissued"] += 1
                    continue
                group["forecasts"] += 1
                group["abstained"] += int(item["direction"] == "保留")
                label = labels.get((symbol, horizon))
                if label:
                    group["evaluated"] += 1
                    hit, covered = label["metrics"]["direction_hit"], label["metrics"]["terminal_covered"]
                    if hit is not None:
                        group["direction_n"] += 1
                        group["direction_hits"] += int(hit)
                    if covered is not None:
                        group["range_n"] += 1
                        group["range_hits"] += int(covered)
                        bounds = item["distribution"]["terminal_range_pct"]
                        group["range_width_total"] += bounds[1] - bounds[0]
    for group in groups.values():
        group["pending"] = group["forecasts"] - group["evaluated"]
        group["direction_hit_rate"] = group["direction_hits"] / group["direction_n"] if group["direction_n"] else None
        group["terminal_coverage"] = group["range_hits"] / group["range_n"] if group["range_n"] else None
        group["mean_range_width_pct"] = group["range_width_total"] / group["range_n"] if group["range_n"] else None
    value = {"generated_at": now(), "mode": "回想・入力隔離順次反復", "prepared_rounds": len(rounds),
             "forecast_rounds": sum(item["issued"] for item in rounds), "rounds": rounds, "groups": groups,
             "limitations": ["前向き運用の成績ではない。入力隔離はモデル内部の未来知識を排除しない。",
                             "全件を含め、横ばい発行も方向の分母へ残す。保留は別記。", "範囲は価格だけの未校正経験分布。後発材料による除外なし。",
                             "長期の未満了分・欠測分は未評価。単発の結果や反復動作を予測優位性の実証としない。"]}
    lines = ["# 反復研究の総括", "", f"- 準備：{len(rounds)}回、固定予測：{value['forecast_rounds']}回。区分：{value['mode']}。"]
    for key, group in groups.items():
        direction = f"{group['direction_hits']}/{group['direction_n']}" if group["direction_n"] else "の採点対象なし"
        coverage = f"{group['range_hits']}/{group['range_n']}" if group["range_n"] else "の採点対象なし"
        lines.append(f"- {key}：発行{group['forecasts']}、未発行{group['unissued']}、評価{group['evaluated']}、保留発行{group['abstained']}、方向一致{direction}、終点範囲内{coverage}、未評価{group['pending']}。")
    lines += [f"- {item}" for item in value["limitations"]]
    for item in rounds:
        prediction_link = f"[予測]({item['prediction']})" if item["prediction"] else "予測未発行"
        lines.append(f"- {item['round']}：{prediction_link}" + (f"／[振り返り]({item['review']})" if item["review"] else "／振り返り未完了"))
    if len(lines) >= 100:
        raise ValueError("総括は100行未満のため、反復を複数の研究単位へ分けてください。")
    write_json(root / "summary.json", value)
    with (root / "総括.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return value


# ----------------------------------------
# 全回終了後の長期追跡。反復中の知見や当日成績へ遡及投入しない


def track_long_term(root: Path, prices: Path, observed_at: datetime,
                    post_articles: list[dict]) -> dict:
    """全回の振り返り完了後、元の長期仮説を変えず満了分を別評価する。

    引数: root (Path) は反復ルート、prices (Path) は読取専用価格、observed_at (datetime) は追跡締切、
    post_articles (list[dict]) は後発記事。元の予測・レビュー・監査・引継ぎは変更しない。
    返り値: dict は元予測ハッシュと仮説、長期の実績・未評価理由、評価ファイルのハッシュ。
    """
    root, prices, observed_at = Path(root), Path(prices), timestamp(observed_at)
    outputs = [root / "長期追跡.json", root / "長期追跡.md"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("長期追跡は上書きできません。別の研究単位へ保存してください。")
    audit_paths = sorted(root.glob("*/audit.jsonl"))
    if not audit_paths:
        raise ValueError("追跡する反復記録がありません。")
    predictions = []
    for audit_path in audit_paths:
        records = _audit(audit_path.parent)
        if records[-1]["phase"] != "review":
            raise ValueError("全回の振り返りを完了してから長期追跡してください。")
        prediction = read_json(audit_path.parent / "予測.json")
        if timestamp(prediction["as_of"]) >= observed_at:
            raise ValueError("追跡締切は全予測の情報締切より後にしてください。")
        predictions.append((audit_path.parent, prediction))
    predictions.sort(key=lambda item: timestamp(item[1]["as_of"]))

    # ----------------------------------------

    rounds, groups, lines = [], {}, ["# 長期追跡", "", f"- 観測締切：{observed_at.isoformat()}。全回の振り返り完了後の補助評価。",
                                     "- 元の予測・振り返り・当日成績を変更せず、追跡結果を反復中の引継ぎへ戻さない。",
                                     "- 基準・終点は同じ先物の最終観測値で、公式終値の代用。経験範囲は未校正。",
                                     "- 当初確認点・予測ハッシュは元予測とJSONを参照。後発記事だけで因果を認定しない。", ""]
    for round_path, prediction in predictions:
        evaluation_path = round_path / "機械予測" / ("evaluation-" + observed_at.strftime("%Y%m%dT%H%M%S%z") + ".json")
        if evaluation_path.exists():
            evaluation = read_json(evaluation_path)
        else:
            evaluation = evaluate(round_path / "機械予測", prices, observed_at, post_articles)
        if evaluation.get("forecast_hash") != prediction["machine_forecast_hash"] or timestamp(evaluation.get("observed_at", "")) != observed_at:
            raise ValueError("長期追跡の機械評価が元予測または観測締切と一致しません。")
        label_map = {}
        for label in evaluation["labels"]:
            if label["horizon"] not in LONG_HORIZONS:
                continue
            key = (label["symbol"], label["horizon"])
            if key in label_map or timestamp(label["label_end"]) > observed_at:
                raise ValueError("長期評価に重複または未満了の実績が含まれています。")
            label_map[key] = label
        record = {"round": round_path.name, "forecast_hash": prediction["forecast_hash"], "as_of": prediction["as_of"],
                  "observed_at": observed_at.isoformat(), "machine_forecast_hash": prediction["machine_forecast_hash"],
                  "input_hypothesis": {symbol: copy.deepcopy(item["long_term"]) for symbol, item in prediction["instruments"].items()},
                  "labels": [], "pending": [], "evaluation_file": evaluation_path.relative_to(root).as_posix(),
                  "evaluation_hash": digest(evaluation),
                  "artifact_hashes": {path.relative_to(root).as_posix(): _file_hash(path) for path in
                                      (round_path / "予測.json", round_path / "review.json", round_path / "audit.jsonl", evaluation_path)}}
        for symbol, instrument in prediction["instruments"].items():
            for horizon in LONG_HORIZONS:
                direction = instrument["long_term"][horizon]["direction"]
                hypothesis = instrument["long_term"][horizon]["pointer"]
                group = groups.setdefault(f"{symbol}／{horizon}", {"forecasts": 0, "evaluated": 0, "direction_n": 0,
                                                                  "direction_hits": 0, "abstained": 0, "range_n": 0, "range_hits": 0})
                group["forecasts"] += 1
                group["abstained"] += int(direction == "保留")
                source = label_map.get((symbol, horizon))
                if source is None:
                    nominal = prediction["targets"][horizon]
                    reason = "名目満期が観測締切より後のため未満了" if timestamp(nominal) > observed_at else "休場・終端観測不足などにより評価終点が未確定"
                    record["pending"].append({"symbol": symbol, "horizon": horizon, "nominal_target": nominal,
                                              "direction": direction, "input_hypothesis": hypothesis, "reason": reason})
                    outcome = f"予測{direction}、未評価（{reason}）"
                else:
                    label = copy.deepcopy(source)
                    actual_direction = label["metrics"]["actual_direction"]
                    label.update(direction=direction, input_hypothesis=hypothesis)
                    label["metrics"] = {key: label["metrics"][key] for key in ("actual_direction", "terminal_covered", "path_covered", "interval_score")}
                    label["metrics"]["direction_hit"] = None if direction == "保留" else direction == actual_direction
                    record["labels"].append(label)
                    group["evaluated"] += 1
                    hit, covered = label["metrics"]["direction_hit"], label["metrics"]["terminal_covered"]
                    if hit is not None:
                        group["direction_n"] += 1
                        group["direction_hits"] += int(hit)
                    if covered is not None:
                        group["range_n"] += 1
                        group["range_hits"] += int(covered)
                    hit_text = "方向保留" if hit is None else "一致" if hit else "不一致"
                    cover_text = "範囲未発行" if covered is None else "終点範囲内" if covered else "終点範囲外"
                    outcome = f"予測{direction}→{label['actual_pct']:+.3f}%（{hit_text}、{cover_text}、終点{label['label_end']}）"
                if horizon == "week":
                    lines.append(f"- [{round_path.name}の予測]({round_path.name}/予測.md)・{symbol}週：{outcome}。")
        rounds.append(record)
    for group in groups.values():
        group["pending"] = group["forecasts"] - group["evaluated"]
        group["direction_hit_rate"] = group["direction_hits"] / group["direction_n"] if group["direction_n"] else None
        group["terminal_coverage"] = group["range_hits"] / group["range_n"] if group["range_n"] else None
    pending_counts = {horizon: sum(group["pending"] for key, group in groups.items() if key.endswith("／" + horizon)) for horizon in LONG_HORIZONS}
    value = {"generated_at": now(), "observed_at": observed_at.isoformat(), "mode": "回想・全回終了後の長期補助評価",
             "rounds": rounds, "groups": groups, "pending_counts": pending_counts,
             "excluded_from_feedback": True, "limitations": ["当日成績と重複集計せず、元予測と振り返りは不変。",
                 "名目満期後の最初の同種セッションを使用。未満了・終端不足は未評価。",
                 "重複期間・少数標本の補助評価であり、予測優位性や単独の因果を実証しない。"]}
    value["tracking_hash"] = digest(value)
    lines += ["", f"- 月の未評価：{pending_counts['month']}件。３か月の未評価：{pending_counts['quarter']}件。",
              "- 詳細仮説・各期間の未評価理由・評価原本ハッシュは[長期追跡JSON](長期追跡.json)に保存。", ""]
    write_json(outputs[0], value)
    with outputs[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return value
