"""結果を未開示のまま３候補を固定し、満了・価格原本を待って別評価する回想研究。"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

from .common import digest, now, read_json, timestamp, write_json
from .engine import evaluate as evaluate_prices
from .engine import forecast, scheduled_end
from .news import load_export, select_articles
from .pdca import SYMBOLS, predict_learned
from .pdca_research import _load_initial_history, validate_annotation
from .prices import load_candles, session_groups
from .replay import _append_audit, _audit, _file_hash, _range_text


AS_OF = "2026-10-16T07:45:00+09:00"
METHODS = ("ridge", "analog", "ridge_scaled")
DEFAULT_NEWS = Path("local/PDCA改善/次回ニュース.json")
DEFAULT_HISTORY = Path("検証/00003_学習期間の延長/学習状態.json")
DEFAULT_PRICES = Path("../FutureAlgorithmResearch")


def _verify_prepared(root: Path) -> dict:
    """準備後の計画・入力・原ニュース・初期教師の差替えを検出する。

    引数: root (Path) は一件の結果未開示追跡フォルダ。
    返り値: dict は照合済み準備記録。
    """
    prepared = read_json(root / "準備.json")
    if digest({key: value for key, value in prepared.items() if key != "prepare_hash"}) != prepared["prepare_hash"]:
        raise ValueError("準備記録のハッシュが一致しません。")
    for relative, expected in prepared["files"].items():
        if _file_hash(root / relative) != expected:
            raise ValueError("準備後に固定入力または計画が変更されています。")
    for source in prepared["sources"].values():
        if _file_hash(Path(source["path"])) != source["hash"]:
            raise ValueError("準備後に原ニュースまたは学習履歴が変更されています。")
    _audit(root)
    return prepared


def prepare(root: Path, news: Path = DEFAULT_NEWS, history: Path = DEFAULT_HISTORY,
            prices: Path = DEFAULT_PRICES) -> dict:
    """締切以前の価格と可知ニュース全文だけで注釈用入力を新規保存する。

    引数: root (Path) は実験計画のある保存先、news/history (Path) は原ニュースと過去教師、
    prices (Path) は読取専用価格。情報締切は2026年10月16日07:45に固定する。
    返り値: dict は自己ハッシュ付き第２版入力。対象セッションの結果は読まない。
    """
    root, news, history, prices = Path(root), Path(news), Path(history), Path(prices)
    if any((root / name).exists() for name in ("準備.json", "masked", "機械基準", "audit.jsonl")):
        raise FileExistsError("準備済み入力は上書きできません。")
    if not (root / "実験計画.md").is_file() or not history.is_file():
        raise ValueError("実験計画と学習履歴の原本が必要です。")
    as_of = timestamp(AS_OF)
    articles, _ = select_articles(load_export(news), as_of)
    machine = forecast(prices, articles, [], as_of, root / "機械基準", history=None, mode="回想")
    instruments = {}
    for symbol in SYMBOLS:
        candles = load_candles(prices, symbol, cutoff=as_of)
        groups = [group for group in session_groups(candles) if scheduled_end(group[0].session) <= as_of]
        recent = []
        for index in range(max(1, len(groups) - 6), len(groups)):
            group, previous = groups[index], groups[index - 1][-1].close
            recent.append({"session": group[0].session, "return_pct": (group[-1].close / previous - 1) * 100,
                           "low_pct": (min(bar.low for bar in group) / previous - 1) * 100,
                           "high_pct": (max(bar.high for bar in group) / previous - 1) * 100})
        instrument = machine["instruments"][symbol]
        instruments[symbol] = {"reference": copy.deepcopy(instrument["reference"]), "recent_sessions": recent,
                               "horizons": {key: {"distribution": copy.deepcopy(value["distribution"])}
                                            for key, value in instrument["horizons"].items()}}
    packet = {key: machine[key] for key in ("as_of", "session", "kind", "targets")}
    packet.update(schema_version=2, instruments=instruments, news=articles)
    packet["packet_hash"] = digest(packet)
    write_json(root / "masked" / "packet.json", packet)
    files = ("実験計画.md", "masked/packet.json", "機械基準/forecast.json", "機械基準/snapshot.json", "機械基準/materials.json", "機械基準/予測.md")
    prepared = {"mode": "回想・開発再生", "as_of": AS_OF, "prepared_at": now(),
                "prices_path": str(prices.resolve()), "packet_hash": packet["packet_hash"],
                "sources": {"news": {"path": str(news.resolve()), "hash": _file_hash(news)},
                            "history": {"path": str(history.resolve()), "hash": _file_hash(history)}},
                "files": {relative: _file_hash(root / relative) for relative in files}}
    prepared["prepare_hash"] = digest(prepared)
    write_json(root / "準備.json", prepared)
    _append_audit(root, "prepare", as_of, [root / "準備.json"] + [root / relative for relative in files])
    return packet


def seal(root: Path) -> dict:
    """結果を参照せず、満了済み教師から３候補を対象終点より前に固定する。

    引数: root (Path) は準備済み入力とannotation.jsonを持つ保存先。
    返り値: dict は実発行時刻、先物基準、固定原本・実装ハッシュ付き３候補。
    """
    root = Path(root)
    if any((root / name).exists() for name in ("予測.json", "予測.md", "固定.json", "実装控え.json")):
        raise FileExistsError("固定済み予測は上書きできません。")
    prepared = _verify_prepared(root)
    if _audit(root)[-1]["phase"] != "prepare":
        raise ValueError("準備直後だけ予測を固定できます。")
    packet, annotation = read_json(root / "masked" / "packet.json"), read_json(root / "annotation.json")
    target = timestamp(packet["targets"]["today"])
    if timestamp(now()) >= target:
        raise ValueError("対象終点を過ぎたため、結果未開示の追跡として固定できません。")
    validate_annotation(annotation, packet)
    history_path = Path(prepared["sources"]["history"]["path"])
    history, metadata = _load_initial_history(history_path, packet)
    predictions = {method: {symbol: predict_learned(annotation, packet, symbol, history, mode=method)
                            for symbol in SYMBOLS} for method in METHODS}
    issued_at = now()
    if timestamp(issued_at) >= target:
        raise ValueError("計算中に対象終点を過ぎたため、結果未開示の追跡として固定できません。")
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    implementation = {"created_at": issued_at, "files": {
        path.name: {"content": path.read_bytes().decode("utf-8"), "hash": _file_hash(path)} for path in code_files}}
    implementation["implementation_hash"] = digest(implementation)
    write_json(root / "実装控え.json", implementation)
    prediction = {"mode": "回想・開発再生", "as_of": packet["as_of"], "issued_at": issued_at,
                  "session": packet["session"], "target": packet["targets"]["today"],
                  "issuance": "セッション開始後の回想。純粋な前向き成績ではない。",
                  "operational_acceptance": False, "adoption_status": "全候補80％未達・不採用継続",
                  "plan_hash": prepared["files"]["実験計画.md"], "packet_hash": packet["packet_hash"],
                  "annotation_hash": digest(annotation), "history_file_hash": metadata["initial_history_file_hash"],
                  "history_normalized_hash": metadata["initial_history_normalized_hash"],
                  "history_counts": metadata["initial_history_counts"],
                  "implementation_hash": implementation["implementation_hash"], "candidates": predictions}
    prediction["forecast_hash"] = digest(prediction)
    write_json(root / "予測.json", prediction)
    lines = ["# 結果未開示の追跡予測", "", f"- 情報締切：{packet['as_of']}。実発行：{issued_at}。",
             f"- 対象：{packet['session']}、終点{prediction['target']}。", "- セッション開始後の回想。純粋な前向き成績ではない。",
             "- 全候補80％未達・不採用継続。保留０。", ""]
    for symbol in SYMBOLS:
        reference = packet["instruments"][symbol]["reference"]
        lines.append(f"- {symbol}：基準{reference['price']}（{reference['time']}、同じ先物の公式終値代用）。")
        for method in METHODS:
            item = predictions[method][symbol]
            estimate = "百分率未発行" if item["prediction_pct"] is None else f"探索的推定{item['prediction_pct']:+.3f}%"
            lines.append(f"  - {method}：{item['direction']}、{estimate}。{item['status']}。")
        lines.append(f"  - 価格だけの経験範囲：{_range_text(packet['instruments'][symbol]['horizons']['today']['distribution'])}。")
    lines += [f"- 固定hash：{prediction['forecast_hash']}。", ""]
    (root / "予測.md").write_text("\n".join(lines), encoding="utf-8")
    sealed_files = ("annotation.json", "予測.json", "予測.md", "実装控え.json")
    fixed = {"forecast_hash": prediction["forecast_hash"], "files": {name: _file_hash(root / name) for name in sealed_files},
             "implementation_files": {path.name: _file_hash(path) for path in code_files}}
    if (root / "actor.json").is_file():
        fixed["files"]["actor.json"] = _file_hash(root / "actor.json")
    fixed["fixed_hash"] = digest(fixed)
    write_json(root / "固定.json", fixed)
    _verify_prepared(root)
    _append_audit(root, "seal", timestamp(packet["as_of"]), [root / "固定.json"] + [root / name for name in fixed["files"]])
    return prediction


def evaluate(root: Path, observed_at: datetime, prices: Path | None = None) -> dict:
    """固定入力を照合後に初めて結果価格を読み、当日２件が揃うまで評価を待つ。

    引数: root (Path) は固定済み予測、observed_at (datetime) は実績観測時刻、
    prices (Path | None) は読取専用価格。省略時は準備時の価格場所。
    返り値: dict は満期・原本待ち、または３候補の固定方向と先物実績の別採点。
    """
    root, observed_at = Path(root), timestamp(observed_at)
    if observed_at > timestamp(now()):
        raise ValueError("結果待ちの評価に未来の観測時刻は指定できません。")
    if (root / "結果.json").exists() or (root / "振り返り.md").exists():
        raise FileExistsError("保存済みの結果・振り返りは上書きできません。")
    prepared = _verify_prepared(root)
    if _audit(root)[-1]["phase"] != "seal":
        raise ValueError("予測固定後だけ結果を取得できます。")
    fixed, prediction = read_json(root / "固定.json"), read_json(root / "予測.json")
    if digest({key: value for key, value in fixed.items() if key != "fixed_hash"}) != fixed["fixed_hash"]:
        raise ValueError("固定記録のハッシュが一致しません。")
    if digest({key: value for key, value in prediction.items() if key != "forecast_hash"}) != prediction["forecast_hash"] or fixed["forecast_hash"] != prediction["forecast_hash"]:
        raise ValueError("予測の固定ハッシュが一致しません。")
    for relative, expected in fixed["files"].items():
        if _file_hash(root / relative) != expected:
            raise ValueError("固定後に予測・注釈または実装控えが変更されています。")
    for name, expected in fixed["implementation_files"].items():
        if _file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError("固定後に実装が変更されています。")

    # ----------------------------------------

    evaluation = evaluate_prices(root / "機械基準", Path(prices) if prices is not None else Path(prepared["prices_path"]), observed_at)
    labels = [label for label in evaluation["labels"] if label["horizon"] == "today"]
    if len(labels) != len(SYMBOLS) or {label["symbol"] for label in labels} != set(SYMBOLS):
        return {"status": "満期/原本待ち", "observed_at": observed_at.isoformat()}
    comparisons = []
    for label in labels:
        if timestamp(label["label_end"]) != timestamp(prediction["target"]):
            raise ValueError("当日の評価終点が固定予測と一致しません。")
        actual = label["actual_pct"]
        direction = "上昇" if actual > 0.05 else "下落" if actual < -0.05 else "横ばい"
        symbol = label["symbol"]
        for method in METHODS:
            candidate = prediction["candidates"][method][symbol]["direction"]
            comparisons.append({"method": method, "symbol": symbol, "direction": candidate,
                                "actual_direction": direction, "actual_pct": actual, "direction_hit": candidate == direction})
    result = {"mode": "回想・開発再生", "forecast_hash": prediction["forecast_hash"], "observed_at": observed_at.isoformat(),
              "labels": labels, "comparisons": comparisons, "adoption_status": "全候補80％未達・不採用継続"}
    result["result_hash"] = digest(result)
    write_json(root / "結果.json", result)
    _append_audit(root, "reveal", observed_at, [root / "結果.json"])
    lines = ["# 結果未開示の追跡：振り返り", "", "- セッション開始後の回想。純粋な前向き成績ではない。",
             "- 単発の結果で採用へ昇格しない。全候補80％未達・不採用継続。", ""]
    for item in comparisons:
        lines.append(f"- {item['symbol']}／{item['method']}：{item['direction']}→{item['actual_pct']:+.3f}%（{'一致' if item['direction_hit'] else '外れ'}）。")
    lines += ["- 原因：未識別。後発材料や単発の一致だけで因果を認定しない。", "- 次回：この固定結果を保存し、未使用期間で全件を継続評価する。", ""]
    (root / "振り返り.md").write_text("\n".join(lines), encoding="utf-8")
    _append_audit(root, "review", observed_at, [root / "振り返り.md"])
    return result


def main() -> None:
    """準備・固定・満期後評価を別コマンドとして実行する。

    引数: なし。コマンド行に操作、root、必要時だけニュース・教師・価格・観測時刻を指定する。
    返り値: None。準備・固定または満期・原本待ちの状態を表示する。
    """
    parser = argparse.ArgumentParser(description="結果未開示の追跡候補を固定する回想研究")
    parser.add_argument("action", choices=("prepare", "seal", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--observed-at", help="評価時の時差付き観測時刻")
    args = parser.parse_args()
    if args.action == "prepare":
        packet = prepare(args.root, args.news, args.history, args.prices or DEFAULT_PRICES)
        print({"status": "準備完了", "as_of": packet["as_of"], "articles": len(packet["news"]), "packet_hash": packet["packet_hash"]})
    elif args.action == "seal":
        prediction = seal(args.root)
        print({"status": "予測固定", "forecast_hash": prediction["forecast_hash"]})
    else:
        result = evaluate(args.root, timestamp(args.observed_at or now()), args.prices)
        print(result.get("status", "結果保存"))


if __name__ == "__main__":
    main()
