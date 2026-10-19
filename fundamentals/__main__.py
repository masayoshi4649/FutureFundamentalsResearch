"""研究用コマンドを日本語で提供する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import digest, now, read_json, timestamp, write_json
from .engine import evaluate, forecast, session_plan, summarize, verify_forecast
from .news import load_export, snapshot_payload
from .prices import inventory


def parser() -> argparse.ArgumentParser:
    """研究コマンドの引数を定義する。

    引数: なし。
    返り値: argparse.ArgumentParser は日本語説明付きの引数定義。
    """
    root = argparse.ArgumentParser(description="ニュースとNK225／TOPIX先物の時点整合研究")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("inventory", help="価格期間・欠測候補を確認")
    command.add_argument("--prices", type=Path, default=Path("../FutureAlgorithmResearch"), help="価格リポジトリ")
    command.add_argument("--out", type=Path, help="集計の保存先")

    command = commands.add_parser("import-news", help="MCP書出しを版ごとの原本へ保存")
    command.add_argument("--input", type=Path, required=True, help="MCP結果の構造化ファイル")
    command.add_argument("--archive", type=Path, default=Path("local/news"), help="版別保存先")

    command = commands.add_parser("snapshot", help="未来本文を除いた時点スナップショット")
    command.add_argument("--news", type=Path, required=True, help="MCPニュース書出し")
    command.add_argument("--as-of", required=True, help="情報締切の時差付き日時")
    command.add_argument("--out", type=Path, required=True, help="新規保存フォルダ")

    command = commands.add_parser("forecast", help="開始前の四期間見通しを固定")
    command.add_argument("--prices", type=Path, default=Path("../FutureAlgorithmResearch"), help="価格リポジトリ")
    command.add_argument("--news", type=Path, required=True, help="MCPニュース書出し")
    command.add_argument("--events", type=Path, required=True, help="材料注釈")
    command.add_argument("--as-of", required=True, help="標準07:45／16:00の情報締切")
    command.add_argument("--out", type=Path, required=True, help="新規予測保存先")
    command.add_argument("--history", type=Path, help="満了した過去予測のルート")
    command.add_argument("--mode", choices=["回想", "運用"], default="回想", help="回想と前向き運用を分離")

    command = commands.add_parser("evaluate", help="固定予測の満了分を全件評価")
    command.add_argument("--run", type=Path, required=True, help="固定予測フォルダ")
    command.add_argument("--prices", type=Path, default=Path("../FutureAlgorithmResearch"), help="価格リポジトリ")
    command.add_argument("--observed-at", required=True, help="評価に使える情報の時刻上限")
    command.add_argument("--news", type=Path, help="後発材料候補を抽出するニュース")

    command = commands.add_parser("walk-forward", help="過去再生で予測固定→評価→次時点を反復")
    command.add_argument("--prices", type=Path, default=Path("../FutureAlgorithmResearch"), help="価格リポジトリ")
    command.add_argument("--news", type=Path, required=True, help="全版保持のニュース原本")
    command.add_argument("--events", type=Path, required=True, help="回想で作成した材料注釈")
    command.add_argument("--schedule", type=Path, required=True, help="時系列の判断日時配列またはas_of付き配列")
    command.add_argument("--out", type=Path, required=True, help="新規の実験ルート")

    command = commands.add_parser("summarize", help="全件成績と材料別反応を保存")
    command.add_argument("--runs", type=Path, required=True, help="予測保存ルート")
    command.add_argument("--out", type=Path, help="新規集計フォルダ")

    command = commands.add_parser("review", help="誤差の根拠と次回の修正案を追記")
    command.add_argument("--run", type=Path, required=True, help="評価済み予測フォルダ")
    command.add_argument("--input", type=Path, required=True, help="根拠付き考察ファイル")
    return root


def load_events(path: Path) -> list[dict]:
    """材料注釈の必須項目を検証して読む。

    引数: path (Path) は材料配列または events 配列を持つ構造化ファイル。
    返り値: list[dict] は注釈。主観点数と記事の公表実数は区別する。
    """
    value = read_json(path)
    events = value.get("events") if isinstance(value, dict) else value
    if not isinstance(events, list):
        raise ValueError("材料は配列で指定してください。")
    required = {"event_id", "story_key", "category", "regime", "known_at", "annotated_at", "evidence",
                "impacts", "confidence", "novelty", "priced_in", "half_life_days", "reason"}
    used = set()
    for event in events:
        if not isinstance(event, dict) or not required.issubset(event):
            raise ValueError("材料注釈の必須項目が不足しています。docs/材料注釈.mdを参照してください。")
        if event["event_id"] in used:
            raise ValueError("材料IDは全版を通じ一意にしてください。更新は別IDと同じstory_keyを使います。")
        used.add(event["event_id"])
        timestamp(event["known_at"])
        timestamp(event["annotated_at"])
        if not isinstance(event["evidence"], list) or not all(isinstance(item, dict) and {"id", "revision_id"} <= item.keys() for item in event["evidence"]):
            raise ValueError("材料根拠には記事IDと改訂版IDの配列が必要です。")
        if not isinstance(event["impacts"], dict) or not {"NK225", "TOPIX"} <= event["impacts"].keys():
            raise ValueError("材料にはNK225とTOPIXそれぞれの方向点数が必要です。")
        for key in ("event_id", "story_key", "category", "regime", "reason"):
            if not isinstance(event[key], str) or not event[key].strip():
                raise ValueError(f"材料の{key}は空でない文字列が必要です。")
    return events


def render_summary(value: dict) -> str:
    """集計結果を読みやすい日本語へ整形する。

    引数: value (dict) は全件集計。
    返り値: str は探索的成績の文書。
    """
    lines = ["# 先物予測の全件評価", "", "公式終値の代用値による評価。回想成績から実運用の優位性は断定しない。", "",
             "| 区分／銘柄／セッション／期間 | 予測 | 満了 | 保留 | 終点範囲包含 | 観測経路包含 | 推定候補方向の一致 |",
             "| --- | ---: | ---: | ---: | --- | --- | --- |"]
    for key, group in value["groups"].items():
        coverage = f"{group['terminal_coverage']:.1%}（{group['terminal_interval_n']}件）" if group["terminal_coverage"] is not None else "未算出"
        path = f"{group['observed_path_coverage']:.1%}（{group['path_interval_n']}件）" if group["observed_path_coverage"] is not None else "未算出"
        hit = f"{group['candidate_direction_hit_rate']:.1%}（{group['candidate_direction_n']}件）" if group["candidate_direction_hit_rate"] is not None else "推定未発行"
        lines.append(f"| {key} | {group['forecasts']} | {group['evaluated']} | {group['abstained']} | {coverage} | {path} | {hit} |")
    lines += ["", "保留は優位方向の保留。候補方向を表示した場合でも、優位性の確立とは扱わない。", ""]
    lines += [f"- {item}" for item in value["limitations"]] + [""]
    return "\n".join(lines)


def save_summary(root: Path, out: Path) -> dict:
    """集計と材料反応の再利用用成果を保存する。

    引数: root (Path) は予測ルート、out (Path) は新規集計先。
    返り値: dict は保存した集計。
    """
    value = summarize(root)
    write_json(out / "summary.json", value)
    (out / "評価.md").write_text(render_summary(value), encoding="utf-8")
    return value


def walk_forward(args: argparse.Namespace) -> dict:
    """指定した時系列で固定予測と評価を反復する。

    引数: args (argparse.Namespace) は価格・ニュース・材料・日程・保存先。
    返り値: dict は回想検証の全件集計。未来の評価を先の予測へ戻さない。
    """
    if args.out.exists():
        raise FileExistsError("過去再生には新しい実験フォルダを指定してください。")
    articles, events = load_export(args.news), load_events(args.events)
    schedule = read_json(args.schedule)
    dates = [timestamp(item["as_of"] if isinstance(item, dict) else item) for item in schedule]
    if dates != sorted(set(dates)) or not dates:
        raise ValueError("日程は空でない昇順・重複なしの日時配列にしてください。")
    created = []
    for index, as_of in enumerate(dates):
        for prior in created:
            evaluate(prior, args.prices, as_of, articles)
        run = args.out / as_of.strftime("%Y%m%dT%H%M%S%z")
        forecast(args.prices, articles, events, as_of, run, args.out, "回想")
        item = schedule[index]
        observed_at = timestamp(item["observed_at"]) if isinstance(item, dict) and "observed_at" in item else session_plan(as_of)["end"]
        if observed_at <= as_of or (index + 1 < len(dates) and observed_at >= dates[index + 1]):
            raise ValueError("評価日時は判断後かつ次回判断前にしてください。")
        evaluate(run, args.prices, observed_at, articles)
        created.append(run)
        print(f"固定・評価済み：{as_of.isoformat()}", flush=True)
    write_json(args.out / "experiment.json", {"mode": "回想・非盲検の再生", "recorded_at": now(),
        "schedule": [value.isoformat() for value in dates], "news_input_hash": digest(articles),
        "events_input_hash": digest(events), "warning": "結果既知の言語モデルによる注釈は、盲検や前向き成績ではない。"})
    return save_summary(args.out, args.out / "集計")


def review(run: Path, path: Path) -> dict:
    """考察を追記し、既存予測や係数を変更しない。

    引数: run (Path) は予測保存先、path (Path) は考察入力。
    返り値: dict は次回以降に検証する修正案の記録。
    """
    fixed = read_json(run / "forecast.json")
    verify_forecast(fixed)
    value = read_json(path)
    required = {"classification", "evidence", "reason", "next_action", "available_at"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("考察にはclassification/evidence/reason/next_action/available_atが必要です。")
    if value["classification"] not in ("既知材料の解釈", "後発材料候補", "モデル仕様", "データ不備", "未識別", "整合を確認"):
        raise ValueError("考察の分類が未対応です。")
    timestamp(value["available_at"])
    if timestamp(value["available_at"]) > timestamp(now()):
        raise ValueError("考察の利用可能時刻を記録時刻より未来にはできません。")
    if value["classification"] == "後発材料候補" and not value["evidence"]:
        raise ValueError("後発材料の考察には時点付きの根拠が必要です。")
    value.update(forecast_hash=fixed["forecast_hash"], recorded_at=now(), status="次回以降の検証案・自動採用なし")
    write_json(run / ("review-" + digest(value)[:16] + ".json"), value)
    return value


def main() -> int:
    """コマンドを実行し、失敗時に日本語の理由を表示する。

    引数: なし。コマンド行引数を利用する。
    返り値: int は成功０、失敗１。
    """
    args = parser().parse_args()
    try:
        if args.command == "inventory":
            value = inventory(args.prices)
            if args.out:
                write_json(args.out, value)
        elif args.command == "import-news":
            records = load_export(args.input)
            saved = 0
            for article in records:
                identity = digest({"id": article["id"], "revision_id": article["revision_id"]})
                path = args.archive / (identity + ".json")
                if path.exists():
                    if read_json(path) != {"articles": [article]}:
                        raise ValueError("同じ記事版IDの本文が変化しています。既存原本は上書きしません。")
                else:
                    write_json(path, {"articles": [article]})
                    saved += 1
            value = {"入力件数": len(records), "新規保存件数": saved, "保存先": str(args.archive)}
        elif args.command == "snapshot":
            if args.out.exists():
                raise FileExistsError("切出し先は新しいフォルダを指定してください。")
            value = snapshot_payload(load_export(args.news), timestamp(args.as_of))
            write_json(args.out / "snapshot.json", value)
            value = {"記事数": len(value["articles"]), "除外数": len(value["excluded"]), "保存先": str(args.out)}
        elif args.command == "forecast":
            value = forecast(args.prices, load_export(args.news), load_events(args.events), timestamp(args.as_of), args.out, args.history, args.mode)
            value = {"保存先": str(args.out), "予測ハッシュ": value["forecast_hash"]}
        elif args.command == "evaluate":
            value = evaluate(args.run, args.prices, timestamp(args.observed_at), load_export(args.news) if args.news else None)
            value = {"満了件数": len(value["labels"]), "未満了件数": len(value["pending"])}
        elif args.command == "walk-forward":
            value = walk_forward(args)
            value = {"集計保存先": str(args.out / "集計"), "集計区分数": len(value["groups"])}
        elif args.command == "summarize":
            value = save_summary(args.runs, args.out) if args.out else summarize(args.runs)
            value = {"集計区分": value["groups"]}
        elif args.command == "review":
            value = review(args.run, args.input)
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"実行できません：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
