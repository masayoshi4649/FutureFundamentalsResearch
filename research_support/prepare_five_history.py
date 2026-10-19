"""五分類再生に使う過去教師を別コピーへ付記し、元入力を変更しない。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.engine import reference_quote, session_plan
from fundamentals.prices import load_candles
from fundamentals.replay import _file_hash
from research_support import five_class as evaluation_module
from research_support.five_class import evaluate_priority_path


def prepare_history(inputs: Path, measured: Path, prices: Path, output: Path, rules: Path) -> dict:
    """初回締切より前の教師だけを新分類で点検し、実測可視入力を別コピーする。

    引数: inputs (Path) は既存masked原本、measured (Path) は実測原本、prices (Path) は元価格、
        output (Path) は未使用の新出力先、rules (Path) は事前に確認した規約JSON。
    返り値: dict はコピー元ハッシュ・時点検査・新教師を束縛した来歴。
    """
    packets = sorted((inputs / "masked").glob("*.json"))
    if not packets or output.exists():
        raise ValueError("可視入力があることと、出力先が未使用であることが必要です。")
    destination = output.resolve()
    for source_root in (inputs, measured, prices):
        source = source_root.resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("出力先と入力・元価格のルートを重ねられません。")
    cutoff = timestamp(read_json(packets[0])["as_of"])
    contract = read_json(rules)
    if contract.get("confirmed") is not True or contract.get("overlap_policy") != "もみ合い優先・初回到達方向":
        raise ValueError("確認済みのもみ合い優先規約が必要です。")
    if timestamp(contract["confirmed_at"]) > timestamp(now()):
        raise ValueError("規約の確認日時が未来です。")
    initial = read_json(measured / "初期教師.json")
    rows = initial["rows"]
    for row in rows:
        if timestamp(row["label_end"]) >= cutoff or timestamp(row["as_of"]) >= timestamp(row["label_end"]):
            raise ValueError("初期教師は初回締切より前に満了した行だけを使用できます。")
    sources = [measured / "初期教師.json", rules]
    sources.extend(measured / path.stem / "入力.json" for path in packets)
    sources.extend([Path(__file__), Path(evaluation_module.__file__)])
    source_hashes = {str(path.resolve()): _file_hash(path) for path in sources}

    # ----------------------------------------
    # 数値結果は時刻検査後だけ解釈する。未確定の経路を一つの正解へ変換しない。

    enriched = []
    for row in rows:
        end, origin = timestamp(row["label_end"]), timestamp(row["as_of"])
        plan = session_plan(origin)
        candles = load_candles(prices, row["symbol"], cutoff=end)
        reference = reference_quote(candles, origin, plan)
        path = [bar for bar in candles if bar.session == plan["session"]]
        evaluation = evaluate_priority_path(reference, path, plan["session"], end, contract["criteria"])
        actual = evaluation["metrics"]["actual_pct"]
        if actual is None or abs(actual - row["actual_pct"]) > 1e-9:
            raise ValueError("初期教師の先物終点と新評価の終点が一致しません。")
        enriched.append({**row, "five_class_evaluation": evaluation})
    for path in packets:
        source = measured / path.stem / "入力.json"
        destination = output / path.stem / "入力.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(output / "初期教師.json", {**initial, "rows": enriched,
               "five_class_rules_hash": _file_hash(rules), "first_as_of": cutoff.isoformat(),
               "label_scope": "終点実績と保存足で判明する経路候補。欠測・順序不明は確定ラベルにしない。"})
    if any(_file_hash(Path(name)) != expected for name, expected in source_hashes.items()):
        raise ValueError("コピー作成中に元の実測入力または規約が変更されています。")
    value = {"created_at": now(), "mode": "回想・過去教師の経路点検", "first_as_of": cutoff.isoformat(),
             "teacher_rows": len(rows), "source_hashes": source_hashes,
             "teacher_hash": _file_hash(output / "初期教師.json"),
             "implementation_hash": _file_hash(Path(__file__))}
    value["provenance_hash"] = digest(value)
    write_json(output / "初期教師の来歴.json", value)
    return value


def main() -> None:
    """原本と別出力先を指定して、過去教師の五分類入力を作る。

    引数: なし。入力・実測・価格・出力・規約のパスをコマンド行で受け取る。
    返り値: None。保存した教師件数を表示する。
    """
    parser = argparse.ArgumentParser(description="元入力を変更しない五分類の初期教師作成")
    for name in ("inputs", "measured", "prices", "out", "rules"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(prepare_history(args.inputs, args.measured, args.prices, args.out, args.rules)["teacher_rows"])


if __name__ == "__main__":
    main()
