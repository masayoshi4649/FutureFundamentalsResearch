"""固定済みの振り返りを変更せず、次の予測へ実際に渡った教師の重みを補記する。"""

from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.replay import _audit, _file_hash
from research_support.five_class_learning import _verify


def append_feedback_notes(root: Path) -> dict:
    """保存済みの次回入力と予測から、今回教師の実際の利用状態だけを追記する。

    引数: root (Path) は正式再生を終えた実験。価格・ニュース・予測規則は変更しない。
    返り値: dict は既存成果物と新しい補足のSHA256を固定した独立した追記記録。
    """
    manifest = _verify(root)
    folders = [root / folder / identity for folder in manifest["variants"].values() for identity in manifest["rounds"]]
    if (root / "振り返り補足の来歴.json").exists() or any((path / "振り返りの補足.md").exists() for path in folders):
        raise FileExistsError("振り返り補足は未使用の保存先へ一度だけ作成してください。")
    for path in folders:
        if _audit(path)[-1]["phase"] != "review":
            raise ValueError("補足の前に全回の固定・結果・振り返りの監査が必要です。")
    record = {"created_at": now(), "implementation_hash": _file_hash(Path(__file__)),
              "manifest_hash": manifest["manifest_hash"], "sources": {}, "notes": {}}
    for folder in manifest["variants"].values():
        for index, identity in enumerate(manifest["rounds"]):
            path = root / folder / identity
            _audit(path)
            source_files = [path / "result.json", path / "review.json", path / "振り返り.md"]
            for source in source_files:
                record["sources"].setdefault(str(source.resolve()), _file_hash(source))
            result = read_json(path / "result.json")
            lines = ["# 次回学習への反映の確認", "", "- 元の振り返りと採点は保持。この補足は保存済みの次回入力・重みの照合。",
                     "- 元文の「教師追加」は過去特徴の追加を含む。材料認定不明の五候補を正解ラベルとして学習しない。", ""]
            if index + 1 < len(manifest["rounds"]):
                next_path = root / folder / manifest["rounds"][index + 1]
                _audit(next_path)
                for source in (next_path / "入力.json", next_path / "予測.json"):
                    record["sources"].setdefault(str(source.resolve()), _file_hash(source))
                visible = read_json(next_path / "入力.json")
                prediction = read_json(next_path / "予測.json")
                for row in result["labels"]:
                    symbol = row["symbol"]
                    teachers = [teacher for teacher in visible["history"] if teacher["symbol"] == symbol
                                and timestamp(teacher["as_of"]) == timestamp(row["evaluation"]["evaluation_window"]["as_of"])]
                    if len(teachers) != 1:
                        raise ValueError("次回履歴に当回の先物教師が一つだけ存在する必要があります。")
                    teacher = teachers[0]
                    model = prediction["instruments"][symbol]
                    position = model["training_row_ends"].index(teacher["label_end"])
                    window = teacher["five_class_evaluation"]["evaluation_window"]
                    weight = model["weights"][position]
                    if (window["adjudication_unknown"] or not window["scoreable"]) and weight != 0:
                        raise ValueError("材料認定不明・失効の教師へ正のラベル重みが付いています。")
                    state = "材料認定不明" if window["adjudication_unknown"] else window["status"]
                    lines += [f"- {symbol}：次回締切{visible['as_of']}で、当回教師は{state}、ラベル重み{weight:g}。",
                              "  - 重み０でも過去特徴の中央値・標準化プールには残る。後日の記事であることを知る前は、当時の根拠だけで評価する。"]
            else:
                lines.append("- この実験の最終回。次回予測は未実行であり、次回の重み・成績を推測しない。")
            note = path / "振り返りの補足.md"
            with note.open("x", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
            record["notes"][str(note.resolve())] = _file_hash(note)
    if any(_file_hash(Path(name)) != expected for name, expected in record["sources"].items()):
        raise ValueError("補足の作成中に固定済みの成果物が変わっています。")
    _verify(root)
    record["record_hash"] = digest(record)
    write_json(root / "振り返り補足の来歴.json", record)
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="次回教師の保存済み重みを振り返りへ追記")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(len(append_feedback_notes(arguments.root)["notes"]))
