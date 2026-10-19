"""時点入力と独立担当の材料判断を照合し、結果を読む前の実例を固定する。"""

import argparse
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.replay import _append_audit, _file_hash
from research_support.material_points import build_material_ledger, render_material_markdown


def _verify_body(article: dict, original: dict, cutoff: str) -> Path | None:
    """追加取得の実記事版と、判断に使った本文区間を照合する。

    引数: article (dict) は注釈根拠、original (dict) は時点内の記事、cutoff (str) は締切。
    返り値: Path | None は取得原本の保存先。抜粋だけの根拠はNone。
    """
    body_path = article.get("body_raw_path")
    if body_path is None:
        raw_text = original.get("body_excerpt", "")
        path = None
    else:
        path = Path(body_path)
        if _file_hash(path) != article.get("body_raw_sha256"):
            raise ValueError("根拠本文の取得原本ハッシュが一致しません。")
        raw = read_json(path)["structuredContent"]["article"]
        if str(raw["id"]) != str(article["id"]) or raw["revision_id"] != article["revision_id"]:
            raise ValueError("追加取得の実記事ID・版が切出と一致しません。")
        for key in ("published_at", "fetched_at", "updated_at"):
            if raw.get(key) is not None and timestamp(raw[key]) > timestamp(cutoff):
                raise ValueError("追加取得の本文版が締切後です。")
        if raw.get("body_offset", 0) != 0:
            raise ValueError("本文先頭からの主文区間を照合できません。")
        raw_text = raw["body_text"]
    scope = article.get("body_scope")
    if not isinstance(scope, dict) or raw_text[scope["start"]:scope["end_exclusive"]] != article.get("body"):
        raise ValueError("実際に判断へ使った本文区間が取得原本または既知抜粋と一致しません。")
    return path


def main() -> None:
    """可知記事版・先物基準を照合し、材料点付き予測と原本のハッシュを保存する。

    引数: なし。時点入力、判断、担当記録、未使用の出力先をコマンド行で受け取る。
    返り値: None。予測固定ハッシュを表示する。価格結果は読まない。
    """
    parser = argparse.ArgumentParser(description="材料点付きの実ニュース予測を固定")
    for name in ("packet", "judgment", "actor", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--equivalent-packet", type=Path)
    parser.add_argument("--narrative", type=Path, help="結果開示前の短い展開予想")
    parser.add_argument("--instructions", type=Path, action="append", default=[], help="固定する判断手順。複数指定可")
    args = parser.parse_args()
    if (args.out / "予測.json").exists():
        raise FileExistsError("材料予測は未使用の保存先へ固定してください。")
    packet, judgment = read_json(args.packet), read_json(args.judgment)
    if packet["packet_hash"] != digest({key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("時点入力の自己ハッシュが一致しません。")
    if timestamp(judgment["as_of"]) != timestamp(packet["as_of"]):
        raise ValueError("判断と入力の情報締切が一致しません。")
    visible = {(str(article["id"]), str(article["revision_id"])): article for article in packet["news"]}
    for article in judgment["articles"]:
        key = str(article["id"]), str(article["revision_id"])
        if key not in visible:
            raise ValueError("切出にない記事・版を予測へ追加できません。")
        source = visible[key]
        for field in ("published_at", "fetched_at", "effective_at", "updated_at", "available_at"):
            if (source.get(field) is None) != (article.get(field) is None):
                raise ValueError("記事の公表・取得・版の時計を変更できません。")
            if source.get(field) is not None and timestamp(source[field]) != timestamp(article[field]):
                raise ValueError("記事の公表・取得・版の時計が原本と一致しません。")
        if article["canonical_url"] != source["canonical_url"]:
            raise ValueError("記事の出典URLが原本と一致しません。")
    for symbol, state in packet["instruments"].items():
        source, fixed = state["reference"], judgment["references"][symbol]
        if source is None:
            raise ValueError("P0不足の場は不足記録へ残し、数値を補完して採点しません。")
        for field in ("price", "time", "session", "contract"):
            if source[field] != fixed[field]:
                raise ValueError("同先物の前セッション基準が原本と一致しません。")
    ledger = build_material_ledger(**judgment, source_packet=packet)
    if ledger["quality"]["reference_snapshot_binding"] != "P0原値・使用記事版の切出入力照合済み":
        raise ValueError("正式固定には時点入力との突合が必要です。")
    body_files = []
    for article in judgment["articles"]:
        body = _verify_body(article, visible[(str(article["id"]), str(article["revision_id"]))], packet["as_of"])
        if body is not None:
            body_files.append(body)
    actor = read_json(args.actor)
    equivalence = None
    if actor["packet_hash"] != packet["packet_hash"]:
        if args.equivalent_packet is None:
            raise ValueError("担当へ渡した入力と正式固定する入力が違います。")
        former = read_json(args.equivalent_packet)
        if actor["packet_hash"] != former["packet_hash"] or former["packet_hash"] != digest({
                key: value for key, value in former.items() if key != "packet_hash"}):
            raise ValueError("元の担当入力の自己ハッシュが一致しません。")
        if former["instruments"] != packet["instruments"] or former["as_of"] != packet["as_of"]:
            raise ValueError("再切出で価格・基準・締切が変わっています。")
        previous = {(str(article["id"]), str(article["revision_id"])): article for article in former["news"]}
        if previous.keys() != visible.keys():
            raise ValueError("再切出で担当の既知記事集合が変わっています。")
        for key, article in visible.items():
            for field in set(article) | set(previous[key]):
                if field in ("published_at", "fetched_at", "updated_at", "effective_at", "available_at"):
                    same = timestamp(article[field]) == timestamp(previous[key][field])
                else:
                    same = article.get(field) == previous[key].get(field)
                if not same:
                    raise ValueError("再切出で担当の既知記事の内容が変わっています。")
        equivalence = {"former_packet_hash": former["packet_hash"], "fixed_packet_hash": packet["packet_hash"],
                       "status": "記事集合・経済情報・時計の瞬間・価格が同一。日時表記と切出実装を訂正。"}
    sources = [args.packet, args.judgment, args.actor, Path(__file__),
               Path("research_support/material_points.py"), Path("research_support/prepare_material_snapshot.py"), *body_files]
    sources += args.instructions
    if args.narrative is not None:
        sources.append(args.narrative)
    sources += sorted(args.judgment.parent.glob(timestamp(packet["as_of"]).strftime("%Y%m%d_%H%M") + "*.json"))
    if args.equivalent_packet is not None:
        sources += [args.equivalent_packet, Path("local/全セッション材料/prepare_material_snapshot_v1.py")]
    source_hashes = {str(path.resolve()): _file_hash(path) for path in sources}
    write_json(args.out / "入力固定.json", {"as_of": packet["as_of"], "session": packet["session"],
               "packet_hash": packet["packet_hash"], "actor": actor, "source_hashes": source_hashes, "input_equivalence": equivalence,
               "recorded_at": now(), "mode": "回想・時点限定の材料点実例。未使用期間の実証ではない。"})
    _append_audit(args.out, "prepare", timestamp(packet["as_of"]), [args.out / "入力固定.json"])
    write_json(args.out / "予測.json", ledger)
    narrative = "" if args.narrative is None else args.narrative.read_text(encoding="utf-8").rstrip() + "\n\n"
    (args.out / "予測.md").write_text(narrative + render_material_markdown(ledger)
        + "\n- [材料・記事・各点の根拠全文](予測.json)。\n", encoding="utf-8")
    _append_audit(args.out, "seal", timestamp(packet["as_of"]), [args.out / "予測.json", args.out / "予測.md"])
    if any(_file_hash(Path(name)) != expected for name, expected in source_hashes.items()):
        raise ValueError("固定中に入力・実装・担当記録が変わっています。")
    print({"session": packet["session"], "ledger_hash": ledger["ledger_hash"]})


if __name__ == "__main__":
    main()
