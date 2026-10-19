# 材料判断JSONの静的組立

`point_judgment` は明示された3点・3根拠・観測欄を写し、`assemble_material_judgment` はpacketからP0と記事メタデータを写す。点の符号から方向を決めない。材料の意味、方向、根拠、本文区間は呼出側が渡す。省略できるのは未作成の週・月・3か月だけで、JSONでは `null` になる。

以下は既存の**合成fixtureだけ**を使う2材料・2先物・当日の例。実記事・実価格・過去予測を読まない。出力先は未使用の場所を指定する。

```python
import copy
from pathlib import Path

from fundamentals.common import digest, read_json, write_json
from research_support.material_judgment import assemble_material_judgment, point_judgment

data = read_json(Path("tests/fixtures/material_points.json"))
news = copy.deepcopy(data["articles"])
for article in news:
    article["body_excerpt"] = article.pop("body")
packet = {
    "as_of": data["as_of"], "session": "2026-09-30_Day", "news": news,
    "instruments": {symbol: {"reference": {key: row[key] for key in
        ("price", "time", "session", "contract")}} for symbol, row in data["references"].items()},
}
packet["packet_hash"] = digest(packet)
materials = copy.deepcopy(data["materials"])
for material in materials:
    material["impacts"] = {symbol: {"today": row["today"]}
                           for symbol, row in material["impacts"].items()}
materials[0]["impacts"]["NK225"]["today"] = point_judgment(
    impact_pt=3, priced_in_before_reference_pt=2, priced_in_after_reference_pt=None,
    impact_reason="合成材料の総影響を主観評価。", before_reason="合成の基準前観測による主観評価。",
    after_reason="基準後の反映量は不明。", transmission="合成の利益期待を通じた経路。",
    before_observations=[], after_observations=[],
)
result = assemble_material_judgment(
    packet,
    article_uses=[{"id": row["id"], "revision_id": row["revision_id"], "mode": "excerpt",
        "scope": {"start": 0, "end_exclusive": 4, "notice": "既知抜粋の先頭4文字だけを使用。"}}
        for row in news],
    reference_details={symbol: {key: row[key] for key in ("source", "proxy")}
                       for symbol, row in data["references"].items()},
    materials=materials,
    forecasts={symbol: {"today": row["today"]} for symbol, row in data["forecasts"].items()},
)
output = Path("local/材料判断合成例")
write_json(output / "packet.json", packet)
write_json(output / "judgment.json", result["judgment"])
write_json(output / "本文出典記録.json", result["source_records"])
```

実運用では新たにpacketを合成せず、担当へ渡された `packet.json` をそのまま渡す。`reference_details` は両先物の価格出典文字列と終値代理の真偽値だけを明示し、P0数値・時計・セッション・限月は変更できない。

取得原本を使う記事の指定は次の形式に替える。`raw_sha256` は既知なら指定でき、省略時も原本バイト列のSHA256を計算して記録する。原本は `structuredContent.article.body_text`、`body_offset=0` の形式を使い、ID・版・時計をpacketと照合する。`scope` はPython文字列の文字位置で、終点を含まない。

```python
{"id": "1", "revision_id": "1", "mode": "raw", "raw_path": "取得原本.json",
 "scope": {"start": 10, "end_exclusive": 80, "notice": "判断に使う主文の明示区間。"}}
```

`judgment.json` は既存の `python -m research_support.material_report --packet ... --judgment ... --actor ... --out ...` の判断入力になる。actorは独立した明示記録が必要で、この補助は生成しない。機械的な出典記録は読了・未来結果の未閲覧を証明しない。現行入口は全選択記事の本文照合を要求するため、本文なしの記事は選択せず別記し、選択する場合は既知抜粋または取得本文の利用区間を明示する。
