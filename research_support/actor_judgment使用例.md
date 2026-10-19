# 担当記録の組立補助

`assemble_actor` は本人が入力した解釈と申告を保存するための辞書を返す。点、方向、事実判定、読了、結果未閲覧を作らない。ファイルの保存や署名も行わない。`author_kind` は「本人明示入力の組立」であり、人間・モデルの属性や独立性の認定ではない。

```python
from research_support.actor_judgment import assemble_actor

# この一覧は実際に本人が確認したIDから作る。metadataから自動生成しない。
confirmed = [{"id": "合成記事1", "revision_id": "合成版1"}]

# 本人の実際の申告値を、6項目とも省略せず入力する。
attestations = {
    "body_curation_completed": False,
    "unmasked_body_displayed": False,
    "unmasked_excerpt_displayed": False,
    "future_result_seen": False,
    "original_price_seen": False,
    "old_trial_seen": False,
}

# 本人が内容を記入する。空配列も本人が「該当記録なし」と指定する場合だけ使う。
analysis = {
    "headline_batches": [{"ids": ["合成記事1"], "reading": "本人が記入した確認記録"}],
    "dedup_groups": [],
    "new_vs_repeat": "本人が記入した新規・継続の判断",
    "p0_reading": "本人が記入した現在地と直近変化",
    "case_comparisons": [],
    "story_lineage": [],
    "external_reaction_records": [],
    "scenario_bounds": "本人が記入した継続・反転の条件",
    "uncertainty": "本人が記入した不明事項",
    "market_calendar": "本人が記入した市場日程の解釈",
    "selected_ids": ["合成記事1"],  # 任意。本人が選んだ場合のみ記入する。
}

result = assemble_actor(
    "合成入力/packet.json",
    actor_name="本人が指定する担当名",
    headline_metadata_path="合成入力/見出し.json",
    confirmed_headline_ids=confirmed,
    read_scopes=[{"path": "合成入力/見出し.json", "scope": "本人が読んだ見出し全件"}],
    curation_manifest_path="合成入力/本文選別.json",
    feedback_snapshot_path="合成入力/反応.json",
    attestations=attestations,
    analysis=analysis,
)
actor = result["actor"]
source_records = result["source_records"]
```

見出し入力は `news`、`articles`、`headlines` のいずれか一つの配列、または配列自体。全件確認は版を含む集合の完全一致で検査する。IDが一意の場合だけ単純なID一覧も受け取る。同ID別版は上の複合指定が必須。見出し出力は `id revision_id headline canonical_url published_at fetched_at effective_at available_at source_id source_name` の10項目に限定する。本文・抜粋は返さない。

本文選別台帳は `records` 配列、反応入力は `cases` 配列を持つJSON。本文遮断版も受け取る。台帳の文章、反応ケースの文章、`query`、`notice` を返さず、ファイルハッシュ、最小限のヘッダ、配列件数を記録する。`independent_sample_count` の `null` は独立数未認定としてそのまま返し、0や場数に補完しない。他の件数は非負整数を要求する。任意の自己ハッシュがあれば検査する。`case_comparisons` は本人入力だけを配線し、`cases` の読了や理解を推測しない。

機械読込したpacket・metadata・台帳・snapshotは `read_files` に自動追加しない。本人が `read_scopes` に明示したファイルだけSHA256を計算し、そのscopeをそのまま記録する。packet内の締切以前の可知価格だけを複製し、外部価格ファイルを探索しない。申告6項目は `actor.attestations` とactorトップの両方へ同じ本人入力値を配線する。申告がFalseでも拒否せず、その値を保持する。組立成功は安全性の証明ではない。

既存 `material_report --actor` には呼出側が別途 `actor` だけを保存して渡せる。`source_records` は機械照合の別記録であり署名ではない。検査は `python -m unittest discover -s tests -p test_actor_judgment.py`。合成記事・合成価格を使う既存入口の接続検査を含む。
