# 五分類の反復検証

- 現在の母集団は[全39場・78対象](../検証/00013_全セッションと材料の織込み/README.md)。旧18回は限定日程であり、全期間の件数ではない。
- `prepare_material_snapshot.py` は締切までの全利用時計で記事版を選び、既知の同先物価格だけを保存する。直近72時間の見出しと全既知記事の別索引を用意する。
- `material_points.py` は材料別・先物別のT/B0/Dと根拠を検査し、既反映合計・P0起点寄与・asof残余を別表示する。
- `build_material_ledger(..., source_packet=packet)` で原P0・記事版を束縛する。五択は材料ID・点の解釈・時間順序を持つ人間／LLM判断を渡す。過去頻度による代用はしない。
- 入力例は`tests/fixtures/material_points.json`（合成）。実際の[材料表の実例](../検証/00013_全セッションと材料の織込み/実例_20260921_1600/要点.md)とは区別する。
- `material_report.py` は切出P0・記事版・取得本文の主文区間を照合し、結果を読む前に材料点付き予測をprepare→sealへ固定する。記事の元抜粋・時計を削除せず、別取得本文と来歴を追加する。
- [先物の展開予想](../docs/先物の展開予想.md)で主導材料・寄付き・継続／反転を判断してから点と五択を結ぶ。`material_report --narrative <短い展開予想.md> --instructions <判断手順>`で本文と手順も固定できる。手順は複数指定可。

```powershell
python -X utf8 -m research_support.prepare_material_snapshot --as-of <締切> --news <版別記事索引JSON> --prices ../FutureAlgorithmResearch --out <未使用の入力先>
python -X utf8 -m research_support.material_report --packet <切出packet.json> --judgment <材料判断JSON> --actor <担当記録JSON> --out <未使用の実験先>
```

- 現在の材料打切り規約は[未知臨時材料前](臨時材料規約.json)。予定指標・会見のサプライズでは終了しない。
- `event_learning.py` は独立材料台帳を結果側だけで用い、確定教師・候補配分を同じ18締切で逐次比較する別版。
- 各締切で満了済みの過去教師を再構成。根拠全記事の収録版の公開・取得・有効時計が締切前の認定だけを教師へ適用する。
- 両方式の当回予測を固定後に主窓・全場参考を計算。主採点と参考採点を混ぜない。
- 材料認定不明・開始前失効は教師ラベル重み０。価格経路の候補配分とは区別する。
- [新規約の実験計画](../検証/00012_臨時材料前の五分類/実験計画.md)。以下の `five_class_replay` は旧全場版の保存手順。

```powershell
python -X utf8 -m research_support.event_learning --root <未使用の新実験> --inputs 検証/00003_学習期間の延長 --measured local/五分類準備/実測原本_v2 --prices ../FutureAlgorithmResearch --ledger local/臨時材料点検/統合認定台帳.json
```

- 有効な規約は[五分類の合格条件](../docs/精度の合格条件.md)。対象は場中±0.5％到達、対象内の無回答は不正解、もみ合い優先。
- `five_class.py` の `evaluate_priority_path` は初回到達方向と最初の反転で一つに分類。欠測・足内順序不明は候補集合のまま保存する。
- 同モジュールの `evaluate_path` は優先順位のない一般的な集合評価。現在の主採点へ代用しない。
- `prepare_five_history.py` は初回締切より前に満了した教師だけへ経路情報を付記し、原本と異なるコピーへ保存する。未確定を正解ラベルに変えない。
- `five_class_replay.py` は規約・実装・静的指示・入力・担当・予測・結果・個別考察を結び付けて固定する。
- 新実験に `実験計画.md`、`予測担当の手順.md`、確認済み `五分類規約.json` を用意。[計画例](../検証/00010_五分類の時点反復/実験計画.md)を参照する。
- 確認済みの[共通規約](五分類規約.json)を新実験へコピーできる。同じ条件での再実行に再確認は不要。条件を変更する場合は別版を固定する。
- 各回は新規担当。読取範囲は当回の切出済み入力と静的指示。予測固定後に結果を開示し、同じ担当が個別考察を保存してから次の回へ進む。
- 対象・対象不明・対象外・無回答を別集計。対象不明を含む保守的下限、真の対象精度の楽観上限、確定対象だけの成績を区別する。
- 同じ入力・対象・分母で五つの定数回答も全て比較する。既知期間の再生を前向き成績と呼ばない。
- 旧実験の予測・結果を上書きしない。

```powershell
python -X utf8 -m research_support.five_class_replay prepare --root <新実験> --inputs <可視原本> --measured <実測入力コピー> --index 1
python -X utf8 -m research_support.five_class_replay seal --root <新実験> --round <当回名> --actor <担当JSON>
python -X utf8 -m research_support.five_class_replay reveal --root <新実験> --round <当回名> --prices ../FutureAlgorithmResearch
python -X utf8 -m research_support.five_class_replay review --root <新実験> --round <当回名> --reflection <個別考察JSON> --actor <考察担当JSON>
python -X utf8 -m research_support.five_class_replay summarize --root <新実験>
```

## 旧三分類向けの手順修正

- `analyst_v2.py` は循環10を変更しない別版。予測精度を改善した実績はまだない。
- 新しい実験フォルダへ計画・静的指示を先に保存する。
- 静的指示には次の文をそのまま含める：`横ばいは-0.05％以上+0.05％以下（境界を含む）。`
- `prepare_v2`：上記境界を検証し、別版実装・指示を固定してから当回入力を作る。
- 子へ入力を渡す前に `準備固定.json` をprepare監査へ束縛し、初回seal前の実装・手順の同時変更も検出する。
- prepare開始時刻と追加固定を終えた最終記録時刻を分けて保存する。
- 新規の独立担当が当回入力だけで予測し、`seal_v2` が実装・指示の固定を担当証跡へ束縛。
- `reveal_v2`：予測固定後に結果だけを開示。原採点は `source_metrics`、当回採点は `metrics` とtop-level。
- 対象終点が実時計でも満了するまで原結果を読まない。
- 同じ当回担当へ入力・固定予測・結果を渡し、当回固有の理由と次回対応を `考察入力.json` へ新規保存させる。
- `finalize_review_v2`：個別考察を固定するまで次の締切へ進めない。同じ担当を別締切の予測へ再利用しない。
- 次回feedbackでは `prediction_evidence_ids` と `reflection_evidence_ids` を別々に保持する。
- 実装・指示・予測・結果・担当・考察の改変は次段階で検出する。
- 個別考察の形式と結び付きは検証するが、理由の因果的な正しさまで自動判定しない。
- 同じ既知18回をさらに再生して80％へ合わせる作業には使っていない。

考察JSONの形式：

```json
{
  "forecast_hash": "固定予測の値",
  "result_hash": "結果JSON全体のdigest",
  "instruments": {
    "NK225": {
      "predicted_direction": "上昇",
      "actual_direction": "下落",
      "causality": "未識別",
      "reason": "当回の既反応・追加反応仮説と実績を比較した200字以内の考察",
      "next_action": "次の締切で確認する観測事実を160字以内で具体的に記述",
      "evidence_ids": []
    },
    "TOPIX": {
      "predicted_direction": "下落",
      "actual_direction": "下落",
      "causality": "仮説",
      "reason": "方向一致でも反対材料を点検し、単独因果と断定しない",
      "next_action": "同じ条件の次回で成立・失効を照合する",
      "evidence_ids": []
    }
  }
}
```
