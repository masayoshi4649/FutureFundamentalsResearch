"""本人が明示した担当記録を、判断や読了を補完せず機械的に組み立てる。

本文・抜粋は出力しない。ファイル保存、署名、安全性の認定も行わない。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from fundamentals.common import digest, timestamp


HEADLINE_FIELDS = ("id", "revision_id", "headline", "canonical_url", "published_at", "fetched_at",
                   "effective_at", "available_at", "source_id", "source_name")
REQUIRED_ATTESTATIONS = ("body_curation_completed", "unmasked_body_displayed", "unmasked_excerpt_displayed",
                         "future_result_seen", "original_price_seen", "old_trial_seen")
REQUIRED_ANALYSIS = ("headline_batches", "dedup_groups", "new_vs_repeat", "p0_reading", "case_comparisons",
                     "story_lineage", "external_reaction_records", "scenario_bounds", "uncertainty", "market_calendar")
SNAPSHOT_HEADERS = ("version", "as_of", "mode", "session_count", "instrument_observation_count",
                    "independent_sample_count", "snapshot_hash")
PRICE_FIELDS = {
    "reference": ("price", "time", "session", "contract", "status"),
    "recent_sessions": ("session", "first_time", "last_time", "open", "close", "observed_high", "observed_low",
                        "observed_rows", "previous_observed_session_close"),
    "price_timeline": ("time", "end", "session", "open", "high", "low", "close", "volume", "contract"),
}


def _load(path: str | Path) -> tuple[dict | list, dict]:
    """明示されたJSONだけを解析し、ファイル単位の出典を作る。

    引数: path (str | Path) は呼出側が指定したファイル。
    返り値: tuple[dict | list, dict] は解析値とパス・バイト列SHA256。内容は表示しない。
    """
    resolved = Path(path).resolve()
    content = resolved.read_bytes()
    return json.loads(content.decode("utf-8-sig")), {
        "path": str(resolved), "sha256": hashlib.sha256(content).hexdigest()}


def _identity(row: dict) -> tuple[str, str]:
    """記事IDと版を文字列の複合キーにする。

    引数: row (dict) は記事または本人の確認指定。
    返り値: tuple[str, str] はIDと版。欠損や真偽値を拒否する。
    """
    if not isinstance(row, dict):
        raise ValueError("記事指定には辞書が必要です。")
    values = [row.get(key) for key in ("id", "revision_id")]
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip()
           for value in values):
        raise ValueError("記事IDと版を明示してください。")
    return tuple(str(value) for value in values)


def _headlines(metadata: dict | list, cutoff: str) -> list[dict]:
    """見出しメタデータから許可された10項目だけを取り出す。

    引数: metadata (dict | list) はnews/articles/headlines配列または配列自体、cutoff (str) は締切。
    返り値: list[dict] は版の重複と未来時計を検査した見出し。本文・抜粋は含めない。
    """
    if isinstance(metadata, dict):
        collections = [metadata[key] for key in ("news", "articles", "headlines") if key in metadata]
        if len(collections) != 1:
            raise ValueError("見出し配列を一つだけ明示してください。")
        rows = collections[0]
    else:
        rows = metadata
    if not isinstance(rows, list):
        raise ValueError("見出しは配列が必要です。")
    result, seen = [], set()
    for row in rows:
        identity = _identity(row)
        if identity in seen:
            raise ValueError("見出しの同一ID・版が重複しています。")
        seen.add(identity)
        for key in HEADLINE_FIELDS:
            value = row.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
                raise ValueError("見出し許可項目へ本文などの入れ子構造を含められません。")
        for key in ("published_at", "fetched_at", "effective_at", "available_at", "updated_at", "retrieved_at"):
            if row.get(key) is not None and timestamp(row[key]) > timestamp(cutoff):
                raise ValueError("締切後の見出し時計は使えません。")
        result.append({key: copy.deepcopy(row[key]) for key in HEADLINE_FIELDS if key in row})
    return result


def _confirmed(values: list, rows: list[dict]) -> list[dict]:
    """本人が確認した全IDを、見出し集合の全件と照合する。

    引数: values (list) はIDまたはID・版辞書の明示一覧、rows (list[dict]) は全見出し。
    返り値: list[dict] は確認指定のID・版。同ID別版があるときは辞書指定を必須とする。
    """
    if not isinstance(values, list):
        raise ValueError("confirmed_headline_idsは本人の明示リストが必要です。")
    by_id = {}
    expected = {_identity(row) for row in rows}
    for identity in expected:
        by_id.setdefault(identity[0], []).append(identity)
    confirmed = []
    for value in values:
        if isinstance(value, dict):
            identity = _identity(value)
        elif not isinstance(value, bool) and isinstance(value, (str, int)):
            matches = by_id.get(str(value), [])
            if len(matches) != 1:
                raise ValueError("不明なIDまたは同ID別版です。IDと版の複合指定が必要です。")
            identity = matches[0]
        else:
            raise ValueError("確認IDはIDまたはID・版の辞書が必要です。")
        confirmed.append(identity)
    if len(set(confirmed)) != len(confirmed) or set(confirmed) != expected:
        raise ValueError("本人の確認IDが見出し全件と一致しません。all_confirmedは作れません。")
    return [{"id": identity[0], "revision_id": identity[1]} for identity in confirmed]


def _prices(instruments: dict, cutoff: str) -> dict:
    """packet内の可知価格だけを許可項目で複製する。

    引数: instruments (dict) はpacketの先物別状態、cutoff (str) は締切。
    返り値: dict は基準・直近場・確定時系列。外部価格を開かず、未入力価格を補完しない。
    """
    if not isinstance(instruments, dict) or any(not isinstance(state, dict) for state in instruments.values()):
        raise ValueError("packetの先物別状態は辞書が必要です。")
    result = {}
    for symbol, state in instruments.items():
        fixed = {}
        for collection, fields in PRICE_FIELDS.items():
            if collection not in state:
                continue
            value = state[collection]
            if value is None:
                fixed[collection] = None
                continue
            rows = [value] if collection == "reference" else value
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("packetの価格形式が不正です。")
            selected = []
            for row in rows:
                if row.get("closed") is False:
                    raise ValueError("packetの未確定価格は複製できません。")
                for clock in ("time", "end", "first_time", "last_time"):
                    if row.get(clock) is not None and timestamp(row[clock]) > timestamp(cutoff):
                        raise ValueError("packetの未来価格は複製できません。")
                selected.append({key: copy.deepcopy(row[key]) for key in fields if key in row})
            fixed[collection] = selected[0] if collection == "reference" else selected
        result[symbol] = fixed
    return result


def assemble_actor(packet_path: str | Path, *, actor_name: str, headline_metadata_path: str | Path,
                   confirmed_headline_ids: list, read_scopes: list[dict], curation_manifest_path: str | Path,
                   feedback_snapshot_path: str | Path, attestations: dict, analysis: dict) -> dict:
    """本人の明示入力を担当記録と機械的な出典記録へ組み立てる。

    引数: packet_path/headline_metadata_path/curation_manifest_path/feedback_snapshot_path (str | Path) は指定入力、
        actor_name (str) は本人が指定する担当名、confirmed_headline_ids (list) は確認した全IDと必要なら版、
        read_scopes (list[dict]) は本人が読んだpath/scopeのみ、attestations (dict) は必須6項目を含む本人bool、
        analysis (dict) はREQUIRED_ANALYSIS全項目を含む本人解釈。空配列や不明も本人が明示する。
    返り値: dict はactorとsource_records。判断・読了・安全性を推測せず、本文を返さず、保存しない。
    """
    if not isinstance(actor_name, str) or not actor_name.strip():
        raise ValueError("担当名を明示してください。")
    if (not isinstance(attestations, dict) or set(REQUIRED_ATTESTATIONS) - set(attestations)
            or any(type(value) is not bool for value in attestations.values())):
        raise ValueError("必須の本人申告を全てboolで明示してください。既定値はありません。")
    if not isinstance(analysis, dict) or set(REQUIRED_ANALYSIS) - set(analysis):
        raise ValueError("本人の分析欄を全て明示してください。解釈は補完しません。")
    digest(analysis)
    packet, packet_record = _load(packet_path)
    if not isinstance(packet, dict) or packet.get("packet_hash") != digest({
            key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("packetの自己ハッシュが一致しません。")
    cutoff = packet["as_of"]
    timestamp(cutoff)
    metadata, metadata_record = _load(headline_metadata_path)
    headlines = _headlines(metadata, cutoff)
    confirmed = _confirmed(confirmed_headline_ids, headlines)
    visible = {_identity(row): row for row in headlines}
    for row in packet["news"]:
        identity = _identity(row)
        if identity not in visible or any(row.get(key) != visible[identity].get(key) for key in HEADLINE_FIELDS):
            raise ValueError("packetと見出しメタデータの記事版が一致しません。")

    # ----------------------------------------

    manifest, manifest_record = _load(curation_manifest_path)
    snapshot, snapshot_record = _load(feedback_snapshot_path)
    for document, label in ((manifest, "本文選別台帳"), (snapshot, "反応snapshot")):
        if not isinstance(document, dict):
            raise ValueError(label + "は辞書が必要です。")
        if document.get("as_of") is not None and timestamp(document["as_of"]) > timestamp(cutoff):
            raise ValueError(label + "の締切がpacketより未来です。")
    for document, hash_key in ((manifest, "manifest_hash"), (snapshot, "snapshot_hash")):
        if hash_key in document and document[hash_key] != digest({
                key: value for key, value in document.items() if key != hash_key}):
            raise ValueError("補助入力の自己ハッシュが一致しません。")
    if not isinstance(manifest.get("records"), list) or not isinstance(snapshot.get("cases"), list):
        raise ValueError("本文選別台帳のrecordsと反応snapshotのcasesは配列が必要です。")
    manifest_record.update(record_count=len(manifest["records"]))
    for key in ("as_of", "candidate_sha256", "manifest_hash"):
        if key in manifest:
            if not isinstance(manifest[key], str):
                raise ValueError("本文選別台帳のヘッダは文字列が必要です。")
            manifest_record[key] = copy.deepcopy(manifest[key])
    for key in SNAPSHOT_HEADERS:
        if key == "independent_sample_count" and snapshot.get(key) is None:
            continue
        if key in snapshot and (isinstance(snapshot[key], bool) or not isinstance(snapshot[key], (str, int))):
            raise ValueError("反応snapshotのヘッダに入れ子構造を含められません。")
    snapshot_record.update({key: copy.deepcopy(snapshot[key]) for key in SNAPSHOT_HEADERS if key in snapshot})
    snapshot_record["case_count"] = len(snapshot["cases"])
    for key in ("session_count", "instrument_observation_count", "independent_sample_count"):
        if key == "independent_sample_count" and snapshot_record.get(key) is None:
            continue
        if key in snapshot_record and (type(snapshot_record[key]) is not int or snapshot_record[key] < 0):
            raise ValueError("反応snapshotの件数は非負整数が必要です。")
    if not isinstance(read_scopes, list):
        raise ValueError("read_scopesは本人が読んだpath/scopeの配列が必要です。")
    read_files = []
    for row in read_scopes:
        if (not isinstance(row, dict) or set(row) != {"path", "scope"}
                or not isinstance(row["scope"], str) or not row["scope"].strip()):
            raise ValueError("読んだpathと空でないscopeだけを明示してください。")
        path = Path(row["path"]).resolve()
        read_files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "scope": row["scope"]})

    # ----------------------------------------

    prices = _prices(packet["instruments"], cutoff)
    actor = {"actor_name": actor_name, "author_kind": "本人明示入力の組立", "packet_hash": packet["packet_hash"],
             "as_of": cutoff, "session": packet["session"], "attestations": copy.deepcopy(attestations),
             "analysis": copy.deepcopy(analysis), "read_files": read_files,
             "headline_review": {"count": len(headlines), "all_confirmed": True, "confirmed_ids": confirmed,
                 "batches": copy.deepcopy(analysis["headline_batches"]), "fields": list(HEADLINE_FIELDS),
                 "all_headlines": headlines, "dedup_groups": copy.deepcopy(analysis["dedup_groups"]),
                 "new_vs_repeat": copy.deepcopy(analysis["new_vs_repeat"])},
             "curation_manifest": copy.deepcopy(manifest_record),
             "reaction_snapshot": {**copy.deepcopy(snapshot_record),
                                   "case_comparisons": copy.deepcopy(analysis["case_comparisons"])},
             "p0_and_current_location": {
                 "references": {symbol: copy.deepcopy(state["reference"]) for symbol, state in prices.items()
                                if "reference" in state},
                 "recent_sessions": {symbol: copy.deepcopy(state["recent_sessions"]) for symbol, state in prices.items()
                                     if "recent_sessions" in state},
                 "last_windows": {symbol: copy.deepcopy(state["price_timeline"]) for symbol, state in prices.items()
                                  if "price_timeline" in state}, "reading": copy.deepcopy(analysis["p0_reading"])}}
    actor.update({key: attestations[key] for key in REQUIRED_ATTESTATIONS})
    if "selected_ids" in analysis:
        actor["headline_review"]["selected_ids"] = copy.deepcopy(analysis["selected_ids"])
    records = {"packet": packet_record, "headline_metadata": metadata_record, "curation_manifest": manifest_record,
               "feedback_snapshot": snapshot_record, "packet_hash": packet["packet_hash"],
               "as_of": cutoff, "headline_count": len(headlines), "read_scope_count": len(read_files),
               "actor_hash": digest(actor), "notice": "ファイル照合記録。本人の読了・申告の真実性や安全性は認定しない。"}
    records["record_hash"] = digest(records)
    return {"actor": actor, "source_records": records}
