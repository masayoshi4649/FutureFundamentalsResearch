"""材料別の点数と、過去に満了した予測からの縮小推定。"""

from __future__ import annotations

import math
from datetime import datetime

from .common import finite, timestamp


def material_features(events: list[dict], articles: list[dict], as_of: datetime,
                      target: datetime, symbol: str, mode: str,
                      issued_at: datetime | None = None) -> tuple[dict, list, list]:
    """時点で利用できる材料から銘柄別の残存点数を作る。

    引数: events (list[dict]) は注釈、articles (list[dict]) は可視記事、as_of/target
    (datetime) は判断と満期、symbol (str) は銘柄、mode (str) は回想または運用、
    issued_at (datetime | None) は発行時刻。運用の注釈処理時間だけを許容する。
    返り値: tuple[dict, list, list] は分類別特徴量、材料内訳、除外理由。
    """
    visible = {(str(a["id"]), a["revision_id"]): a for a in articles}
    features, details, excluded, used = {}, [], [], set()
    for event in sorted(events, key=lambda e: (timestamp(e["known_at"]), timestamp(e["annotated_at"]), e["event_id"]), reverse=True):
        identity = event["event_id"]
        known = timestamp(event["known_at"])
        annotated = timestamp(event["annotated_at"])
        evidence = event["evidence"]
        if not evidence or any((str(e["id"]), e["revision_id"]) not in visible for e in evidence):
            excluded.append({"event_id": identity, "reason": "根拠記事の指定版が判断時点に存在しない"})
            continue
        available = max([known] + [timestamp(visible[(str(e["id"]), e["revision_id"])]["available_at"]) for e in evidence])
        if available > as_of or (mode == "運用" and annotated > (issued_at or as_of)):
            excluded.append({"event_id": identity, "reason": "材料または注釈が判断後に利用可能"})
            continue
        if event["story_key"] in used:
            excluded.append({"event_id": identity, "reason": "同一材料の重複"})
            continue
        used.add(event["story_key"])
        confidence = finite(event["confidence"], 0, 1, "確信度")
        novelty = finite(event["novelty"], 0, 1, "新規性")
        priced = finite(event["priced_in"], 0, 1, "織込度")
        half_life = finite(event["half_life_days"], 0.01, 3650, "半減期")
        impact = finite(event["impacts"][symbol], -1, 1, "銘柄別方向点数")
        age = max(0, (as_of - known).total_seconds() / 86400)
        future = max(0, (target - as_of).total_seconds() / 86400)
        score = impact * confidence * novelty * (1 - priced) * 2 ** (-age / half_life)
        key = event["category"] + "／" + event["regime"]
        remaining = score * 2 ** (-future / half_life)
        features[key] = features.get(key, 0) + remaining
        details.append({"event_id": identity, "story_key": event["story_key"], "category": event["category"],
                        "regime": event["regime"], "score": score, "remaining_score": remaining,
                        "evidence": evidence, "reason": event["reason"],
                        "annotation_retrospective": mode == "回想" and annotated > as_of})
    return {key: max(-3, min(3, value)) for key, value in features.items()}, details, excluded


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """小さい連立一次方程式を部分ピボットで解く。

    引数: matrix (list[list[float]]) は正則係数、vector (list[float]) は右辺。
    返り値: list[float] は解。特異の場合は例外。
    """
    size = len(vector)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for index in range(size):
        pivot = max(range(index, size), key=lambda row: abs(augmented[row][index]))
        augmented[index], augmented[pivot] = augmented[pivot], augmented[index]
        scale = augmented[index][index]
        if abs(scale) < 1e-12:
            raise ValueError("推定行列を解けません。")
        augmented[index] = [value / scale for value in augmented[index]]
        for row in range(size):
            if row != index:
                ratio = augmented[row][index]
                augmented[row] = [a - ratio * b for a, b in zip(augmented[row], augmented[index])]
    return [row[-1] for row in augmented]


def fit_predict(rows: list[dict], features: dict, minimum: int = 12, penalty: float = 4.0) -> dict:
    """満了済み過去例の材料点数から寄与率を縮小推定する。

    引数: rows (list[dict]) は features と actual_pct、features (dict) は今回の点数、
    minimum (int) は最低事例数、penalty (float) は固定した縮小強度。
    返り値: dict は推定状態、騰落率と分類別寄与。推定は因果効果を意味しない。
    """
    keys = sorted({key for row in rows for key in row["features"]})
    result = {"n": len(rows), "minimum": minimum, "penalty": penalty,
              "prediction_pct": None, "contributions_pct": {}, "status": "標本不足",
              "unseen_categories": sorted(set(features) - set(keys))}
    if any(abs(features[key]) > 1e-12 for key in result["unseen_categories"]):
        result["status"] = "未学習の材料分類あり"
        return result
    result["category_observations"] = {key: sum(abs(row["features"].get(key, 0)) > 1e-12 for row in rows) for key in keys}
    if any(abs(features.get(key, 0)) > 1e-12 and result["category_observations"][key] < 3 for key in keys):
        result["status"] = "分類別の有効標本不足"
        return result
    if len(rows) < max(minimum, len(keys) * 3) or not keys:
        return result
    vectors = [[row["features"].get(key, 0.0) for key in keys] for row in rows]
    matrix = [[sum(v[i] * v[j] for v in vectors) + (penalty if i == j else 0)
               for j in range(len(keys))] for i in range(len(keys))]
    rhs = [sum(v[i] * row["actual_pct"] for v, row in zip(vectors, rows)) for i in range(len(keys))]
    coefficients = solve(matrix, rhs)
    contributions = {key: coefficient * features.get(key, 0) for key, coefficient in zip(keys, coefficients)}
    result.update(prediction_pct=sum(contributions.values()), contributions_pct=contributions,
                  coefficients_pct=dict(zip(keys, coefficients)), status="探索的推定")
    return result


def wilson_lower(hits: int, total: int) -> float | None:
    """方向一致率の参考下側信頼限界を求める。

    引数: hits (int) は一致数、total (int) は標本数。
    返り値: float | None は両側９５％区間の下端。独立性は別途検証が必要。
    """
    if not total:
        return None
    z = 1.96
    rate = hits / total
    return (rate + z*z/(2*total) - z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total))) / (1+z*z/total)
