"""回想研究専用の固定規則と、満了履歴だけによる逐次推定。

履歴の利用条件はlabel_endだけであり、available_atやsimulated_available_atによる
当時の受信・評価記録を再現しない。前向き運用の学習履歴とは区別する。
"""

from __future__ import annotations

import math

from .common import digest, finite, timestamp
from .model import solve


CHANNELS = ("growth", "discount", "fx", "cost_risk")
FEATURES = CHANNELS + ("previous_return", "is_night", "intercept")
SYMBOLS = ("NK225", "TOPIX")
PENALTY = 4.0
MINIMUM = 8
BOUNDARY = 0.05
PRIOR = {key: 0.25 if key in CHANNELS else 0.0 for key in FEATURES}


def _direction(value: float) -> str:
    """固定境界によって上昇、下落、横ばいを判定する。

    引数: value (float) は定性点数、または明示した単位の推定騰落率。
    返り値: str は上昇、下落、横ばいのいずれか。
    """
    return "上昇" if value > BOUNDARY else "下落" if value < -BOUNDARY else "横ばい"


def _features(value: dict) -> dict:
    """固定した数値特徴だけを検証する。

    引数: value (dict) は４材料経路、直前騰落、夜間フラグ、定数項。
    返り値: dict は固定順序の検証済み有限値。不正な入力は例外。
    """
    if not isinstance(value, dict) or set(value) != set(FEATURES):
        raise ValueError("特徴量は固定した７項目だけを指定してください。")
    result = {key: finite(value[key], -1, 1, key) for key in FEATURES}
    if result["is_night"] not in (0.0, 1.0) or result["intercept"] != 1.0:
        raise ValueError("夜間フラグは０または１、定数項は１で指定してください。")
    return result


def feature_vector(annotation: dict, packet: dict, symbol: str) -> dict:
    """回想研究でラベルを受け取らず、可視注釈と直前先物から固定特徴を作る。

    引数: annotation (dict) は指数別の４経路注釈、packet (dict) は締切付き可視入力、
    symbol (str) は NK225 または TOPIX。根拠記事と注釈時刻は呼出側で監査する。
    返り値: dict は信頼度・新規性・未織込度を掛けた４点数、直前騰落／３、夜間フラグ、定数項。
    """
    if symbol not in SYMBOLS:
        raise ValueError("対象先物はNK225またはTOPIXです。")
    as_of = timestamp(packet["as_of"])
    if packet.get("kind") not in ("Day", "Night"):
        raise ValueError("セッション種別はDayまたはNightが必要です。")
    channels = annotation["instruments"][symbol]["channels"]
    if not isinstance(channels, dict) or set(channels) != set(CHANNELS):
        raise ValueError("材料経路はgrowth、discount、fx、cost_riskの４項目が必要です。")
    result = {}
    for channel in CHANNELS:
        item = channels[channel]
        if not isinstance(item, dict):
            raise ValueError("各材料経路には構造化した採点が必要です。")
        score = finite(item["score"], -1, 1, "材料点数")
        confidence = finite(item["confidence"], 0, 1, "確信度")
        novelty = finite(item["novelty"], 0, 1, "新規性")
        priced_in = finite(item["priced_in"], 0, 1, "織込度")
        result[channel] = score * confidence * novelty * (1 - priced_in)

    # ----------------------------------------

    instrument = packet["instruments"][symbol]
    reference = instrument["reference"]
    if timestamp(reference["time"]) > as_of:
        raise ValueError("基準先物価格が情報締切より後です。")
    recent = instrument["recent_sessions"]
    if not recent or recent[-1]["session"] != reference["session"]:
        raise ValueError("直前セッションの先物騰落率が基準価格と一致しません。")
    previous_return = finite(recent[-1]["return_pct"], -math.inf, math.inf, "直前先物騰落率")
    result.update(previous_return=max(-1.0, min(1.0, previous_return / 3)),
                  is_night=float(packet["kind"] == "Night"), intercept=1.0)
    return _features(result)


def predict_rule(annotation: dict, packet: dict, symbol: str) -> dict:
    """回想研究で４材料経路の平均による固定した定性方向を全件発行する。

    引数: annotation (dict) は固定済み材料注釈、packet (dict) は可視入力、
    symbol (str) は対象先物。入力に評価ラベルは必要ない。
    返り値: dict は定性方向、点数、特徴、固定係数、学習件数０と未実証の表示。
    """
    features = feature_vector(annotation, packet, symbol)
    score = sum(features[key] for key in CHANNELS) / len(CHANNELS)
    return {"model": "固定材料平均", "mode": "回想・開発再生", "symbol": symbol,
            "as_of": timestamp(packet["as_of"]).isoformat(),
            "direction": _direction(score), "score": score, "score_unit": "定性材料点数",
            "prediction_pct": None, "features": features, "coefficients": dict(PRIOR),
            "direction_boundary": BOUNDARY, "boundary_unit": "定性材料点数",
            "training_n": 0, "eligible_n": 0, "fitted_n": 0,
            "training_label_hashes": [], "training_labels_hash": digest([]),
            "status": "固定規則・優位性未検証", "statistical_advantage": "未検証",
            "annotation_status": "材料点数・新規性・織込度は未校正の注釈判断",
            "history_time_contract": "回想専用。label_endのみ制約し、当時の受信・評価記録時刻は未検証",
            "range_model": "価格だけの経験分布は別処理"}


def _history_rows(history: list[dict], packet: dict, symbol: str) -> list[dict]:
    """締切までに満了した同じ先物の学習行を検証して順序を固定する。

    引数: history (list[dict]) はsymbol、features、actual_pct、label_endを持つ履歴、
    packet (dict) は情報締切、symbol (str) は今回の対象先物。
    返り値: list[dict] は同先物の正規化行。未来ラベルや重複終点は例外。
    available_atとsimulated_available_atは契約に含めず、前向き運用の再現には使わない。
    """
    if not isinstance(history, list):
        raise ValueError("学習履歴は配列が必要です。")
    as_of = timestamp(packet["as_of"])
    selected, seen = [], set()
    for row in history:
        if not isinstance(row, dict) or row.get("symbol") not in SYMBOLS:
            raise ValueError("各学習行には対象先物symbolが必要です。")
        label_end = timestamp(row["label_end"])
        if label_end > as_of:
            raise ValueError("情報締切より後に満了するラベルは学習へ渡せません。")
        if row["symbol"] != symbol:
            continue
        key = label_end.isoformat()
        if key in seen:
            raise ValueError("同じ先物・同じ評価終点の学習ラベルは重複できません。")
        seen.add(key)
        selected.append({"symbol": symbol, "features": _features(row["features"]),
                         "actual_pct": finite(row["actual_pct"], -math.inf, math.inf, "実績騰落率"),
                         "label_end": key})
    return sorted(selected, key=lambda row: timestamp(row["label_end"]))


def _predict_analog(result: dict, rows: list[dict], packet: dict) -> dict:
    """近い過去５類例の時間減衰した方向重みから参考方向を選ぶ。

    引数: result (dict) は固定規則と共通履歴情報、rows (list[dict]) は検証済み同先物履歴、
    packet (dict) は今回締切。８件以上の適格履歴がある場合だけ呼び出す。
    返り値: dict は近傍・方向別重み・参考重み比を追加した係数未使用の予測。
    """
    current = result["features"]
    candidates = []
    for row in rows:
        distance = sum((current[key] - row["features"][key]) ** 2 for key in CHANNELS) / len(CHANNELS)
        distance += (current["previous_return"] - row["features"]["previous_return"]) ** 2
        distance += (current["is_night"] - row["features"]["is_night"]) ** 2
        candidates.append((distance, row))
    candidates.sort(key=lambda item: (item[0], -timestamp(item[1]["label_end"]).timestamp()))
    weights = {direction: 0.0 for direction in ("上昇", "下落", "横ばい")}
    analogs = []
    for distance, row in candidates[:5]:
        age = (timestamp(packet["as_of"]).date() - timestamp(row["label_end"]).date()).days
        weight = math.exp(-distance) * 2 ** (-age / 7)
        direction = _direction(row["actual_pct"])
        weights[direction] += weight
        analogs.append({"symbol": row["symbol"], "label_end": row["label_end"],
                        "actual_pct": row["actual_pct"], "direction": direction, "distance": distance,
                        "age_calendar_days": age, "weight": weight, "hash": digest(row)})
    highest = max(weights.values())
    tied = [direction for direction, weight in weights.items() if weight == highest]
    direction = result["direction"] if result["direction"] in tied else tied[0]
    total = sum(weights.values())
    result.update(direction=direction, score=None, score_unit="方向別の参考重みを比較・定性点数なし",
                  prediction_pct=None, coefficients={}, prior_coefficients={}, contributions_pct={},
                  direction_boundary=BOUNDARY, boundary_unit="過去実績の騰落率％",
                  analogs=analogs, analogs_n=len(analogs), direction_weights=weights,
                  direction_weight_shares={key: value / total if total else None for key, value in weights.items()},
                  weight_share_status="少数類例の参考重み比・確率校正済みではない",
                  tie_break="固定規則が同点集合にあれば採用、他は上昇・下落・横ばい順",
                  analog_spec={"maximum": 5, "half_life_calendar_days": 7,
                               "distance": "４材料差二乗平均＋直前先物差二乗＋夜間フラグ差二乗"},
                  status="反応類例・係数未使用・少数類例による探索・優位性未検証")
    return result


def _predict_scaled(result: dict, rows: list[dict]) -> dict:
    """過去行だけで６特徴を標準化し、等価な事前係数へ縮小推定する。

    引数: result (dict) は検証済み現在特徴と共通履歴情報、rows (list[dict]) は満了済み同先物履歴。
    返り値: dict は元尺度係数・騰落率と、過去だけの標準化統計・標準化係数。
    """
    keys = FEATURES[:-1]
    means = {key: math.fsum(row["features"][key] for row in rows) / len(rows) for key in keys}
    deviations = {key: math.sqrt(math.fsum((row["features"][key] - means[key]) ** 2 for row in rows) / len(rows))
                  for key in keys}
    zero_variance = [key for key in keys if deviations[key] <= 1e-12]
    scales = {key: 1.0 if key in zero_variance else deviations[key] for key in keys}
    train_min = {key: min(row["features"][key] for row in rows) for key in keys}
    train_max = {key: max(row["features"][key] for row in rows) for key in keys}
    outside = [key for key in keys if result["features"][key] < train_min[key] or result["features"][key] > train_max[key]]
    vectors = [[(row["features"][key] - means[key]) / scales[key] for key in keys] + [1.0] for row in rows]
    teachers = [max(-1.0, min(1.0, row["actual_pct"] / 3)) for row in rows]
    prior_z = {key: PRIOR[key] * scales[key] for key in keys}
    prior_z["intercept"] = PRIOR["intercept"] + sum(PRIOR[key] * means[key] for key in keys)
    matrix = [[sum(vector[i] * vector[j] for vector in vectors) + (PENALTY if i == j else 0.0)
               for j in range(len(FEATURES))] for i in range(len(FEATURES))]
    rhs = [sum(vector[i] * teacher for vector, teacher in zip(vectors, teachers)) + PENALTY * prior_z[key]
           for i, key in enumerate(FEATURES)]
    standardized = dict(zip(FEATURES, solve(matrix, rhs)))

    # ----------------------------------------

    coefficients = {key: standardized[key] / scales[key] for key in keys}
    coefficients["intercept"] = standardized["intercept"] - sum(coefficients[key] * means[key] for key in keys)
    contributions = {key: 3 * coefficients[key] * result["features"][key] for key in FEATURES}
    prediction_pct = sum(contributions.values())
    result.update(direction=_direction(prediction_pct), score=prediction_pct / 3,
                  score_unit="正規化した探索的推定値", prediction_pct=prediction_pct,
                  coefficients=coefficients, contributions_pct=contributions,
                  direction_boundary=BOUNDARY, boundary_unit="騰落率％",
                  scaling={"means": means, "scales": scales, "zero_variance_features": zero_variance,
                           "train_min": train_min, "train_max": train_max, "out_of_range_features": outside,
                           "extrapolation_status": "学習域外の未校正の外挿" if outside else "学習範囲内・優位性未検証",
                           "trained_on_past_only": True},
                  standardized_coefficients=standardized, standardized_prior_coefficients=prior_z,
                  status="過去特徴の標準化による探索的推定・優位性未検証")
    return result


def predict_learned(annotation: dict, packet: dict, symbol: str, history: list[dict],
                    mode: str = "ridge") -> dict:
    """回想研究で満了済み同先物だけを使い、固定事前係数へ縮小推定する。

    引数: annotation (dict) は材料注釈、packet (dict) は締切付き可視入力、symbol (str)
    は先物、history (list[dict]) は満了履歴、mode (str) はridge、analog、ridge_scaled。
    返り値: dict は方向、特徴、履歴ハッシュと、ridgeの係数・探索的騰落率、
    analogの少数反応類例・未校正の方向重み比、またはridge_scaledの過去だけの標準化統計。
    ８件未満は固定規則の冷開始とし、点数を予測騰落率へ置き換えない。
    履歴はlabel_endだけを制約し、当時の受信・評価記録済み状態を保証しない。
    """
    if mode not in ("ridge", "analog", "ridge_scaled"):
        raise ValueError("推定方式はridge、analog、ridge_scaledを指定してください。")
    result = predict_rule(annotation, packet, symbol)
    rows = _history_rows(history, packet, symbol)
    label_hashes = [digest(row) for row in rows]
    result.update(model="材料・直前先物の逐次縮小推定", training_n=len(rows), eligible_n=len(rows),
                  fitted_n=len(rows) if len(rows) >= MINIMUM else 0,
                  training_label_hashes=label_hashes, training_labels_hash=digest(label_hashes),
                  minimum=MINIMUM, penalty=PENALTY, prior_coefficients=dict(PRIOR),
                  training_last_label_end=rows[-1]["label_end"] if rows else None,
                  teacher="実績騰落率／３を－１から１へ制限", cold_start=len(rows) < MINIMUM)
    if mode == "analog":
        result.update(model="反応類例・係数未使用", coefficients={}, prior_coefficients={},
                      penalty=None, teacher="過去実績の方向を±0.05％で分類",
                      analogs=[], analogs_n=0, weight_share_status="少数類例の参考重み比・確率校正済みではない")
    if mode == "ridge_scaled":
        result.update(model="材料・直前先物の標準化逐次縮小推定",
                      scaling={"means": {}, "scales": {}, "zero_variance_features": [],
                               "train_min": {}, "train_max": {}, "out_of_range_features": [],
                               "trained_on_past_only": True, "status": "冷開始のため未計算"},
                      standardized_coefficients={}, standardized_prior_coefficients={})
    if len(rows) < MINIMUM:
        result["status"] = "冷開始・固定材料平均を使用・優位性未検証"
        return result
    if mode == "analog":
        return _predict_analog(result, rows, packet)
    if mode == "ridge_scaled":
        return _predict_scaled(result, rows)

    # ----------------------------------------

    vectors = [[row["features"][key] for key in FEATURES] for row in rows]
    teachers = [max(-1.0, min(1.0, row["actual_pct"] / 3)) for row in rows]
    matrix = [[sum(vector[i] * vector[j] for vector in vectors) + (PENALTY if i == j else 0.0)
               for j in range(len(FEATURES))] for i in range(len(FEATURES))]
    rhs = [sum(vector[i] * teacher for vector, teacher in zip(vectors, teachers)) + PENALTY * PRIOR[key]
           for i, key in enumerate(FEATURES)]
    coefficients = dict(zip(FEATURES, solve(matrix, rhs)))
    contributions = {key: 3 * coefficients[key] * result["features"][key] for key in FEATURES}
    prediction_pct = sum(contributions.values())
    result.update(direction=_direction(prediction_pct), score=prediction_pct / 3,
                  score_unit="正規化した探索的推定値", prediction_pct=prediction_pct,
                  coefficients=coefficients, contributions_pct=contributions,
                  direction_boundary=BOUNDARY, boundary_unit="騰落率％",
                  status="探索的推定・優位性未検証")
    return result
