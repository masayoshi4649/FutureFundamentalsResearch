"""実測反応とニュース特徴から、満了済み履歴だけで３方向を縮小分類する。"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev

from .common import digest, timestamp
from .model import solve


OBSERVED_FEATURES = (
    "previous_return_pct", "close_location_centered", "range_pct",
    "late_change_pct", "relative_return_pct", "is_night",
)
MODE_FEATURES = {
    "observed": OBSERVED_FEATURES,
    "raw_news": OBSERVED_FEATURES + ("growth", "discount", "fx", "cost_risk"),
    "market_news": OBSERVED_FEATURES + (
        "us_nasdaq_return_pct", "us_dow_return_pct", "usdjpy_change_pct",
    ),
}
DIRECTIONS = ("上昇", "下落", "横ばい")
SYMBOLS = ("NK225", "TOPIX")
MINIMUM = 8
PENALTY = 4.0
BOUNDARY = 0.05
MISSING_SUFFIX = "__missing"


def _number(value: object, label: str, allow_missing: bool = False) -> float | None:
    """欠測と有限の実数を区別して入力を検証する。

    引数: value (object) は入力値、label (str) は項目名、allow_missing (bool) は欠測許可。
    返り値: float | None は有限の数値または許可された欠測。
    """
    if value is None and allow_missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} は有限の数値または許可された欠測が必要です。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} は有限の数値が必要です。")
    return result


def _features(value: dict, kind: str, keys: tuple[str, ...]) -> dict:
    """方式別の固定特徴だけを検証し、欠測をそのまま保存する。

    引数: value (dict) は全方式を含み得る特徴、kind (str) は対象場、keys (tuple[str, ...]) は採用列。
    返り値: dict は採用列の有限値またはNone。採用しない列は解釈しない。
    """
    if kind not in ("Day", "Night"):
        raise ValueError("場種別kindはDayまたはNightが必要です。")
    if not isinstance(value, dict):
        raise ValueError("特徴featuresは辞書が必要です。")
    result = {key: _number(value.get(key), key, allow_missing=True) for key in keys}
    expected_night = float(kind == "Night")
    if result["is_night"] is not None and result["is_night"] != expected_night:
        raise ValueError("場種別kindと特徴is_nightが一致しません。")
    return result


def _direction(value: float) -> str:
    """固定した騰落率境界で教師方向を分類する。

    引数: value (float) は実績の騰落率％。
    返り値: str は上昇、下落または横ばい。境界ちょうどは横ばい。
    """
    return "上昇" if value > BOUNDARY else "下落" if value < -BOUNDARY else "横ばい"


def _select_direction(scores: dict, counts: dict) -> str:
    """最大クラススコアを選び、同数時は過去多数と固定順で解決する。

    引数: scores (dict) は方向別スコア、counts (dict) は過去の方向件数。
    返り値: str は選択した方向。スコアを確率として扱わない。
    """
    highest = max(scores.values())
    candidates = [direction for direction in DIRECTIONS if scores[direction] == highest]
    most = max(counts[direction] for direction in candidates)
    return next(direction for direction in candidates if counts[direction] == most)


def _prepare_columns(rows: list[dict], current: dict, keys: tuple[str, ...]) -> tuple:
    """過去だけで中央値補完と標準化を行い、欠測フラグを独立列にする。

    引数: rows (list[dict]) は同先物の過去履歴、current (dict) は今回の特徴、keys (tuple[str, ...]) は採用列。
    返り値: tuple は学習行列、今回ベクトル、列名、特徴統計、学習域外列、全欠測からの新観測列。
    """
    columns, statistics, outside, newly_observed = {}, {}, [], []
    expanded = tuple(item for key in keys for item in (key, key + MISSING_SUFFIX))
    for key in keys:
        observed = [row["features"][key] for row in rows if row["features"][key] is not None]
        middle = median(observed) if observed else None
        fill = middle if middle is not None else 0.0
        imputed = [row["features"][key] if row["features"][key] is not None else fill for row in rows]
        missing = [float(row["features"][key] is None) for row in rows]
        current_missing = current[key] is None
        current_imputed = fill if current_missing else current[key]
        low, high = (min(observed), max(observed)) if observed else (None, None)
        out_of_range = bool(observed and not current_missing and (current[key] < low or current[key] > high))
        if out_of_range:
            outside.append(key)
        if not observed and not current_missing:
            newly_observed.append(key)
        for name, values, present in (
            (key, imputed, current_imputed),
            (key + MISSING_SUFFIX, missing, float(current_missing)),
        ):
            average = mean(values)
            deviation = pstdev(values)
            scale = deviation if deviation > 0 else 1.0
            standardized = [(value - average) / scale for value in values]
            current_standardized = (present - average) / scale
            if not all(math.isfinite(value) for value in standardized + [average, deviation, current_standardized]):
                raise ValueError(f"{name} の標準化結果が有限値ではありません。")
            columns[name] = {"history": standardized, "current": current_standardized}
            statistics[name] = {
                "mean": average, "std": deviation, "scale": scale,
                "constant": deviation == 0, "train_min": min(values), "train_max": max(values),
                "current_imputed": present, "current_standardized": current_standardized,
            }
        statistics[key].update(
            observed_n=len(observed), missing_n=len(rows) - len(observed),
            median=middle, imputation_value=fill, all_missing=not observed,
            observed_min=low, observed_max=high, current_value=current[key],
            current_missing=current_missing, out_of_range=out_of_range,
        )
        missing_key = key + MISSING_SUFFIX
        statistics[missing_key]["out_of_range"] = (
            float(current_missing) < min(missing) or float(current_missing) > max(missing)
        )
        if statistics[missing_key]["out_of_range"]:
            outside.append(missing_key)

    # ----------------------------------------

    matrix = [[columns[key]["history"][index] for key in expanded] for index in range(len(rows))]
    vector = [columns[key]["current"] for key in expanded]
    return matrix, vector, expanded, statistics, outside, newly_observed


def predict_measured(current: dict, history: list[dict], as_of: str, mode: str = "observed") -> dict:
    """満了済み同先物履歴から、固定特徴の３クラス縮小最小二乗分類を行う。

    引数: current (dict) はsymbol・kind・features、history (list[dict]) は同項目とactual_pct・label_end、
    as_of (str) は時差付き情報締切、mode (str) はobserved、raw_newsまたはmarket_news。
    返り値: dict は方向、未校正のクラススコア、過去だけの前処理統計、係数・寄与と学習証跡。
    全履歴の満了時刻を最初に検査し、締切以降に満了する値を特徴・実績の解釈前に拒否する。
    """
    cutoff = timestamp(as_of)
    if not isinstance(history, list):
        raise ValueError("学習履歴は配列が必要です。")
    timed = []
    for row in history:
        if not isinstance(row, dict) or "label_end" not in row:
            raise ValueError("各学習行には満了時刻label_endが必要です。")
        end = timestamp(row["label_end"])
        if end >= cutoff:
            raise ValueError("情報締切以降に満了するラベルは学習へ渡せません。")
        timed.append((end, row))

    # ----------------------------------------

    if mode not in MODE_FEATURES:
        raise ValueError("方式はobserved、raw_news、market_newsを指定してください。")
    if not isinstance(current, dict) or current.get("symbol") not in SYMBOLS:
        raise ValueError("今回入力には対象先物symbolが必要です。")
    if "as_of" in current and timestamp(current["as_of"]) != cutoff:
        raise ValueError("今回入力と引数の情報締切as_ofが一致しません。")
    symbol, kind = current["symbol"], current.get("kind")
    keys = MODE_FEATURES[mode]
    features = _features(current.get("features"), kind, keys)
    selected, seen = [], set()
    for end, row in timed:
        if row.get("symbol") not in SYMBOLS:
            raise ValueError("各学習行には対象先物symbolが必要です。")
        if row["symbol"] != symbol:
            continue
        identity = end.isoformat()
        if identity in seen:
            raise ValueError("同じ先物・同じ評価終点の学習ラベルは重複できません。")
        seen.add(identity)
        if "as_of" in row and timestamp(row["as_of"]) >= end:
            raise ValueError("学習行の情報締切as_ofは満了時刻label_endより前が必要です。")
        actual = _number(row.get("actual_pct"), "実績騰落率")
        selected.append({
            "symbol": symbol, "kind": row.get("kind"), "label_end": identity,
            "features": _features(row.get("features"), row.get("kind"), keys),
            "actual_pct": actual, "direction": _direction(actual),
        })
    selected.sort(key=lambda row: timestamp(row["label_end"]))
    counts = {direction: sum(row["direction"] == direction for row in selected) for direction in DIRECTIONS}
    hashes = [digest(row) for row in selected]
    result = {
        "model": "実測反応・３方向縮小分類", "variant": mode, "mode": "回想・開発再生",
        "as_of": cutoff.isoformat(), "symbol": symbol, "kind": kind,
        "direction": "保留", "class_scores": {direction: None for direction in DIRECTIONS},
        "prediction_pct": None, "feature_keys": list(keys), "features": features,
        "expanded_feature_keys": [], "expanded_features": {}, "feature_stats": {},
        "out_of_range_features": [], "newly_observed_features": [],
        "training_n": len(selected), "fitted_n": 0, "minimum": MINIMUM,
        "penalty": PENALTY, "direction_boundary": BOUNDARY, "boundary_unit": "実績騰落率％",
        "cold_start": len(selected) < MINIMUM, "training_counts": counts,
        "coefficients": {}, "contributions": {},
        "coefficient_basis": "過去だけで標準化した原特徴・欠測フラグ。interceptは非罰則。",
        "training_label_hashes": hashes, "training_labels_hash": digest(hashes),
        "training_last_label_end": selected[-1]["label_end"] if selected else None,
        "teacher": "実績騰落率を±0.05％で３方向に分類したonehot値",
        "score_semantics": "クラススコアは未校正。確率ではなく、推定後は０未満・１超も許容する。",
        "tie_break": "最大スコアの同数時は過去多数方向、なお同数なら上昇・下落・横ばい順",
        "history_time_contract": "label_endのみを締切未満へ制約。受信・評価記録時刻は未検証。",
        "statistical_advantage": "未検証", "status": "履歴なしのため保留",
    }
    if not selected:
        return result

    # ----------------------------------------

    matrix, vector, expanded, statistics, outside, newly_observed = _prepare_columns(selected, features, keys)
    result.update(
        expanded_feature_keys=list(expanded), expanded_features=dict(zip(expanded, vector)),
        feature_stats=statistics, out_of_range_features=outside, newly_observed_features=newly_observed,
    )
    if len(selected) < MINIMUM:
        scores = {direction: counts[direction] / len(selected) for direction in DIRECTIONS}
        result.update(
            direction=_select_direction(scores, counts), class_scores=scores,
            status="少数標本のため過去最多方向を使用・係数未学習",
            score_semantics="少数標本のクラススコアは過去構成比。予測確率として校正していない。",
        )
        return result

    # ----------------------------------------

    size = len(expanded)
    gram = [[math.fsum(row[i] * row[j] for row in matrix) + (PENALTY if i == j else 0.0)
             for j in range(size)] for i in range(size)]
    scores, coefficients, contributions = {}, {}, {}
    for direction in DIRECTIONS:
        base = counts[direction] / len(selected)
        teachers = [float(row["direction"] == direction) - base for row in selected]
        rhs = [math.fsum(row[index] * teacher for row, teacher in zip(matrix, teachers)) for index in range(size)]
        slopes = solve(gram, rhs)
        coefficients[direction] = {"intercept": base, **dict(zip(expanded, slopes))}
        contributions[direction] = {
            "intercept": base, **{key: slope * value for key, slope, value in zip(expanded, slopes, vector)},
        }
        scores[direction] = math.fsum(contributions[direction].values())
        if not math.isfinite(scores[direction]):
            raise ValueError("クラススコアが有限値ではありません。")
    result.update(
        direction=_select_direction(scores, counts), class_scores=scores,
        fitted_n=len(selected), coefficients=coefficients, contributions=contributions,
        status="探索的３方向分類・優位性未検証",
    )
    return result
