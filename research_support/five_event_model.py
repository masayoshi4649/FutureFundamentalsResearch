"""未知の臨時材料前のラベル窓に対応し、旧版の推定式を変更しない５分類モデル別版。"""

from __future__ import annotations

import math
import re
from decimal import Decimal

from fundamentals.common import digest, timestamp
from fundamentals.engine import scheduled_end, session_plan
from fundamentals.model import solve
from fundamentals.pdca_measured_model import OBSERVED_FEATURES, SYMBOLS, _features, _prepare_columns
from research_support.five_class import CLASSES, validate_contract
from research_support.event_window import _pending_applies, _validate_event


FEATURE_KEYS = OBSERVED_FEATURES + ("growth", "discount", "fx", "cost_risk",
                                    "us_nasdaq_return_pct", "us_dow_return_pct", "usdjpy_change_pct")
MODES = ("確定教師", "候補配分")
CLASSIFICATION_POLICY = "もみ合い優先・初回到達方向と最初の反転"
MINIMUM = 8
PENALTY = 4.0
BASE_MODEL_SHA256 = "db8195823debdc88d9481816594885cd0f29e7a5c484463bde6886a01a64e047"
ESTIMATION_FORMULA_SHA256 = "d169653414456c2bb9e719263844494218cb3342aaf85aade2d510f22dc162f1"


def _window(evaluation: dict, symbol: str, kind: str, label_end: str, forecast_as_of: str) -> dict:
    """評価の名目終点・材料公表・教師利用可能時計を分離して検証する。

    引数: evaluation (dict) は材料前評価、symbol/kind (str) は先物と場、label_end (str) は教師利用可能時計、
        forecast_as_of (str) は当該教師を予測した時点。
    返り値: dict は検証済み窓。公表前の終点へ教師の利用可能時刻を繰り上げない。
    """
    window = evaluation.get("evaluation_window")
    required = {"version", "status", "as_of", "scheduled_full_end", "nominal_window_end", "source_event_time",
                "scoreable", "planned_count", "selected_event", "pending_events", "adjudication_unknown", "label_available_at"}
    if not isinstance(window, dict) or not required <= set(window) or window["version"] != "臨時材料前１":
        raise ValueError("教師には臨時材料前１の評価窓と利用可能時刻が必要です。")
    session, target_end = evaluation.get("session"), evaluation.get("target_end")
    if (not isinstance(session, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}_(?:Day|Night)", session)
            or not isinstance(target_end, str) or not isinstance(forecast_as_of, str)):
        raise ValueError("経路評価には対象セッションと時差付き名目終点が必要です。")
    origin, available = timestamp(forecast_as_of), timestamp(window["label_available_at"])
    full_end, nominal = timestamp(window["scheduled_full_end"]), timestamp(window["nominal_window_end"])
    plan = session_plan(origin)
    if (evaluation.get("symbol") != symbol or not session.endswith("_" + kind) or plan["session"] != session
            or scheduled_end(session) != full_end or timestamp(target_end) != nominal
            or timestamp(label_end) != available or timestamp(window["as_of"]) != origin):
        raise ValueError("評価窓の先物・場・公式終了・名目終点・利用可能時刻が教師行と一致しません。")
    if (type(window["scoreable"]) is not bool or type(window["adjudication_unknown"]) is not bool
            or window["planned_count"] is not True or not isinstance(window["pending_events"], list)):
        raise ValueError("評価窓の採点可否・認定不明・予定回・保留一覧の指定が不正です。")
    if "maturity_required_at" in window and timestamp(window["maturity_required_at"]) != available:
        raise ValueError("評価窓の満了必須時計と教師利用可能時計が一致しません。")
    event = window["selected_event"]
    if event is None:
        if (window["source_event_time"] is not None or nominal != full_end or available != full_end
                or not window["scoreable"] or window["status"] != "全セッション"):
            raise ValueError("材料なしの教師は公式終点と同じ窓・利用可能時刻が必要です。")
    else:
        assessed, published = _validate_event(event)
        if (event["decision"] != "採用" or event["schedule_status"] != "臨時・予見不能"
                or assessed != origin or symbol not in event["affected_symbols"]
                or not origin < published <= full_end or timestamp(window["source_event_time"]) != published
                or available != published or nominal != published.replace(second=0, microsecond=0)):
            raise ValueError("採用材料の予定・判定締切・公表時計・分切下げ・利用可能時刻が一致しません。")
        expired = published <= plan["start"]
        if (window["scoreable"] == expired or nominal > full_end
                or window["status"] != ("開始前失効" if expired else "材料前まで")):
            raise ValueError("開始前失効と材料前の採点窓が不整合です。")
    for pending in window["pending_events"]:
        if not _pending_applies(pending, origin, symbol, available):
            raise ValueError("保留材料の判定締切・先物・公表時計が評価窓と一致しません。")
    if window["adjudication_unknown"] != bool(window["pending_events"]):
        raise ValueError("材料認定不明フラグと窓内の保留材料一覧が一致しません。")
    return window


def _label(evaluation: object, symbol: str, kind: str, label_end: str, forecast_as_of: str) -> dict:
    """新規約の確定正解と未確定候補を検証し、真ラベルへ読み替えず正規化する。

    引数: evaluation (object) は経路評価、symbol/kind (str) は教師の先物と場、label_end (str) は利用可能時刻、
        forecast_as_of (str) は教師の元予測時点。
    返り値: dict は確定集合・未確定集合・固定順候補と正規化契約。
    """
    if not isinstance(evaluation, dict) or evaluation.get("classification_policy") != CLASSIFICATION_POLICY:
        raise ValueError("教師はもみ合い優先・初回到達方向の新規約による評価が必要です。")
    window = _window(evaluation, symbol, kind, label_end, forecast_as_of)
    if evaluation.get("target_status") not in ("対象", "対象外", "未確定"):
        raise ValueError("経路評価の対象状態が不正です。")
    labels = []
    for name in ("correct_candidates", "uncertain_candidates"):
        values = evaluation.get(name)
        if (not isinstance(values, list) or any(not isinstance(value, str) or value not in CLASSES for value in values)
                or len(values) != len(set(values))):
            raise ValueError("確定正解・未確定候補は重複のない指定５種類の配列が必要です。")
        labels.append(set(values))
    certain, uncertain = labels
    if certain & uncertain or len(certain) > 1 or (certain and uncertain):
        raise ValueError("優先分類の確定１種類と未確定候補は混在・重複できません。")
    if not window["scoreable"] and (certain or uncertain):
        raise ValueError("開始前失効の評価に分類候補を付けられません。")
    if window["adjudication_unknown"] and (certain or evaluation["target_status"] != "未確定"):
        raise ValueError("材料認定不明の評価を確定クラスへ変換できません。")
    criteria = validate_contract(evaluation.get("contract"))
    expected = {"movement_threshold_pct": Decimal("0.5"), "initial_excursion_pct": Decimal("0.5"),
                "reversal_threshold_pct": Decimal("0.5"), "range_threshold_pct": Decimal("1"),
                "flat_terminal_threshold_pct": Decimal("0.5")}
    if (criteria.target_basis != "場中到達" or criteria.reversal_requires_reference_cross
            or any(getattr(criteria, name) != value for name, value in expected.items())):
        raise ValueError("教師の対象0.5％・反転0.5％・もみ合い1％/0.5％の規約が一致しません。")
    reported = {"certain": [name for name in CLASSES if name in certain],
                "uncertain": [name for name in CLASSES if name in uncertain]}
    exclusion = "開始前失効" if not window["scoreable"] else "材料認定不明" if window["adjudication_unknown"] else None
    if exclusion is not None:
        certain, uncertain = set(), set()
    return {"certain": [name for name in CLASSES if name in certain],
            "uncertain": [name for name in CLASSES if name in uncertain],
            "candidates": [name for name in CLASSES if name in certain | uncertain], "reported_candidates": reported,
            "label_exclusion": exclusion, "evaluation_window": window,
            "target_status": evaluation["target_status"], "contract": criteria.to_dict(),
            "classification_policy": CLASSIFICATION_POLICY}


def _choose(scores: dict, mass: dict) -> str:
    """最大スコア、過去重み付きクラス質量、固定クラス順で一つを選ぶ。

    引数: scores (dict) は未校正スコア、mass (dict) は過去の行重み付き目的値の和。
    返り値: str は事前固定した同点規則で選んだクラス。
    """
    maximum = max(scores.values())
    tied = [name for name in CLASSES if scores[name] == maximum]
    most = max(mass[name] for name in tied)
    return next(name for name in tied if mass[name] == most)


def _finish(result: dict) -> dict:
    """決定時刻を加えず、同じ入力に同じ結果ハッシュを付ける。

    引数: result (dict) はハッシュ追加前の診断付き予測。
    返り値: dict は内容ハッシュを含む予測。
    """
    result["hash"] = digest(result)
    return result


def predict_event_five(current: dict, history: list[dict], as_of: str, mode: str) -> dict:
    """満了済み同先物の５分類を、確定教師または候補配分の重み付きリッジで推定する。

    引数: current (dict) はsymbol・kind・as_of・features13、history (list[dict]) は同特徴とlabel_end・経路評価、
        as_of (str) は時差付き情報締切、mode (str) は確定教師または候補配分。
    返り値: dict は方向または無回答None、未校正スコア、共通前処理、行重み・目的値・係数・寄与と証跡。
        候補配分は候補k個に目的値1/k、行重み1/kを別々に設定し、各クラスへの質量寄与は1/k²。
        未確定候補は真ラベルでも確率でもなく、欠測・収集方法の偏りを学ぶ危険がある。
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
            raise ValueError("情報締切以降に満了する教師は特徴・候補の数値解釈前に拒否します。")
        timed.append((end, row))
    for end, row in timed:
        evaluation = row.get("five_class_evaluation")
        window = evaluation.get("evaluation_window") if isinstance(evaluation, dict) else None
        if not isinstance(window, dict) or "label_available_at" not in window:
            raise ValueError("全教師に評価窓の利用可能時刻が必要です。")
        available = timestamp(window["label_available_at"])
        if available >= cutoff:
            raise ValueError("情報締切以降に利用可能な材料前教師は数値解釈前に拒否します。")
        if available != end:
            raise ValueError("教師label_endと評価窓label_available_atが一致しません。")

    # ----------------------------------------

    if mode not in MODES:
        raise ValueError("方式は確定教師または候補配分を指定してください。")
    if not isinstance(current, dict) or current.get("symbol") not in SYMBOLS:
        raise ValueError("今回入力には対象先物symbolが必要です。")
    if (set(current) - {"symbol", "kind", "as_of", "features"}
            or not isinstance(current.get("features"), dict) or set(current["features"]) - set(FEATURE_KEYS)):
        raise ValueError("今回の予測入力へ材料台帳・評価窓など指定外の情報を渡せません。")
    if "as_of" in current and timestamp(current["as_of"]) != cutoff:
        raise ValueError("今回入力と引数の情報締切as_ofが一致しません。")
    symbol, kind = current["symbol"], current.get("kind")
    features = _features(current.get("features"), kind, FEATURE_KEYS)
    selected, seen, shared_contract = [], set(), None
    seen_origins, seen_sessions = set(), set()
    for end, row in timed:
        if row.get("symbol") not in SYMBOLS:
            raise ValueError("各教師行には対象先物symbolが必要です。")
        if row["symbol"] != symbol:
            continue
        identity = end.isoformat()
        if identity in seen:
            raise ValueError("同じ先物・同じ満了時刻の教師は重複できません。")
        seen.add(identity)
        if "as_of" in row and timestamp(row["as_of"]) >= end:
            raise ValueError("教師の情報締切as_ofは満了時刻より前が必要です。")
        if not isinstance(row.get("as_of"), str):
            raise ValueError("教師には元予測の情報締切as_ofが必要です。")
        origin = timestamp(row["as_of"]).isoformat()
        source_session = row["five_class_evaluation"].get("session")
        if not isinstance(source_session, str):
            raise ValueError("教師には元予測の対象セッションが必要です。")
        if origin in seen_origins or source_session in seen_sessions:
            raise ValueError("同じ先物・同じ元予測締切または対象セッションの教師は重複できません。")
        seen_origins.add(origin)
        seen_sessions.add(source_session)
        row_features = _features(row.get("features"), row.get("kind"), FEATURE_KEYS)
        label = _label(row.get("five_class_evaluation"), symbol, row.get("kind"), identity, row.get("as_of"))
        if shared_contract is not None and label["contract"] != shared_contract:
            raise ValueError("同じ学習集合へ異なる５分類契約を混在できません。")
        shared_contract = label["contract"]
        selected.append({"symbol": symbol, "kind": row["kind"], "label_end": identity,
                         "features": row_features, "label": label})
    selected.sort(key=lambda row: timestamp(row["label_end"]))
    row_weights, targets = [], []
    for row in selected:
        label = row["label"]
        candidates = label["certain"] if mode == "確定教師" else label["candidates"]
        count = len(candidates)
        weight = 1.0 if mode == "確定教師" and count else 1.0 / count if count else 0.0
        row_weights.append(weight)
        targets.append({name: 1.0 / count if name in candidates else 0.0 for name in CLASSES})
    mass = {name: math.fsum(weight * target[name] for weight, target in zip(row_weights, targets)) for name in CLASSES}
    effective = math.fsum(row_weights)
    positive_n = sum(weight > 0 for weight in row_weights)
    hashes = [digest(row) for row in selected]
    result = {
        "model": "臨時材料前５分類・重み付きリッジ作業用モデル", "variant": mode, "mode": "回想・開発再生",
        "base_model_sha256": BASE_MODEL_SHA256, "estimation_formula_sha256": ESTIMATION_FORMULA_SHA256,
        "delta_scope": "評価窓・利用可能時計・失効/認定不明の教師不採用と予測入力遮断。推定式は旧版を複製。",
        "as_of": cutoff.isoformat(), "symbol": symbol, "kind": kind, "classification_policy": CLASSIFICATION_POLICY,
        "direction": None, "class_scores": {name: None for name in CLASSES}, "prediction_pct": None,
        "features": features, "feature_keys": list(FEATURE_KEYS), "expanded_feature_keys": [], "expanded_features": {},
        "scaler": {}, "scaler_weighting": "同じ先物の全過去行を等重み。目的値未付与・行重み０の行も含む。",
        "out_of_range_features": [], "newly_observed_features": [],
        "training_n": len(selected), "positive_weight_n": positive_n, "fitted_n": 0, "effective_weight": effective,
        "minimum": MINIMUM, "penalty": PENALTY, "cold_start": positive_n < MINIMUM,
        "label_counts": {"certain": {name: sum(name in row["label"]["certain"] for row in selected) for name in CLASSES},
                         "candidate_membership": {name: sum(name in row["label"]["candidates"] for row in selected) for name in CLASSES},
                         "weighted_class_mass": mass, "zero_weight_rows": len(selected) - positive_n},
        "label_exclusion_counts": {reason: sum(row["label"]["label_exclusion"] == reason for row in selected)
                                   for reason in ("開始前失効", "材料認定不明")},
        "weights": row_weights, "targets": targets, "coefficients": {}, "contributions": {},
        "training_row_ends": [row["label_end"] for row in selected], "training_label_hashes": hashes,
        "training_labels_hash": digest(hashes), "last_label_end": selected[-1]["label_end"] if selected else None,
        "last_positive_label_end": next((row["label_end"] for row, weight in reversed(list(zip(selected, row_weights))) if weight > 0), None),
        "contract": shared_contract,
        "teacher_semantics": "確定教師は一意正解のonehot。候補配分は候補k個へ目的値1/k・行重み1/k、各class質量1/k²。",
        "teacher_risk": "未確定候補は真ラベル・確率ではない。価格欠測や足内順序が作る収集上の偏りを学ぶ危険がある。",
        "score_semantics": "クラススコアは未校正で予測確率ではない。０未満・１超も許容する。",
        "coefficient_basis": "全過去行で補完・標準化後、正重み行を重み付き中心化。interceptは非罰則。",
        "tie_break": "最大スコア→過去weighted_class_mass最大→固定CLASSES順。厳密同値の場合に適用。",
        "history_time_contract": "全行のlabel_end<as_ofを他先物除外・特徴・候補の解釈より先に確認。受信・評価記録時刻は未検証。",
        "status": "正重みの教師なし・無回答", "statistical_advantage": "未検証",
    }
    if not selected:
        return _finish(result)

    # ----------------------------------------

    matrix, vector, expanded, statistics, outside, newly_observed = _prepare_columns(selected, features, FEATURE_KEYS)
    result.update(expanded_feature_keys=list(expanded), expanded_features=dict(zip(expanded, vector)),
                  scaler=statistics, out_of_range_features=outside, newly_observed_features=newly_observed)
    if effective == 0:
        return _finish(result)
    if positive_n < MINIMUM:
        scores = {name: mass[name] / effective for name in CLASSES}
        result.update(direction=_choose(scores, mass), class_scores=scores, status="正重み８行未満・重み付き過去構成比",
                      score_semantics="少数教師のスコアは重み付き目的値の構成比。予測確率として校正していない。")
        return _finish(result)

    # ----------------------------------------

    size = len(expanded)
    averages = [math.fsum(weight * row[index] for row, weight in zip(matrix, row_weights)) / effective for index in range(size)]
    centered = [[value - average for value, average in zip(row, averages)] for row in matrix]
    gram = [[math.fsum(weight * row[i] * row[j] for row, weight in zip(centered, row_weights))
             + (PENALTY if i == j else 0.0) for j in range(size)] for i in range(size)]
    scores, coefficients, contributions = {}, {}, {}
    for name in CLASSES:
        average_target = mass[name] / effective
        rhs = [math.fsum(weight * row[index] * (target[name] - average_target)
                         for row, weight, target in zip(centered, row_weights, targets)) for index in range(size)]
        slopes = solve(gram, rhs)
        intercept = average_target - math.fsum(slope * average for slope, average in zip(slopes, averages))
        coefficients[name] = {"intercept": intercept, **dict(zip(expanded, slopes))}
        contributions[name] = {"intercept": intercept,
                               **{key: slope * value for key, slope, value in zip(expanded, slopes, vector)}}
        scores[name] = math.fsum(contributions[name].values())
        if not all(math.isfinite(value) for value in [scores[name], *coefficients[name].values()]):
            raise ValueError("重み付き推定の係数・スコアが有限値ではありません。")
    result.update(direction=_choose(scores, mass), class_scores=scores, coefficients=coefficients,
                  contributions=contributions, fitted_n=positive_n, weighted_feature_means=dict(zip(expanded, averages)),
                  status="探索的５分類・重み付きリッジ・優位性未検証")
    return _finish(result)
