"""先物前セッションの観測終値を基準に予測と順次評価を保存する。"""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import asdict
from datetime import datetime, time, timedelta
from pathlib import Path

from .common import JST, digest, now, quantile, read_json, timestamp, write_json
from .model import fit_predict, material_features, wilson_lower
from .news import snapshot_payload
from .prices import Candle, SYMBOLS, load_candles, session_groups
from .reactions import reaction_statistics

HORIZONS = {"today": "対象セッション", "week": "１週間", "month": "１か月", "quarter": "３か月"}
ENGINE_VERSION = "0.1.1"
MODEL_SPEC = "前セッション代理基準・残存点数・期間別縮小推定１"


def session_plan(as_of: datetime) -> dict:
    """開始前の判断時刻から対象セッションと公式終了予定を決める。

    引数: as_of (datetime) は日本時間の判断時刻。
    返り値: dict は対象名、開始、終了予定。休日の営業有無は価格観測で別途確認する。
    """
    local = timestamp(as_of)
    clock = local.time().replace(tzinfo=None)
    if time(6) <= clock < time(8, 45):
        kind, start, end = "Day", time(8, 45), time(15, 45)
        end_date = local.date()
    elif time(15, 45) <= clock < time(17):
        kind, start, end = "Night", time(17), time(6)
        end_date = local.date() + timedelta(days=1)
    else:
        raise ValueError("開始前予測は06:00〜08:44または15:45〜16:59を指定してください（標準07:45／16:00）。")
    return {"session": local.date().isoformat() + "_" + kind, "kind": kind,
            "start": datetime.combine(local.date(), start, JST),
            "end": datetime.combine(end_date, end, JST)}


def add_months(value: datetime, count: int) -> datetime:
    """暦月を加算し、月末を補正する。

    引数: value (datetime) は起点、count (int) は月数。
    返り値: datetime は同じ時刻の満期予定。
    """
    index = value.year * 12 + value.month - 1 + count
    year, month = divmod(index, 12)
    month += 1
    return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))


def targets(plan: dict) -> dict[str, datetime]:
    """当日・週・月・四半期の名目満期を作る。

    引数: plan (dict) は対象セッション計画。
    返り値: dict[str, datetime] は対象終了予定から７暦日／１暦月／３暦月先。
    """
    end = plan["end"]
    return {"today": end, "week": end + timedelta(days=7),
            "month": add_months(end, 1), "quarter": add_months(end, 3)}


def scheduled_end(session: str) -> datetime:
    """保存セッション名から公式終了予定を求める。

    引数: session (str) は日付と Day または Night。
    返り値: datetime は日中15:45、夜間翌06:00の予定。
    """
    day = datetime.strptime(session[:10], "%Y-%m-%d").date()
    if session.endswith("_Night"):
        return datetime.combine(day + timedelta(days=1), time(6), JST)
    return datetime.combine(day, time(15, 45), JST)


def candle_hash(candles: list[Candle]) -> str:
    """利用価格の値と出典をハッシュ化する。

    引数: candles (list[Candle]) は時系列価格。
    返り値: str は変更検知用ハッシュ。第三者による時刻証明ではない。
    """
    values = []
    for candle in candles:
        row = asdict(candle)
        row["time"], row["end"] = candle.time.isoformat(), candle.end.isoformat()
        values.append(row)
    return digest(values)


def reference_quote(candles: list[Candle], as_of: datetime, plan: dict) -> Candle:
    """同じ先物の前セッション最終観測値を選ぶ。

    引数: candles (list[Candle]) は可視価格、as_of (datetime) は判断時刻、plan (dict) は対象。
    返り値: Candle は現物前日比を使わない先物基準。公式終値の代用である。
    """
    expected_kind = "Night" if plan["kind"] == "Day" else "Day"
    groups = [group for group in session_groups(candles) if scheduled_end(group[0].session) <= as_of]
    if not groups:
        raise ValueError("前セッションの先物価格がありません。")
    group = groups[-1]
    if not group[0].session.endswith("_" + expected_kind):
        raise ValueError("直前の期待セッションが欠損しています。古い同種セッションへ後退しません。")
    if plan["kind"] == "Night" and group[0].session[:10] != as_of.date().isoformat():
        raise ValueError("夜間向け基準となる当日日中セッションがありません。")
    end = scheduled_end(group[0].session)
    eligible = [bar for bar in group if bar.end <= end]
    if not eligible or end - eligible[-1].end > timedelta(minutes=10):
        raise ValueError("前セッション末尾の観測が不足しています。")
    if as_of - end > timedelta(days=4):
        raise ValueError("先物基準価格が４日以上古いため予測を保留します。")
    return eligible[-1]


def endpoint(candles: list[Candle], nominal: datetime, kind: str,
             observed_at: datetime, exact_session: str | None = None) -> tuple[Candle, datetime] | None:
    """満了した同種セッションの末尾観測値を探す。

    引数: candles (list[Candle]) は価格、nominal (datetime) は名目満期、kind (str) は種類、
    observed_at (datetime) は評価時点、exact_session (str | None) は当日固定対象。
    返り値: tuple[Candle, datetime] | None は観測足と予定終了。未満了や欠測は None。
    """
    for group in session_groups(candles):
        name = group[0].session
        end = scheduled_end(name)
        if not name.endswith("_" + kind) or end < nominal or end > observed_at:
            continue
        if exact_session is not None and name != exact_session:
            continue
        if end - nominal > timedelta(days=7):
            continue
        eligible = [bar for bar in group if bar.end <= end]
        if eligible and end - eligible[-1].end <= timedelta(minutes=10):
            return eligible[-1], end
        return None
    return None


def movement(candles: list[Candle], reference: Candle, end: Candle) -> dict:
    """前セッション先物基準の終点騰落と観測中の高安を計算する。

    引数: candles (list[Candle]) は対象価格、reference/end (Candle) は基準と終点。
    返り値: dict は百分率騰落、高安、観測件数、欠測候補を含む値。
    """
    path = [bar for bar in candles if reference.end <= bar.time and bar.end <= end.end]
    if not path:
        raise ValueError("基準後の価格経路がありません。")
    missing = sum(max(0, round((b.time - a.end).total_seconds() / 60))
                  for a, b in zip(path, path[1:]) if a.session == b.session)
    return {"actual_pct": (end.close / reference.close - 1) * 100,
            "low_pct": (min([reference.close] + [bar.low for bar in path]) / reference.close - 1) * 100,
            "high_pct": (max([reference.close] + [bar.high for bar in path]) / reference.close - 1) * 100,
            "path_rows": len(path), "missing_interval_slots": missing,
            "path_hash": candle_hash(path)}


def empirical_distribution(candles: list[Candle], as_of: datetime, horizon: str, plan: dict) -> dict:
    """過去の同じ開始前時刻・セッションから期間別の経験範囲を求める。

    引数: candles (list[Candle]) は判断までの足、as_of (datetime) は判断、horizon (str) は期間、
    plan (dict) は対象の種類。期間をまたぐ外挿は行わない。
    返り値: dict は件数、終点騰落の１０／９０分位、経路高安の経験範囲。
    """
    examples = []
    for group in session_groups(candles):
        if not group[0].session.endswith("_" + plan["kind"]):
            continue
        day = datetime.strptime(group[0].session[:10], "%Y-%m-%d").date()
        origin = datetime.combine(day, as_of.timetz()).astimezone(JST)
        if origin >= as_of:
            continue
        historical_plan = session_plan(origin)
        nominal = targets(historical_plan)[horizon]
        found = endpoint(candles, nominal, plan["kind"], as_of,
                         historical_plan["session"] if horizon == "today" else None)
        if found is None:
            continue
        try:
            reference = reference_quote([bar for bar in candles if bar.end <= origin], origin, historical_plan)
        except ValueError:
            continue
        actual, end = found
        example = movement(candles, reference, actual)
        example.update(origin=origin.isoformat(), label_end=end.isoformat())
        examples.append(example)
    sufficient = len(examples) >= 8
    return {"n": len(examples), "minimum": 8, "nominal_coverage": 0.8,
            "status": "過去の観測経験範囲・予測区間未校正" if sufficient else "標本不足",
            "terminal_range_pct": [quantile([e["actual_pct"] for e in examples], p) for p in (0.1, 0.9)] if sufficient else None,
            "path_range_pct": [quantile([e["low_pct"] for e in examples], 0.1), quantile([e["high_pct"] for e in examples], 0.9)] if sufficient else None,
            "last_label_end": max((e["label_end"] for e in examples), default=None),
            "missing_interval_slots": sum(e["missing_interval_slots"] for e in examples)}


# ----------------------------------------
# 固定した予測と、判断時刻までに満了した学習履歴


def verify_forecast(value: dict) -> None:
    """予測本文の変更を検知する。

    引数: value (dict) は保存予測。
    返り値: None。ハッシュ不一致なら例外。
    """
    body = {key: item for key, item in value.items() if key != "forecast_hash"}
    if digest(body) != value["forecast_hash"]:
        raise ValueError("保存予測のハッシュが一致しません。")


def history_rows(root: Path | None, as_of: datetime, mode: str) -> list[dict]:
    """予測より前に満了・評価された別の予測だけを収集する。

    引数: root (Path | None) は履歴ルート、as_of (datetime) は上限、mode (str) は検証区分。
    返り値: list[dict] は重複を除いた材料特徴と観測結果。
    """
    if root is None or not root.exists():
        return []
    rows, used = [], set()
    records = [(path, read_json(path)) for path in root.rglob("forecast.json")]
    records.sort(key=lambda pair: (timestamp(pair[1]["as_of"]), timestamp(pair[1].get("issued_at", pair[1]["generated_at"])), pair[1]["forecast_hash"]))
    for path, forecast in records:
        verify_forecast(forecast)
        if forecast.get("model_spec", "前セッション代理基準・残存点数・期間別縮小推定１") != MODEL_SPEC:
            continue
        if timestamp(forecast["as_of"]) >= as_of or forecast["mode"] != mode:
            continue
        if mode == "運用" and timestamp(forecast["issued_at"]) > as_of:
            continue
        for evaluation_path in sorted(path.parent.glob("evaluation-*.json")):
            evaluation = read_json(evaluation_path)
            if evaluation["forecast_hash"] != forecast["forecast_hash"]:
                raise ValueError("評価が別の予測を参照しています。")
            if timestamp(evaluation["observed_at"]) > as_of:
                continue
            if mode == "運用" and timestamp(evaluation["recorded_at"]) > as_of:
                continue
            for label in evaluation["labels"]:
                if timestamp(label["label_end"]) > as_of:
                    continue
                symbol, horizon = label["symbol"], label["horizon"]
                key = (symbol, horizon, forecast["session"], forecast["targets"][horizon])
                if key in used:
                    continue
                used.add(key)
                prediction = forecast["instruments"][symbol]["horizons"][horizon]
                rows.append({"symbol": symbol, "horizon": horizon, "kind": forecast["kind"],
                             "features": prediction["features"], "actual_pct": label["actual_pct"],
                             "predicted_pct": prediction["model"]["prediction_pct"],
                             "forecast_hash": forecast["forecast_hash"], "label_end": label["label_end"],
                             "metrics": label["metrics"]})
    return rows


def source_hash() -> str:
    """実行した研究コードの版を内容から識別する。

    引数: なし。
    返り値: str はモジュール内容のハッシュ。
    """
    return digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(Path(__file__).parent.glob("*.py"))})


def forecast(prices: Path, articles: list[dict], events: list[dict], as_of: datetime,
             out: Path, history: Path | None = None, mode: str = "回想") -> dict:
    """可視情報だけで先物の四期間予測を固定保存する。

    引数: prices (Path) は価格ルート、articles/events (list[dict]) は原本と材料注釈、
    as_of (datetime) は判断、out (Path) は新規保存先、history (Path | None) は過去評価、
    mode (str) は回想または運用。
    返り値: dict はハッシュ付き固定予測。現物前日比は一切参照しない。
    """
    as_of = timestamp(as_of)
    if mode not in ("回想", "運用"):
        raise ValueError("区分は回想または運用です。")
    if mode == "運用" and not 0 <= (timestamp(now()) - as_of).total_seconds() <= 300:
        raise ValueError("運用区分の情報締切は実行時刻以前の５分以内にしてください。過去再生は回想区分です。")
    if out.exists():
        raise FileExistsError("予測保存先は新しいフォルダを指定してください。")
    plan = session_plan(as_of)
    issued_at = timestamp(now())
    if mode == "運用" and issued_at >= plan["start"]:
        raise ValueError("対象セッション開始後には開始前予測を発行できません。")
    expiry = targets(plan)
    snapshot = snapshot_payload(articles, as_of)
    past = history_rows(history, as_of, mode)
    value = {"version": ENGINE_VERSION, "model_spec": MODEL_SPEC, "source_hash": source_hash(), "generated_at": now(), "issued_at": issued_at.isoformat(),
             "as_of": as_of.isoformat(), "mode": mode, "session": plan["session"], "kind": plan["kind"],
             "session_start": plan["start"].isoformat(), "targets": {key: end.isoformat() for key, end in expiry.items()},
             "snapshot_hash": snapshot["canonical_hash"], "instruments": {},
             "limitations": ["騰落基準は同じ先物の前セッション観測最終値。現物の前日比は使用しない。",
                             "保存データに公式クロージング価格がなく、基準と評価終点は観測最終値による暫定値。",
                             "CSVに限月・受信時刻・改訂履歴がない。同一限月の確認と厳密な過去受信状態の復元は未完了。",
                             "回想注釈は言語モデル内部の未来知識まで隠せない。前向き運用成績と混ぜない。",
                             "方向とレンジの優位性は未検証。長期の未知材料を事前に当てたとは扱わない。"]}
    for symbol in SYMBOLS:
        visible = load_candles(prices, symbol, cutoff=as_of)
        reference = reference_quote(visible, as_of, plan)
        instrument = {"reference": {"price": reference.close, "time": reference.end.isoformat(),
                                    "session": reference.session, "source": "candle_1min",
                                    "is_official_close": False, "contract": reference.contract,
                                    "description": "同じ先物の前セッション観測最終値（公式終値の代用）"},
                      "visible_price_hash": candle_hash(visible), "visible_price_rows": len(visible), "horizons": {}}
        for horizon, target in expiry.items():
            features, details, excluded = material_features(events, snapshot["articles"], as_of, target, symbol, mode, issued_at)
            rows = [row for row in past if row["symbol"] == symbol and row["horizon"] == horizon and row["kind"] == plan["kind"]]
            model = fit_predict(rows, features)
            distribution = empirical_distribution(visible, as_of, horizon, plan)
            center = model["prediction_pct"]
            candidate = "上昇候補" if center is not None and center > 0.05 else "下落候補" if center is not None and center < -0.05 else "保留"
            instrument["horizons"][horizon] = {"features": features, "materials": details,
                "excluded_materials": excluded, "material_score": sum(features.values()),
                "remaining_material_score": sum(item["remaining_score"] for item in details),
                "model": model, "direction": "保留（優位性未検証）", "model_direction": candidate,
                "p_up": None, "distribution": distribution,
                "training_forecast_hashes": sorted({row["forecast_hash"] for row in rows}),
                "hypothesis": "予測時点の固定材料のみ。期中新規材料は別途記録する。"}
        value["instruments"][symbol] = instrument
    issued_at = timestamp(now())
    if mode == "運用" and (not 0 <= (issued_at - as_of).total_seconds() <= 300 or issued_at >= plan["start"]):
        raise ValueError("計算完了時に５分以内・セッション開始前の発行条件を満たしていません。")
    value["issued_at"] = issued_at.isoformat()
    value["generated_at"] = issued_at.isoformat()
    value["forecast_hash"] = digest(value)
    write_json(out / "snapshot.json", snapshot)
    write_json(out / "forecast.json", value)
    used = {material["event_id"] for instrument in value["instruments"].values()
            for item in instrument["horizons"].values() for material in item["materials"]}
    write_json(out / "materials.json", [event for event in events if event["event_id"] in used])
    (out / "予測.md").write_text(render_forecast(value), encoding="utf-8")
    return value


# ----------------------------------------
# 結果判定、後発材料候補、全件集計


def interval_score(actual: float, bounds: list[float] | None) -> float | None:
    """８０％終点区間の幅と逸脱を同時に採点する。

    引数: actual (float) は実績百分率、bounds (list[float] | None) は両端。
    返り値: float | None は小さいほど良い区間得点。区間未発行では None。
    """
    if bounds is None:
        return None
    lower, upper = bounds
    return upper - lower + 10 * max(lower - actual, 0) + 10 * max(actual - upper, 0)


def evaluate(run: Path, prices: Path, observed_at: datetime, post_articles: list[dict] | None = None) -> dict:
    """固定予測を結果価格に照合し、後発材料は候補として別記する。

    引数: run (Path) は固定予測、prices (Path) は価格、observed_at (datetime) は評価時刻、
    post_articles (list[dict] | None) は後発記事。記事があっても失敗は免除しない。
    返り値: dict は満了分の全件評価。未満了は pending に残す。
    """
    observed_at = timestamp(observed_at)
    value = read_json(run / "forecast.json")
    verify_forecast(value)
    as_of = timestamp(value["as_of"])
    if observed_at <= as_of:
        raise ValueError("評価時刻は判断より後にしてください。")
    if value["mode"] == "運用" and observed_at > timestamp(now()):
        raise ValueError("運用評価には未来の観測時刻を指定できません。")
    evaluation = {"forecast_hash": value["forecast_hash"], "observed_at": observed_at.isoformat(),
                  "recorded_at": now(), "labels": [], "pending": [],
                  "attribution": "未判定。後発記事と値動きの因果関係は自動認定しない。"}
    for symbol, instrument in value["instruments"].items():
        visible_before = load_candles(prices, symbol, cutoff=as_of)
        if candle_hash(visible_before) != instrument["visible_price_hash"]:
            raise ValueError(f"{symbol} の予測入力価格が変更されています。別版で再研究してください。")
        all_visible = load_candles(prices, symbol, cutoff=observed_at)
        reference = next(bar for bar in visible_before if bar.session == instrument["reference"]["session"]
                         and bar.end == timestamp(instrument["reference"]["time"]))
        for horizon, item in instrument["horizons"].items():
            nominal = timestamp(value["targets"][horizon])
            found = endpoint(all_visible, nominal, value["kind"], observed_at,
                             value["session"] if horizon == "today" else None)
            if found is None:
                evaluation["pending"].append({"symbol": symbol, "horizon": horizon, "reason": "未満了・休場・終端観測不足のいずれか"})
                continue
            quote, end = found
            actual = movement(all_visible, reference, quote)
            center = item["model"]["prediction_pct"]
            bounds = item["distribution"]["terminal_range_pct"]
            path_bounds = item["distribution"]["path_range_pct"]
            actual_pct = actual["actual_pct"]
            sign = "上昇" if actual_pct > 0.05 else "下落" if actual_pct < -0.05 else "横ばい"
            metrics = {"actual_direction": sign,
                       "direction_hit": (abs(actual_pct) > 0.05 and (center > 0) == (actual_pct > 0)) if center is not None and abs(center) > 0.05 else None,
                       "absolute_error_pct": abs(center - actual_pct) if center is not None else None,
                       "zero_change_absolute_error_pct": abs(actual_pct),
                       "terminal_covered": bounds[0] <= actual_pct <= bounds[1] if bounds else None,
                       "path_covered": path_bounds[0] <= actual["low_pct"] and actual["high_pct"] <= path_bounds[1] if path_bounds else None,
                       "interval_score": interval_score(actual_pct, bounds), "brier": None}
            candidates = []
            if post_articles:
                later = snapshot_payload(post_articles, end)["articles"]
                for article in later:
                    public = timestamp(article.get("published_at") or article["fetched_at"])
                    available = timestamp(article["available_at"])
                    if as_of < available <= end:
                        candidates.append({"id": article["id"], "revision_id": article["revision_id"],
                                           "available_at": article["available_at"],
                                           "classification": "後発公表材料候補" if public > as_of else "既存公表・取得遅延または後日改訂候補"})
            evaluation["labels"].append({"symbol": symbol, "horizon": horizon, "nominal_target": nominal.isoformat(),
                "label_end": end.isoformat(), "quote_time": quote.end.isoformat(), "quote_price": quote.close,
                "target_is_official_close": False, "reference_price": reference.close,
                "calendar_shift_days": (end - nominal).total_seconds() / 86400, **actual,
                "material_residual_pct": actual_pct - center if center is not None else None,
                "metrics": metrics, "later_material_candidates": candidates})
    name = "evaluation-" + observed_at.strftime("%Y%m%dT%H%M%S%z") + ".json"
    write_json(run / name, evaluation)
    return evaluation


def render_forecast(value: dict) -> str:
    """固定予測を日本語の閲覧用文書へ整形する。

    引数: value (dict) は固定予測。
    返り値: str は先物基準と未検証を明示した文書。
    """
    lines = [f"# 先物見通し：{value['as_of']}", "", f"対象：{value['session']} ／ 区分：{value['mode']}", "",
             "騰落は同じ先物の前セッション観測最終値から測定する。公式終値の未収録による暫定評価。", "",
             "| 銘柄 | 期間 | 満期予定 | 優位方向 | 推定候補 | 材料点数 | 終点経験範囲 | 経路経験範囲 | 標本数 |",
             "| --- | --- | --- | --- | --- | ---: | --- | --- | ---: |"]
    for symbol, instrument in value["instruments"].items():
        for horizon, item in instrument["horizons"].items():
            distribution = item["distribution"]
            ranges = []
            for key in ("terminal_range_pct", "path_range_pct"):
                bounds = distribution[key]
                ranges.append(f"{bounds[0]:+.2f}% 〜 {bounds[1]:+.2f}%" if bounds else "標本不足")
            lines.append(f"| {symbol} | {HORIZONS[horizon]} | {value['targets'][horizon]} | {item['direction']} | {item['model_direction']} | {item['material_score']:+.3f} | {ranges[0]} | {ranges[1]} | {distribution['n']} |")
    lines += ["", "経験範囲は過去標本の１０／９０分位。将来８０％を保証する予測区間ではない。経路高安は欠測中の極値を復元しない。", ""]
    for symbol, instrument in value["instruments"].items():
        reference = instrument["reference"]
        lines += [f"## {symbol} の基準と材料", "", f"基準：{reference['session']} ／ {reference['time']} ／ {reference['price']}（先物・公式終値の代用）。", ""]
        for material in instrument["horizons"]["today"]["materials"]:
            lines.append(f"- {material['category']}：{material['score']:+.3f}点。{material['reason']}（材料ID：{material['event_id']}）")
        if not instrument["horizons"]["today"]["materials"]:
            lines.append("- 判断時点で利用可能な材料注釈なし。")
    lines += ["", "## 制約", ""] + [f"- {item}" for item in value["limitations"]] + ["", f"変更検知用ハッシュ：{value['forecast_hash']}", ""]
    return "\n".join(lines)


def summarize(root: Path) -> dict:
    """全保存予測の結果を区分・銘柄・期間・セッション別に集計する。

    引数: root (Path) は保存ルート。
    返り値: dict は保留を含む分母と、未知材料で除外しない成績。
    """
    groups, used, comparisons = {}, set(), []
    forecasts = list(root.rglob("forecast.json"))
    records = [(path, read_json(path)) for path in forecasts]
    records.sort(key=lambda pair: (timestamp(pair[1]["as_of"]), timestamp(pair[1].get("issued_at", pair[1]["generated_at"])), pair[1]["forecast_hash"]))
    duplicates = []
    for path, value in records:
        verify_forecast(value)
        latest = {}
        for evaluation_path in sorted(path.parent.glob("evaluation-*.json")):
            evaluation = read_json(evaluation_path)
            if evaluation["forecast_hash"] != value["forecast_hash"]:
                raise ValueError("評価と予測のハッシュが一致しません。")
            for label in evaluation["labels"]:
                latest[(label["symbol"], label["horizon"])] = label
        for symbol, instrument in value["instruments"].items():
            for horizon, item in instrument["horizons"].items():
                identity = (value["mode"], symbol, horizon, value["session"], value["targets"][horizon])
                if identity in used:
                    duplicates.append({"forecast_hash": value["forecast_hash"], "symbol": symbol, "horizon": horizon,
                                       "reason": "同一対象の最初の固定予測を主採点。再発行は主分母へ重複計上しない。"})
                    continue
                used.add(identity)
                key = "／".join((value["mode"], symbol, value["kind"], horizon))
                group = groups.setdefault(key, {"forecasts": 0, "evaluated": 0, "abstained": 0, "labels": []})
                group["forecasts"] += 1
                group["abstained"] += int(item["direction"].startswith("保留"))
                label = latest.get((symbol, horizon))
                if label:
                    group["evaluated"] += 1
                    group["labels"].append(label)
                    for material in item["materials"]:
                        comparisons.append({"forecast_hash": value["forecast_hash"], "symbol": symbol, "horizon": horizon,
                                            "mode": value["mode"], "kind": value["kind"], "as_of": value["as_of"],
                                            "category": material["category"], "regime": material["regime"],
                                            "score": material["score"], "remaining_score": material["remaining_score"],
                                            "actual_pct": label["actual_pct"], "interpretation": "同時発生材料を含む関連。単独の因果効果ではない。"})
    for group in groups.values():
        labels = group.pop("labels")
        hits = [label["metrics"]["direction_hit"] for label in labels if label["metrics"]["direction_hit"] is not None]
        covers = [label["metrics"]["terminal_covered"] for label in labels if label["metrics"]["terminal_covered"] is not None]
        path_covers = [label["metrics"]["path_covered"] for label in labels if label["metrics"]["path_covered"] is not None]
        errors = [label["metrics"]["absolute_error_pct"] for label in labels if label["metrics"]["absolute_error_pct"] is not None]
        interval_scores = [label["metrics"]["interval_score"] for label in labels if label["metrics"]["interval_score"] is not None]
        group.update(pending=group["forecasts"] - group["evaluated"], candidate_direction_n=len(hits),
                     candidate_direction_hit_rate=sum(hits)/len(hits) if hits else None,
                     candidate_direction_wilson_lower=wilson_lower(sum(hits), len(hits)),
                     terminal_interval_n=len(covers), terminal_coverage=sum(covers)/len(covers) if covers else None,
                     path_interval_n=len(path_covers), observed_path_coverage=sum(path_covers)/len(path_covers) if path_covers else None,
                     mean_absolute_error_pct=sum(errors)/len(errors) if errors else None,
                     mean_interval_score=sum(interval_scores)/len(interval_scores) if interval_scores else None)
    return {"generated_at": now(), "forecast_files": len(forecasts), "groups": groups, "duplicate_predictions": duplicates,
            "source_versions": sorted({value["source_hash"] for _, value in records}),
            "material_reactions": comparisons,
            "reaction_statistics": reaction_statistics(comparisons),
            "limitations": ["全件を含み、後発材料を理由に誤予測を除外しない。", "同時発生材料の反応は関連であり因果ではない。",
                            "長期重複窓や同一日の依存性があり、単純な方向一致率で優位性は確立しない。",
                            "候補方向の一致率と、優位方向を実際に発行した割合を混同しない。"]}
