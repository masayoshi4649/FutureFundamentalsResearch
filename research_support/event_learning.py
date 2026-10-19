"""未知の臨時材料による有効期間を結果側だけで決める、五分類の逐次比較。"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.engine import reference_quote, session_plan
from fundamentals.prices import SYMBOLS, load_candles
from fundamentals.replay import _append_audit, _audit, _file_hash
from research_support.event_window import evaluate_event_window, select_event
from research_support.five_class import CLASSES, evaluate_priority_path, score_prediction
from research_support.five_class_learning import (
    VARIANTS, _input_provenance, _numerical_reason, _verify, _verify_packet,
)
from research_support.five_class_replay import _add_score, _finish_rates, _new_group
from research_support.five_event_model import predict_event_five


def _freeze(root: Path, inputs: Path, measured: Path, prices: Path, ledger: Path) -> dict:
    """旧入力の来歴と新規約・材料認定・別版実装を予測開始前にまとめて固定する。

    引数: root (Path) は新比較、inputs (Path) は可視原本、measured (Path) は元実測コピー、
        prices (Path) は元価格、ledger (Path) は予測と独立に認定した材料台帳。
    返り値: dict は固定した全日程・実装・認定証跡の一覧。
    """
    if (root / "manifest.json").exists():
        raise FileExistsError("新規約の比較は未使用の保存先で実行してください。")
    packets = sorted((inputs / "masked").glob("*.json"))
    if not packets:
        raise ValueError("事前に固定した可視入力が必要です。")
    for name in ("五分類規約.json", "臨時材料規約.json"):
        if _file_hash(root / name) != _file_hash(Path(__file__).with_name(name)):
            raise ValueError("確認済みの共通規約と一致しません。")
    provenance = _input_provenance(inputs, measured, packets, root / "五分類規約.json")
    evidence = read_json(ledger)
    if (evidence.get("ledger_hash") != digest({key: item for key, item in evidence.items() if key != "ledger_hash"})
            or not isinstance(evidence.get("events"), list)):
        raise ValueError("材料台帳の構造または自己ハッシュが不正です。")
    evidence_files = evidence.get("source_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise ValueError("材料台帳のニュース原本・認定記録のSHAが必要です。")
    for name, expected in evidence_files.items():
        if not Path(name).is_file() or _file_hash(Path(name)) != expected:
            raise ValueError("材料台帳のニュース・認定原本が変更されています。")
    public_clocks = evidence.get("article_public_clocks", {})
    available_clocks = evidence.get("article_available_clocks", {})
    for event in evidence["events"]:
        if any(str(identity) not in available_clocks for identity in event["article_ids"]):
            raise ValueError("材料の全根拠記事について収録版の取得・利用可能時計が必要です。")
        _event_evidence_time(event, public_clocks, available_clocks)
    for packet in packets:
        value = read_json(packet)
        for symbol in SYMBOLS:
            select_event(evidence["events"], timestamp(value["as_of"]), symbol, timestamp(value["targets"]["today"]))
    files = [root / "実験計画.md", root / "五分類規約.json", root / "臨時材料規約.json", ledger]
    files.extend(Path(__file__).parent / name for name in (
        "event_learning.py", "event_window.py", "five_event_model.py", "five_class.py", "five_class_model.py",
        "five_class_learning.py", "five_class_replay.py"))
    files.extend(Path("fundamentals") / name for name in ("prices.py", "engine.py", "common.py", "model.py", "pdca_measured_model.py", "replay.py"))
    for symbol in SYMBOLS:
        files.extend(sorted((prices / "candle_1min" / symbol).glob("*.csv")))
    hashes = {**provenance["files"], **evidence_files,
              **{str(path.resolve()): _file_hash(path) for path in files}}
    if any(_file_hash(Path(name)) != expected for name, expected in hashes.items()):
        raise ValueError("開始時の照合中に原本が変更されています。")
    value = {"mode": "回想・未知臨時材料前の開発比較", "independent_validation": False,
             "created_at": now(), "rounds": [path.stem for path in packets], "variants": VARIANTS,
             "prices": str(prices.resolve()), "ledger": str(ledger.resolve()), "files": hashes,
             "history_policy": "各締切より前に満了した過去場を、その締切で根拠全記事の収録版が取得済みの認定だけで再構成。",
             "availability_limitation": "収録版の公開・取得・有効時計を照合した開発回想。未収録ニュースと人手認定の当時の再現性は未実証。"}
    value["manifest_hash"] = digest(value)
    write_json(root / "manifest.json", value)
    return _verify(root)


def _labels(prices: Path, origin: str, session: str, rules: dict, events: list[dict],
            references: dict | None = None, full_only: bool = False) -> list[dict]:
    """結果側でだけ材料時計を選び、必要な窓の終了までの元価格を読む。

    引数: prices (Path) は元価格、origin/session (str) は締切と場、rules (dict) は五分類規約、
        events (list[dict]) は結果専用台帳、references (dict | None) は固定基準、full_only (bool) は参考全場評価。
    返り値: list[dict] は先物別の評価。窓短縮の主評価では終了後の価格を数値解釈しない。
    """
    as_of = timestamp(origin)
    plan = session_plan(as_of)
    if plan["session"] != session:
        raise ValueError("対象場が締切と一致しません。")
    rows = []
    for symbol in SYMBOLS:
        event = None if full_only else select_event(events, as_of, symbol, plan["end"])
        available = plan["end"] if event is None else timestamp(event["first_public_at"])
        if available > timestamp(now()):
            raise ValueError("実時計で評価窓が未満了のため元価格を読めません。")
        nominal = plan["end"] if event is None else available.replace(second=0, microsecond=0)
        candles = load_candles(prices, symbol, cutoff=nominal)
        reference = reference_quote(candles, as_of, plan)
        if references is not None:
            fixed = references[symbol]["reference"]
            if (reference.close != fixed["price"] or reference.end != timestamp(fixed["time"])
                    or reference.session != fixed["session"]):
                raise ValueError("先物の基準が予測前に固定した値と一致しません。")
        path = [bar for bar in candles if bar.session == session]
        evaluation = (evaluate_priority_path(reference, path, session, plan["end"], rules["criteria"]) if full_only
                      else evaluate_event_window(reference, path, session, as_of, events, rules["criteria"]))
        source = prices / "candle_1min" / symbol / (session + ".csv")
        rows.append({"symbol": symbol, "label_end": available.isoformat(), "evaluation": evaluation,
                     "source_file": str(source.resolve()), "source_file_hash": _file_hash(source) if source.exists() else None})
    return rows


def _event_evidence_time(event: dict, article_clocks: dict | None = None,
                         available_clocks: dict | None = None) -> datetime:
    """認定に使う記事のうち最も遅い公開時計を、回想での利用開始下限として返す。

    引数: event (dict) は独立認定した材料、article_clocks (dict | None) はニュース索引の公開時計、
        available_clocks (dict | None) は収録版の公開・取得・有効時計の最大。
        記事時計は実材料の初報時計へ代入しない。
    返り値: datetime は根拠全記事が使えた時計の最大。記事時計がない場合は利用を拒否する。
        収録されていないニュースまで網羅した証明には使わない。
    """
    timing = event["timing_evidence"]
    clocks, checked = [], set()
    if isinstance(timing, dict):
        if timing.get("article_public_upper_bound") is not None:
            clocks.append(timestamp(timing["article_public_upper_bound"]))
        for key in ("related_article_clocks", "article_clocks"):
            for article in timing.get(key, []):
                if isinstance(article, dict):
                    public = article.get("published_at", article.get("published_at_jst"))
                    if public is not None:
                        clock = timestamp(public)
                        identity = str(article["article_id"])
                        if article_clocks is not None and (identity not in article_clocks
                                or timestamp(article_clocks[identity]) != clock):
                            raise ValueError("材料認定の根拠記事時計がニュース索引と一致しません。")
                        checked.add(identity)
                        clocks.append(clock)
    article_ids = set(map(str, event["article_ids"]))
    if not checked <= article_ids:
        raise ValueError("根拠時計に列挙した記事は全てarticle_idsへ含める必要があります。")
    if article_clocks is not None:
        for identity in article_ids:
            if identity not in article_clocks:
                raise ValueError("材料の全根拠記事に公開時計が必要です。")
            public = timestamp(article_clocks[identity])
            clocks.append(public)
            if available_clocks is not None:
                if identity not in available_clocks or timestamp(available_clocks[identity]) < public:
                    raise ValueError("根拠記事の利用可能時計が公開時計と不整合です。")
                clocks.append(timestamp(available_clocks[identity]))
        checked.update(article_ids)
    if not article_ids or not article_ids <= checked:
        raise ValueError("材料の全根拠記事の公開時計を確認する必要があります。")
    if not clocks:
        raise ValueError("教師へ使う材料認定には根拠記事の公開時計が必要です。")
    return max(clocks)


def _history_as_of(sources: list[dict], prices: Path, rules: dict, events: list[dict], cutoff: str,
                   cache: dict, article_clocks: dict | None = None,
                   available_clocks: dict | None = None) -> list[dict]:
    """満了済みの過去教師を、その予測締切で公表済みの材料根拠だけで再構成する。

    引数: sources (list[dict]) は過去特徴、prices (Path) は元価格、rules (dict) は五分類規約、
        events (list[dict]) は結果側台帳、cutoff (str) は今回締切、cache (dict) は同じ過去窓の再計算控え、
        article_clocks (dict | None) は全根拠記事の公開時計、available_clocks (dict | None) は収録版の取得等を含む時計。
    返り値: list[dict] は今回知り得た記事だけで訂正した教師。後日の保留の存在も早めて使わない。
    """
    deadline = timestamp(cutoff)
    for row in sources:
        if session_plan(timestamp(row["as_of"]))["end"] >= deadline:
            raise ValueError("今回締切までに満了していない過去教師が混入しています。")
    known = [event for event in events if _event_evidence_time(event, article_clocks, available_clocks) < deadline]
    enriched = []
    for source in sources:
        origin = source["as_of"]
        applicable = [event for event in known if timestamp(event["assessment_as_of"]) == timestamp(origin)]
        key = (origin, digest(applicable))
        if key not in cache:
            plan = session_plan(timestamp(origin))
            cache[key] = {row["symbol"]: row for row in _labels(prices, origin, plan["session"], rules, applicable)}
        label = cache[key][source["symbol"]]
        if timestamp(label["label_end"]) >= deadline:
            raise ValueError("過去教師の評価終点が今回締切以降です。")
        enriched.append({**{key: copy.deepcopy(source[key]) for key in ("symbol", "kind", "as_of", "features")},
                         "label_end": label["label_end"], "actual_pct": label["evaluation"]["metrics"]["actual_pct"],
                         "five_class_evaluation": copy.deepcopy(label["evaluation"])})
    return enriched


def _score(answer: str | None, evaluation: dict) -> dict:
    """開始前失効だけを別状態にし、評価対象内の無回答は不正解にする。

    引数: answer (str | None) は固定回答、evaluation (dict) は材料前の評価。
    返り値: dict は通常の採点、または全予定に残す開始前失効。
    """
    if evaluation["evaluation_window"]["scoreable"]:
        return score_prediction(answer, evaluation)
    return {"prediction": answer, "status": "開始前失効", "target_status": "価格評価期間なし",
            "correct_lower": False, "correct_upper": False, "denominator": False, "unanswered": answer is None}


def _add(group: dict, answer: str | None, evaluation: dict, score: dict) -> None:
    """全予定回を維持し、開始前失効と通常の対象不明を分けて加算する。

    引数: group (dict) は集計、answer (str | None) は回答、evaluation/score (dict) は結果と採点。
    返り値: None。失効も件数として残す。
    """
    if not evaluation["evaluation_window"]["scoreable"]:
        group["scheduled"] += 1
        group["unanswered"] += int(answer is None)
        group["expired_preopen"] += 1
    else:
        _add_score(group, answer, evaluation, score)
    window = evaluation["evaluation_window"]
    group["material_truncated"] += int(window["status"] == "材料前まで")
    group["material_unknown"] += int(window["adjudication_unknown"])


def _group() -> dict:
    """通常の精度集計へ、材料前終了・認定不明・開始前失効の件数を追加する。

    引数: なし。
    返り値: dict は空の集計。
    """
    return {**_new_group(), "expired_preopen": 0, "material_truncated": 0, "material_unknown": 0}


def _review(path: Path, current: dict, primary: list[dict], full: list[dict], prediction: dict) -> list[dict]:
    """材料前の最終主採点と全場の参考採点を分離し、当回の振り返りを保存する。

    引数: path (Path) は当回、current (dict) は可視入力、primary/full (list[dict]) は主窓と参考全場、
        prediction (dict) は結果を読む前に固定した予測。
    返り値: list[dict] は最終主窓の教師形式の控え。次回学習へ直接渡さない。
    """
    full_by_symbol = {row["symbol"]: row for row in full}
    rows, teachers = [], []
    lines = [f"# 振り返り：{current['session']}", "", "- 主評価は未知臨時材料前。全場は参考。市場反応の因果は未識別。", ""]
    for label in primary:
        symbol, evaluation = label["symbol"], label["evaluation"]
        model = prediction["instruments"][symbol]
        score = _score(model["direction"], evaluation)
        full_label = full_by_symbol[symbol]
        full_score = score_prediction(model["direction"], full_label["evaluation"])
        window = evaluation["evaluation_window"]
        rows.append({**label, "direction": model["direction"], "score": score,
                     "full_session_reference": {**full_label, "score": full_score}})
        metrics = evaluation["metrics"]
        actual = metrics["actual_pct"]
        terminal = "未観測" if actual is None else f"{actual:+.3f}％"
        reason, action = _numerical_reason(model, evaluation)
        status = "材料認定不明" if window["adjudication_unknown"] else window["status"]
        lines += [f"- {symbol}：{model['direction'] or '無回答'}→{score['status']}。窓：{status}。",
                  f"  - 名目終点：{evaluation['target_end']}。観測終点：{metrics['quote_time'] or 'なし'}、騰落{terminal}（認定不明時は暫定値）。",
                  f"  - 全場参考：{full_score['status']}。主採点へ混ぜない。",
                  "  - 診断：" + reason, "  - 次回：" + action]
        teachers.append({**copy.deepcopy(current["instruments"][symbol]), "label_end": label["label_end"],
                         "actual_pct": actual, "five_class_evaluation": copy.deepcopy(evaluation)})
    result = {"forecast_hash": prediction["forecast_hash"], "read_after_seal_at": now(), "labels": rows}
    write_json(path / "result.json", result)
    available = max(timestamp(row["label_end"]) for row in primary)
    _append_audit(path, "reveal", available, [path / "result.json"])
    write_json(path / "review.json", {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
               "available_at": available.isoformat(), "recorded_at": now(), "final_primary_labels": teachers,
               "next_training_policy": "次回締切で根拠記事公開時計を検査し、過去教師を別途再構成する。",
               "causality": "未識別"})
    (path / "振り返り.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(path, "review", available, [path / "review.json", path / "振り返り.md"])
    return teachers


def _summarize(root: Path, identities: list[str]) -> dict:
    """主窓と全場参考を別集計し、五つの定数回答を同じ窓・分母で比較する。

    引数: root (Path) は一方式、identities (list[str]) は全予定回。
    返り値: dict は対象・失効・材料不明と精度の境界、価格品質上の証明可能率。
    """
    groups, full_groups, rows = {}, {}, []
    for identity in identities:
        path = root / identity
        if _audit(path)[-1]["phase"] != "review":
            raise ValueError("全予定回の振り返りが終わっていません。")
        visible, result = read_json(path / "入力.json"), read_json(path / "result.json")
        for row in result["labels"]:
            evaluation, answer = row["evaluation"], row["direction"]
            score = _score(answer, evaluation)
            if score != row["score"]:
                raise ValueError("保存した材料前採点が再計算と一致しません。")
            rows.append({"round": identity, "kind": visible["kind"], **row})
            for key in (row["symbol"], row["symbol"] + "／" + visible["kind"]):
                group = groups.setdefault(key, {**_group(), "certifiable_targets": 0,
                    "baselines": {constant: _group() for constant in CLASSES}})
                _add(group, answer, evaluation, score)
                group["certifiable_targets"] += int(
                    evaluation["evaluation_window"]["scoreable"] and evaluation["target_status"] == "対象"
                    and len(evaluation["correct_candidates"]) == 1)
                for constant in CLASSES:
                    _add(group["baselines"][constant], constant, evaluation, _score(constant, evaluation))
                reference = row["full_session_reference"]
                full_group = full_groups.setdefault(key, _new_group())
                _add_score(full_group, answer, reference["evaluation"], score_prediction(answer, reference["evaluation"]))
    for group in groups.values():
        _finish_rates(group)
        group["certifiable_rate_ceiling"] = group["certifiable_targets"] / group["denominator"] if group["denominator"] else None
        for baseline in group["baselines"].values():
            _finish_rates(baseline)
    for group in full_groups.values():
        _finish_rates(group)
    value = {"mode": "回想・未知臨時材料前の開発比較", "independent_validation": False,
             "operational_acceptance": False, "groups": groups, "full_session_reference_groups": full_groups,
             "rows": rows, "target_reached": all(groups[s]["accuracy_lower"] is not None
                 and groups[s]["accuracy_lower"] >= 0.8 for s in SYMBOLS),
             "ceiling_semantics": "価格・材料認定から正解を確定できる率の上限。真の予測精度上限ではない。"}
    write_json(root / "summary.json", value)
    return value


def run_event_comparison(root: Path, inputs: Path, measured: Path, prices: Path, ledger: Path) -> dict:
    """各締切の過去教師を再構成し、両方式の予測固定後に当回の最終主評価を作る。

    引数: root (Path) は新実験、inputs (Path) は可視原本、measured (Path) は元実測、
        prices (Path) は元価格、ledger (Path) は独立認定済みの結果専用材料台帳。
    返り値: dict は確定教師・候補配分の全予定回の比較成績。
    """
    manifest = _freeze(root, inputs, measured, prices, ledger)
    rules = read_json(root / "五分類規約.json")
    evidence = read_json(ledger)
    events, article_clocks = evidence["events"], evidence.get("article_public_clocks", {})
    available_clocks = evidence.get("article_available_clocks", {})
    first = read_json(inputs / "masked" / (manifest["rounds"][0] + ".json"))["as_of"]
    sources = read_json(measured / "初期教師.json")["rows"]
    if any(timestamp(row["label_end"]) >= timestamp(first) for row in sources):
        raise ValueError("初期教師は最初の締切より前に満了している必要があります。")
    cache = {}
    history = _history_as_of(sources, prices, rules, events, first, cache, article_clocks, available_clocks)
    write_json(root / "初期教師.json", {"rows": history, "source_teacher_hash": _file_hash(measured / "初期教師.json"),
               "first_as_of": first, "rule_hash": _file_hash(root / "臨時材料規約.json")})
    manifest["files"][str((root / "初期教師.json").resolve())] = _file_hash(root / "初期教師.json")
    manifest["manifest_hash"] = digest({key: item for key, item in manifest.items() if key != "manifest_hash"})
    write_json(root / "manifest.json", manifest, exclusive=False)
    _verify(root)
    for identity in manifest["rounds"]:
        _verify(root)
        packet = read_json(inputs / "masked" / (identity + ".json"))
        current = read_json(measured / identity / "入力.json")
        _verify_packet(packet, current)
        cutoff = timestamp(current["as_of"])
        history = _history_as_of(sources, prices, rules, events, current["as_of"], cache, article_clocks, available_clocks)
        used_events = {}
        for row in history:
            window = row["five_class_evaluation"]["evaluation_window"]
            for event in ([window["selected_event"]] if window["selected_event"] is not None else []) + window["pending_events"]:
                key = event["assessment_as_of"] + "／" + event["event_id"]
                used_events[key] = {"event_id": event["event_id"], "assessment_as_of": event["assessment_as_of"],
                                    "evidence_available_at": _event_evidence_time(event, article_clocks, available_clocks).isoformat()}
        predictions = {}
        for mode, folder in VARIANTS.items():
            path = root / folder / identity
            visible = {"as_of": current["as_of"], "session": current["session"], "kind": packet["kind"],
                       "instruments": copy.deepcopy(current["instruments"]), "history": copy.deepcopy(history),
                       "source_input_hash": _file_hash(measured / identity / "入力.json")}
            write_json(path / "入力.json", visible)
            write_json(path / "教師来歴.json", {"as_of": current["as_of"], "history_hash": digest(history),
                       "known_evidence": list(used_events.values()), "source_rows": len(sources),
                       "policy": manifest["history_policy"], "availability_limitation": manifest["availability_limitation"]})
            _append_audit(path, "prepare", cutoff, [path / "入力.json", path / "教師来歴.json"])
            models = {symbol: predict_event_five(current["instruments"][symbol], history, current["as_of"], mode) for symbol in SYMBOLS}
            prediction = {"as_of": current["as_of"], "session": current["session"], "issued_at": now(),
                          "mode": "回想・未知臨時材料前の開発比較", "variant": mode,
                          "input_hash": digest(visible), "instruments": models}
            prediction["forecast_hash"] = digest(prediction)
            write_json(path / "予測.json", prediction)
            lines = [f"# 予測：{current['session']}", "", f"- 締切：{current['as_of']}。方式：{mode}。既知期間の開発比較。",
                     "- 未知臨時材料で有効期間終了。予定指標・会見のサプライズでは終了しない。",
                     "- 当回の未来材料・打切り時計は予測入力へ渡していない。", ""]
            for symbol, model in models.items():
                reference = packet["instruments"][symbol]["reference"]
                lines += [f"- {symbol}：{model['direction'] or '無回答'}。基準{reference['price']}（{reference['time']}、同先物終値代理）。",
                          f"  - 過去教師{model['training_n']}行、正重み{model['positive_weight_n']}行。スコアは確率ではない。"]
            lines += ["", "- 長期・確率・ニュース条件付き価格帯は未実証。", f"- 固定ハッシュ：{prediction['forecast_hash']}。"]
            (path / "予測.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            _append_audit(path, "seal", cutoff, [path / "予測.json", path / "予測.md"])
            predictions[mode] = prediction

        # ----------------------------------------
        # 両方の当回予測が固定された後にだけ、当回材料で主窓を決定する。

        primary = _labels(prices, current["as_of"], current["session"], rules, events, packet["instruments"])
        full = _labels(prices, current["as_of"], current["session"], rules, [], packet["instruments"], full_only=True)
        final_teachers = None
        for mode, folder in VARIANTS.items():
            teachers = _review(root / folder / identity, current, primary, full, predictions[mode])
            if final_teachers is None:
                final_teachers = teachers
            elif final_teachers != teachers:
                raise ValueError("二方式の次回教師が一致しません。")
        sources.extend(copy.deepcopy(list(current["instruments"].values())))
    _verify(root)
    summary = {mode: _summarize(root / folder, manifest["rounds"]) for mode, folder in VARIANTS.items()}
    write_json(root / "学習状態.json", {"source_rows": sources, "last_prediction_history": history,
               "mode": "各締切で過去教師を再構成した回想・開発用。運用取得履歴ではない。"})
    lines = ["# 未知臨時材料前の五分類比較", "", f"- 締切時点の過去教師再構成→予測固定→当回主窓評価→振り返りを全{len(manifest['rounds'])}回で反復。",
             "- 予定済み指標・会見は打切り不可。材料不明と開始前失効を別に保持。", "",
             "| 方式 | 先物 | 対象 | 対象不明 | 材料認定不明 | 開始前失効 | 正解下限／分母 | 下限精度 | 楽観上限 | 無回答 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, result in summary.items():
        for symbol in SYMBOLS:
            group = result["groups"][symbol]
            low = "算出不能" if group["accuracy_lower"] is None else f"{group['accuracy_lower']:.1%}"
            upper = "算出不能" if group["accuracy_upper"] is None else f"{group['accuracy_upper']:.1%}"
            lines.append(f"| {mode} | {symbol} | {group['target']} | {group['target_unknown']} | {group['material_unknown']} | {group['expired_preopen']} | {group['correct_lower']}/{group['denominator']} | {low} | {upper} | {group['unanswered']} |")
    lines += ["", "- 楽観上限は達成率ではない。未確定を正解へ繰り上げない。全場参考と五定数比較は各summary.jsonへ別保存。",
              "- 教師は根拠全記事の収録版が取得済みになった次の締切から訂正。未収録ニュースの網羅性は未実証。",
              "- 同じ既知期間の開発比較。80％に到達しても未使用期間の確認まで運用採用しない。"]
    (root / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """独立材料台帳と固定可視入力を指定し、二方式の反復比較を一度だけ実行する。

    引数: なし。実験・可視原本・実測・価格・台帳をコマンド行から受け取る。
    返り値: None。先物別の主精度下限を表示する。
    """
    parser = argparse.ArgumentParser(description="未知臨時材料前の五分類を逐次検証")
    for name in ("root", "inputs", "measured", "prices", "ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_event_comparison(args.root, args.inputs, args.measured, args.prices, args.ledger)
    print({mode: {symbol: value["groups"][symbol]["accuracy_lower"] for symbol in SYMBOLS} for mode, value in result.items()})


if __name__ == "__main__":
    main()
