"""固定した二つの五分類学習を、当回予測の保存後だけ実績を開く順序で比較する。"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from fundamentals.common import digest, now, read_json, timestamp, write_json
from fundamentals.engine import reference_quote, session_plan
from fundamentals.prices import SYMBOLS, load_candles
from fundamentals.replay import _append_audit, _audit, _file_hash
from research_support.five_class import CLASSES, evaluate_priority_path, score_prediction
from research_support.five_class_model import predict_five
from research_support.five_class_replay import _add_score, _finish_rates, _new_group


VARIANTS = {"確定教師": "12_確定教師", "候補配分": "13_候補配分"}
FEATURE_NAMES = {"previous_return_pct": "直前先物騰落", "close_location_centered": "直前終値位置",
                 "range_pct": "直前値幅", "late_change_pct": "終盤変化", "relative_return_pct": "指数間相対騰落",
                 "is_night": "夜間区分", "growth": "成長材料点", "discount": "金利材料点",
                 "fx": "為替材料点", "cost_risk": "費用リスク材料点", "us_nasdaq_return_pct": "米NASDAQ騰落",
                 "us_dow_return_pct": "米ダウ騰落", "usdjpy_change_pct": "ドル円変化", "intercept": "切片"}


def _past_initial(rows: object, first: str) -> None:
    """初期教師全行の満了だけを先に調べ、未来値の数値解釈を防ぐ。

    引数: rows (object) は未検証の教師配列、first (str) は最初の情報締切。
    返り値: None。全行の満了・予測時点と重複を検証する。
    """
    if not isinstance(rows, list):
        raise ValueError("初期教師はrows配列が必要です。")
    cutoff = timestamp(first)
    for row in rows:
        if not isinstance(row, dict) or "label_end" not in row:
            raise ValueError("初期教師の全行に満了時刻が必要です。")
        if timestamp(row["label_end"]) >= cutoff:
            raise ValueError("最初の締切以降に満了する初期教師は数値解釈前に拒否します。")
    seen = set()
    for row in rows:
        end = timestamp(row["label_end"])
        if timestamp(row["as_of"]) >= end:
            raise ValueError("初期教師の情報締切は満了より前が必要です。")
        identity = row.get("symbol"), end.isoformat()
        if row.get("symbol") not in SYMBOLS or row.get("kind") not in ("Day", "Night") or identity in seen:
            raise ValueError("初期教師の先物・場が不正、または満了実績が重複しています。")
        seen.add(identity)


def _verify_packet(packet: dict, current: dict) -> None:
    """実測コピーを可視原本の自己ハッシュ・全体ハッシュと同じ先物基準へ束縛する。

    引数: packet (dict) は可視原本、current (dict) は元実測とバイト一致を確認する入力コピー。
    返り値: None。時点・対象・商品・状態の基準が一致しない場合は拒否する。
    """
    if packet.get("packet_hash") != digest({key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("可視原本packetの自己ハッシュが一致しません。")
    if current.get("source_packet_hash") != digest(packet):
        raise ValueError("実測入力のsource_packet_hashが可視原本全体に結び付いていません。")
    origin = timestamp(packet["as_of"])
    plan = session_plan(origin)
    if (packet.get("session") != plan["session"] or packet.get("kind") != plan["kind"]
            or timestamp(packet["targets"]["today"]) != plan["end"]
            or timestamp(current["as_of"]) != origin or current.get("session") != packet["session"]):
        raise ValueError("可視原本と実測入力の時点・対象場・終了予定が一致しません。")
    if (set(current.get("instruments", {})) != set(SYMBOLS) or set(current.get("states", {})) != set(SYMBOLS)
            or set(packet.get("instruments", {})) != set(SYMBOLS)):
        raise ValueError("可視原本・実測入力・状態は両先物だけが必要です。")
    for symbol in SYMBOLS:
        instrument, state = current["instruments"][symbol], current["states"][symbol]
        if (instrument.get("symbol") != symbol or instrument.get("kind") != plan["kind"]
                or timestamp(instrument["as_of"]) != origin):
            raise ValueError("実測特徴の先物・場・情報締切が可視原本と一致しません。")
        fixed, actual = packet["instruments"][symbol]["reference"], state["reference"]
        for reference in (fixed, actual):
            price = reference.get("price")
            if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
                raise ValueError("先物の基準価格は有限の正数が必要です。")
        if (actual.get("price") != fixed.get("price") or actual.get("session") != fixed.get("session")
                or timestamp(actual["time"]) != timestamp(fixed["time"]) or timestamp(fixed["time"]) > origin):
            raise ValueError("実測状態の先物基準価格・時刻・セッションが可視原本と一致しません。")
        if "target_session" in state and state["target_session"] != packet["session"]:
            raise ValueError("実測状態の対象セッションが可視原本と一致しません。")


def _input_provenance(inputs: Path, measured: Path, packets: list[Path], rules: Path) -> dict:
    """生成済み教師の来歴と元特徴・実測コピーを、比較の新規固定より先に照合する。

    引数: inputs (Path) は可視原本、measured (Path) は教師と実測コピー、packets (list[Path]) は固定日程、
        rules (Path) は比較へ置いた確認済み規約。
    返り値: dict は照合した全入力SHAと教師生成来歴。元価格や今回の原結果は読まない。
    """
    visible = [read_json(path) for path in packets]
    cutoffs = [timestamp(packet["as_of"]) for packet in visible]
    if any(previous >= current for previous, current in zip(cutoffs, cutoffs[1:])):
        raise ValueError("可視原本の日程は締切の厳密な昇順が必要です。")
    first = cutoffs[0].isoformat()
    teacher_path, provenance_path = measured / "初期教師.json", measured / "初期教師の来歴.json"
    teacher = read_json(teacher_path)
    _past_initial(teacher.get("rows"), first)

    # ----------------------------------------
    # 時刻検査後に、元データの保全と新たな固定で正当化できない来歴を照合する。

    provenance = read_json(provenance_path)
    if (provenance.get("provenance_hash") != digest({key: value for key, value in provenance.items() if key != "provenance_hash"})
            or provenance.get("mode") != "回想・過去教師の経路点検"):
        raise ValueError("初期教師の生成来歴の自己ハッシュまたは作成方式が不正です。")
    sources = provenance.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("初期教師の生成元SHA一覧が必要です。")
    for name, expected in sources.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not Path(name).is_file() or _file_hash(Path(name)) != expected:
            raise ValueError("初期教師の生成元ファイルが作成時のSHAと一致しません。")
    implementation = Path(__file__).with_name("prepare_five_history.py").resolve()
    engine = Path(__file__).with_name("five_class.py").resolve()
    implementation_hash, engine_hash, rules_hash = _file_hash(implementation), _file_hash(engine), _file_hash(rules)
    if (provenance.get("implementation_hash") != implementation_hash
            or sources.get(str(implementation)) != implementation_hash or sources.get(str(engine)) != engine_hash):
        raise ValueError("初期教師生成器または５分類エンジンが生成来歴と一致しません。")
    source_teachers = [Path(name) for name in sources if Path(name).name == "初期教師.json"]
    source_rules = [Path(name) for name in sources if Path(name).name == "五分類規約.json"]
    if (len(source_teachers) != 1 or source_teachers[0].resolve() == teacher_path.resolve()
            or len(source_rules) != 1 or sources[str(source_rules[0])] != rules_hash):
        raise ValueError("生成前の初期教師と確認済み規約が一意に結び付いていません。")
    if (provenance.get("teacher_hash") != _file_hash(teacher_path)
            or teacher.get("five_class_rules_hash") != rules_hash
            or timestamp(teacher["first_as_of"]) != cutoffs[0] or timestamp(provenance["first_as_of"]) != cutoffs[0]
            or provenance.get("teacher_rows") != len(teacher["rows"])):
        raise ValueError("初期教師のバイト列・件数・初回締切・規約が生成来歴と一致しません。")
    original_teacher = read_json(source_teachers[0])
    _past_initial(original_teacher.get("rows"), first)
    if len(original_teacher["rows"]) != len(teacher["rows"]):
        raise ValueError("元教師の行数を増減した初期教師は使えません。")
    for original, enriched in zip(original_teacher["rows"], teacher["rows"]):
        preserved = {key: value for key, value in enriched.items() if key != "five_class_evaluation"}
        if not isinstance(enriched.get("five_class_evaluation"), dict) or digest(preserved) != digest(original):
            raise ValueError("初期教師の元の数値・特徴・行情報がそのまま保全されていません。")
    generated_keys = {"rows", "first_as_of", "five_class_rules_hash", "label_scope"}
    if any(key not in generated_keys and teacher.get(key) != value for key, value in original_teacher.items()):
        raise ValueError("元教師の付帯情報が初期教師の作成時に変更されています。")

    # ----------------------------------------

    verified = {str(Path(name).resolve()): expected for name, expected in sources.items()}
    verified.update({str(teacher_path.resolve()): provenance["teacher_hash"], str(provenance_path.resolve()): _file_hash(provenance_path),
                     str(rules.resolve()): rules_hash})
    for path, packet in zip(packets, visible):
        original = source_teachers[0].parent / path.stem / "入力.json"
        copied = measured / path.stem / "入力.json"
        expected = sources.get(str(original.resolve()))
        if expected is None or _file_hash(copied) != expected:
            raise ValueError("実測入力コピーが生成時の元実測ファイルとバイト一致しません。")
        current = read_json(copied)
        _verify_packet(packet, current)
        verified[str(path.resolve())] = _file_hash(path)
        verified[str(copied.resolve())] = expected
    if any(_file_hash(Path(name)) != expected for name, expected in verified.items()):
        raise ValueError("初期教師来歴の照合中に入力ファイルが変更されています。")
    return {"first_as_of": first, "teacher_rows": len(teacher["rows"]),
            "provenance_hash": provenance["provenance_hash"], "files": verified}


def _verify(root: Path) -> dict:
    """事前に固定した入力・規約・実装が比較中に変わっていないことを照合する。

    引数: root (Path) は比較実験のルート。
    返り値: dict は照合済みの開始記録。
    """
    value = read_json(root / "manifest.json")
    if digest({key: item for key, item in value.items() if key != "manifest_hash"}) != value["manifest_hash"]:
        raise ValueError("比較計画の自己ハッシュが一致しません。")
    for name, expected in value["files"].items():
        if _file_hash(Path(name)) != expected:
            raise ValueError("比較前に固定した計画・入力・実装が変更されています。")
    return value


def _freeze(root: Path, inputs: Path, measured: Path, prices: Path) -> dict:
    """全日程と学習設定を固定し、既存の比較出力への上書きを拒否する。

    引数: root (Path) は新実験、inputs (Path) は旧masked原本、measured (Path) は実測入力コピー、
        prices (Path) は読み取り専用価格ルート。原結果の内容はここで読まない。
    返り値: dict は開始時点のハッシュ一覧。
    """
    if (root / "manifest.json").exists():
        raise FileExistsError("比較実験は新しい保存先で実行してください。")
    packets = sorted((inputs / "masked").glob("*.json"))
    if not packets:
        raise ValueError("事前に固定した可視原本の日程が必要です。")
    files = [root / "実験計画.md", root / "五分類規約.json", measured / "初期教師.json",
             measured / "初期教師の来歴.json"]
    files.extend(Path(__file__).parent / name for name in (
        "five_class.py", "five_class_model.py", "five_class_replay.py", "five_class_learning.py"))
    files.extend(Path("fundamentals") / name for name in (
        "pdca_measured_model.py", "model.py", "common.py", "replay.py", "prices.py", "engine.py"))
    for packet in packets:
        files.extend((packet, measured / packet.stem / "入力.json"))
    rules = read_json(root / "五分類規約.json")
    if (rules.get("confirmed") is not True or rules.get("overlap_policy") != "もみ合い優先・初回到達方向"
            or _file_hash(root / "五分類規約.json") != _file_hash(Path(__file__).with_name("五分類規約.json"))):
        raise ValueError("共通の確認済み五分類規約を固定してください。")
    checked = _input_provenance(inputs, measured, packets, root / "五分類規約.json")
    fixed = dict(checked["files"])
    for path in files:
        name, actual = str(path.resolve()), _file_hash(path)
        if name in fixed and fixed[name] != actual:
            raise ValueError("生成来歴の照合後に入力が変更されています。")
        fixed[name] = actual
    value = {"created_at": now(), "mode": "回想・既知期間の開発比較", "independent_validation": False,
             "rounds": [path.stem for path in packets], "variants": VARIANTS, "prices": str(prices.resolve()),
             "input_provenance": {key: item for key, item in checked.items() if key != "files"}, "files": fixed}
    value["manifest_hash"] = digest(value)
    write_json(root / "manifest.json", value)
    return _verify(root)


def _read_labels(prices: Path, packet: dict, rules: dict) -> list[dict]:
    """両方式の予測固定後に呼び、満了した当回の結果だけを検証して返す。

    引数: prices (Path) は元価格、packet (dict) は当回の固定した基準・締切・終点、rules (dict) は固定規約。
    返り値: list[dict] は旧予測・旧採点を取り除いた当回の経路実績。
    """
    target = timestamp(packet["targets"]["today"])
    if target > timestamp(now()):
        raise ValueError("実時計で未満了のため結果を読み取れません。")
    result = []
    for symbol in SYMBOLS:
        candles = load_candles(prices, symbol, cutoff=target)
        origin = timestamp(packet["as_of"])
        reference = reference_quote(candles, origin, session_plan(origin))
        fixed = packet["instruments"][symbol]["reference"]
        if (reference.close != fixed["price"] or reference.end != timestamp(fixed["time"])
                or reference.session != fixed["session"]):
            raise ValueError("先物の基準が事前に固定した入力と一致しません。")
        path = [bar for bar in candles if bar.session == packet["session"]]
        evaluation = evaluate_priority_path(reference, path, packet["session"], target, rules["criteria"])
        source = prices / "candle_1min" / symbol / (packet["session"] + ".csv")
        result.append({"symbol": symbol, "label_end": target.isoformat(), "evaluation": evaluation,
                       "source_file": str(source.resolve()), "source_file_hash": _file_hash(source) if source.exists() else None})
    return result


def _numerical_reason(model: dict, evaluation: dict) -> tuple[str, str]:
    """固定予測のクラス差と実績候補を対照し、当回固有の数値理由を作る。

    引数: model (dict) は結果開示前に固定したモデル出力、evaluation (dict) は当回の経路評価。
    返り値: tuple[str, str] はスコア差の説明と次の再学習で確認する具体的な項目。
    """
    chosen = model["direction"]
    if chosen is None:
        return ("正の重みを持つ過去教師がなく、分類を発行できなかった。",
                "今回の分類候補を追加し、正の重みの教師が得られるか確認。対象内の無回答は不正解として残す。")
    certain = evaluation["correct_candidates"]
    scores = model["class_scores"]
    other = certain[0] if len(certain) == 1 and certain[0] != chosen else max(
        (name for name in CLASSES if name != chosen), key=lambda name: scores[name])
    gap = scores[chosen] - scores[other]
    contributions = model.get("contributions", {})
    if not contributions:
        return (f"８行未満のため過去構成比で選択。{chosen}のスコア{scores[chosen]:.3f}、比較{other}は{scores[other]:.3f}。",
                f"今回の{('／'.join(certain or evaluation['uncertain_candidates'])) or '不明'}を追加し、少数教師の多数側へ偏る判断が維持されるか照合。")
    differences = {key: value - contributions[other].get(key, 0) for key, value in contributions[chosen].items()}
    leading = sorted(differences, key=lambda key: abs(differences[key]), reverse=True)[:3]
    pieces = []
    for key in leading:
        base = key.removesuffix("__missing")
        name = FEATURE_NAMES.get(base, base) + ("の欠測表示" if key.endswith("__missing") else "")
        pieces.append(f"{name}{differences[key]:+.3f}")
    reason = f"{chosen}−{other}の固定スコア差は{gap:+.3f}。差の主な項は{'、'.join(pieces)}。市場反応の因果ではない。"
    leading_feature = next((key for key in leading if key != "intercept"), "intercept")
    factor = FEATURE_NAMES.get(leading_feature.removesuffix("__missing"), leading_feature)
    action = f"当回教師追加後の再学習で、{factor}と{other}に対するスコア差の更新を照合。係数を当回だけに合わせて手修正しない。"
    return reason, action


def _review(path: Path, current: dict, labels: list[dict], prediction: dict, source_hash: str) -> dict:
    """当回の経路と予測を比較し、数値的な誤りと次に渡す教師を個別に保存する。

    引数: path (Path) は当回、current (dict) は可視実測入力、labels (list[dict]) は開示した経路、
        prediction (dict) は固定予測、source_hash (str) は元結果のハッシュ。
    返り値: dict は次回教師と当回の短い振り返り。因果の説明は断定しない。
    """
    rows, teachers, lines = [], [], [f"# 振り返り：{current['session']}", "", "- 決定論的モデルの数値診断。市場反応の因果は未識別。", ""]
    for label in labels:
        symbol, evaluation = label["symbol"], label["evaluation"]
        model = prediction["instruments"][symbol]
        score = score_prediction(model["direction"], evaluation)
        row = {**label, "direction": model["direction"], "score": score}
        rows.append(row)
        metrics = evaluation["metrics"]
        candidates = evaluation["correct_candidates"] or evaluation["uncertain_candidates"]
        actual = metrics["actual_pct"]
        terminal = "未観測" if actual is None else f"{actual:+.3f}％"
        width = metrics["session_range_pct"]
        width_text = "未観測" if width is None else f"{width:.3f}％"
        reason, action = _numerical_reason(model, evaluation)
        lines += [f"- {symbol}：{model['direction'] or '無回答'}→{score['status']}。終点{terminal}、観測幅{width_text}。",
                  f"  - 実績候補：{'／'.join(candidates) or '判明せず'}。もみ合い：{evaluation.get('flat_status', '不明')}。",
                  "  - 診断：" + reason, "  - 次回：" + action]
        teachers.append({**copy.deepcopy(current["instruments"][symbol]), "label_end": label["label_end"],
                         "actual_pct": actual, "five_class_evaluation": copy.deepcopy(evaluation)})
    result = {"forecast_hash": prediction["forecast_hash"], "source_result_hash": source_hash,
              "read_after_seal_at": now(), "labels": rows}
    write_json(path / "result.json", result)
    end = timestamp(labels[0]["label_end"])
    _append_audit(path, "reveal", end, [path / "result.json"])
    review = {"forecast_hash": prediction["forecast_hash"], "result_hash": digest(result),
              "available_at": end.isoformat(), "recorded_at": now(), "next_training": teachers,
              "causality": "未識別"}
    write_json(path / "review.json", review)
    (path / "振り返り.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_audit(path, "review", end, [path / "review.json", path / "振り返り.md"])
    return review


def _summarize(root: Path, identities: list[str]) -> dict:
    """固定済みの全回を再採点し、五つの定数回答と同じ分母で比較する。

    引数: root (Path) は一方式のルート、identities (list[str]) は固定した全日程。
    返り値: dict は先物別・場別の対象件数と精度境界。
    """
    groups, rows = {}, []
    for identity in identities:
        path = root / identity
        if _audit(path)[-1]["phase"] != "review":
            raise ValueError("全回の比較が完了していません。")
        current, result = read_json(path / "入力.json"), read_json(path / "result.json")
        for row in result["labels"]:
            evaluation, answer = row["evaluation"], row["direction"]
            score = score_prediction(answer, evaluation)
            if score != row["score"]:
                raise ValueError("保存した新採点が再計算と一致しません。")
            rows.append({"round": identity, "kind": current["kind"], **row})
            for key in (row["symbol"], row["symbol"] + "／" + current["kind"]):
                group = groups.setdefault(key, {**_new_group(), "baselines": {item: _new_group() for item in CLASSES}})
                _add_score(group, answer, evaluation, score)
                for constant in CLASSES:
                    _add_score(group["baselines"][constant], constant, evaluation, score_prediction(constant, evaluation))
    for group in groups.values():
        _finish_rates(group)
        for baseline in group["baselines"].values():
            _finish_rates(baseline)
    value = {"mode": "回想・既知期間の開発比較", "independent_validation": False, "operational_acceptance": False,
             "groups": groups, "rows": rows, "target_reached": all(
                 groups[s]["accuracy_lower"] is not None and groups[s]["accuracy_lower"] >= 0.8 for s in SYMBOLS)}
    write_json(root / "summary.json", value)
    return value


def run_comparison(root: Path, inputs: Path, measured: Path, prices: Path) -> dict:
    """二方式を当回結果の読取前に両方固定し、同じ履歴へ一度だけ実績を追加する。

    引数: root (Path) は新比較、inputs (Path) は可視原本、measured (Path) は実測コピー、
        prices (Path) は読み取り専用価格。並行する別の予測担当へ結果を渡さない。
    返り値: dict は二方式の集計。既知期間の結果を独立検証とはしない。
    """
    manifest = _freeze(root, inputs, measured, prices)
    rules = read_json(root / "五分類規約.json")
    history = read_json(measured / "初期教師.json")["rows"]
    for identity in manifest["rounds"]:
        _verify(root)
        packet = read_json(inputs / "masked" / (identity + ".json"))
        current = read_json(measured / identity / "入力.json")
        cutoff = timestamp(packet["as_of"])
        if current["as_of"] != packet["as_of"] or current["session"] != packet["session"]:
            raise ValueError("可視原本と実測入力の時点が一致しません。")
        if any(timestamp(row["label_end"]) >= cutoff for row in history):
            raise ValueError("締切以降に満了する教師を学習へ渡せません。")
        predictions = {}
        for mode, folder in VARIANTS.items():
            path = root / folder / identity
            visible = {"as_of": current["as_of"], "session": current["session"], "kind": packet["kind"],
                       "instruments": current["instruments"], "history": copy.deepcopy(history),
                       "source_input_hash": _file_hash(measured / identity / "入力.json"),
                       "rule_hash": _file_hash(root / "五分類規約.json")}
            write_json(path / "入力.json", visible)
            _append_audit(path, "prepare", cutoff, [path / "入力.json"])
            models = {symbol: predict_five(current["instruments"][symbol], history, current["as_of"], mode) for symbol in SYMBOLS}
            value = {"mode": "回想・既知期間の開発比較", "variant": mode, "as_of": current["as_of"],
                     "session": current["session"], "issued_at": now(), "input_hash": digest(visible), "instruments": models}
            value["forecast_hash"] = digest(value)
            write_json(path / "予測.json", value)
            lines = [f"# 予測：{current['session']}", "", f"- 方式：{mode}。締切：{current['as_of']}。既知期間の開発再生。",
                     "- 同じ先物の前セッション終値代理が基準。もみ合い優先。", ""]
            for symbol, model in models.items():
                reference = packet["instruments"][symbol]["reference"]
                lines += [f"- {symbol}：{model['direction'] or '無回答'}。基準{reference['price']}（{reference['time']}）。",
                          f"  - 過去教師{model['training_n']}行、正重み{model['positive_weight_n']}行、回帰使用{model['fitted_n']}行。クラススコアは確率ではない。"]
            lines += ["", "- 長期・確率・ニュース条件付き価格帯は未実証。", f"- 固定ハッシュ：{value['forecast_hash']}。"]
            (path / "予測.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            _append_audit(path, "seal", cutoff, [path / "予測.json", path / "予測.md"])
            predictions[mode] = value

        # ----------------------------------------
        # 当回の二つの予測が固定された後でだけ、結果を開いて教師を追加する。

        labels = _read_labels(prices, packet, rules)
        source_hash = digest(labels)
        additions = None
        for mode, folder in VARIANTS.items():
            review = _review(root / folder / identity, current, labels, predictions[mode], source_hash)
            if additions is None:
                additions = review["next_training"]
            elif additions != review["next_training"]:
                raise ValueError("二方式で次回教師が一致しません。")
        history.extend(additions)
    _verify(root)
    summary = {mode: _summarize(root / folder, manifest["rounds"]) for mode, folder in VARIANTS.items()}
    write_json(root / "学習状態.json", {"rows": history, "last_label_end": max(row["label_end"] for row in history),
               "mode": "回想・開発用。実取得時計を復元した運用履歴ではない。"})
    lines = ["# 五分類を直接学ぶ比較", "", "- 全18回・２先物を二方式で逐次予測。初期24教師は分母に含めない。",
             "- 当回の両方式の固定後だけ結果を開示。同じ過去特徴プールで前処理。", "",
             "| 方式 | 先物 | 確定対象 | 対象不明 | 下限正解／主分母 | 下限精度 | 楽観上限 | 無回答 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, result in summary.items():
        for symbol in SYMBOLS:
            group = result["groups"][symbol]
            low = "算出不能" if group["accuracy_lower"] is None else f"{group['accuracy_lower']:.1%}"
            high = "算出不能" if group["accuracy_upper"] is None else f"{group['accuracy_upper']:.1%}"
            lines.append(f"| {mode} | {symbol} | {group['target']} | {group['target_unknown']} | {group['correct_lower']}/{group['denominator']} | {low} | {high} | {group['unanswered']} |")
    lines += ["", "- いずれも既知期間の開発比較。未使用期間の80％達成・運用採用とは扱わない。",
              "- 候補均等化は学習上の仮定。真のclass頻度や確率ではなく、欠測の癖を学ぶ危険がある。"]
    (root / "総括.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """新実験の規約・入力・完了済み原結果を指定して比較を一度だけ実行する。

    引数: なし。ルート・可視原本・実測コピー・元価格をコマンド行で受け取る。
    返り値: None。二方式の先物別正解下限を表示する。
    """
    parser = argparse.ArgumentParser(description="五分類を直接学ぶ二方式の逐次比較")
    for name in ("root", "inputs", "measured", "prices"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    value = run_comparison(args.root, args.inputs, args.measured, args.prices)
    print({mode: {s: result["groups"][s]["accuracy_lower"] for s in SYMBOLS} for mode, result in value.items()})


if __name__ == "__main__":
    main()
