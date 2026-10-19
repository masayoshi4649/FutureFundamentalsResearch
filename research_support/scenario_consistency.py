"""本人の主シナリオだけを五分類と照合し、市場の方向や材料点は生成しない。"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from fundamentals.common import now, read_json, timestamp, write_json
from research_support.five_class import CLASSES, _decimal, _priority_price, validate_contract


def validate_center_scenario(path_pct: list, answer: str, bounds: dict, contract: dict) -> dict:
    """本人が指定した連続経路の頂点列と主回答・数値帯の内部整合を検査する。

    引数: path_pct (list) は寄付きから終点までのP0比％の頂点列、answer (str) は主回答、
        bounds (dict) はopen/high/low/end/rangeの想定帯、contract (dict) は既存五分類のcriteria。
    返り値: dict は整合した中心経路の分類と数値。実市場での精度や帯の被覆率は評価しない。
    """
    rules = validate_contract(contract)
    if (rules.target_basis != "場中到達" or rules.movement_threshold_pct != rules.initial_excursion_pct
            or rules.reversal_requires_reference_cross):
        raise ValueError("既存の場中到達・終点跨ぎ不要の優先分類契約が必要です。")
    if answer not in CLASSES or not isinstance(path_pct, list) or len(path_pct) < 2:
        raise ValueError("五択の主回答と、寄付き・終点を含む２点以上の頂点列が必要です。")
    values = [_decimal(value, "中心経路の頂点") for value in path_pct]
    if min(values) <= -100:
        raise ValueError("P0比−100％以下の価格は指定できません。")
    metrics = {"open": values[0], "high": max(values), "low": min(values),
               "end": values[-1], "range": max(values) - min(values)}
    if (metrics["range"] <= rules.range_threshold_pct
            and abs(metrics["end"]) <= rules.flat_terminal_threshold_pct):
        classification = "もみ合い"
    else:
        state = (0, False, Decimal(0))
        for value in values:
            state = _priority_price(state, Decimal(100) + value,
                                    Decimal(100) + rules.initial_excursion_pct,
                                    Decimal(100) - rules.initial_excursion_pct, rules.reversal_threshold_pct)
        names = {(1, False): "上がる", (-1, False): "下がる",
                 (1, True): "上がってから下がる", (-1, True): "下がってから上がる"}
        if state[:2] not in names:
            raise ValueError("中心経路の到達方向を分類できません。")
        classification = names[state[:2]]
    if answer != classification:
        raise ValueError(f"中心経路は『{classification}』ですが、主回答は『{answer}』です。")

    # ----------------------------------------
    # 帯の周辺被覆や同時実現を仮定せず、指定中心経路だけが帯内にあるか確認する。
    if not isinstance(bounds, dict) or not set(metrics) <= set(bounds):
        raise ValueError("寄付き・高値・安値・終点・幅の５つの想定帯が必要です。")
    normalized = {}
    for key, actual in metrics.items():
        band = bounds[key]
        if isinstance(band, dict) and set(band) == {"min", "max"}:
            lo, hi = band["min"], band["max"]
        elif isinstance(band, list) and len(band) == 2:
            lo, hi = band
        else:
            raise ValueError(f"{key}の想定帯は[min,max]かmin/max辞書で指定してください。")
        low, high = _decimal(lo, key), _decimal(hi, key)
        if low > high or not low <= actual <= high:
            raise ValueError(f"中心経路の{key}={actual}が想定帯と整合しません。")
        normalized[key] = [str(low), str(high)]
    return {"classification": classification, "center_vertices_pct": [str(value) for value in values],
            "center_metrics_pct": {key: str(value) for key, value in metrics.items()},
            "bounds_pct": normalized,
            "notice": "指定頂点間を連続して結ぶ主シナリオだけの検査。未指定の往復、実市場の予測精度、確率、帯の被覆率は証明しない。"}


def main() -> None:
    """本人の数値入力だけを読み、検査成功時の証跡を新規ファイルへ固定する。

    引数: なし。CLIの--inputは本人入力JSON、--outは新規保存先、--contractは固定規約、
        --judgment/--actorは固定前の最終判断と担当記録。
    返り値: None。既存の予測・判断・材料点・数値入力は変更しない。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("research_support/五分類規約.json"))
    args = parser.parse_args()
    data, contract = read_json(args.input), read_json(args.contract)
    if (set(data) != {"as_of", "session", "scenarios"}
            or set(data["scenarios"]) != {"NK225", "TOPIX"}
            or contract.get("confirmed") is not True
            or contract.get("overlap_policy") != "もみ合い優先・初回到達方向"):
        raise ValueError("当回・両先物の本人入力と確認済み優先分類契約が必要です。")
    results = validate_final_scenarios(data, read_json(args.judgment), read_json(args.actor), contract["criteria"])

    # ----------------------------------------
    # 元入力・規約・実行コードを本文固定対象JSONに複製し、後日の変更と分ける。
    paths = [args.input, args.contract, Path(__file__), Path("research_support/five_class.py")]
    sources = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "utf8_text": path.read_text(encoding="utf-8")} for path in paths]
    output = {"version": "主シナリオ整合性１", "recorded_at": now(), "as_of": data["as_of"],
              "session": data["session"], "results": results, "sources": sources,
              "final_bindings": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                 for path in (args.judgment, args.actor)]}
    write_json(args.out, output)
    print(json.dumps({"保存": str(args.out), "中心経路の分類": {key: value["classification"]
                                                          for key, value in results.items()}}, ensure_ascii=False))


def validate_final_scenarios(data: dict, judgment: dict, actor: dict, contract: dict) -> dict:
    """中心経路の主回答・帯・対象時点を最終判断と担当記録へ結び付ける。

    引数: data (dict) は本人の中心経路入力、judgment (dict) は最終材料判断、actor (dict) は担当記録、
        contract (dict) は五分類criteria。
    返り値: dict は両先物の整合検査結果。元の入力や回答は変更しない。
    """
    if (set(data) != {"as_of", "session", "scenarios"}
            or set(data["scenarios"]) != {"NK225", "TOPIX"}
            or data["session"] != actor["session"]
            or timestamp(data["as_of"]) != timestamp(actor["as_of"])
            or timestamp(data["as_of"]) != timestamp(judgment["as_of"])):
        raise ValueError("中心経路入力と最終判断の対象場・時点が一致しません。")
    results = {}
    for symbol, item in data["scenarios"].items():
        if set(item) != {"answer", "center_vertices_pct", "bounds_pct"}:
            raise ValueError("各先物は主回答・本人の頂点列・本人の想定帯だけを指定してください。")
        if item["answer"] != judgment["forecasts"][symbol]["today"]["direction"]:
            raise ValueError("中心経路の主回答が最終予測と一致しません。")
        result = validate_center_scenario(item["center_vertices_pct"], item["answer"], item["bounds_pct"], contract)
        recorded = actor["analysis"]["scenario_bounds"][symbol]
        for key, expected in result["bounds_pct"].items():
            band = recorded[key]
            pair = [band["min"], band["max"]] if isinstance(band, dict) else band
            if (not isinstance(pair, list) or len(pair) != 2
                    or [_decimal(value, key) for value in pair] != [Decimal(value) for value in expected]):
                raise ValueError("中心経路の想定帯が最終担当記録と一致しません。")
        results[symbol] = result
    return results


if __name__ == "__main__":
    main()
