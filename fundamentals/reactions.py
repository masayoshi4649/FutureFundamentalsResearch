"""材料別の観測反応を、重複予測と因果効果から区別して集計する。"""

from __future__ import annotations

from .common import quantile


def reaction_statistics(comparisons: list[dict]) -> dict:
    """同じ予測内の同分類材料をまとめ、局面別の反応分布を求める。

    引数: comparisons (list[dict]) は材料と対象期間の観測騰落の対応。
    返り値: dict は分類別の標本数、中央値、分位、上昇頻度。同じ騰落を材料数で水増ししない。
    """
    observations = {}
    for item in comparisons:
        category = "／".join((item["mode"], item["symbol"], item["kind"], item["horizon"], item["category"], item["regime"]))
        identity = (category, item["forecast_hash"])
        row = observations.setdefault(identity, {"score": 0, "actual_pct": item["actual_pct"], "as_of": item["as_of"]})
        row["score"] += item["remaining_score"]
    grouped = {}
    for (category, _), row in observations.items():
        grouped.setdefault(category, []).append(row)
    result = {}
    for category, rows in grouped.items():
        values = [row["actual_pct"] for row in rows]
        result[category] = {"forecast_observations": len(rows),
            "distinct_dates": len({row["as_of"][:10] for row in rows}),
            "first_as_of": min(row["as_of"] for row in rows), "last_as_of": max(row["as_of"] for row in rows),
            "median_actual_pct": quantile(values, 0.5),
            "actual_range_pct": [quantile(values, 0.1), quantile(values, 0.9)],
            "up_frequency": sum(value > 0.05 for value in values) / len(values),
            "limitation": "全材料が共存する期間の反応。単独因果ではなく、重複する長期窓は独立標本ではない。"}
    return result
