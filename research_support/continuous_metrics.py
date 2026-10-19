"""固定分母の成績と、0.5％対象内の条件付き上限を区別する。"""

from collections import Counter


def summarize_scores(scores: list[dict]) -> dict:
    """未確定対象を消さず、正答率の分母と上限の意味を分けて集計する。

    引数: scores (list[dict]) は固定予測の採点結果。
    返り値: dict は件数、固定分母の下限・可能率、対象内精度の楽観上限。
        楽観上限では、対象未確定で正解不能の回が対象外だった可能性も残す。
        回ごとの候補間の制約を越えた確定正解は作らない。
    """
    included = [row for row in scores if row["denominator"]]
    if any(row["target_status"] not in ("対象", "未確定") for row in included):
        raise ValueError("成績分母には確定対象か対象未確定だけを含めます。")
    if any(row["correct_lower"] and (not row["correct_upper"] or row["target_status"] != "対象") for row in scores):
        raise ValueError("確定正解と対象・可能正解が不整合です。")
    certain = [row for row in included if row["target_status"] == "対象"]
    unknown = [row for row in included if row["target_status"] == "未確定"]
    lower = sum(bool(row["correct_lower"]) for row in included)
    possible = sum(bool(row["correct_upper"]) for row in included)
    unknown_possible = sum(bool(row["correct_upper"]) for row in unknown)
    optimistic_denominator = len(certain) + unknown_possible
    optimistic_correct = sum(bool(row["correct_upper"]) for row in certain) + unknown_possible

    # ----------------------------------------

    return {"planned": len(scores), "statuses": dict(Counter(row["status"] for row in scores)),
            "denominator": len(included), "certain_target": len(certain), "unknown_target": len(unknown),
            "unanswered": sum(bool(row["unanswered"]) for row in scores),
            "unanswered_in_target_denominator": sum(bool(row["unanswered"]) for row in included),
            "correct_lower": lower, "correct_upper_on_fixed_denominator": possible,
            "fixed_denominator_lower_pct": 100 * lower / len(included) if included else None,
            "fixed_denominator_possible_pct": 100 * possible / len(included) if included else None,
            "conditional_target_optimistic_upper_pct": 100 * optimistic_correct / optimistic_denominator if optimistic_denominator else None,
            "optimistic_target_denominator": optimistic_denominator,
            "notice": "固定分母の可能正解率と、真に0.5％へ到達した対象内精度の楽観上限は別。どちらも実際の達成率とは限らない。"}
