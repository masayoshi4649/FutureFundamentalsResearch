"""開示済みの当回価格だけから、観測上の到達・反転の時間帯を補足する。"""

import argparse
from decimal import Decimal
from pathlib import Path

from fundamentals.common import now, read_json, timestamp, write_json
from fundamentals.prices import Candle, _read_file
from fundamentals.replay import _audit, _file_hash


def bar_record(bar: Candle, field: str) -> dict:
    """観測足の時間帯と対象価格を記録する。

    引数: bar (Candle) は確定足、field (str) はhigh/low等の価格欄。
    返り値: dict は足の開始・終了と価格。約定の厳密時計ではない。
    """
    return {"start": bar.time.isoformat(), "end": bar.end.isoformat(),
            "field": field, "price": getattr(bar, field)}


def observed_path(bars: list[Candle], reference: float) -> dict:
    """観測した足だけで初回到達と別足の反転を探す。

    引数: bars (list[Candle]) は時間順の当回確定足、reference (float) は固定P0。
    返り値: dict は観測上の到達足と、先行する別足の極値から0.5％反転した足。
        欠測中の初回・反転や同一足の高安順序を補完せず、五分類を再採点しない。
    """
    base = Decimal(str(reference))
    if not base.is_finite() or base <= 0:
        raise ValueError("固定P0は有限の正数が必要です。")
    if any(not bar.closed for bar in bars):
        raise ValueError("確定足だけを補足へ渡してください。")
    if any(previous.end > current.time for previous, current in zip(bars, bars[1:])):
        raise ValueError("観測足の順序または時間帯が不整合です。")
    threshold = base * Decimal("0.005")
    result = {}
    for name, field, initial, reverse_field in (
            ("up", "high", base + threshold, "low"),
            ("down", "low", base - threshold, "high")):
        index = next((i for i, bar in enumerate(bars)
                      if (Decimal(str(bar.high)) >= initial if name == "up"
                          else Decimal(str(bar.low)) <= initial)), None)
        entry = {"first_observed_touch": None, "later_bar_reversal": None}
        if index is not None:
            extreme = bars[index]
            entry["first_observed_touch"] = bar_record(extreme, field)
            for bar in bars[index + 1:]:
                prior = Decimal(str(getattr(extreme, field)))
                current = Decimal(str(getattr(bar, reverse_field)))
                if (prior - current if name == "up" else current - prior) >= threshold:
                    entry["later_bar_reversal"] = {"prior_extreme": bar_record(extreme, field),
                                                   "reversal_bar": bar_record(bar, reverse_field)}
                    break
                if (getattr(bar, field) > getattr(extreme, field) if name == "up"
                        else getattr(bar, field) < getattr(extreme, field)):
                    extreme = bar
        result[name] = entry
    return result


def build_observed_supplement(root: Path) -> dict:
    """価格開示後だけ、固定した当回CSVから観測時刻の補足を作る。

    引数: root (Path) は当回記録先。
    返り値: dict は観測経路・原本SHA・限界。元の価格や採点を変更しない。
    """
    audit = _audit(root)
    if not audit or audit[-1]["phase"] not in ("reveal", "review"):
        raise ValueError("価格開示後の場だけを補足できます。")
    result_path = root / "result.json"
    result = read_json(result_path)
    sources, rows = {str(result_path.resolve()): _file_hash(result_path)}, []
    for label in result["labels"]:
        original = label["full_reference"]
        path = Path(original["source_file"])
        if _file_hash(path) != original["source_file_hash"]:
            raise ValueError("価格CSVが結果固定時と一致しません。")
        bars = [bar for bar in _read_file(path, label["symbol"], 1, timestamp(original["label_end"])) if bar.closed]
        base = original["evaluation"]["metrics"]["reference_price"]
        rows.append({"symbol": label["symbol"], "reference_price": base,
                     "observed_path": observed_path(bars, base)})
        sources[str(path.resolve())] = original["source_file_hash"]

    # ----------------------------------------

    value = {"created_at": now(), "rows": rows, "source_hashes": sources,
             "notice": "全場参考の観測足だけの補足。最初に観測した足は真の初回を保証しない。別足の極値からの反転だけを提示し、足内順序・欠測・材料因果を補完しない。主採点と全場参考のラベルは変更しない。"}
    return value


def main() -> None:
    """価格開示後だけ補足JSONを未使用の保存先へ書き出す。

    引数: なし。root (Path) はコマンド行で指定する当回記録先。
    返り値: None。既存の結果・ラベル・振り返りは変更しない。
    """
    parser = argparse.ArgumentParser(description="開示済み当回の観測経路だけを補足")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = build_observed_supplement(args.root)
    write_json(args.root / "観測経路の補足.json", value)
    print(value["rows"])


if __name__ == "__main__":
    main()
