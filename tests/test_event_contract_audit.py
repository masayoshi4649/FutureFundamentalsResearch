"""材料認定の未了と同じ元予測の二重学習を、合成価格だけで独立監査する。"""

import copy
import unittest

import test_event_window as price_fixture_module
from research_support.event_window import evaluate_event_window
from research_support.five_event_model import FEATURE_KEYS, predict_event_five


class EventContractAuditTests(unittest.TestCase):
    """ヘルパーの正規出力を境界で変更し、学習側の検証漏れを確認する。"""

    def setUp(self) -> None:
        """合成価格と13列だけを使い、実ニュース・原価格へのアクセスを避ける。

        引数: self (EventContractAuditTests) はテスト。
        返り値: None。
        """
        self.price = price_fixture_module.EventWindowTests()
        self.price.setUp()
        self.cutoff = "2026-10-15T07:45:00+09:00"
        self.features = dict.fromkeys(FEATURE_KEYS, 0.0)
        self.current = {"symbol": "NK225", "kind": "Day", "as_of": self.cutoff, "features": dict(self.features)}

    def _teacher(self, event: dict) -> dict:
        """指定材料の前までの正規評価から、学習へ渡す一行を作る。

        引数: self (EventContractAuditTests) はテスト、event (dict) は合成材料。
        返り値: dict は利用可能時計と評価窓を持つ教師。
        """
        evaluation = evaluate_event_window(self.price.reference, self.price.bars, self.price.session,
            self.price.as_of, [event], self.price.contract)
        return {"symbol": "NK225", "kind": "Day", "as_of": self.price.as_of.isoformat(),
                "features": dict(self.features), "label_end": evaluation["evaluation_window"]["label_available_at"],
                "five_class_evaluation": evaluation}

    def test_pending_list_cannot_be_enabled_as_teacher_by_clearing_only_flag(self) -> None:
        """先行保留の一覧があるのに認定不明フラグだけ落とした教師を拒否する。

        引数: self (EventContractAuditTests) はテスト。
        返り値: None。
        """
        pending = {**self.price.event, "decision": "保留", "schedule_status": "不明"}
        row = self._teacher(pending)
        accepted = predict_event_five(self.current, [row], self.cutoff, "候補配分")
        self.assertEqual(accepted["weights"], [0.0])
        row["five_class_evaluation"]["evaluation_window"]["adjudication_unknown"] = False
        with self.assertRaises(ValueError):
            predict_event_five(self.current, [row], self.cutoff, "候補配分")

    def test_one_forecast_cannot_be_counted_twice_at_different_event_clocks(self) -> None:
        """同一先物・同一元締切の教師を、公表時計だけ変えて二標本へ増やせない。

        引数: self (EventContractAuditTests) はテスト。
        返り値: None。
        """
        early = self._teacher(self.price.event)
        late_event = {**copy.deepcopy(self.price.event), "event_id": "別の材料", "first_public_at": "2026-10-01T10:00:30+09:00"}
        late = self._teacher(late_event)
        self.assertNotEqual(early["label_end"], late["label_end"])
        self.assertEqual(early["as_of"], late["as_of"])
        for mode in ("確定教師", "候補配分"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                predict_event_five(self.current, [early, late], self.cutoff, mode)


if __name__ == "__main__":
    unittest.main()
