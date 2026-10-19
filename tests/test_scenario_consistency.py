"""境界・経路順序・帯と主回答の不整合を合成入力で確認する。"""

import copy
import unittest

from research_support.scenario_consistency import validate_center_scenario, validate_final_scenarios


CONTRACT = {"target_basis": "場中到達", "movement_threshold_pct": "0.5", "direction_threshold_pct": "0.5",
            "initial_excursion_pct": "0.5", "reversal_threshold_pct": "0.5", "reversal_requires_reference_cross": False,
            "range_threshold_pct": "1", "flat_terminal_threshold_pct": "0.5", "interval_minutes": 1,
            "reference_max_lag_minutes": 10, "endpoint_max_lag_minutes": 10}
BOUNDS = {"open": [-2, 2], "high": [-2, 2], "low": [-2, 2], "end": [-2, 2], "range": [0, 4]}


class CenterScenarioTests(unittest.TestCase):
    """本人が指定したシナリオの論理だけを検証する。"""

    def test_flat_wins_at_boundary(self):
        """幅1％・終点0.5％の重複はもみ合いになる。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        result = validate_center_scenario(["0.5", "-0.5", "0.5"], "もみ合い", BOUNDS, CONTRACT)
        self.assertEqual(result["center_metrics_pct"]["range"], "1.0")

    def test_no_need_to_cross_reference(self):
        """負終点でも極値から0.5％の反発なら下後上になる。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        self.assertEqual(validate_center_scenario([0, -1.2, -0.7], "下がってから上がる", BOUNDS, CONTRACT)
                         ["classification"], "下がってから上がる")

    def test_initial_direction_wins_multiple_swings(self):
        """最初の到達側を、その後の再反転で変更しない。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        self.assertEqual(validate_center_scenario([0, 0.6, -0.6, 0.8], "上がってから下がる", BOUNDS, CONTRACT)
                         ["classification"], "上がってから下がる")

    def test_opening_gap_is_first_trade(self):
        """P0を場内高安へ含めず、寄付きギャップから最初の到達を判定する。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        result = validate_center_scenario([-0.8, -0.9, -0.6], "下がる", BOUNDS, CONTRACT)
        self.assertEqual(result["center_metrics_pct"]["high"], "-0.6")
        self.assertEqual(result["center_metrics_pct"]["range"], "0.3")

    def test_no_rounding_to_half(self):
        """0.499999％を0.5％に丸めて先行側にしない。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        self.assertEqual(validate_center_scenario([0, "0.499999", -0.7, -0.6], "下がる", BOUNDS, CONTRACT)
                         ["classification"], "下がる")

    def test_inconsistent_answer_rejected(self):
        """中心経路がもみ合いなのに反転型とする回答を拒否する。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        with self.assertRaisesRegex(ValueError, "もみ合い"):
            validate_center_scenario([-0.1, 0.4, -0.6, -0.05], "下がってから上がる", BOUNDS, CONTRACT)

    def test_center_outside_envelope_rejected(self):
        """帯内にない中心高値を拒否する。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        bounds = {**BOUNDS, "high": {"min": 0, "max": 0.4}}
        with self.assertRaisesRegex(ValueError, "high"):
            validate_center_scenario([0, 0.8, 0.7], "上がる", bounds, CONTRACT)

    def test_redundant_range_bound_is_not_impossible(self):
        """中心が整合すれば周辺帯の冗長な幅上限だけで不可能扱いしない。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        bounds = {"open": [-0.1, 0.1], "high": [0.7, 0.9], "low": [-0.1, 0],
                  "end": [0.6, 0.8], "range": [0.7, 1.5]}
        self.assertEqual(validate_center_scenario([0, 0.8, 0.7], "上がる", bounds, CONTRACT)
                         ["classification"], "上がる")

    def test_nonfinite_and_boolean_rejected(self):
        """非有限・真偽値の頂点を拒否する。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        for value in (True, "NaN", "Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_center_scenario([0, value], "もみ合い", BOUNDS, CONTRACT)

    def test_final_values_must_match(self):
        """反証後の最終回答・帯・時点が変われば再検査を要求する。

        引数: self (CenterScenarioTests)。返り値: None。
        """
        data = {"as_of": "2026-09-29T16:00:00+09:00", "session": "2026-09-29_Night",
                "scenarios": {symbol: {"answer": "上がる", "center_vertices_pct": [0, 0.8, 0.7],
                                       "bounds_pct": copy.deepcopy(BOUNDS)} for symbol in ("NK225", "TOPIX")}}
        judgment = {"as_of": data["as_of"], "forecasts": {symbol: {"today": {"direction": "上がる"}}
                                                        for symbol in ("NK225", "TOPIX")}}
        actor = {"as_of": data["as_of"], "session": data["session"], "analysis": {"scenario_bounds": {
            symbol: {key: {"min": value[0], "max": value[1]} for key, value in BOUNDS.items()}
            for symbol in ("NK225", "TOPIX")}}}
        self.assertEqual(len(validate_final_scenarios(data, judgment, actor, CONTRACT)), 2)
        changes = [("回答", lambda value: value[0]["forecasts"]["NK225"]["today"].update(direction="下がる")),
                   ("帯", lambda value: value[1]["analysis"]["scenario_bounds"]["TOPIX"]["high"].update(max=3)),
                   ("時点", lambda value: value[1].update(as_of="2026-09-30T16:00:00+09:00"))]
        for name, change in changes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                pair = [copy.deepcopy(judgment), copy.deepcopy(actor)]
                change(pair)
                validate_final_scenarios(data, *pair, CONTRACT)


if __name__ == "__main__":
    unittest.main()
