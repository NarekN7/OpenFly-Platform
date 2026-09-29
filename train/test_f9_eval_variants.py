"""Regression checks that custom F9 evals retain the corrected eval protocol."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
SCRIPTS = ROOT / "scripts"


def _without_module_docstring(source: str) -> str:
    tree = ast.parse(source)
    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return source.lstrip("\n")
    return "".join(source.splitlines(keepends=True)[first.end_lineno :]).lstrip("\n")


class F9EvalVariantsTest(unittest.TestCase):
    def test_distance_variants_are_corrected_eval_plus_displacement_only(self):
        base = (TRAIN / "eval.py").read_text(encoding="utf-8")
        map_old = (
            '"10": np.array([0, 27, 0, 0, 0, 0, 0, 0]).astype(np.float32),  '
            "# move forward 27 (9x3)"
        )
        pose_old = (
            "    elif action == 10:\n"
            "        x += step_size * math.cos(yaw) * 9\n"
            "        y += step_size * math.sin(yaw) * 9"
        )
        self.assertEqual(base.count(map_old), 1)
        self.assertEqual(base.count(pose_old), 1)

        for meters, multiplier in ((15, 5), (18, 6), (21, 7)):
            with self.subTest(meters=meters):
                expected = base.replace(
                    map_old,
                    f'"10": np.array([0, {meters}, 0, 0, 0, 0, 0, 0]).astype(np.float32),  '
                    f"# move forward {meters} ({multiplier}x3); default is 27 (9x3)",
                    1,
                ).replace(
                    pose_old,
                    "    elif action == 10:\n"
                    f"        # {meters} m forward ({multiplier} * step_size); "
                    "eval.py uses * 9 (= 27 m)\n"
                    f"        x += step_size * math.cos(yaw) * {multiplier}\n"
                    f"        y += step_size * math.sin(yaw) * {multiplier}",
                    1,
                )
                actual = (TRAIN / f"eval_f9_{meters}m.py").read_text(encoding="utf-8")
                self.assertEqual(
                    _without_module_docstring(actual),
                    _without_module_docstring(expected),
                )

    def test_short_forward_keeps_corrected_action_protocol(self):
        source = (TRAIN / "eval_f9_short_forward.py").read_text(encoding="utf-8")
        for required in (
            "action_stopping_criteria",
            "parsed_action = _parse_vln_action_id(text_out)",
            "prediction_detail=action_detail",
            "qwen_action_outputs.append(action_detail)",
            '"qwen_action_parser"',
            '"qwen_n_invalid_actions"',
            '"stopped_on_invalid_action"',
            "simulator_action=model_action",
            '"f9_kind"',
            '"num_history_actions"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn('re.search(r"[0-9]"', source)

    def test_short_forward_rewrite_table(self):
        source = (TRAIN / "eval_f9_short_forward.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "resolve_f9_substitution"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {
            "List": List,
            "Sequence": Sequence,
            "Tuple": Tuple,
        }
        exec(compile(ast.fix_missing_locations(module), "<f9-rewrite>", "exec"), namespace)
        resolve = namespace["resolve_f9_substitution"]
        cases = (
            ([], 2, (2, [2], "passthrough")),
            ([], 10, (8, [8], "append")),
            ([8], 10, (1, [9], "replace")),
            ([9], 10, (8, [9, 8], "append")),
            ([9, 8], 10, (1, [9, 9], "replace")),
            ([9, 9], 10, (8, [9, 9, 8], "append")),
            ([9, 9, 8], 10, (1, [10], "collapse")),
            ([2, 9, 9, 8], 10, (1, [2, 10], "collapse")),
        )
        for history, prediction, expected in cases:
            with self.subTest(history=history, prediction=prediction):
                self.assertEqual(resolve(history, prediction), expected)

    def test_launchers_delegate_to_corrected_generic_runner(self):
        for name in ("15m", "18m", "21m", "short_forward"):
            with self.subTest(name=name):
                source = (
                    SCRIPTS / f"run_qwen_eval_f9_{name}_per_airsim_env.sh"
                ).read_text(encoding="utf-8")
                self.assertIn("run_qwen_eval_per_airsim_env.sh", source)
                self.assertIn("seenx9.json", source)
                self.assertNotIn("openfly_of3_env.sh", source)

    def test_combined_runner_defaults_to_x9_data(self):
        source = (SCRIPTS / "run_qwen_combined_eval.sh").read_text(encoding="utf-8")
        for expected in (
            "data_curated/seenx9.json",
            "left_evaluation_skill_validation_x9.json",
            "right_evaluation_skill_validation_x9.json",
            "stop_evaluation_skill_validation_x9.json",
            "left_evaluation_skill_test_x9.json",
            "right_evaluation_skill_test_x9.json",
            "stop_evaluation_skill_test_x9.json",
            "trainx9_intrain_eval_500.json",
            "OPENFLY_COMBINED_SKIP_CLOSED_LOOP",
            "OPENFLY_EVAL_PY",
            "export DATA_DIR=",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
