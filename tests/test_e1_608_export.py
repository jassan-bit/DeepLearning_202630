"""Focused metadata guards; no fitted model or fabricated closed execution.

Toy numeric values exercise helpers only and never create a public report,
registration receipt, model checkpoint, or market data file.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

MODULE = Path(__file__).resolve().parents[1] / "scripts/export_e1_608_remediation.py"
SPEC = importlib.util.spec_from_file_location("e1_608_export_test", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class Public608ExportGuards(unittest.TestCase):
    @staticmethod
    def panel():
        return pd.DataFrame([{"model": model, "seed": seed, "study_id": exporter.STUDY,
            "execution_id": "unit-test-only", "source_signature": "unit-test-sources"}
            for model in exporter.MODELS for seed in (18101, 18102)])

    def test_missing_or_mixed_execution_identity_rejected(self):
        frame = self.panel()
        exporter.identity(frame, "unit-test-only", "unit-test-sources")
        for field in ("study_id", "execution_id", "source_signature"):
            changed = frame.copy()
            changed.loc[0, field] = "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                exporter.identity(changed, "unit-test-only", "unit-test-sources")
        with self.assertRaises(ValueError):
            exporter.identity(frame.drop(columns="execution_id"), "unit-test-only")

    def test_exact_three_family_panel_and_integer_seeds(self):
        frame = self.panel()
        exporter.panel(frame, (18101, 18102))
        for changed in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                frame.assign(model=frame.model.replace("MLP", "TCN")), frame.assign(seed=18101.5)):
            with self.subTest(rows=len(changed)), self.assertRaises(ValueError):
                exporter.panel(changed, (18101, 18102))

    def test_persistence_has_one_reference(self):
        frame = pd.concat([self.panel(), pd.DataFrame([{"model": "Persistence", "seed": 0}])])
        exporter.panel(frame, (18101, 18102), persistence=True)
        with self.assertRaises(ValueError):
            exporter.panel(pd.concat([frame, frame.iloc[-1:]]), (18101, 18102), persistence=True)

    def test_plan_uses_actual_required_R_and_registered_disjoint_seeds(self):
        plan = {"R": 30, "R_final": 30, "R_required": 12, "seeds": list(range(18101, 18131))}
        self.assertEqual(exporter.verify_plan(plan), 30)
        for override in ({"R_required": 31}, {"R": 31, "R_final": 31},
                {"seeds": list(range(8101, 8131))}, {"R": 30.5}, {"R": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                exporter.verify_plan({**plan, **override})

    def test_complete608_support_and_one_scoring_pass_are_required(self):
        score = {"trajectories": 90, "dates": 608, "prediction_rows": 608*90,
            "scoring_passes_per_checkpoint": 1, "historical_exploratory_not_confirmatory": True}
        exporter.verify_support(score, 30)
        for override in ({"dates": 60}, {"prediction_rows": 608*89},
                {"trajectories": 89}, {"scoring_passes_per_checkpoint": 2}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                exporter.verify_support({**score, **override}, 30)

    def test_phase_timeline_rejects_retroactive_publication_or_access(self):
        # These are helper inputs only, not execution receipts or closure evidence.
        points = {"source_published": "2020-01-01T00:00:01+00:00",
            "hpo": "2020-01-01T00:00:02+00:00", "pilot": "2020-01-01T00:00:03+00:00",
            "plan_published": "2020-01-01T00:00:04+00:00", "final_fit": "2020-01-01T00:00:05+00:00",
            "frozen_weights": "2020-01-01T00:00:06+00:00", "price_access": "2020-01-01T00:00:07+00:00"}
        exporter.verify_timeline(points)
        for key in ("source_published", "plan_published", "frozen_weights"):
            changed = {**points, key: "2020-01-01T00:00:08+00:00"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "chronology"):
                exporter.verify_timeline(changed)
        with self.assertRaises(ValueError):
            exporter.moment("2020-01-01T00:00:00")

    def test_private_columns_and_unknown_columns_rejected(self):
        for key in ("y_true", "outcome_day", "weights", "mase_train_scale", "fitted_scaler_mean", "daily_prediction"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                exporter.public_frame(pd.DataFrame({key: [1]}))
        with self.assertRaisesRegex(ValueError, "closed metadata allowlist"):
            exporter.public_frame(pd.DataFrame({"unreviewed_column": [1]}), exporter.ANALYSIS_FIELDS)
        safe = exporter.projected_runs(pd.DataFrame([{"model": "MLP", "rmse": .1, "mase_scale": .2, "y_pred": .3}]))
        self.assertEqual(set(safe), {"model", "rmse"})
        with self.assertRaises(ValueError):
            exporter.no_private_metadata({"runtime": {"scalers": [1, 2]}})

    def test_refit_has_no_validation_and_inner_losses_are_observed(self):
        curve = pd.DataFrame({"epoch": [1, 2], "stage": ["refit"]*2,
            "optimization_mse_scaled": [.3, .2], "train_mse_scaled_eval": [.2, .1],
            "validation_mse_scaled_eval": [np.nan]*2, "best_epoch_so_far": [1, 2],
            "learning_rate": [.001]*2})
        exporter.validate_curve(curve, "refit", 2)
        with self.assertRaises(ValueError):
            exporter.validate_curve(curve.fillna(.1), "refit", 2)
        inner = curve.assign(stage="inner", validation_mse_scaled_eval=[.3, .2])
        exporter.validate_curve(inner, "inner", 2)
        with self.assertRaises(ValueError):
            exporter.validate_curve(inner.assign(train_mse_scaled_eval=[.1, np.inf]), "inner", 2)

    def test_curve_aggregation_reports_surviving_runs(self):
        curves = pd.DataFrame([{"model": "GRU", "stage": "inner", "epoch": epoch,
            "optimization_mse_scaled": value, "train_mse_scaled_eval": value,
            "validation_mse_scaled_eval": value} for epoch, value in ((1, 1.), (1, 3.), (2, 2.))])
        result = exporter.aggregate_curves(curves)
        rows = result.loc[result.metric.eq("train_mse_scaled_eval")].set_index("epoch")
        self.assertEqual(rows.n_available.to_dict(), {1: 2, 2: 1})
        self.assertEqual(rows.loc[1, "mean"], 2.)
        self.assertEqual(rows.loc[1, "q25"], 1.5)
        self.assertEqual(rows.loc[1, "q75"], 2.5)

    def test_qq_requires_actual_observed_difference_values(self):
        from itertools import combinations
        runs = pd.DataFrame([{"model": model, "seed": seed, "rmse": .1 + index*.02 + j*.001*(index+1)}
            for index, model in enumerate(exporter.MODELS) for j, seed in enumerate((18101, 18102, 18103))])
        pivot = runs.pivot(index="seed", columns="model", values="rmse")
        rows = []
        for a, b in combinations(exporter.MODELS, 2):
            values = (pivot[a] - pivot[b]).to_numpy()
            x, y = stats.probplot(values, fit=False)
            rows.extend({"model_a": a, "model_b": b, "n_paired": 3, "order_index": index,
                "theoretical_normal_quantile": xx, "ordered_difference_a_minus_b": yy,
                "ordered_standardized_difference": (yy-values.mean())/values.std(ddof=1)}
                for index, (xx, yy) in enumerate(zip(x, y), 1))
        qq = pd.DataFrame(rows)
        exporter.verify_qq(qq, runs, (18101, 18102, 18103))
        qq.loc[0, "ordered_difference_a_minus_b"] += .01
        with self.assertRaisesRegex(ValueError, "observed paired"):
            exporter.verify_qq(qq, runs, (18101, 18102, 18103))

    def test_hash_boundaries_symlinks_and_missing_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, source = root / "private", root / "source"
            private.mkdir()
            source.mkdir()
            payload = private / "metadata.txt"
            payload.write_text("unit test metadata only")
            mapping = {"metadata.txt": exporter.sha(payload)}
            exporter.verify_mapping(private, mapping)
            with self.assertRaises(ValueError):
                exporter.verify_mapping(private, {"../escape": "0"*64})
            with self.assertRaises(ValueError):
                exporter.bound_file(private, mapping, "analysis/summary.json")
            payload.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                exporter.verify_mapping(private, mapping)
            link = root / "linked"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                exporter.destination(link, source, private)
            with self.assertRaises(ValueError):
                exporter.destination(private / "public", source, private)
            with self.assertRaises(ValueError):
                exporter.verify(source, private)

    def test_public_allowlist_contains_full_curves_registered_diagnostics_and_costs(self):
        self.assertTrue({"learning_curves_by_run.csv", "learning_curve_summary.csv", "qq_data.csv",
            "friedman_rmse.csv", "wilcoxon_rmse.csv", "first_seed_learning_curves.csv",
            "run_costs.csv", "pilot_summary.csv", "cost_summary.csv"} <= exporter.PUBLIC_FILES)
        self.assertFalse(any("prediction" in name or name.endswith(".pt") for name in exporter.PUBLIC_FILES))


if __name__ == "__main__":
    unittest.main()
