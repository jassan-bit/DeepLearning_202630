"""Pure stdlib checks for enumerations, sampling and experiment accounting."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e2_search_plan.py"
SPEC = importlib.util.spec_from_file_location("e2_search_plan", SCRIPT)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def fixture():
    models = []
    for i in range(11):
        models.append({
            "model": "LSTM-autoencoder con predictor" if i == 3 else f"Candidate{i}",
            "search_recipe_id": f"recipe-{i}-v1", "nas": i < 6,
            "auxiliary_phases": 2 if i == 3 else 0,
            "same_window_moments": 10 if 4 <= i < 10 else 0,
            "hpo": {"dropout_values": [] if i < 6 else [0, 0.1, 0.3, 0.5],
                    "extra_axes": {"warmup_ratio": [0, 0.05, 0.1],
                                   "normalization": ["none", "layer"]}},
        })
    refs = [{"model": name, "search_recipe_id": name.lower()+"-v1", "nas": False,
             "auxiliary_phases": 0, "same_window_moments": 0,
             "hpo": {"dropout_values": [0], "extra_axes": {}}}
            for name in ["MLP16", "LSTM32", "GRU32"]]
    return {
        "schema_version": 1, "models": models, "neural_reference_profiles": refs,
        "common_hpo": {"trials": 18, "seed": 271101,
            "learning_rate": [1e-5, 0.003], "weight_decay_positive": [1e-6, 0.01],
            "zero_weight_decay_trials": 3, "batch_sizes": [16, 32, 64],
            "optimization_seeds": [271201, 271202], "max_epochs": 200,
            "patience_epochs": 20},
        "nas": {"widths": [16, 32, 64, 128, 256], "dropout_rate": 0.2,
            "phase1_seed": 271301, "phase2_seeds": list(range(271401, 271406)),
            "reference": {"widths": [64, 64, 64], "dropout_pattern": [0, 1, 0]},
            "k_options": [20, 40], "phase1_epochs": 50, "phase2_epochs": 200},
        "budget": {"selection_origins": 11, "outer_cv_origins": 10,
            "final_selection_origins": 1, "neural_references": 3,
            "min_final_seeds": 6, "max_final_seeds": 30, "noise_pilot_architectures": 12,
            "noise_pilot_seeds": 5, "power_pilot_models": 3, "power_pilot_seeds": 5,
            "power_pilot_model_names": ["Candidate0", "Candidate10", "GRU32"],
            "profiling_fixed_neural_stages": 8, "profiling_classical_fits": 2,
            "noise_pilot_seed_values": list(range(271501, 271506)),
            "profiling_seed": 271601, "power_pilot_seed_values": list(range(271701, 271706)),
            "final_seeds": list(range(271801, 271831))},
        "hardware": {"short_stage_wall_limit_seconds": 1800, "full_stage_wall_limit_seconds": 7200,
                     "fit_process_memory_limit_bytes": 12*1024**3, "cgroup_memory_bytes": 16*1024**3},
    }


class EnumerationTests(unittest.TestCase):
    def test_exhaustive_space_includes_asymmetric_extremes_and_reference(self):
        protocol = fixture()
        rows = planner.enumerate_architectures(protocol, "Candidate0")
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({r["architecture_id"] for r in rows}), 1000)
        self.assertEqual(len({(tuple(r["widths"]), tuple(r["dropout_pattern"])) for r in rows}), 1000)
        self.assertEqual(sum(not any(r["dropout_pattern"]) for r in rows), 125)
        self.assertEqual(sum(r["is_reference"] for r in rows), 1)
        widths = {tuple(r["widths"]) for r in rows}
        self.assertIn((16, 32, 256), widths)
        self.assertIn((256, 32, 16), widths)
        self.assertIn((16, 256, 16), widths)
        self.assertEqual(rows[0]["widths"], [16, 16, 16])
        self.assertEqual(rows[-1]["widths"], [256, 256, 256])
        self.assertEqual(planner.enumerate_architectures(protocol, "Candidate10"), [])

    def test_noise_pilot_has_shared_reference_extremes_and_unique_stratified_patterns(self):
        p = fixture()
        patterns = planner.noise_pilot_architectures(p)
        self.assertEqual(len(patterns), 12)
        self.assertEqual(len({row["pattern_id"] for row in patterns}), 12)
        self.assertEqual(patterns, planner.noise_pilot_architectures(deepcopy(p)))
        self.assertTrue(patterns[0]["is_reference"])
        self.assertEqual((patterns[1]["widths"], patterns[1]["dropout_pattern"]), ([16]*3, [0]*3))
        self.assertEqual((patterns[2]["widths"], patterns[2]["dropout_pattern"]), ([256]*3, [1]*3))
        space = {(tuple(row["widths"]), tuple(row["dropout_pattern"]))
                 for row in planner.enumerate_architectures(p, "Candidate0")}
        self.assertTrue(all((tuple(row["widths"]), tuple(row["dropout_pattern"])) in space for row in patterns))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = fixture()

    def test_distinct_reproducible_balanced_trials(self):
        a = planner.generate_hpo(self.protocol, "Candidate10")
        b = planner.generate_hpo(deepcopy(self.protocol), "Candidate10")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 18)
        self.assertEqual(len({json.dumps(r, sort_keys=True) for r in a}), 18)
        self.assertEqual(Counter(r["batch_size"] for r in a), {16: 6, 32: 6, 64: 6})
        for axis in ["dropout", "warmup_ratio", "normalization"]:
            counts = Counter(r[axis] for r in a)
            self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        changed = deepcopy(self.protocol)
        changed["common_hpo"]["seed"] += 1
        self.assertNotEqual(a, planner.generate_hpo(changed, "Candidate10"))

    def test_log_strata_and_positive_wd_mixture(self):
        rows = planner.generate_hpo(self.protocol, "Candidate10")
        lo, hi = map(math.log, self.protocol["common_hpo"]["learning_rate"])
        bins = []
        for row in rows:
            rate = row["learning_rate"]
            self.assertGreaterEqual(rate, math.exp(lo))
            self.assertLessEqual(rate, math.exp(hi))
            bins.append(min(17, int((math.log(rate)-lo)/(hi-lo)*18)))
        self.assertEqual(sorted(bins), list(range(18)))
        self.assertEqual(sum(r["weight_decay"] == 0 for r in rows), 3)
        positive = [r["weight_decay"] for r in rows if r["weight_decay"] > 0]
        lo, hi = map(math.log, self.protocol["common_hpo"]["weight_decay_positive"])
        bins = [min(14, int((math.log(v)-lo)/(hi-lo)*15)) for v in positive]
        self.assertEqual(sorted(bins), list(range(15)))

    def test_nas_hpo_keeps_architecture_reference_and_omits_dropout_axis(self):
        rows = planner.generate_hpo(self.protocol, "Candidate0")
        for row in rows:
            self.assertNotIn("dropout", row)
            self.assertEqual(row["architecture_reference"], {
                "widths": [64, 64, 64], "dropout_pattern": [0, 1, 0], "dropout_rate": 0.2})
            self.assertEqual(row["optimization_seeds"], [271201, 271202])
        self.assertNotEqual(rows, planner.generate_hpo(self.protocol, "Candidate1"))


class BudgetTests(unittest.TestCase):
    def test_counts_include_auxiliary_phases_all_origins_and_global_pilots(self):
        b = planner.budget(fixture(), 20, 30)
        full, short = b["full_training_phases"], b["short_training_phases"]
        self.assertEqual(full["hpo"], {"supervised": 5544, "auxiliary": 792, "total": 6336})
        self.assertEqual(short["nas_phase1"]["total"], 88000)
        self.assertEqual(full["nas_phase2_upper_bound"]["total"], 9240)
        self.assertEqual(full["neural_refits"]["total"], 5280)
        self.assertEqual(short["noise_pilot_short_global"]["total"], 96)
        self.assertEqual(full["noise_pilot_full_global"]["total"], 480)
        self.assertEqual(full["fixed_neural_resource_profiles_global"], {"supervised": 8, "auxiliary": 0, "total": 8})
        self.assertEqual(full["power_pilot_non_sae_models_global"]["total"], 15)
        self.assertEqual(b["full_phase_total"], 21359)
        self.assertEqual(b["short_phase_total"], 88096)
        self.assertEqual(b["classical_adjustments"]["total"], 420)
        self.assertEqual(b["classical_adjustments"]["resource_profiling"], 2)
        self.assertEqual(b["classical_adjustments"]["deterministic_refits"], 22)

    def test_high_noise_scenario_changes_only_phase2_and_cap_is_enforced(self):
        p = fixture()
        a, b = planner.budget(p, 20, 6), planner.budget(p, 40, 6)
        self.assertEqual(b["full_training_phases"]["nas_phase2_upper_bound"]["total"], 18040)
        self.assertEqual(a["short_phase_total"], b["short_phase_total"])
        self.assertEqual(b["full_phase_total"]-a["full_phase_total"], 8800)
        with self.assertRaises(ValueError):
            planner.budget(p, 20, 31)
        with self.assertRaises(ValueError):
            planner.budget(p, 10, 6)
        with self.assertRaises(ValueError):
            planner.budget(p, 20, 5)


class ValidationTests(unittest.TestCase):
    def test_seed_collisions_and_unknown_versions_fail_closed(self):
        for value in [True, 0, 2, "1", None]:
            p = fixture()
            p["schema_version"] = value
            with self.assertRaises(ValueError):
                planner.validate_protocol(p)
        p = fixture()
        p["nas"]["phase2_seeds"][0] = p["nas"]["phase1_seed"]
        with self.assertRaises(ValueError):
            planner.validate_protocol(p)
        p = fixture()
        p["nas"]["phase2_seeds"][0] = p["common_hpo"]["optimization_seeds"][0]
        with self.assertRaises(ValueError):
            planner.validate_protocol(p)

    def test_all_pilot_and_refit_seed_sets_are_disjoint_and_profiles_correct(self):
        for key in ["noise_pilot_seed_values", "power_pilot_seed_values", "final_seeds"]:
            p = fixture()
            p["budget"][key][0] = p["nas"]["phase1_seed"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                planner.validate_protocol(p)
        p = fixture()
        p["budget"]["profiling_seed"] = p["budget"]["power_pilot_seed_values"][0]
        with self.assertRaises(ValueError):
            planner.validate_protocol(p)
        p = fixture()
        p["budget"]["power_pilot_model_names"][0] = "LSTM-autoencoder con predictor"
        with self.assertRaises(ValueError):
            planner.validate_protocol(p)
        p = fixture()
        p["budget"]["profiling_fixed_neural_stages"] = 7
        with self.assertRaises(ValueError):
            planner.validate_protocol(p)

    def test_short_full_caps_are_consistent_and_memory_leaves_headroom(self):
        for change in [lambda p: p["nas"].update(phase1_epochs=200),
                       lambda p: p["nas"].update(phase2_epochs=300),
                       lambda p: p["hardware"].update(short_stage_wall_limit_seconds=7200),
                       lambda p: p["hardware"].update(fit_process_memory_limit_bytes=16*1024**3)]:
            p = fixture()
            change(p)
            with self.subTest(change=change), self.assertRaises(ValueError):
                planner.validate_protocol(p)

    def test_invalid_space_or_duplicate_identity_is_rejected(self):
        cases = [lambda p: p["nas"].update(widths=[16, 32, 64, 128]),
                 lambda p: p["common_hpo"].update(learning_rate=[0, 0.003]),
                 lambda p: p["common_hpo"].update(trials=12),
                 lambda p: p["models"][0]["hpo"]["extra_axes"].update(batch_size=[16]),
                 lambda p: p["models"][0].update(search_recipe_id="bad/id"),
                 lambda p: p["models"][0].update(model=p["models"][1]["model"]),
                 lambda p: p["budget"].update(selection_origins=10)]
        for change in cases:
            p = fixture()
            change(p)
            with self.subTest(change=change), self.assertRaises(ValueError):
                planner.validate_protocol(p)


class CLITests(unittest.TestCase):
    def test_cli_writes_only_new_manifest_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "protocol.json"
            output = root / "manifest.json"
            protocol.write_text(json.dumps(fixture()), encoding="utf-8")
            cmd = [sys.executable, str(SCRIPT), "--protocol", str(protocol), "--output", str(output)]
            first = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(result["hpo"]), 14)
            self.assertEqual(len(result["nas"]), 6)
            self.assertEqual(result["execution"]["fits_executed"], 0)
            self.assertEqual(len(result["classical_domains"]["ridge_alpha"]), 18)
            self.assertEqual(len(result["classical_domains"]["arima_orders"]), 18)
            self.assertEqual(result["classical_domains"]["seasonal_naive_periods"], [7, 14, 28])
            before = output.read_bytes()
            second = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            self.assertIn("Refusing to replace", second.stderr)


if __name__ == "__main__":
    unittest.main()
