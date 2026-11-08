"""Public download refresh tests with synthetic archives, never market data."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refresh_e1_native_downloads as refresh


def public_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))


def remediation_public_fixture(r=30, required=12, baseline="MLP"):
    """In-memory toy public aggregates for packaging guards; never real closure evidence."""
    import csv
    import io
    families = ("GRU", "LSTM", "MLP")
    execution = "d" * 64
    source_map = {"scripts/unit_test_only.py": "a" * 64}
    design_sha = "b" * 64
    source_signature = refresh.legacy.digest(refresh.legacy.encode({"sources_sha256": source_map,
        "design_sha256": design_sha, "source_commit": refresh.REMEDIATION_SOURCE}))
    identity = {"study_id": refresh.REMEDIATION_STUDY, "execution_id": execution,
                "source_signature": source_signature}
    recipes = {model: {"model": model, "history_days": 7, "width": width,
        "parameters": parameters, "candidate_id": "c00", "learning_rate": .001, "dropout": 0.}
        for model, width, parameters in (("GRU", 32, 3777), ("LSTM", 32, 5025), ("MLP", 16, 593))}
    summary = {**identity, "source_commit": refresh.REMEDIATION_SOURCE,
        "models": list(families), "core_models": list(families), "R_final": r, "R_required": required,
        "dates_per_trajectory": 608, "neural_prediction_rows": 1824*r,
        "definitive_trajectories": 3*r, "definitive_fit_stages": 6*r,
        "hpo_trajectories": 24, "hpo_blocks": 240, "hpo_fit_stages": 480,
        "pilot_trajectories": 30, "pilot_fit_stages": 60,
        "known_reserve_exploratory": True, "fresh_confirmatory_holdout": False,
        "selection_using_holdout": False, "private_payloads_exported": False,
        "source_publication_precedes_new_fits": True, "plan_publication_precedes_definitive_fits": True,
        "scoring_after_all_new_weights_frozen": True, "baseline_selected_before_reserve": baseline}
    records = {"summary.json": summary,
        "registration.json": {"study_id": refresh.REMEDIATION_STUDY, "registration_signature": execution,
            "source_signature": source_signature, "source_commit": refresh.REMEDIATION_SOURCE,
            "sources_sha256": source_map, "design_sha256": design_sha,
            "design": {"models": list(families), "history_days": 7,
                "reserve": {"start": "2025-01-01", "end": "2026-08-31"}}},
        "selection.json": {"registration_signature": execution, "recipes": recipes, "baseline_model": baseline},
        "plan.json": {"registration_signature": execution, "R": r, "R_final": r,
            "R_required": required, "seeds": list(range(18101, 18101+r))},
        "plan_publication.json": {"registration_signature": execution},
        "source_manifest.json": {"source_commit": refresh.REMEDIATION_SOURCE,
            "sources_sha256": source_map, "design_sha256": design_sha}}
    files = {}
    for name in refresh.REMEDIATION_PUBLIC_FILES:
        files[name] = (public_png() if name.endswith(".png") else
            refresh.legacy.encode(records.get(name, {"unit_test_public_metadata": True})) if name.endswith(".json") else
            b"model,unit_test_value\nGRU,1\n" if name.endswith(".csv") else b"Unit test public metadata only; never a scientific execution.\n")
    def table(name, rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        files[name] = stream.getvalue().encode()
    runs = [{**identity, "model": model, "seed": seed, "n_dates": 608,
        **{metric: .005 for metric in ("rmse", "mae", "mse", "r2", "mape", "mase")}}
        for model in families for seed in range(18101, 18101+r)]
    table("run_metrics.csv", [*runs, {**runs[0], "model": "Persistence", "seed": 0}])
    table("final_fit_summary.csv", runs)
    table("pilot_metrics.csv", [{**identity, "model": model, "seed": seed}
        for model in families for seed in range(17101, 17111)])
    table("run_costs.csv", [{"model": row["model"], "seed": row["seed"], "n_dates": 608,
        "training_seconds": 1., "inference_seconds": .01} for row in runs])
    table("hpo_run_metrics.csv", [{**identity, "model": model, "candidate_id": f"c{candidate:02d}",
        "trajectory_seed": seed} for model in families for candidate in range(4) for seed in (16101, 16102)])
    pairs = (("GRU", "LSTM"), ("GRU", "MLP"), ("LSTM", "MLP"))
    for name in ("pairwise_rmse.csv", "wilcoxon_rmse.csv"):
        table(name, [{"model_a": a, "model_b": b, "n_paired": r} for a, b in pairs])
    table("qq_data.csv", [{"model_a": a, "model_b": b, "n_paired": r, "order_index": index}
        for a, b in pairs for index in range(1, r+1)])
    table("friedman_rmse.csv", [{"n_paired": r, "n_models": 3}])
    temporal_pairs = (*pairs, *((model, "Persistence") for model in families))
    table("dm_pairs.csv", [{"model_a": a, "model_b": b, "n_dates": 608,
        "baseline_selected_before_reserve": baseline} for a, b in temporal_pairs])
    table("temporal_bootstrap.csv", [{"model_a": a, "model_b": b, "n_dates": 608,
        "block_length": block, "baseline_selected_before_reserve": baseline}
        for a, b in temporal_pairs for block in (7, 3, 14)])
    table("diagnostics.csv", [{"model": row["model"], "seed": row["seed"], "n_residuals": 608,
        "diagnostic": diagnostic, "lag": lag} for row in runs
        for diagnostic, lag in (("BDS", 2), ("Ljung-Box", 1), ("Ljung-Box", 5))])
    curves = [{"model": row["model"], "seed": row["seed"], "stage": stage, "epoch": 1,
        "validation_mse_scaled_eval": .1 if stage == "inner" else ""}
        for row in runs for stage in ("inner", "refit")]
    table("learning_curves_by_run.csv", curves)
    table("first_seed_learning_curves.csv", [row for row in curves if row["seed"] == 18101])
    files["manifest.json"] = refresh.legacy.encode({"unit_test_only": True})
    return files


class NativeDownloadRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.output = Path(self.temporary.name) / "delivery"
        self.addCleanup(patch.stopall)
        patch.object(refresh, "ROOT", self.root).start()
        patch.object(refresh.legacy, "ROOT", self.root).start()
        self.write(".gitignore", "book/_build/\ndata/raw/\n")
        self.write("BUILD.md", "Current public build instructions\n")
        self.write("README.md", "Current repository documentation\n")
        self.write("requirements-e1-supplement.lock.txt", "supplement-package==1.0\n")
        self.write("requirements-e1-curves.lock.txt", "curve-package==1.0\n")
        for name in refresh.REPRODUCTION_PROTOCOL_FILES:
            self.write(name, '{"new_development_reproduction": true}\n' if name.endswith(".json")
                       else "Separately registered development curve reproduction\n")
        self.write("scripts/current.py", "print('public current code')\n")
        self.write("tests/test_current.py", "# current public verification\n")
        for name in refresh.MANUAL_REACQUISITION_FILES:
            self.write(name, json.dumps({"nbformat": 4, "nbformat_minor": 5,
                "metadata": {}, "cells": [{"cell_type": "code", "metadata": {},
                    "source": ["print('download exact original hashes')"],
                    "execution_count": None, "outputs": []}]}) + "\n")
        self.write("book/references.bib", "@article{current, title={Current literature}}\n")
        self.write("book/_static/literature_e1/README.md", "Current literature stage index\n")
        self.stage_directories = {}
        for stage, names in refresh.STAGES.items():
            prefix = "book/_static/literature_e1/" + stage
            directory = self.root / prefix
            for name in names:
                if name == "summary.json":
                    value = json.dumps({"stage": stage, "reviewed_records": 2}) + "\n"
                elif name.endswith(".json"):
                    value = json.dumps({"public_metadata": True}) + "\n"
                elif name.endswith(".csv"):
                    value = "record_id,title,doi\nR1,Public title,10.1000/example\n"
                elif name.endswith(".bib"):
                    value = "@article{new, title={Primary reviewed paper}}\n"
                elif name.endswith(".svg"):
                    value = '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
                else:
                    value = "Public sanitized literature metadata\n"
                self.write(prefix + "/" + name, value)
            mapping = {name: refresh.legacy.digest((directory / name).read_bytes()) for name in names}
            self.write(prefix + "/manifest.json", json.dumps({"files_sha256": mapping}) + "\n")
            self.stage_directories[stage] = directory
        self.dynamic_stage_names = {}
        self.provisional_summary = {
            "schema_version": 1, "log_status": "PROVISIONAL_EDITABLE", "row_count": 5,
            "status_counts": {"provisional_lower_bound": 4, "planned_not_executed": 1},
            "new_native_search_executions": 0, "verified_native_search_totals_added": 0,
            "verified_native_search_dates_added": 0, "recovered_historical_query_literals_added": 0,
            "provisional_export_compatible_lower_bounds_by_source_key": {
                "W02": 8, "W01": 59, "S01": 19, "S02": 158},
            "ieee": {"status": "planned_not_executed", "export_compatible_lower_bound": None,
                     "native_total_verified": None, "native_search_date_verified": None,
                     "execution_accredited": False},
            "preserved_native_import": {"exported_occurrences": 244, "candidate_groups_provisional": 182},
            "historical_e1": {"crypto_full_text_denominator_preserved": 7,
                              "frequency_update_performed": False},
            "excluded_from_prisma_totals": True,
        }
        for stage in refresh.MANIFEST_STAGES:
            prefix = "book/_static/literature_e1/" + stage
            directory = self.root / prefix
            names = {"README.md", "summary.json", "part_a/reviews.json", "query_log.json"}
            if stage == "open_access_replacement_20261107":
                names |= {"source_manifest.json", "evidence_matrix.json", "metrics.csv",
                          "references.bib", "inputs/prior_review_registry.json",
                          "part_upload_psogru/review.json", "part_upload_psogru/README.md"}
            elif stage == "provisional_native_log_20261107":
                names |= {"provenance.json", "provisional_log.csv"}
            elif stage == "free_model_expansion_20261107":
                names |= {"references.csv", "references.bib", "readings.json"}
            elif stage == "homogeneous_expansion_20261107":
                names |= {"adjudications.csv", "frequencies.csv", "comparator_quality.csv"}
            elif stage == "core_selection_20261107":
                names |= {"ranking.csv", "parameter_candidates.csv", "source_anchors.csv"}
            for name in names:
                if stage == "provisional_native_log_20261107" and name == "summary.json":
                    value = json.dumps(self.provisional_summary) + "\n"
                else:
                    value = ('{"public_metadata": true}\n' if name.endswith(".json")
                             else "Reviewed nested metadata only\n")
                self.write(prefix + "/" + name, value)
            mapping = {name: {"sha256": refresh.legacy.digest((directory / name).read_bytes()),
                              "bytes": (directory / name).stat().st_size} for name in names}
            self.write(prefix + "/manifest.json", json.dumps({
                "publication_schema": refresh.PUBLICATION_SCHEMA,
                "approved_public_files": sorted(names), "files": mapping}) + "\n")
            self.dynamic_stage_names[stage] = names
            self.stage_directories[stage] = directory
        self.supplement_directories = {}
        self.supplement_names = {}
        for label, spec in refresh.SUPPLEMENTS.items():
            if label == "remediation_608":
                fixture = remediation_public_fixture()
                names = set(fixture) - {"manifest.json"}
                directory = self.root / spec["directory"]
                for name in names:
                    self.write(spec["directory"] + "/" + name, fixture[name])
                mapping = {name: {"sha256": refresh.legacy.digest(fixture[name]), "bytes": len(fixture[name])}
                           for name in names}
                self.write(spec["directory"] + "/manifest.json", refresh.legacy.encode({
                    "publication_schema": refresh.PUBLICATION_SCHEMA, "approved_public_files": sorted(names), "files": mapping}))
                self.supplement_directories[label] = directory
                self.supplement_names[label] = names
                continue
            prefix, names = spec["directory"], spec["files"]
            if names is None:
                names = ({"README.md", "summary.json", "comparison.csv", "manifest.json"}
                         | set(spec.get("configuration_files", ()))) - {spec["manifest"]}
            if label in {"historical_core", "primary_608"}:
                names |= {"run_metrics.csv"}
            directory = self.root / prefix
            for name in names:
                if label == "primary_608" and name == "summary.json":
                    recipes = {model: {"model": model, "history_days": 7, "width": width,
                        "learning_rate": lr, "dropout": 0.1, "parameters": parameters}
                        for model, width, lr, parameters in (("GRU", 32, .003, 3777),
                            ("LSTM", 32, .003, 5025), ("MLP", 16, .001, 593))}
                    historical = self.supplement_directories["historical_core"]
                    value = json.dumps({"study_id": "e1_primary_608_20261107",
                        "models": ["LSTM", "MLP", "GRU"], "baseline_model": "GRU",
                        "selected_recipes": recipes, "baseline_recipe": recipes["GRU"],
                        "R": 10, "R_per_neural": 10, "n_dates": 608, "dates_per_trajectory": 608,
                        "phases": {"cv": {"start": "2024-01-01", "end": "2024-12-31", "n_dates": 366},
                            "test": {"start": "2025-01-01", "end": "2026-08-31", "n_dates": 608}},
                        "primary_scope": "historical_known_fixed_recipes", "HPO_not_equal": True,
                        "editorial_promotion_after_results_seen": True, "new_model_fits": 0, "new_forecasts": 0,
                        "source_summary_sha256": refresh.legacy.digest((historical / "summary.json").read_bytes()),
                        "source_score_rows_sha256": refresh.legacy.digest((historical / "run_metrics.csv").read_bytes())}) + "\n"
                elif label in {"primary_608", "historical_core"} and name == "run_metrics.csv":
                    value = "phase,history_days,model,seed,n_dates,parameters,rmse\n"
                    for phase, days in (("cv", 366), ("test", 608)):
                        for model, parameters in (("GRU", 3777), ("LSTM", 5025), ("MLP", 593)):
                            for seed in range(4101, 4111):
                                value += f"{phase},7,{model},{seed},{days},{parameters},0.005\n"
                        value += f"{phase},7,Persistence,,{days},0,0.004\n"
                elif label == "core_mlp_extension" and name in {"summary.json", "selection.json", "plan.json"}:
                    execution = "d" * 64
                    recipes = {model: {"model": model, "history_days": 7, "width": 32,
                                       "mean_rmse": 0.001 + index * 0.0001}
                               for index, model in enumerate(("GRU", "LSTM", "MLP"))}
                    payloads = {
                        "summary.json": {"study_id": "e1_core_mlp_extension_20261107",
                            "execution_id": execution, "source_commit": "e" * 40,
                            "core_models": ["GRU", "LSTM", "MLP"], "R_final": 465,
                            "definitive_trajectories": 1395, "baseline_selected_before_reserve": "GRU",
                            "known_reserve_exploratory": True, "fresh_confirmatory_holdout": False},
                        "selection.json": {"registration_signature": execution,
                            "baseline_model": "GRU", "recipes": recipes},
                        "plan.json": {"registration_signature": execution, "R_final": 465, "R": 465},
                    }
                    value = json.dumps(payloads[name]) + "\n"
                elif name == "summary.json":
                    value = json.dumps({"supplement": label, "post_hoc": True}) + "\n"
                elif name.endswith(".json"):
                    value = json.dumps({"public_metrics": True}) + "\n"
                elif name.endswith(".csv"):
                    value = "model,rmse\nGRU,0.005\n"
                elif name.endswith(".png"):
                    value = public_png()
                else:
                    value = "Public post-hoc evidence; no new fits\n"
                self.write(prefix + "/" + name, value)
            mapping = {name: {"sha256": refresh.legacy.digest((directory / name).read_bytes()),
                              "bytes": (directory / name).stat().st_size} for name in names}
            manifest = {"files": mapping}
            if spec["files"] is None:
                manifest.update(publication_schema=refresh.PUBLICATION_SCHEMA,
                                approved_public_files=sorted(names))
            self.write(prefix + "/" + spec["manifest"], json.dumps(manifest) + "\n")
            self.supplement_directories[label] = directory
            self.supplement_names[label] = names
        self.base = self.root / "book/_static/e1_downloads"
        self.base.mkdir(parents=True)
        records = []
        self.base_payloads = {}
        for item, filename in refresh.ITEMS.items():
            payload = {"original_metadata.json": b'{"historical": true}\n'}
            if item == "8.2":
                payload.update({"scripts/obsolete.py": b"# stale code\n",
                                "tests/test_obsolete.py": b"# stale test\n",
                                "experiments/original/design.json": b'{"seed": 73}\n',
                                "requirements-training.lock.txt": b"historical==1.0\n",
                                "BUILD.md": b"Old build\n", "README.md": b"Old README\n"})
            if item == "8.4":
                payload.update({"book/references.bib": b"@article{old}\n",
                    "book/_static/literature_e1/history_20261102/studies.json": b'{"n": 7}\n'})
            if item == "8.6":
                payload.update({"experiments/original/design.json": b'{"original_config": true}\n',
                                "requirements-training.lock.txt": b"original-runtime==1.0\n"})
            if item == "8.7":
                payload.update({"literature_refresh_20261106.json": b'{"prior_round": true}\n',
                                "AMPLIACION_LITERARIA_20261106.md": b"Prior dated literature round\n",
                                "literature_refresh_20261107.json": b'{"prior_same_day_round": true}\n',
                                "AMPLIACION_LITERARIA_20261107.md": b"Prior same-day literature round\n",
                                "literature_refresh_20261107_free_models.json": b'{"previous_free_models_round": true}\n'})
            self.base_payloads[item] = payload
            records.append(refresh.legacy.archive(self.base / filename, payload, item, "Product " + item,
                                                  "a" * 40, "Original phase I/II notes\n"))
        self.catalog = {"source_commit": "a" * 40, "models": ["GRU", "LSTM", "TCN"],
            "active_entrypoint": "scripts/run_gru_lstm_tcn.py",
            "active_entrypoints": {"phase_I": "scripts/run_gru_lstm_tcn.py",
                                    "phase_II": "scripts/run_gru_lstm_tcn_planned.py"},
            "baseline_selected_by_cv": {"model": "GRU", "seed_count": 10},
            "planned_D7_phase": {"definitive_seeds_per_family": 465, "temporal_power_claimed": False},
            "scope": "Two separate historical experimental phases",
            "declaration_manifest_sha256": "c" * 64,
            "presentation_refresh": {"source_commit": "a" * 40,
                                     "preserved_archive_sha256": {"8_6_configuraciones.zip": "b" * 64}},
            "packages": records, "complete_package": {"title": "Complete E1"},
            "scientific_limitations_preserved": True}
        self.write("book/_static/e1_downloads/catalogo.json", json.dumps(self.catalog) + "\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit()
        self.build()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE, text=True).strip()

    def commit(self):
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "--quiet", "-m", "Public source")

    def build(self):
        self.write("book/_build/html/_static/eda/results/summary.json", '{"private": "one"}')
        self.write("book/_build/html/_static/eda_temporal_reference/summary.json", '{"private": "two"}')
        self.write("book/_build/html/_static/e1_downloads/8_5_benchmark.zip", b"must be excluded")
        self.write("book/_build/html/intro.html", '<html><a href="_static/e1_downloads/8_5_benchmark.zip">Download</a></html>')

    def test_refresh_preserves_science_and_updates_only_reviewed_delivery(self):
        baseline_archives = {item: (self.base / refresh.ITEMS[item]).read_bytes()
                             for item in refresh.PRESERVED_ITEMS}
        with patch.object(refresh.legacy, "load_crypto_stage", side_effect=AssertionError("Legacy validator must not run")):
            catalog = refresh.package(self.output)
        for item, data in baseline_archives.items():
            self.assertEqual((self.output / refresh.ITEMS[item]).read_bytes(), data)
        for key in ("models", "active_entrypoint", "active_entrypoints", "baseline_selected_by_cv",
                    "planned_D7_phase", "scope", "declaration_manifest_sha256"):
            self.assertEqual(catalog["historical_study_metadata"][key], self.catalog[key])
        self.assertEqual(catalog["models"], ["LSTM", "MLP", "GRU"])
        self.assertEqual(catalog["complementary_models"], ["TCN"])
        self.assertEqual(catalog["active_entrypoint"], "scripts/run_e1_608_remediation.py")
        self.assertEqual(catalog["baseline_selected_by_cv"]["parameters"], 593)
        self.assertEqual(catalog["current_study"]["n_dates"], 608)
        self.assertEqual(catalog["current_study"]["R"], 30)
        self.assertEqual(catalog["complementary_study_metadata"]["current_study"]["execution_id"], "d" * 64)
        self.assertEqual(catalog["complementary_study_metadata"]["current_study"]["source_commit"], "e" * 40)
        self.assertTrue(catalog["current_study"]["known_reserve_exploratory"])
        self.assertFalse(catalog["current_study"]["fresh_confirmatory_holdout"])
        self.assertNotIn("planned_D7_phase", catalog)
        self.assertNotIn("declaration_manifest_sha256", catalog)
        self.assertEqual(catalog["source_commit"], self.git("rev-parse", "HEAD"))
        for row in catalog["packages"] + [catalog["complete_package"]]:
            self.assertEqual(row["url"], refresh.legacy.PUBLIC_BOOK + "_static/e1_downloads/" + row["file"])
        self.assertFalse(catalog["literature_refresh"]["retrospective_preregistration"])
        self.assertNotIn("presentation_refresh", catalog)
        self.assertEqual(catalog["prior_presentation_refreshes"], [self.catalog["presentation_refresh"]])
        self.assertEqual(catalog["new_model_fits_by_packaging"], 0)
        with ZipFile(self.output / refresh.ITEMS["8.2"]) as saved:
            self.assertEqual(saved.read("scripts/current.py"), (self.root / "scripts/current.py").read_bytes())
            for name in refresh.MANUAL_REACQUISITION_FILES:
                self.assertEqual(saved.read(name), (self.root / name).read_bytes())
            self.assertNotIn("scripts/obsolete.py", saved.namelist())
            self.assertNotIn("tests/test_obsolete.py", saved.namelist())
            self.assertEqual(saved.read("experiments/original/design.json"), b'{"seed": 73}\n')
            self.assertEqual(saved.read("requirements-training.lock.txt"), b"historical==1.0\n")
            self.assertEqual(saved.read("BUILD.md"), b"Current public build instructions\n")
            self.assertEqual(saved.read("requirements-e1-supplement.lock.txt"), b"supplement-package==1.0\n")
            self.assertEqual(saved.read("requirements-e1-curves.lock.txt"), b"curve-package==1.0\n")
        with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
            self.assertEqual(saved.read("book/references.bib"), (self.root / "book/references.bib").read_bytes())
            self.assertEqual(saved.read("book/_static/literature_e1/history_20261102/studies.json"), b'{"n": 7}\n')
            for stage, names in refresh.STAGES.items():
                for name in names | {"manifest.json"}:
                    self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            for stage, names in self.dynamic_stage_names.items():
                for name in names | {"manifest.json"}:
                    self.assertEqual(saved.read("book/_static/literature_e1/" + stage + "/" + name),
                                     (self.stage_directories[stage] / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
            for stage, names in refresh.STAGES.items():
                for name in ("summary.json", "manifest.json", "source_manifest.json"):
                    if name == "manifest.json" or name in names:
                        self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            for stage in refresh.MANIFEST_STAGES:
                for name in ("summary.json", "manifest.json"):
                    self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            self.assertEqual(saved.read("original_metadata.json"), b'{"historical": true}\n')
            self.assertEqual(saved.read("literature_refresh_20261107.json"),
                             b'{"prior_same_day_round": true}\n')
            self.assertEqual(saved.read("AMPLIACION_LITERARIA_20261107.md"),
                             b"Prior same-day literature round\n")
            self.assertEqual(saved.read("literature_refresh_20261107_free_models.json"),
                             b'{"previous_free_models_round": true}\n')
            self.assertIn("prerregistro retrospectivo", saved.read("AMPLIACION_LITERARIA_" + refresh.REFRESH_ID + ".md").decode())
            self.assertIn("SUPLEMENTO_EVIDENCIAS_" + refresh.REFRESH_ID + ".md", saved.namelist())
            for label, spec in refresh.SUPPLEMENTS.items():
                for name in ("summary.json", spec["manifest"]):
                    self.assertIn(spec["directory"] + "/" + name, saved.namelist())
        for label, spec in refresh.SUPPLEMENTS.items():
            with ZipFile(self.output / refresh.ITEMS[spec["item"]]) as saved:
                for name in self.supplement_names[label] | {spec["manifest"]}:
                    self.assertEqual(saved.read(spec["directory"] + "/" + name),
                                     (self.supplement_directories[label] / name).read_bytes())
                # Supplemental analyses add separate namespaces; original science remains byte-identical.
                for name, data in self.base_payloads[spec["item"]].items():
                    self.assertEqual(saved.read(name), data)
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["statistics"]["product"], "8.5")
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["curves"]["product"], "8.5")
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["academic_margin"]["product"], "8.7")
        self.assertEqual(catalog["evidence_refresh"]["date"], refresh.REFRESH_DATE)
        for label in ("replay_forensics", "replay_repeatability", "historical_core", "core_mlp_extension", "primary_608"):
            self.assertEqual(catalog["evidence_refresh"]["supplements"][label]["product"], "8.5")
        with ZipFile(self.output / refresh.ITEMS["8.6"]) as saved:
            for name, data in self.base_payloads["8.6"].items():
                self.assertEqual(saved.read(name), data)
            for name in refresh.REPRODUCTION_PROTOCOL_FILES:
                self.assertEqual(saved.read(name), (self.root / name).read_bytes())
            self.assertEqual(saved.read("requirements-e1-curves.lock.txt"), b"curve-package==1.0\n")
            curve_prefix = refresh.SUPPLEMENTS["curves"]["directory"] + "/"
            for name in ("reproduction_registration.json", "runtime.json", "execution.json"):
                self.assertEqual(saved.read(curve_prefix + name),
                    (self.supplement_directories["curves"] / name).read_bytes())
            for label, spec in refresh.SUPPLEMENTS.items():
                for name in spec.get("configuration_files", ()):
                    self.assertEqual(saved.read(spec["directory"] + "/" + name),
                        (self.supplement_directories[label] / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
            for name in refresh.REPRODUCTION_PROTOCOL_FILES:
                self.assertEqual(saved.read(name), (self.root / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.1"]) as saved:
            self.assertNotIn("informe/_static/e1_downloads/8_5_benchmark.zip", saved.namelist())
            self.assertNotIn("informe/_static/eda/results/summary.json", saved.namelist())
            self.assertIn(refresh.legacy.PUBLIC_BOOK.encode(), saved.read("informe/intro.html"))
        with ZipFile(self.output / "entregable1_completo.zip") as saved:
            self.assertEqual(set(saved.namelist()), set(refresh.ITEMS.values()) |
                             {"CATALOGO_PRODUCTOS.json", "LEEME.md", "MANIFIESTO.json"})
            self.assertEqual(json.loads(saved.read("CATALOGO_PRODUCTOS.json")), catalog["packages"])
        for line in (self.output / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split("  ")
            self.assertEqual(refresh.legacy.digest((self.output / name).read_bytes()), expected)
        for filename in [*refresh.ITEMS.values(), "entregable1_completo.zip"]:
            self.assertEqual(refresh.verify_archive(self.output / filename)["member_hashes"], "PASS")
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_rejects_dirty_source_and_mismatched_commit(self):
        self.write("README.md", "Uncommitted edit\n")
        with self.assertRaisesRegex(ValueError, "Commit final source"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_new_core_cannot_ship_without_its_public_planning_gate(self):
        spec = refresh.SUPPLEMENTS["core_mlp_extension"]
        directory = self.supplement_directories["core_mlp_extension"]
        path = directory / spec["manifest"]
        changed = json.loads(path.read_text())
        changed["files"].pop("plan_publication.json")
        changed["approved_public_files"].remove("plan_publication.json")
        path.write_text(json.dumps(changed))
        self.git("add", ".")
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "configuration files must be publicly approved"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())
        self.git("checkout", "--", "README.md")
        with self.assertRaisesRegex(subprocess.CalledProcessError, "git"):
            refresh.source_revision("does-not-exist")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                 "commit", "--allow-empty", "--quiet", "-m", "Second commit")
        with self.assertRaisesRegex(ValueError, "current HEAD"):
            refresh.source_revision("HEAD~1")

    def test_reviewed_additions_route_metadata_and_current_builders(self):
        additions = {
            "open_access_replacement_20261107": "scripts/build_e1_open_access_replacement.py",
            "provisional_native_log_20261107": "scripts/build_e1_provisional_native_log.py",
            "homogeneous_synthesis_20261107": "scripts/build_e1_homogeneous_synthesis.py",
            "native_execution_20261107": "scripts/ingest_e1_native_execution.py",
            "free_model_expansion_20261107": "scripts/build_e1_free_model_expansion.py",
            "homogeneous_expansion_20261107": "scripts/build_e1_homogeneous_expansion.py",
            "core_selection_20261107": "scripts/build_e1_core_selection.py",
        }
        for stage, builder in additions.items():
            self.assertIn(stage, refresh.MANIFEST_STAGES)
            self.write(builder, "# Build reviewed literature metadata without training.\n")
            self.git("add", builder)
        self.commit()
        self.build()
        catalog = refresh.package(self.output)
        for stage, builder in additions.items():
            with self.subTest(stage=stage):
                prefix = "book/_static/literature_e1/" + stage + "/"
                directory = self.stage_directories[stage]
                approved = self.dynamic_stage_names[stage] | {"manifest.json"}
                with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
                    included = {name[len(prefix):] for name in saved.namelist() if name.startswith(prefix)}
                    self.assertEqual(included, approved)
                    for name in approved:
                        self.assertEqual(saved.read(prefix + name), (directory / name).read_bytes())
                with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
                    included = {name[len(prefix):] for name in saved.namelist() if name.startswith(prefix)}
                    expected_roots = approved & {"summary.json", "source_manifest.json", "manifest.json"}
                    self.assertEqual(included, expected_roots)
                    if stage == "provisional_native_log_20261107":
                        # Export-compatible bounds and unknown native IEEE data must
                        # survive packaging as metadata, without becoming totals.
                        self.assertEqual(json.loads(saved.read(prefix + "summary.json")),
                                         self.provisional_summary)
                with ZipFile(self.output / refresh.ITEMS["8.2"]) as saved:
                    self.assertEqual(saved.read(builder), (self.root / builder).read_bytes())
                info = catalog["literature_refresh"]["stages"][stage]
                self.assertEqual(info["public_files"], len(approved))
                if stage == "provisional_native_log_20261107":
                    self.assertEqual(info["summary"], self.provisional_summary)
        for key in ("new_model_fits_by_packaging", "new_checkpoint_forecasts_by_packaging",
                    "raw_market_archive_reads_by_packaging", "publications_by_packaging"):
            self.assertEqual(catalog[key], 0)

    def test_rejects_unbuilt_stale_report_and_unsafe_destination(self):
        intro = self.root / "book/_build/html/intro.html"
        intro.unlink()
        with self.assertRaisesRegex(ValueError, "regular public source"):
            refresh.package(self.output)
        self.build()
        self.write("book/references.bib", "@article{newer, title={Newer source}}\n")
        self.git("add", "book/references.bib")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Rebuild E1"):
            refresh.package(self.output)
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            refresh.package(self.root / "inside")
        self.output.mkdir()
        (self.output / "existing.txt").write_text("Do not overwrite")
        with self.assertRaisesRegex(ValueError, "empty output"):
            refresh.package(self.output)
        self.assertEqual((self.output / "existing.txt").read_text(), "Do not overwrite")

    def test_current_catalog_rejects_old_families_mixed_execution_and_fresh_holdout_claim(self):
        directory = self.supplement_directories["core_mlp_extension"]
        files, _ = refresh.public_stage(directory, None)
        cases = (("summary.json", "core_models", ["GRU", "LSTM", "TCN"]),
                 ("plan.json", "registration_signature", "f" * 64),
                 ("summary.json", "fresh_confirmatory_holdout", True))
        for filename, key, value in cases:
            with self.subTest(filename=filename, key=key):
                changed = dict(files)
                record = json.loads(changed[filename])
                record[key] = value
                changed[filename] = json.dumps(record).encode()
                with self.assertRaisesRegex(ValueError, "consistently identified"):
                    refresh.complementary_core_catalog(changed)

    def test_primary_catalog_rejects_changed_period_R_models_and_source_mixing(self):
        files, _ = refresh.public_stage(self.supplement_directories["primary_608"], None)
        historical, _ = refresh.public_stage(self.supplement_directories["historical_core"], None)
        for key, value in (("n_dates", 60), ("R", 465), ("models", ["GRU", "LSTM", "TCN"]),
                           ("source_score_rows_sha256", "f" * 64)):
            with self.subTest(key=key):
                changed = dict(files)
                summary = json.loads(files["summary.json"])
                summary[key] = value
                changed["summary.json"] = json.dumps(summary).encode()
                with self.assertRaisesRegex(ValueError, "saved source proof"):
                    refresh.current_core_catalog(changed, historical)
        for old, new in ((b"test,7,MLP,4101,608", b"test,7,MLP,4101,60"),
                         (b"test,7,MLP,4101,608", b"test,7,MLP,8101,608"),
                         (b"test,7,MLP,4101,608,593", b"test,7,MLP,4101,608,1185")):
            with self.subTest(mixed=new):
                changed = {**files, "run_metrics.csv": files["run_metrics.csv"].replace(old, new)}
                with self.assertRaisesRegex(ValueError, "mix dates, recipes, seeds or models"):
                    refresh.current_core_catalog(changed, historical)

        changed = {**files, "run_metrics.csv": files["run_metrics.csv"].replace(b",0.005", b",0.006", 1)}
        with self.assertRaisesRegex(ValueError, "saved historical source rows"):
            refresh.current_core_catalog(changed, historical)

    def test_remediation_primary_uses_calculated_R_and_any_frozen_cv_family(self):
        for r, required, baseline in ((30, 12, "MLP"), (37, 37, "LSTM"), (41, 41, "GRU")):
            with self.subTest(r=r, baseline=baseline):
                current = refresh.remediation_core_catalog(remediation_public_fixture(r, required, baseline))
                self.assertEqual(current["current_study"]["R"], r)
                self.assertEqual(current["current_study"]["definitive_trajectories"], 3*r)
                self.assertEqual(current["baseline_selected_by_cv"]["model"], baseline)
                self.assertEqual(current["active_entrypoint"], "scripts/run_e1_608_remediation.py")

    def test_remediation_primary_rejects_wrong_source_cohort_R_baseline_and_scope(self):
        files = remediation_public_fixture()
        cases = (("summary.json", "source_commit", "e"*40),
            ("summary.json", "source_signature", "f"*64),
            ("summary.json", "dates_per_trajectory", 60),
            ("summary.json", "neural_prediction_rows", 608*89),
            ("summary.json", "baseline_selected_before_reserve", "GRU"),
            ("summary.json", "fresh_confirmatory_holdout", True),
            ("plan.json", "R_required", 31),
            ("plan.json", "seeds", list(range(8101, 8131))))
        for name, key, value in cases:
            changed = dict(files)
            record = json.loads(changed[name])
            record[key] = value
            changed[name] = refresh.legacy.encode(record)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                refresh.remediation_core_catalog(changed)

    def test_remediation_primary_rejects_mixed_row_sources_seeds_phase_and_prices(self):
        files = remediation_public_fixture()
        replacements = (("run_metrics.csv", b",18101,608,", b",8101,608,"),
            ("run_metrics.csv", b",608,", b",60,"),
            ("run_metrics.csv", b"d"*64, b"e"*64),
            ("diagnostics.csv", b",608,", b",60,"),
            ("hpo_run_metrics.csv", b"16101", b"6101"))
        for name, old, new in replacements:
            self.assertIn(old, files[name])
            changed = {**files, name: files[name].replace(old, new, 1)}
            with self.subTest(name=name, replacement=new), self.assertRaises(ValueError):
                refresh.remediation_core_catalog(changed)
        import csv
        import io
        for key, value in (("phase", "cv"), ("y_true", "0.1"), ("daily_close", "1"), ("fitted_scaler_mean", "2")):
            rows = list(csv.DictReader(io.StringIO(files["run_metrics.csv"].decode())))
            for row in rows:
                row[key] = value
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                refresh.remediation_core_catalog({**files, "run_metrics.csv": out.getvalue().encode()})

    def test_remediation_requires_actual_closed_stage_without_old_result_fallback(self):
        files = remediation_public_fixture()
        with self.assertRaisesRegex(ValueError, "exactly33"):
            refresh.remediation_core_catalog({name: data for name, data in files.items() if name != "qq_data.csv"})
        manifest = self.supplement_directories["remediation_608"] / "manifest.json"
        manifest.unlink()
        self.git("add", ".")
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "regular public source"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_full_prior_primary_and_complement_snapshots_survive_new_primary(self):
        prior_files, _ = refresh.public_stage(self.supplement_directories["primary_608"], None)
        history_files, _ = refresh.public_stage(self.supplement_directories["historical_core"], None)
        complement_files, _ = refresh.public_stage(self.supplement_directories["core_mlp_extension"], None)
        prior = {**self.catalog, **refresh.current_core_catalog(prior_files, history_files),
            "historical_study_metadata": {"frozen_original": {"models": ["GRU", "LSTM", "TCN"]}},
            "complementary_study_metadata": {**refresh.complementary_core_catalog(complement_files),
                "full_prior_note": {"provenance": "retained byte-for-byte"}},
            "reviewed_primary_note": {"not_discarded": True}}
        self.write("book/_static/e1_downloads/catalogo.json", refresh.legacy.encode(prior))
        self.git("add", ".")
        self.commit()
        self.build()
        current = refresh.package(self.output)
        for key, value in prior.items():
            self.assertEqual(current["previous_primary_study_metadata"][key], value)
        self.assertEqual(current["historical_study_metadata"], prior["historical_study_metadata"])
        self.assertEqual(current["complementary_study_metadata"], prior["complementary_study_metadata"])
        self.assertEqual(current["dated_study_records"]["e1_primary_608_20261107"]["current_study"]["R"], 10)
        self.assertEqual(current["dated_study_records"]["e1_core_mlp_extension_20261107"]["current_study"]["definitive_seeds_per_family"], 465)
        self.assertEqual(current["current_study"]["study_id"], refresh.REMEDIATION_STUDY)

    def test_prior_60_date_metadata_and_historical_snapshot_survive_promotion(self):
        files, _ = refresh.public_stage(self.supplement_directories["core_mlp_extension"], None)
        previous = {**self.catalog, **refresh.complementary_core_catalog(files),
                    "historical_study_metadata": {"original": "frozen", "models": ["GRU", "LSTM", "TCN"]}}
        self.write("book/_static/e1_downloads/catalogo.json", json.dumps(previous) + "\n")
        self.git("add", ".")
        self.commit()
        self.build()
        current = refresh.package(self.output)
        for key in ("current_study", "active_entrypoint", "active_entrypoints", "scope",
                    "baseline_selected_by_cv", "historical_study_metadata"):
            self.assertEqual(current["complementary_study_metadata"][key], previous[key])
        self.assertEqual(current["historical_study_metadata"], previous["historical_study_metadata"])
        self.assertEqual(current["current_study"]["n_dates"], 608)

    def test_manual_notebook_cannot_publish_execution_outputs(self):
        name = next(iter(refresh.MANUAL_REACQUISITION_FILES))
        notebook = json.loads((self.root / name).read_text())
        notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["private downloaded observations"]}]
        self.write(name, json.dumps(notebook) + "\n")
        self.git("add", name)
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "unexecuted and have no outputs"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_baseline_archive_hash_mismatch(self):
        (self.base / refresh.ITEMS["8.5"]).write_bytes(b"tampered archive")
        with self.assertRaisesRegex(ValueError, "does not match catalog"):
            refresh.baseline_products(self.base)

    def test_rejects_manifest_path_escape_and_unreviewed_member(self):
        directory = self.stage_directories["native_search_20261106"]
        manifest = directory / "manifest.json"
        record = json.loads(manifest.read_text())
        record["files_sha256"]["../private.csv"] = "0" * 64
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "approved public files"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])
        for name in ("../private.csv", "/tmp/file", "C:/file", "a\\b", "a//b", "./a", "a\nfile"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                refresh.safe_member(name)

    def test_rejects_changed_allowlisted_file_and_symlink(self):
        directory = self.stage_directories["native_search_20261106"]
        target = directory / "records.csv"
        original = target.read_bytes()
        target.write_bytes(original + b"unexpected change\n")
        with self.assertRaisesRegex(ValueError, "allowlist hash mismatch"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])
        target.unlink()
        outside = Path(self.temporary.name) / "outside.csv"
        outside.write_bytes(original)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes|Symlink"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])

    def test_accepts_sized_native_manifest_and_checks_byte_counts(self):
        directory = self.stage_directories["native_search_20261106"]
        path = directory / "manifest.json"
        mapping = json.loads(path.read_text())["files_sha256"]
        record = {"files": {name: {"sha256": sha, "bytes": (directory / name).stat().st_size}
                            for name, sha in mapping.items()}}
        path.write_text(json.dumps(record))
        files, info = refresh.public_stage(directory, refresh.STAGES[directory.name])
        self.assertIn("records.csv", files)
        self.assertEqual(info["public_files"], len(mapping) + 1)
        record["files"]["records.csv"]["bytes"] += 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])

    def test_unlisted_uploads_are_never_copied_or_opened(self):
        # Ignored files mimic private uploads kept alongside public working data.
        stage = self.stage_directories["native_search_20261106"]
        self.write(".gitignore", (self.root / ".gitignore").read_text() + "raw_upload.csv\noriginal.pdf\n")
        self.git("add", ".gitignore")
        self.commit()
        self.build()
        upload = stage / "raw_upload.csv"
        upload.write_text("Abstract,Author Email\nPrivate abstract,private@example.test\n")
        nested_uploads = set()
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            nested_upload = directory / "raw_upload.csv"
            nested_upload.write_text("Abstract,Author Email\nPrivate abstract,private@example.test\n")
            original_pdf = directory / "original.pdf"
            original_pdf.write_bytes(b"Private original supplied separately")
            nested_uploads.update({nested_upload, original_pdf})
        market = self.write("data/raw/market.csv", "Do not read market observations\n")
        original_read_bytes = Path.read_bytes
        def public_reads_only(path):
            if path in {upload, market} | nested_uploads:
                raise AssertionError("Private input was opened: " + str(path))
            return original_read_bytes(path)
        with patch.object(Path, "read_bytes", public_reads_only):
            refresh.package(self.output)
        with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
            self.assertFalse(any("raw_upload" in name for name in saved.namelist()))
            self.assertFalse(any(name.endswith(".pdf") for name in saved.namelist()))

    def test_rejects_abstract_contact_fields_in_metadata(self):
        cases = {"records.csv": b"Title,Abstract\nPublic title,Private abstract\n",
                 "source_manifest.json": b'{"records": [{"author_email": "private@example.test"}]}',
                 "references.bib": b"@article{x,\n abstract = {Private abstract}\n}\n",
                 "reviews.json": b'{"raw_abstract": "Exported copyrighted text"}'}
        for name, data in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Private/abstract"):
                refresh.validate_public_bytes(name, data)

    def test_nested_stage_needs_explicit_closed_approval(self):
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            path = directory / "manifest.json"
            original = json.loads(path.read_text())
            files, _ = refresh.public_stage(directory, None)
            self.assertIn("part_a/reviews.json", files)
            for change in ("missing_schema", "unapproved_file", "duplicate_approval",
                           "unsorted_approval", "hash_mismatch", "byte_mismatch"):
                changed = json.loads(json.dumps(original))
                if change == "missing_schema":
                    changed.pop("publication_schema")
                elif change == "unapproved_file":
                    changed["files"]["other.json"] = {"bytes": 1, "sha256": "a" * 64}
                elif change == "duplicate_approval":
                    changed["approved_public_files"].append("summary.json")
                elif change == "unsorted_approval":
                    changed["approved_public_files"].reverse()
                elif change == "byte_mismatch":
                    changed["files"]["part_a/reviews.json"]["bytes"] += 1
                else:
                    changed["files"]["part_a/reviews.json"]["sha256"] = "a" * 64
                path.write_text(json.dumps(changed))
                with self.subTest(stage=stage, change=change), self.assertRaises(ValueError):
                    refresh.public_stage(directory, None)

    def test_nested_approval_cannot_ship_private_payload_or_pdf(self):
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            path = directory / "manifest.json"
            original = json.loads(path.read_text())
            for name in ("predictions.csv", "part_a/paper.pdf", "../escape.json"):
                changed = json.loads(json.dumps(original))
                changed["files"][name] = {"bytes": 1, "sha256": "a" * 64}
                changed["approved_public_files"] = sorted(changed["files"])
                path.write_text(json.dumps(changed))
                # Failure must precede reading any payload named by the bad entry.
                with self.subTest(stage=stage, name=name), self.assertRaisesRegex(
                        ValueError, "Private experimental|Unsupported public|Unsafe"):
                    refresh.public_stage(directory, None)

    def test_reproduction_protocol_must_be_committed_before_any_output(self):
        name = sorted(refresh.REPRODUCTION_PROTOCOL_FILES)[0]
        self.git("rm", name)
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "Reproduction protocol source is not committed"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_statistics_manifest_is_closed_and_public_payload_is_validated(self):
        spec = refresh.SUPPLEMENTS["statistics"]
        directory = self.supplement_directories["statistics"]
        files, _ = refresh.public_stage(directory, spec["files"], spec["manifest"])
        self.assertIn("qq_differences.png", files)
        path = directory / spec["manifest"]
        original = json.loads(path.read_text())
        changed = json.loads(path.read_text())
        changed["files"]["predictions.csv"] = {"sha256": "a" * 64, "bytes": 1}
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "approved public files"):
            refresh.public_stage(directory, spec["files"], spec["manifest"])
        private = b"model,Author Email\nGRU,private@example.test\n"
        (directory / "pairwise_sensitivity.csv").write_bytes(private)
        original["files"]["pairwise_sensitivity.csv"] = {
            "sha256": refresh.legacy.digest(private), "bytes": len(private)}
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "Private/abstract"):
            refresh.public_stage(directory, spec["files"], spec["manifest"])

    def test_diagnostic_supplements_require_closed_allowlists_and_all_configuration_records(self):
        for label in ("replay_forensics", "replay_repeatability"):
            spec = refresh.SUPPLEMENTS[label]
            directory = self.supplement_directories[label]
            path = directory / spec["manifest"]
            original = json.loads(path.read_text())
            changed = json.loads(path.read_text())
            changed.pop("approved_public_files")
            path.write_text(json.dumps(changed))
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "explicit closed public approval"):
                refresh.public_stage(directory, None, spec["manifest"])
            path.write_text(json.dumps(original))
        spec = refresh.SUPPLEMENTS["replay_repeatability"]
        directory = self.supplement_directories["replay_repeatability"]
        path = directory / spec["manifest"]
        changed = json.loads(path.read_text())
        changed["files"].pop("publication_verification.json")
        changed["approved_public_files"].remove("publication_verification.json")
        path.write_text(json.dumps(changed))
        self.git("add", ".")
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "configuration files must be publicly approved"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_broken_plot_trailing_payload_and_unsupported_format(self):
        refresh.validate_public_bytes("qq.png", public_png())
        with self.assertRaisesRegex(ValueError, "Trailing data"):
            refresh.validate_public_bytes("qq.png", public_png() + b"hidden payload")
        corrupted = bytearray(public_png())
        corrupted[20] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            refresh.validate_public_bytes("qq.png", bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "Unsupported public evidence extension"):
            refresh.validate_public_bytes("model.pt", b"checkpoint")

    def test_rejects_archive_manifest_mismatch_unsafe_path_and_symlink(self):
        path = Path(self.temporary.name) / "test.zip"
        refresh.legacy.archive(path, {"public.txt": b"public"}, "8.4", "Literature", "a" * 40, "Readme")
        with ZipFile(path) as saved:
            files = {name: saved.read(name) for name in saved.namelist()}
        files["public.txt"] = b"tampered"
        with ZipFile(path, "w") as saved:
            for name, data in files.items():
                saved.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "member hash mismatch"):
            refresh.verify_archive(path)
        refresh.legacy.archive(path, {"../escape.txt": b"public"}, "8.4", "Literature", "a" * 40, "Readme")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            refresh.verify_archive(path)
        with ZipFile(path, "w") as saved:
            link = ZipInfo("link.txt")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            saved.writestr(link, "../outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            refresh.verify_archive(path)


if __name__ == "__main__":
    unittest.main()
