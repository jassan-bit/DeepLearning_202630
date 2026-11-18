"""Pruebas stdlib de barreras; sin mercado, entrenamiento, red ni publicación."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "e3_colab_preflight_under_test", HERE / "colab_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def git_command(root, *args, date="2020-01-01T00:00:00+00:00"):
    """Git local y efímero, con identidad/fechas ajenas a la configuración real."""
    env = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_DATE=date,
        GIT_COMMITTER_DATE=date,
    )
    return subprocess.check_output(
        [
            "git", "-C", str(root),
            "-c", "user.name=Fixture local",
            "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false",
            *args,
        ],
        env=env,
        text=True,
        stderr=subprocess.PIPE,
    )


def seal_fixture(
    root,
    *,
    annotated=True,
    frozen_at="2020-01-02T00:00:00Z",
    tag_at="2020-01-01T00:00:00+00:00",
    first_fit="2030-01-03T00:00:00Z",
    datasets_empty=False,
):
    """Manifiesto mínimo de prueba, sellado sin acceso a repositorios externos.

    El contenido de evidencia es artificial. Este fixture comprueba la barrera
    de metadatos y orden temporal; no admite datasets científicos reales.
    """
    git_command(root, "init", "-q")
    evidence = json.dumps({
        "fixture_only": True,
        "same_information_sets_all_17": True,
        "train_only_scalers": True,
        "selection_before_evaluation": True,
        "calibration_policy_frozen": True,
        "all_17_systems_covered": True,
        "same_resource_and_search_policy": True,
        "failed_and_aborted_jobs_charged": True,
        "old_ledgers_reset": False,
        "prospective_for_this_campaign": True,
        "observed_effect_used_as_planning_alternative": False,
        "rivals": 16,
        "procedures_per_rival": 3,
        "full_battery_dimensioned": True,
        "minimum_individual_power": 0.95,
        "joint_power": 0.95,
    }, sort_keys=True).encode("utf-8")
    (root / "evidence.txt").write_bytes(evidence)
    bound = {"path": "evidence.txt", "sha256": digest(evidence)}
    datasets = []
    for index in range(3):
        item = {
            "id": f"fixture-{index}",
            "status": "admitted",
            "target": "population_sigma7_log_close_returns_ddof0_unannualized",
            "horizon": 1,
            "clock": "UTC_calendar_day",
        }
        for key in (
            "source_manifest", "license_evidence", "technical_admission", "exposure_log"
        ):
            item[key] = deepcopy(bound)
        datasets.append(item)
    manifest = {
        "candidate": deepcopy(preflight.CANDIDATE_PROFILE),
        "family": list(preflight.FAMILY),
        "datasets": [] if datasets_empty else datasets,
        "protocol_freeze": {
            "annotated_git_tag": "fixture-protocol",
            "manifest_path_in_tag": "campaign.json",
            "frozen_at_utc": frozen_at,
        },
        "planned_first_real_fit_or_evaluation_utc": first_fit,
        "real_fits_or_evaluation_already_started": False,
    }
    for key in (
        "candidate_source_configuration", "common_treatment", "budget_17_systems",
        "power_individual_and_joint", "robustness_and_calibration",
    ):
        manifest[key] = deepcopy(bound)
    (root / "campaign.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    git_command(root, "add", "--", "campaign.json", "evidence.txt")
    git_command(root, "commit", "-q", "-m", "Fixture temporal sin publicación")
    args = ("tag", "-a", "fixture-protocol", "-m", "Fixture anotado") if annotated else (
        "tag", "fixture-protocol"
    )
    git_command(root, *args, date=tag_at)
    return manifest


def campaign_at_fixture_time(manifest, root=preflight.ROOT):
    """Reloj fijo: la prueba de orden no caduca cuando cambie la fecha real."""
    with mock.patch.object(preflight, "datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
        return preflight.campaign_gate(manifest, root=root)


class ColabPreflightGateTests(unittest.TestCase):
    def test_gpu_hardware_metadata_does_not_certify_model_execution(self):
        # All hardware and worker responses here are deliberately mocked.
        # This tests receipt semantics; it supplies no GPU execution evidence.
        cuda = SimpleNamespace(is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="Fixture GPU", total_memory=1024),
            get_device_capability=lambda _: (8, 9), device_count=lambda: 1)
        torch = SimpleNamespace(__version__="2.8.0+cu126", cuda=cuda,
            version=SimpleNamespace(cuda="12.6"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)))
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(preflight.importlib.metadata, "version", side_effect=lambda name: preflight.APPROVED_DEPENDENCIES[name]), \
                mock.patch.object(preflight.subprocess, "check_output", return_value="Fixture GPU; no hardware executed"), \
                mock.patch.object(preflight.sys, "version_info", (3, 12, 12)):
            hardware = preflight.environment("cuda:0")
        self.assertTrue(hardware["cuda_available"])
        self.assertEqual(hardware["GPU"]["name"], "Fixture GPU")
        self.assertNotIn("GPU_execution_verified", hardware)
        for case, candidate_device, expected in (("one_neural_CPU", "cpu", False),
                                                  ("all_fourteen_neural_GPU", "cuda:0", True)):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = SimpleNamespace(output=root / "output", device="cuda:0", package_root=root / "D02")
                def synthetic_worker(command, **kwargs):
                    name = command[command.index("--worker") + 1]
                    output = Path(command[command.index("--output") + 1])
                    device = candidate_device if name == preflight.CANDIDATE else (
                        "cpu_classical" if name in preflight.CLASSICAL else "cuda:0")
                    preflight.write_new(output / "receipt.json", {
                        "model": name, "status": "passed", "device": device,
                        "synthetic_backward_executed": name not in preflight.CLASSICAL,
                        "checkpoint_reload_identical": True, "fixture_only_no_GPU_executed": True})
                    return SimpleNamespace(returncode=0)
                with mock.patch.object(preflight, "environment", return_value=hardware), \
                        mock.patch.object(preflight, "source_gate", return_value={"passed": True}), \
                        mock.patch.object(preflight, "git", return_value="fixture-HEAD"), \
                        mock.patch.object(preflight, "portable_inventory", return_value=({}, {}, {})), \
                        mock.patch.object(preflight.subprocess, "run", side_effect=synthetic_worker) as worker, \
                        mock.patch("builtins.print"):
                    self.assertEqual(preflight.run(args), 0)
                self.assertEqual(worker.call_count, 17)
                before = json.loads((args.output / "hardware_and_source_receipt.json").read_text())
                after = json.loads((args.output / "preflight_receipt.json").read_text())
                self.assertFalse(before["GPU_execution_verified"])
                self.assertNotIn("GPU_execution_verified", before["environment"])
                self.assertEqual(after["GPU_execution_verified"], expected)
                self.assertNotIn("GPU_execution_verified", after["environment"])

    def test_source_gate_rejects_changed_source_with_unchanged_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = "src/e3_innovation_model.py"
            source = root / selected
            original = b"SOURCE_ID = 'reviewed'\n"
            for relative in preflight.LOCKED_SOURCE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original)
            lock = {
                "reviewed_source_commit": "a" * 40,
                "files": {
                    relative: digest(original)
                    for relative in sorted(preflight.LOCKED_SOURCE_FILES)
                },
            }
            (root / "colab_source_lock.json").write_text(json.dumps(lock))
            with mock.patch.object(preflight, "HERE", root):
                self.assertTrue(preflight.source_gate(root)["passed"])
                source.write_bytes(b"SOURCE_ID = 'altered'\n")
                result = preflight.source_gate(root)
            self.assertFalse(result["passed"])
            failures = [item for item in result["files"] if not item["verified"]]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["path"], selected)
            self.assertEqual(failures[0]["expected_sha256"], digest(original))
            lock["files"] = {}
            (root / "colab_source_lock.json").write_text(json.dumps(lock))
            with mock.patch.object(preflight, "HERE", root):
                self.assertFalse(preflight.source_gate(root)["passed"])

    def test_campaign_gate_fails_closed_without_complete_dataset_admission(self):
        for manifest in (None, {}, [], {"datasets": []}, {"datasets": [None]}):
            with self.subTest(manifest=manifest):
                result = campaign_at_fixture_time(manifest)
                self.assertFalse(result["allowed"])
                self.assertTrue(result["blockers"])
                self.assertFalse(result["real_training_dispatched"])
        if shutil.which("git") is None:
            self.skipTest("Git no disponible para comprobar el manifiesto sellado")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = seal_fixture(root, datasets_empty=True)
            result = campaign_at_fixture_time(manifest, root=root)
            self.assertFalse(result["allowed"])
            self.assertTrue(any("tres datasets" in reason for reason in result["blockers"]))
            self.assertFalse(any("tag Git anotado" in reason for reason in result["blockers"]))
            self.assertFalse(result["real_training_dispatched"])

    @unittest.skipUnless(shutil.which("git"), "Git requerido para verificar tags locales")
    def test_campaign_gate_requires_annotated_tag_and_freeze_before_first_fit(self):
        cases = (
            ("control_valido", {}, True, None),
            ("tag_ligero", {"annotated": False}, False, "tag Git anotado"),
            (
                "congelamiento_posterior_al_primer_ajuste",
                {"frozen_at": "2031-01-04T00:00:00Z"},
                False,
                "preceder al primer ajuste",
            ),
            (
                "tag_posterior_al_primer_ajuste",
                {"tag_at": "2031-01-04T00:00:00+00:00"},
                False,
                "preceder al primer ajuste",
            ),
            (
                "primer_ajuste_planificado_en_el_pasado",
                {"first_fit": "2021-01-03T00:00:00Z"},
                False,
                "seguir en el futuro",
            ),
            (
                "congelamiento_todavia_futuro",
                {
                    "frozen_at": "2099-01-02T00:00:00Z",
                    "first_fit": "2099-01-03T00:00:00Z",
                },
                False,
                "congelamiento debe existir ahora",
            ),
        )
        for name, options, expected, reason in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = seal_fixture(root, **options)
                result = campaign_at_fixture_time(manifest, root=root)
                self.assertEqual(result["allowed"], expected, result["blockers"])
                if reason is None:
                    self.assertEqual(result["blockers"], [])
                else:
                    self.assertTrue(any(reason in item for item in result["blockers"]))
                self.assertFalse(result["real_training_dispatched"])

    def test_safe_extract_rejects_traversal_before_extracting_any_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            destination = root / "extracted"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("D02/safe.txt", "No debe extraerse parcialmente")
                stream.writestr("../escaped.txt", "Fuera del destino")
            with self.assertRaisesRegex(ValueError, "Unsafe package member"):
                preflight.safe_extract(archive, destination)
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse((destination / "D02/safe.txt").exists())
            self.assertFalse(destination.exists())

            outside = root / "outside"
            outside.mkdir()
            destination.mkdir()
            (destination / "link").symlink_to(outside, target_is_directory=True)
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("link/escaped.txt", "No debe seguir el enlace existente")
            with self.assertRaisesRegex(ValueError, "new destination"):
                preflight.safe_extract(archive, destination)
            self.assertFalse((outside / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
