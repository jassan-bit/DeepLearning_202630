"""Portable data guards; synthetic prices/arrays do not certify market evidence."""
import ast
import contextlib
import csv
from datetime import date, timedelta
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e2_standalone_dataset", ROOT / "scripts/e2_standalone_dataset.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def npy(shape, dtype, values, *, fortran=False):
    if dtype == "<U10":
        raw = b"".join(value.ljust(10, "\0").encode("utf-32-le") for value in values)
    else:
        raw = struct.pack("<" + ("f" if dtype == "<f4" else "d") * len(values), *values)
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": tuple(shape)}).encode("ascii")
    header += b" " * ((-10 - len(header) - 1) % 16) + b"\n"
    payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw
    digest_header = json.dumps({"shape": shape or [1], "dtype": dtype}, sort_keys=True).encode()
    return payload, {"shape": list(shape), "dtype": dtype,
                     "array_sha256": runner.sha256(digest_header + b"\0" + raw)}


def csv_bytes(header, rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue().encode()


def synthetic_current():
    """Full contract geometry from invented prices, never source certification."""
    days = runner.calendar(runner.START, runner.EVALUATION_END)
    bars, sigmas, rows = [], [], []
    for i, day in enumerate(days):
        if day == date(2023, 3, 24):
            bar = None
        else:
            close = 100 + i * .02 + math.sin(i / 9)
            bar = [close - .1, close + 1, close - 1, close, 1000 + i]
        bars.append(bar)
        values = bars[i - 7:i + 1] if i >= 7 else []
        sigma = None
        if len(values) == 8 and all(v is not None for v in values):
            returns = [math.log(values[k][3]) - math.log(values[k - 1][3]) for k in range(1, 8)]
            mean = math.fsum(returns) / 7
            sigma = math.sqrt(math.fsum((r - mean) ** 2 for r in returns) / 7)
        sigmas.append(sigma)
        rows.append([day.isoformat() + "T00:00:00+00:00", *(bar or [""] * 5)])
    ids = {"training": [d.isoformat() for d, y in zip(days, sigmas) if y is not None and d <= date(2024, 10, 31)],
           "calibration": [d.isoformat() for d in days if date(2024, 11, 8) <= d <= date(2024, 12, 30)],
           "evaluation": [d.isoformat() for d in days if d >= runner.EVALUATION_START]}
    assert {k: len(v) for k, v in ids.items()} == runner.COHORT_COUNTS
    partitions = {"training": {"training_sample_IDs": ids["training"]},
                  "calibration": {"sample_IDs": ids["calibration"]}}
    original = {"daily_market_development.csv": csv_bytes(runner.MARKET_HEADER,
                    [row + ([10] if row[1] != "" else [""]) for row in rows[:731]]),
                "partition_audit.json": json.dumps({"final_partition": partitions}).encode()}
    arrays, metadata, keys = {}, {}, {}
    positions = {d.isoformat(): i for i, d in enumerate(days)}
    scale = math.fsum(sigmas[positions[d]] for d in ids["training"]) / 655
    for cohort, dates in ids.items():
        count = len(dates)
        targets = [sigmas[positions[d]] for d in dates]
        raw = [v for d in dates for bar in bars[positions[d] - 7:positions[d]] for v in bar]
        entries = {"dates": ([count], "<U10", dates), "raw": ([count, 7, 5], "<f8", raw),
                   "X_relative": ([count, 7, 5], "<f8", [0.] * count * 35),
                   "X_scaled": ([count, 7, 5], "<f4", [0.] * count * 35),
                   "y_original": ([count], "<f8", targets),
                   "y_scaled": ([count], "<f4", [y / scale for y in targets])}
        keys[cohort] = {}
        for role, (shape, dtype, values) in entries.items():
            key = cohort + "_" + role
            arrays[key], metadata[key] = npy(shape, dtype, values)
            keys[cohort][role] = key
    for key, shape, values in [("training_target_scale", [], [scale]),
                               ("training_channel_mean", [5], [0.] * 5),
                               ("training_channel_std", [5], [1.] * 5)]:
        arrays[key], metadata[key] = npy(shape, "<f8", values)
    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, data in arrays.items():
            archive.writestr(key + ".npy", data)
    npz = compressed.getvalue()
    scaler = {"target_scale": scale, "mean": [0.] * 5, "std": [1.] * 5,
              "train_dates": ids["training"], "training_sha256": "a" * 64}
    scaler_bytes = json.dumps(scaler).encode()
    payloads = {"daily_market_evaluation.csv": csv_bytes(runner.MARKET_HEADER[:-1], rows[731:]),
                "current_cohort_arrays.npz": npz,
                "cohort_array_manifest.json": json.dumps({"sha256": runner.sha256(npz), "arrays": metadata}).encode(),
                "reconstructed_training_scaler.json": scaler_bytes,
                "audit_result.json": json.dumps({"actual_cohorts_audited": runner.COHORT_COUNTS,
                    "data_ready_for_bounded_controls": True, "arrays_npz_sha256": runner.sha256(npz),
                    "array_keys": keys, "reconstructed_scaler_sha256": runner.sha256(scaler_bytes)}).encode()}
    bundle = {"version": runner.CURRENT_VERSION, "cohorts": runner.COHORT_COUNTS, "array_keys": keys,
              "files": [{"path": name, "sha256": runner.sha256(data), "bytes": len(data), "role": "synthetic guard"}
                        for name, data in sorted(payloads.items())]}
    bundle_bytes = json.dumps(bundle).encode()
    return original, (bundle, payloads, [], bundle_bytes)


class PortableCurrentGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.current = synthetic_current()

    def test_full_synthetic_geometry_checks_without_claiming_transforms_or_models(self):
        report, dataset = runner.audit_current_dataset(self.original, self.current)
        self.assertEqual({k: v["count"] for k, v in report["cohorts"].items()}, runner.COHORT_COUNTS)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(dataset.decode())))), 608)
        self.assertEqual(report["arrays"]["count"], 21)
        self.assertFalse(report["scaler"]["fitting_repeated"])
        self.assertFalse(report["scaler"]["model_specific_scalers_checked"])
        self.assertTrue(any("X_relative/X_scaled" in s for s in report["limits"]))

    def test_npy_object_fortran_nonfinite_and_truncation_are_rejected(self):
        for payload in (npy([1], "<f8", [1.], fortran=True)[0],
                        npy([1], "<f8", [float("nan")])[0],
                        npy([1], "<f8", [1.])[0][:-1],
                        npy([1], "<f8", [1.])[0].replace(b"<f8", b"|O8")):
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                runner.parse_npy(payload)

    def test_npy_unicode_calendar_and_scalar_are_decoded(self):
        scalar = runner.parse_npy(npy([], "<f8", [.25])[0])
        self.assertEqual(scalar["shape"], [])
        self.assertEqual(scalar["values"], [.25])
        self.assertEqual(runner.parse_npy(npy([1], "<U10", ["2025-01-01"])[0])["values"], ["2025-01-01"])

    def test_raw_array_digest_change_is_detected(self):
        manifest = json.loads(self.current[1]["cohort_array_manifest.json"])
        key = next(iter(manifest["arrays"]))
        manifest["arrays"][key]["array_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            runner.read_npz(self.current[1]["current_cohort_arrays.npz"], manifest)

    def test_duplicate_or_traversing_npz_members_are_rejected(self):
        manifest = json.loads(self.current[1]["cohort_array_manifest.json"])
        names = list(manifest["arrays"])
        for members in (["../training_raw.npy"], [names[0] + ".npy"] * 21):
            data = io.BytesIO()
            with warnings.catch_warnings(), zipfile.ZipFile(data, "w") as archive:
                # Duplicate ZIP names are intentionally invalid synthetic input.
                warnings.simplefilter("ignore", UserWarning)
                for name in members:
                    archive.writestr(name, npy([1], "<f8", [1.])[0])
            with self.assertRaises(ValueError):
                runner.read_npz(data.getvalue(), manifest)

    def test_target_and_raw_support_mismatch_are_detected_after_valid_hashes(self):
        for change in ("dates", "raw", "y_original"):
            arrays = runner.read_npz(self.current[1]["current_cohort_arrays.npz"],
                                     json.loads(self.current[1]["cohort_array_manifest.json"]))
            key = self.current[0]["array_keys"]["evaluation"][change]
            if change == "dates":
                arrays[key]["values"][0] = "2024-12-31"
            else:
                arrays[key]["values"][0] += 1.
            with self.subTest(change=change), patch.object(runner, "read_npz", return_value=arrays), self.assertRaises(ValueError):
                runner.audit_current_dataset(self.original, self.current)

    def test_scaler_training_dates_must_match_training_only(self):
        bundle, payloads, records, raw_bundle = self.current
        changed = dict(payloads)
        scaler = json.loads(changed["reconstructed_training_scaler.json"])
        scaler["train_dates"][-1] = "2025-01-01"
        changed["reconstructed_training_scaler.json"] = json.dumps(scaler).encode()
        audit = json.loads(changed["audit_result.json"])
        audit["reconstructed_scaler_sha256"] = runner.sha256(changed["reconstructed_training_scaler.json"])
        changed["audit_result.json"] = json.dumps(audit).encode()
        with self.assertRaises(ValueError):
            runner.audit_current_dataset(self.original, (bundle, changed, records, raw_bundle))

    def test_bundle_is_optional_but_invalid_present_bundle_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            self.assertIsNone(runner.read_current_bundle(inputs))
            (inputs / runner.CURRENT_BUNDLE).write_text('{"version":"wrong"}')
            with self.assertRaises((ValueError, KeyError)):
                runner.read_current_bundle(inputs)

    def test_bundle_hash_tampering_and_duplicate_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            for name, payload in self.current[1].items():
                (inputs / name).write_bytes(payload)
            (inputs / runner.CURRENT_BUNDLE).write_bytes(self.current[3])
            with patch.object(runner, "EVALUATION_DAILY_SHA", runner.sha256(self.current[1]["daily_market_evaluation.csv"])):
                self.assertIsNotNone(runner.read_current_bundle(inputs))
                (inputs / "current_cohort_arrays.npz").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    runner.read_current_bundle(inputs)
            bundle = json.loads(self.current[3])
            bundle["files"][-1] = bundle["files"][0]
            (inputs / runner.CURRENT_BUNDLE).write_text(json.dumps(bundle))
            with self.assertRaises(ValueError):
                runner.read_current_bundle(inputs)

    def test_paths_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            (inputs / "good").write_bytes(b"input")
            (inputs / "link").symlink_to(inputs / "good")
            for name in ("../good", "/good", "x/../good", "./good", "x\\good", "link"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.safe_input_path(inputs, name)

    def test_evaluation_csv_cannot_compress_a_calendar_gap(self):
        payload = self.current[1]["daily_market_evaluation.csv"].splitlines(keepends=True)
        del payload[4]
        with self.assertRaises(ValueError):
            runner.parse_daily(b"".join(payload), runner.MARKET_HEADER[:-1], runner.EVALUATION_START, runner.EVALUATION_END)

    def test_runner_uses_only_standard_library_imports(self):
        tree = ast.parse((ROOT / "scripts/e2_standalone_dataset.py").read_text())
        imports = [name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        imports += [node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertTrue(set(imports) <= sys.stdlib_module_names)


class ExistingOfflineCompatibility(unittest.TestCase):
    def test_six_original_pins_and_offline_outputs_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = Path(directory) / "inputs", Path(directory) / "output"
            inputs.mkdir()
            for name, source in runner.SOURCES.items():
                (inputs / name).write_bytes((ROOT / source["repository_path"]).read_bytes())
            with patch.object(runner, "build_opener", side_effect=AssertionError("Offline means no network")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--input-dir", str(inputs), "--output", str(output)]), 0)
            self.assertEqual({p.name for p in output.iterdir()}, {"daily_market_development.csv", "daily_volatility_development.csv",
                "development_dataset.csv", "dataset_audit.json", "dataset_audit.html"})
            report = json.loads((output / "dataset_audit.json").read_text())
            self.assertEqual(report["mode"], "offline_development_validation")
            self.assertNotIn("current_dataset", report)
            self.assertEqual(report["development"]["valid_sigma7_labels"], 716)
            for name in ("daily_market_development.csv", "daily_volatility_development.csv"):
                self.assertEqual((output / name).read_bytes(), (inputs / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
