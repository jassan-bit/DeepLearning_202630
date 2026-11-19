"""Pinned-data parser regression tests; no market models or fits are run."""
from contextlib import redirect_stdout
import csv
from datetime import date, timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import worker_compat as compat

HERE = Path(__file__).resolve().parent


def scientific_checkout():
    explicit = os.environ.get("E3_SCIENTIFIC_CHECKOUT")
    if explicit is not None:
        candidate = Path(explicit).expanduser().absolute()
        if not explicit.strip() or not (candidate / compat.SCOPE / "evaluation/run.py").is_file():
            raise RuntimeError("E3_SCIENTIFIC_CHECKOUT must name the scientific checkout containing "
                               + compat.SCOPE + "/evaluation/run.py")
        return candidate
    candidates = [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents,
                  Path("/content") / ("e3-extended-source-" + compat.SOURCE),
                  Path("/workspace/e3_review9")]
    for candidate in candidates:
        if (candidate / compat.SCOPE / "evaluation/run.py").is_file():
            return candidate
    raise RuntimeError("Scientific checkout not found; set E3_SCIENTIFIC_CHECKOUT to its root")


def pinned_npz(checkout):
    explicit = os.environ.get("E3_NPZ_PATH")
    if explicit is not None:
        candidate = Path(explicit).expanduser().absolute()
        if not explicit.strip() or not candidate.is_file():
            raise RuntimeError("E3_NPZ_PATH must name the existing pinned current_cohort_arrays.npz file")
        return candidate
    candidates = [
        checkout / "book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz",
        Path("/content/e3-training-source-186d473fdfdf.npz"),
        Path("/workspace/e3_frozen_eval/inputs/current_cohort_arrays.npz"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Pinned NPZ not found; set E3_NPZ_PATH to current_cohort_arrays.npz")


ROOT = scientific_checkout()
RUN = ROOT / compat.SCOPE / "evaluation/run.py"
INPUTS = ROOT / compat.SCOPE / "admission/d00_inputs"
NPZ = pinned_npz(ROOT)
FIXTURES = HERE / "fixtures"


class ParserCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = compat.load_frozen(RUN)
        cls.history_api = cls.original._d00_compat_history
        cls.paths = [INPUTS / s["file"] for s in cls.history_api.SOURCES]
        cls.expected = [s["sha256"] for s in cls.history_api.SOURCES]
        cls.before_parser = staticmethod(cls.original.data_api.btc_history)
        cls.parser = staticmethod(compat.install_parser(cls.original))

    def arguments(self, output):
        return ["--output", str(output), "--source-commit", compat.SOURCE,
                "--admission", str(FIXTURES / "admission/admission.json"),
                "--protocol", str(FIXTURES / "protocol.json"),
                "--freeze-receipt", str(FIXTURES / "freeze_receipt.json"),
                "--npz", str(NPZ), "--btc-history", *map(str, self.paths),
                "--device", "cuda:0"]

    def test_original_parser_reproduces_exact_blank_float_error(self):
        with self.assertRaisesRegex(ValueError, "could not convert string to float: ''"):
            self.before_parser(self.paths, self.expected)

    def test_real_pinned_gap_is_retained_and_complete_bars_are_unchanged(self):
        rows = self.history_api.history(self.paths)
        self.assertEqual(len(rows), 1339)
        self.assertEqual([r["date"] for r in rows if r["missing"]], ["2023-03-24"])
        result = self.parser(self.paths, self.expected)
        self.assertEqual(len(result), 1324)
        for k in range(8):
            self.assertNotIn((date(2023, 3, 24) + timedelta(days=k)).isoformat(), result)
        self.assertIn("2023-04-01", result)

    def test_numpy_arithmetic_matches_original_algorithm_bit_for_bit(self):
        closes = {}
        for path in self.paths:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["close"]:
                        closes[row["market_day_utc"][:10]] = float(row["close"])
        expected = {}
        for day in sorted(closes):
            end = date.fromisoformat(day)
            support = [(end - timedelta(days=j)).isoformat() for j in range(7, -1, -1)]
            if all(d in closes for d in support):
                expected[day] = float(np.std(np.diff(np.log([closes[d] for d in support])), ddof=0))
        self.assertEqual(self.parser(self.paths, self.expected), expected)

    def test_changed_input_bytes_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / self.paths[0].name
            raw = self.paths[0].read_bytes() + b"\n"
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "pinned daily BTC"):
                self.parser([path, self.paths[1]], self.expected)
            self.assertEqual(path.read_bytes(), raw)

    def test_partial_missing_bar_is_refused_by_reused_validator(self):
        history = self.history_api
        source = {"header": ("market_day_utc", *history.FIELDS), "rows": 1,
                  "start": "2023-03-24", "end": "2023-03-24"}
        raw = b"market_day_utc,open,high,low,close,volume\n2023-03-24 00:00:00+00:00,1,2,1,,5\n"
        with self.assertRaisesRegex(ValueError, "partially missing"):
            history.parse_daily(raw, source)

    def test_changed_frozen_parser_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = Path(temp)
            for relative in compat.FROZEN_FILE_SHA256:
                destination = scope / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(RUN.parent.parent / relative, destination)
            (scope / "evaluation/data.py").write_bytes(b"raise AssertionError('must not import')\n")
            with self.assertRaisesRegex(ValueError, "Frozen source digest differs: evaluation/data.py"):
                compat.load_frozen(scope / "evaluation/run.py")

    def test_symlinked_input_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            link = Path(temp) / "npz"
            link.symlink_to(NPZ)
            with self.assertRaisesRegex(ValueError, "symlink"):
                compat.regular(link)

    def test_data_only_preflight_checks_real_npz_without_fits_or_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            job = output / "jobs/D01/example/result.json"
            job.parent.mkdir(parents=True)
            job.write_bytes(b'{"status":"completed","test_fixture_only":true}\n')
            before = job.read_bytes()
            original = self.original
            with patch.object(original.fit_api, "fit", side_effect=AssertionError("No fitting allowed")), \
                 patch.object(original.fit_api, "predict", side_effect=AssertionError("No predictions allowed")), \
                 patch.object(original.fit_api, "_factory", side_effect=AssertionError("No model construction allowed")), \
                 patch.object(original.classical, "predictions", side_effect=AssertionError("No classical predictions allowed")), \
                 redirect_stdout(io.StringIO()):
                report = compat.preflight(original, compat.parse_forwarded(self.arguments(output)))
            self.assertEqual(report["status"], "passed_data_only")
            self.assertEqual(report["cohorts"], {"fit":532,"validation":116,"calibration":53,"test":608})
            self.assertEqual(report["cohort_parity"]["known_targets_compared"], 1316)
            self.assertEqual(report["missing_required_control_lags"], [])
            self.assertEqual(report["new_real_data_fits"], 0)
            self.assertEqual(report["models_constructed"], 0)
            self.assertFalse(report["R3_accredited"])
            self.assertEqual(job.read_bytes(), before)
            self.assertEqual(sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()),
                             ["jobs/D01/example/result.json", "parser_compatibility/d00_preflight.json"])

    def test_repeated_preflight_preserves_first_receipt_as_jobs_accumulate(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp)
            args = compat.parse_forwarded(self.arguments(output))
            first = compat.preflight(self.original, args)
            raw = (output / "parser_compatibility/d00_preflight.json").read_bytes()
            job = output / "jobs/D00/newly_completed/result.json"
            job.parent.mkdir(parents=True)
            job.write_text('{"status":"completed","test_fixture_only":true}\n')
            second = compat.preflight(self.original, args)
            self.assertEqual(second, first)
            self.assertEqual((output / "parser_compatibility/d00_preflight.json").read_bytes(), raw)
            self.assertTrue(job.exists())

    def test_changed_protocol_refused_before_report_write(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            args = compat.parse_forwarded(self.arguments(output))
            bad = output / "changed_protocol.json"
            bad.write_bytes((FIXTURES / "protocol.json").read_bytes() + b"\n")
            args.protocol = str(bad)
            with self.assertRaisesRegex(ValueError, "protocol SHA256 differs"):
                compat.preflight(self.original, args)
            self.assertFalse((output / "parser_compatibility").exists())

    def test_original_cli_flags_forwarded_verbatim_and_main_called_once(self):
        remaining = self.arguments(Path("/content/existing-output"))
        remaining += ["--package-path", "/content/a path containing spaces"]
        fake = type("Frozen", (), {})()
        fake.main = lambda: tuple(sys.argv)
        with patch.object(compat, "load_frozen", return_value=fake) as load, \
             patch.object(compat, "install_parser") as install, patch.object(sys, "argv", ["old"]):
            result = compat.main(["--frozen-run", str(RUN), *remaining])
            self.assertEqual(result, (str(RUN), *remaining))
            load.assert_called_once_with(str(RUN))
            install.assert_called_once_with(fake)

    def test_required_lag_is_checked_instead_of_deleting_heldout_date(self):
        original = self.original
        data_function = original.data_api.btc
        def remove_required_lag(*args, **kwargs):
            panel = data_function(*args, **kwargs)
            axis = original.classical._axis(panel)
            target = str(panel["val_dates"][0])
            panel["history"].pop(axis[axis.index(target)-1])
            return panel
        with tempfile.TemporaryDirectory() as temp, patch.object(original.data_api, "btc", side_effect=remove_required_lag):
            args = compat.parse_forwarded(self.arguments(Path(temp)))
            with self.assertRaisesRegex(ValueError, "Missing daily sigma7 states were altered"):
                compat.preflight(original, args)
            self.assertFalse((Path(temp) / "parser_compatibility").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
