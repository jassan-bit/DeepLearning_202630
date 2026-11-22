"""Offline software checks for the interactive development helper.

Run with stdlib unittest. The original engine is imported only in the isolated
CPU runtime. All identities and all price bars here are synthetic fixtures;
these tests do not make an actor declaration or a market admission claim.
Network calls and access to a final partition fail immediately in subprocesses.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get(
    "E3_HELPER_TEST_SOURCE",
    "/workspace/e3_engine_published_source_cache/source-c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c",
))
PYTHON = Path(os.environ.get(
    "E3_HELPER_TEST_PYTHON",
    "/workspace/e3_engine_clean_cache/venv-cpu-652abf93aba295c3-py312/bin/python",
))
SPEC = importlib.util.spec_from_file_location("development_owner_helper", HERE / "prepare_development_owner.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

RULE = {
    "version": "development-calendar-proportions-1", "fit_fraction": .60,
    "selection_fraction": .10, "planning_fraction": .20, "calibration_fraction": .10,
    "embargo_bars": 0,
    "basis": "expected exchange sessions only; no market prices, labels or performance used",
    "support_rule": "raw disjoint bounds; engine purges endpoints until all 15 closes fit the assigned support",
}
ACTOR = ("SYNTHETIC SOFTWARE FIXTURE", "fixture@example.invalid", "test fixture")
SECRET = "SYNTHETIC_API_KEY_DO_NOT_LOG_978321"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshot(root):
    """Include bytes and modification times to detect silent rewrites."""
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in Path(root).rglob("*") if p.is_file()}


# Guard the actual engine, rather than substituting its integrity functions.
# The path guard also catches accidental reads outside data.load_partition.
GUARDS = r'''
import json, sys
from pathlib import Path
from engine import data
network_calls = []
partition_loads = []
def no_network(*args, **kwargs):
    network_calls.append(True)
    raise AssertionError('NETWORK_FORBIDDEN_IN_SOFTWARE_TEST')
data._network_get = no_network
import urllib.request, socket
urllib.request.urlopen = no_network
socket.create_connection = no_network
real_load = data.load_partition
def development_load(panel_dir, part, **kwargs):
    assert part in ('fit','selection','planning','calibration'), 'TEST_ACCESS_FORBIDDEN'
    assert not kwargs.get('allow_test'), 'TEST_ACCESS_FORBIDDEN'
    partition_loads.append(part)
    return real_load(panel_dir, part, **kwargs)
data.load_partition = development_load
real_open = Path.open
def development_open(path, *args, **kwargs):
    assert path.name != 'test.npz', 'TEST_FILE_ACCESS_FORBIDDEN'
    return real_open(path, *args, **kwargs)
Path.open = development_open
'''
TRACE = "\nsys.stderr.write(json.dumps(dict(network_calls=network_calls, partition_loads=partition_loads))+'\\n')\n"


class DevelopmentHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir() or not PYTHON.is_file():
            raise RuntimeError("Pinned source/runtime unavailable; set E3_HELPER_TEST_SOURCE and E3_HELPER_TEST_PYTHON")
        cls.proposals = {p["id"]: p for p in read_json(SOURCE / "configs/data_panels.json")["panels"]}
        cls.template = read_json(SOURCE / "docs/OWNER_EXPOSURE_FORM.json")
        program = """
from engine import data
import json, sys
proposals = json.load(sys.stdin)
print(json.dumps({p['id']:data.expected_sessions(p['proposed_development_start'],p['proposed_development_end'],p['calendar']) for p in proposals}))
"""
        out = cls.run_engine(program, list(cls.proposals.values()))
        cls.sessions = json.loads(out.stdout)

    @staticmethod
    def run_engine(program, payload):
        env = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
                   NASDAQ_DATA_LINK_API_KEY=SECRET, QUANDL_API_KEY=SECRET)
        return subprocess.run([str(PYTHON), "-B", "-c", GUARDS + program + TRACE],
                              input=json.dumps(payload), capture_output=True, text=True,
                              cwd=SOURCE, env=env, check=True, timeout=60)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-helper-software-")
        self.addCleanup(self.temporary.cleanup)
        self.private = Path(self.temporary.name) / "private"
        self.private.mkdir()
        self.rows = []
        for panel in helper.PANELS:
            p = copy.deepcopy(self.proposals[panel])
            folder = self.private / "inputs" / panel
            folder.mkdir(parents=True)
            row = dict(panel=panel, proposal=p, folder=str(folder),
                       raw=str(self.private / "raw" / panel / "development-v1"),
                       panel_dir=str(self.private / "panels" / panel / "development-v1"))
            license_path = SOURCE / p["rights_snapshot_path"]
            row["blank_owner"] = dict(self.template, panel_ids=[panel], reserve_mode="development_only",
                source_url=p["rights_source_url"], source_license_evidence_path=str(license_path),
                source_license_evidence_sha256=helper._sha(license_path),
                dataset_scope=dict(panel=panel, provider=p["provider"], instrument=p["instrument"],
                    start=p["proposed_development_start"], end=p["proposed_development_end"]))
            write_json(folder / "owner-development.json", row["blank_owner"])
            write_json(folder / "acquisition.json", dict(provider=p["provider"], instrument=p["instrument"],
                development_start=p["proposed_development_start"], development_end=p["proposed_development_end"]))
            n = len(self.sessions[panel])
            ends = [int(n*.60), int(n*.70), int(n*.90), n]
            bounds = {name: [lo, hi-1] for name, lo, hi in zip(helper.PARTS, [0] + ends[:-1], ends)}
            write_json(folder / "development-split.json", dict(raw_start=p["proposed_development_start"],
                raw_end=p["proposed_development_end"], calendar=p["calendar"], raw_bounds=bounds,
                embargo_bars=0, prospective_split_rule=RULE, metadata_only_expected_raw_bars=n))
            self.rows.append(row)
        write_json(self.private / "development-split-rule.json", RULE)
        map_path = self.private / "development-map.json"
        write_json(map_path, {r["panel"]: r["panel_dir"] for r in self.rows})
        self.calls = []
        self.private_results = []
        self.namespace = dict(SOURCE=SOURCE, SOURCE_COMMIT=helper.SOURCE_COMMIT,
            SETTINGS=dict(source_commit=helper.SOURCE_COMMIT, source_root=str(SOURCE)),
            PRIVATE=self.private, definitions=copy.deepcopy(self.rows), DEVELOPMENT_MAP=map_path,
            run_data=self.no_data, run_private_program=self.run_private)

    def no_data(self, args, provider=None):
        self.fail("Unexpected engine command: " + args[0])

    def run_private(self, program, payload):
        output = self.run_engine(program, payload)
        self.private_results.append(output)
        return output

    def checked(self, rows=None):
        return helper._check(self.namespace, self.rows if rows is None else rows, RULE)

    def invoke(self, answers=()):
        output = io.StringIO()
        with mock.patch("builtins.input", side_effect=answers) as prompt, contextlib.redirect_stdout(output):
            result = helper.prepare_development(self.namespace)
        return result, output.getvalue(), prompt

    def complete_owner(self, row=None, **changes):
        row = self.rows[1] if row is None else row
        owner = dict(row["blank_owner"], kind="SYNTHETIC_SOFTWARE_TEST_NOT_A_HUMAN_DECLARATION",
            actor_name=ACTOR[0], actor_contact=ACTOR[1], actor_role=ACTOR[2],
            statement_utc="2026-11-22T00:00:00+00:00", observation_cutoff_utc="2026-11-22T00:00:00+00:00",
            evidence_origin="Synthetic test fixture only", acknowledge_prior_D00_D03_exposed=True,
            all_data_declared_exposed_development=True, source_terms_accepted_by_real_actor=True,
            allowed_processing_scope=helper.SCOPE, allow_derived_publication=True,
            allow_raw_redistribution=False, AI_completed_signature=False, authorize_test_preparation=False)
        owner.update(changes)
        write_json(Path(row["folder"]) / "owner-development.json", owner)
        return owner

    def add_evidence(self, row=None):
        row = self.rows[1] if row is None else row
        folder = Path(row["folder"])
        owner = read_json(folder / "owner-development.json")
        p = row["proposal"]
        write_json(folder / "rights.json", dict(status="approved", actor_name=owner["actor_name"],
            actor_contact=owner["actor_contact"], evidence_path=owner["source_license_evidence_path"],
            evidence_sha256=owner["source_license_evidence_sha256"], scope=helper.SCOPE,
            source_url=p["rights_source_url"], license=p["license"], allow_derived_publication=True,
            allow_raw_redistribution=False, statement_utc=owner["statement_utc"]))
        self.run_engine("""
payload = json.load(sys.stdin)
data.record_owner_evidence(payload['owner'],payload['owner'],payload['receipt'])
""", dict(owner=str(folder / "owner-development.json"), receipt=str(folder / "exposure-development.json")))

    def states(self, **changes):
        state = dict(admitted=False, raw_complete=False, blank_owner=True,
                     owner_receipt=False, rights_present=False)
        state.update(changes)
        return {p: dict(state) for p in helper.PANELS}

    def assert_rejected_without_changes(self, message, rows=None):
        before = snapshot(self.private)
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.checked(rows)
        self.assertIn(message, failure.exception.stdout + failure.exception.stderr)
        self.assertEqual(before, snapshot(self.private))
        self.assertNotIn(SECRET, failure.exception.stdout + failure.exception.stderr)

    def test_real_metadata_check_is_read_only_and_uses_frozen_calendars(self):
        before = snapshot(self.private)
        state = self.checked()
        self.assertEqual(state, self.states())
        self.assertEqual(before, snapshot(self.private))
        for row in self.rows:
            split = read_json(Path(row["folder"]) / "development-split.json")
            self.assertEqual(split["metadata_only_expected_raw_bars"], len(self.sessions[row["panel"]]))
            self.assertNotIn("test", split["raw_bounds"])
        trace = json.loads(self.private_results[-1].stderr.splitlines()[-1])
        self.assertEqual(trace, dict(network_calls=[], partition_loads=[]))

    def test_cancel_keeps_every_byte_and_never_calls_data(self):
        before = snapshot(self.private)
        result, output, prompt = self.invoke((*ACTOR, "NO"))
        self.assertEqual(result, self.states())
        self.assertEqual(prompt.call_count, 4)
        self.assertEqual(before, snapshot(self.private))
        self.assertIn("Cancelado", output)
        self.assertNotIn(ACTOR[0], output)
        self.assertNotIn(ACTOR[1], output)
        self.assertNotIn(SECRET, output)

    def test_missing_interactive_identity_rejects_before_any_write(self):
        for answers in (("", ACTOR[1], ACTOR[2]), (ACTOR[0], "", ACTOR[2]), (ACTOR[0], ACTOR[1], "")):
            with self.subTest(answers=answers):
                before = snapshot(self.private)
                with self.assertRaisesRegex(RuntimeError, "nombre, contacto y función"):
                    self.invoke(answers)
                self.assertEqual(before, snapshot(self.private))

    def test_rejects_wrong_source_commit_and_settings_source(self):
        for key in ("SOURCE_COMMIT", "settings_commit", "settings_root"):
            with self.subTest(key=key):
                namespace = dict(self.namespace, SETTINGS=dict(self.namespace["SETTINGS"]))
                if key == "SOURCE_COMMIT":
                    namespace[key] = "0" * 40
                elif key == "settings_commit":
                    namespace["SETTINGS"]["source_commit"] = "0" * 40
                else:
                    namespace["SETTINGS"]["source_root"] = "/tmp/another-source"
                before = snapshot(self.private)
                with mock.patch("builtins.input", side_effect=AssertionError("Should fail before prompting")), \
                        self.assertRaises(RuntimeError):
                    helper.prepare_development(namespace)
                self.assertEqual(before, snapshot(self.private))

    def test_rejects_mutated_source_bytes_and_manifest(self):
        copied = Path(self.temporary.name) / "source-copy"
        shutil.copytree(SOURCE, copied, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        namespace = dict(self.namespace, SOURCE=copied,
                         SETTINGS=dict(source_commit=helper.SOURCE_COMMIT, source_root=str(copied)))
        before = snapshot(self.private)
        candidate = copied / "engine/data.py"
        original = candidate.read_bytes()
        candidate.write_bytes(original + b"\n# software-test mutation\n")
        with self.assertRaisesRegex(RuntimeError, "Archivo fuente diferente"):
            helper.prepare_development(namespace)
        candidate.write_bytes(original)
        (copied / "SOURCE_MANIFEST.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(RuntimeError, "Manifest de fuente diferente"):
            helper.prepare_development(namespace)
        self.assertEqual(before, snapshot(self.private))

    def test_rejects_wrong_anchor_before_prompt(self):
        original_read = helper._read
        def changed_anchor(path):
            value = original_read(path)
            return dict(value, source_commit="0" * 40) if Path(path).name == "PUBLIC_SOURCE_ANCHOR.json" else value
        before = snapshot(self.private)
        with mock.patch.object(helper, "_read", side_effect=changed_anchor), \
                self.assertRaisesRegex(RuntimeError, "Ancla de fuente diferente"):
            self.invoke()
        self.assertEqual(before, snapshot(self.private))

    def test_rejects_changed_notebook_proposal_map_and_rule(self):
        before = snapshot(self.private)
        self.namespace["definitions"][0]["proposal"]["instrument"] = "OTHER"
        with self.assertRaisesRegex(RuntimeError, "Propuesta del cuaderno diferente"):
            self.invoke()
        self.namespace["definitions"] = copy.deepcopy(self.rows)
        map_path = self.namespace["DEVELOPMENT_MAP"]
        mapping = read_json(map_path)
        write_json(map_path, dict(mapping, F00="/tmp/other"))
        with self.assertRaisesRegex(RuntimeError, "Mapa de paneles diferente"):
            self.invoke()
        write_json(map_path, mapping)
        write_json(self.private / "development-split-rule.json", dict(RULE, embargo_bars=1))
        with self.assertRaisesRegex(RuntimeError, "Regla de partición diferente"):
            self.invoke()
        write_json(self.private / "development-split-rule.json", RULE)
        self.assertEqual({p: data[0] for p, data in before.items()},
                         {p: data[0] for p, data in snapshot(self.private).items()})

    def test_rejects_partial_raw_and_partial_admission_without_overwriting(self):
        row = self.rows[0]
        for key, name, message in (("raw", "interrupted.zip", "partial raw acquisition"),
                                   ("panel_dir", "fit.npz", "partial admission")):
            with self.subTest(kind=key):
                directory = Path(row[key])
                directory.mkdir(parents=True)
                (directory / name).write_bytes(b"synthetic interrupted bytes")
                self.assert_rejected_without_changes(message)
                shutil.rmtree(directory)

    def test_attempt_log_alone_is_retriable_and_preserved(self):
        directory = Path(self.rows[0]["raw"])
        directory.mkdir(parents=True)
        (directory / "acquisition_attempts.jsonl").write_text('{"synthetic":"failed"}\n')
        before = snapshot(self.private)
        self.assertEqual(self.checked(), self.states())
        self.assertEqual(before, snapshot(self.private))

    def test_rejects_modified_acquisition_and_split(self):
        row = self.rows[1]
        for name, key, value, message in (
            ("acquisition.json", "instrument", "OTHER", "acquisition differs"),
            ("development-split.json", "embargo_bars", 1, "split differs"),
            ("development-split.json", "raw_bounds", {"test": [0, 20]}, "split differs"),
        ):
            with self.subTest(name=name, key=key):
                path = Path(row["folder"]) / name
                original = path.read_bytes()
                write_json(path, dict(read_json(path), **{key: value}))
                self.assert_rejected_without_changes(message)
                path.write_bytes(original)

    def test_rejects_owner_scope_source_and_final_authorization_changes(self):
        cases = dict(panel_ids=["F00"], reserve_mode="post_protocol", dataset_scope={"instrument": "OTHER"},
            source_url="https://example.invalid/other", source_license_evidence_path="/tmp/other",
            source_license_evidence_sha256="0"*64, authorize_test_preparation=True)
        for key, value in cases.items():
            with self.subTest(key=key):
                self.complete_owner(**{key: value})
                self.assert_rejected_without_changes("final authorization forbidden" if key == "authorize_test_preparation" else "owner scope/source differs")

    def test_rejects_each_missing_owner_identity_field(self):
        for key in ("actor_name", "actor_contact", "actor_role", "statement_utc", "evidence_origin", "observation_cutoff_utc"):
            with self.subTest(field=key):
                self.complete_owner(**{key: None})
                self.assert_rejected_without_changes("incomplete owner field " + key)

    def test_rejects_owner_timestamp_exposure_signature_and_permission_errors(self):
        cases = (
            ({"statement_utc": "2026-11-22T00:00:00"}, "invalid owner timestamps"),
            ({"observation_cutoff_utc": "2026-11-23T00:00:00+00:00"}, "invalid owner timestamps"),
            ({"observation_cutoff_utc": "1995-01-01T00:00:00+00:00"}, "cutoff before development"),
            ({"acknowledge_prior_D00_D03_exposed": False}, "missing exposure designation"),
            ({"all_data_declared_exposed_development": False}, "missing exposure designation"),
            ({"AI_completed_signature": True}, "missing authentic source acceptance"),
            ({"source_terms_accepted_by_real_actor": False}, "missing authentic source acceptance"),
            ({"allow_derived_publication": None}, "explicit owner publication permissions"),
            ({"allow_raw_redistribution": "no"}, "explicit owner publication permissions"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                self.complete_owner(**changes)
                self.assert_rejected_without_changes(message)

    def test_real_owner_receipt_and_primary_rights_are_accepted_without_logging_actor(self):
        row = self.rows[1]
        self.complete_owner(row)
        self.add_evidence(row)
        before = snapshot(self.private)
        state = self.checked([row])["F01"]
        self.assertEqual(state, dict(admitted=False, raw_complete=False, blank_owner=False,
                                     owner_receipt=True, rights_present=True))
        self.assertEqual(before, snapshot(self.private))
        output = self.private_results[-1].stdout + self.private_results[-1].stderr
        self.assertNotIn(ACTOR[0], output)
        self.assertNotIn(ACTOR[1], output)
        self.assertNotIn(SECRET, output)

    def test_rejects_owner_edits_after_receipt_and_receipt_tampering(self):
        row = self.rows[1]
        self.complete_owner(row)
        self.add_evidence(row)
        folder = Path(row["folder"])
        receipt_path = folder / "exposure-development.json"
        receipt_original = receipt_path.read_bytes()
        owner_path = folder / "owner-development.json"
        owner_original = owner_path.read_bytes()
        write_json(owner_path, dict(read_json(owner_path), actor_role="changed role"))
        self.assert_rejected_without_changes("Evidence bytes/hash mismatch", [row])
        owner_path.write_bytes(owner_original)
        write_json(receipt_path, dict(read_json(receipt_path), form_sha256="0"*64))
        self.assert_rejected_without_changes("owner changed after receipt", [row])
        receipt_path.write_bytes(receipt_original)
        write_json(receipt_path, dict(read_json(receipt_path), final_reserve_claim=True))
        self.assert_rejected_without_changes("no final reserve permitted", [row])

    def test_rejects_rights_different_actor_primary_source_and_excess_permissions(self):
        for panel, changes, message in (
            ("F01", {"actor_contact": "other@example.invalid"}, "rights actor differs"),
            ("F01", {"source_url": "https://example.invalid/other"}, "rights are not bound"),
            ("F01", {"license": "unrelated license"}, "rights are not bound"),
            ("F01", {"allow_raw_redistribution": True}, "permission exceeds"),
            ("F00", {"allow_raw_redistribution": True}, "permission exceeds"),
        ):
            with self.subTest(panel=panel, changes=changes):
                row = next(r for r in self.rows if r["panel"] == panel)
                self.complete_owner(row)
                self.add_evidence(row)
                rights_path = Path(row["folder"]) / "rights.json"
                write_json(rights_path, dict(read_json(rights_path), **changes))
                self.assert_rejected_without_changes(message, [row])
                (Path(row["folder"]) / "exposure-development.json").unlink()
                rights_path.unlink()

    def test_consent_writes_forms_and_real_receipts_before_development_commands(self):
        initial = self.states()
        evidence = self.states(blank_owner=False, owner_receipt=True, rights_present=True)
        final = self.states(admitted=True, raw_complete=True, blank_owner=False,
                            owner_receipt=True, rights_present=True)
        blank_bytes = {r["panel"]: (Path(r["folder"]) / "owner-development.json").read_bytes() for r in self.rows}
        def run_data(args, provider=None):
            self.calls.append((list(args), provider))
            if args[0] == "record-owner-evidence":
                options = dict(zip(args[1::2], args[2::2]))
                self.run_engine("""
payload = json.load(sys.stdin)
data.record_owner_evidence(payload['--form'],payload['--evidence'],payload['--output'])
""", options)
            else:
                self.assertIn(args[0], ("collect-development", "prepare-development"))
                for row in self.rows:
                    self.assertTrue((Path(row["folder"]) / "exposure-development.json").is_file())
                if args[0] == "collect-development":
                    self.assertIn("--enable-market-acquisition", args)
        self.namespace["run_data"] = run_data
        with mock.patch.object(helper, "_check", side_effect=[initial, evidence, final]):
            result, output, _ = self.invoke((*ACTOR, "ACEPTO DESARROLLO"))
        self.assertEqual(result, final)
        commands = [args[0] for args, _ in self.calls]
        self.assertEqual(commands[:4], ["record-owner-evidence"] * 4)
        self.assertEqual(commands[4:], ["collect-development", "prepare-development"] * 4)
        collectors = [provider for args, provider in self.calls if args[0] == "collect-development"]
        self.assertEqual(collectors, ["nasdaq_wiki"] * 3 + ["binance_vision"])
        for row in self.rows:
            folder = Path(row["folder"])
            self.assertEqual((folder / "owner-development.blank-template.json").read_bytes(), blank_bytes[row["panel"]])
            owner = read_json(folder / "owner-development.json")
            receipt = read_json(folder / "exposure-development.json")
            self.assertEqual(owner["actor_name"], ACTOR[0])
            self.assertEqual(owner["actor_contact"], ACTOR[1])
            self.assertFalse(owner["AI_completed_signature"])
            self.assertFalse(owner["authorize_test_preparation"])
            self.assertEqual(receipt["form"], owner)
            self.assertEqual(receipt["form_sha256"], identity(owner))
            self.assertFalse(receipt["cryptographic_signature_verified"])
            self.assertFalse(receipt["final_reserve_claim"])
        self.assertIn("D00–D03", output)
        for value in (*ACTOR[:2], SECRET):
            self.assertNotIn(value, output)
        # Verify all actual forms, receipts and rights with the unmodified checker.
        self.assertEqual(self.checked(), evidence)
        completed_bytes = snapshot(self.private)
        prior_commands = list(self.calls)
        with mock.patch.object(helper, "_check", return_value=final):
            rerun, _, prompt = self.invoke()
        self.assertEqual(rerun, final)
        prompt.assert_not_called()
        self.assertEqual(self.calls, prior_commands)
        self.assertEqual(snapshot(self.private), completed_bytes)

    def test_filled_owner_missing_rights_requires_matching_actor_before_any_write(self):
        for row in self.rows:
            self.complete_owner(row)
        before = snapshot(self.private)
        with self.assertRaisesRegex(RuntimeError, "no coincide"):
            self.invoke(("OTHER SOFTWARE FIXTURE", ACTOR[1], ACTOR[2]))
        self.assertEqual(before, snapshot(self.private))

    def test_all_admitted_is_idempotent_no_prompt_no_network_and_no_rewrites(self):
        final = self.states(admitted=True, raw_complete=True, blank_owner=False,
                            owner_receipt=True, rights_present=True)
        # The real single-panel admission path is exercised separately below.
        before = snapshot(self.private)
        with mock.patch.object(helper, "_check", return_value=final) as check:
            result, output, prompt = self.invoke()
        check.assert_called_once()
        self.assertEqual(result, final)
        prompt.assert_not_called()
        self.assertEqual(before, snapshot(self.private))
        self.assertIn("ya están verificados", output)

    def test_provider_failure_keeps_local_bytes_and_hides_credential_in_traceback(self):
        row = self.rows[1]
        raw = Path(row["raw"])
        raw.mkdir(parents=True)
        (raw / "acquisition_attempts.jsonl").write_text('{"synthetic":"prior attempt"}\n')
        before = snapshot(self.private)
        def provider_failure(*args, **kwargs):
            raise RuntimeError("https://data.nasdaq.com/private?api_key=" + SECRET)
        with self.assertRaisesRegex(RuntimeError, "descarga detenida") as failure:
            helper._progress_call({"run_data": provider_failure}, ["collect-development"], row,
                                  provider="nasdaq_wiki")
        printed = "".join(traceback.format_exception(failure.exception))
        self.assertNotIn(SECRET, printed)
        self.assertNotIn("https://data.nasdaq.com/private", printed)
        self.assertIn("acquisition_attempts.jsonl", printed)
        self.assertEqual(before, snapshot(self.private))

    def test_real_generated_equity_admission_checks_only_development_and_preserves_bytes(self):
        row = self.rows[1]  # Full frozen 1996..2013 XNYS calendar, generated prices.
        self.complete_owner(row)
        self.add_evidence(row)
        self.run_engine(r'''
import math
payload = json.load(sys.stdin)
row = payload['row']
folder, raw, target = (Path(row[k]) for k in ('folder','raw','panel_dir'))
p = row['proposal']
raw.mkdir(parents=True)
dates = data.expected_sessions(p['proposed_development_start'],p['proposed_development_end'],p['calendar'])
values = []
for index, date in enumerate(dates):
    close = 100 + .03*index + 2*math.sin(index/7)
    values.append([p['instrument'],date,close-.1,close+1,close-1,close,1000+index,1.])
columns = ['ticker','date','open','high','low','close','volume','split_ratio']
raw_bytes = json.dumps({'datatable':{'columns':[{'name':x} for x in columns], 'data':values}, 'meta':{'synthetic_software_fixture':True}}).encode()
name = 'synthetic-wiki-software-fixture.json'
(raw/name).write_bytes(raw_bytes)
manifest = dict(provider=p['provider'],instrument=p['instrument'],scope='development_only',
    rights=json.loads((folder/'rights.json').read_text()),exposure=json.loads((folder/'exposure-development.json').read_text()),
    acquired_utc='2026-11-22T00:00:00+00:00',
    files=[dict(path=name,sha256=data.sha256_bytes(raw_bytes),url='https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES.json?ticker=IBM&date.gte=1996-01-02&date.lte=2013-03-01&api_key=REDACTED')])
manifest['manifest_sha256'] = data.identity(manifest)
manifest_path = raw/'raw_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
admission = data.prepare_development(manifest_path,json.loads((folder/'development-split.json').read_text()),target,
    rights=manifest['rights'],exposure=manifest['exposure'])
assert set(admission['partition_paths']) == {'fit','selection','planning','calibration'}
# A forbidden sentinel exists, so an accidental test read cannot pass silently.
real_open(target/'test.npz','wb').write(b'FORBIDDEN FINAL-PARTITION SOFTWARE SENTINEL')
print(json.dumps({'synthetic_raw_bars':len(dates),'partition_counts':admission['partition_counts']}))
''', dict(row=row))
        before = snapshot(self.private)
        states = self.checked([row])
        self.assertEqual(states["F01"], dict(admitted=True, raw_complete=True, blank_owner=False,
                                             owner_receipt=True, rights_present=True))
        self.assertEqual(snapshot(self.private), before)
        trace = json.loads(self.private_results[-1].stderr.splitlines()[-1])
        self.assertEqual(trace["partition_loads"], list(helper.PARTS))
        self.assertEqual(trace["network_calls"], [])
        self.checked([row])
        self.assertEqual(snapshot(self.private), before)
        payload = Path(row["panel_dir"]) / "fit.npz"
        payload.write_bytes(payload.read_bytes() + b"tampered")
        self.assert_rejected_without_changes("Partition byte SHA256 mismatch", [row])


if __name__ == "__main__":
    unittest.main(verbosity=2)
