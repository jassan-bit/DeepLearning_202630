"""Independent bootstrap checks: fake HTTP/processes; no market/network access."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import bootstrap_colab as bootstrap


COMMIT = "a"*40


class _PrivateInput(io.StringIO):
    def close(self):
        self.submitted=self.getvalue()
        super().close()


class _ProcessFixture:
    def __init__(self, output, returncode=0):
        self.stdin=_PrivateInput()
        self.stdout=io.StringIO(output)
        self.returncode=returncode
        self.waited=False

    def wait(self,timeout=None):
        self.waited=True
        return self.returncode

    def poll(self):
        return self.returncode if self.waited else None


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = {"engine/cli.py": b'"""SOFTWARE FIXTURE; NOT EXECUTED"""\n',
                      "requirements-cpu.lock.txt": b"fixture-package==1.0\n"}
        self.manifest = {"algorithm": "sha256", "files": {p: bootstrap.sha(v) for p, v in self.files.items()},
                         "sizes": {p: len(v) for p, v in self.files.items()}}
        self.raw = json.dumps(self.manifest, sort_keys=True).encode()
        self.base = f"https://raw.githubusercontent.com/{bootstrap.REPOSITORY}/{COMMIT}/{bootstrap.PREFIX}/"
        self.urls = []

    def tearDown(self):
        self.tmp.cleanup()

    def http(self, request, timeout):
        self.assertEqual(timeout, 90)
        self.assertTrue(request.full_url.startswith(self.base), request.full_url)
        self.urls.append(request.full_url)
        relative = request.full_url[len(self.base):]
        return io.BytesIO(self.raw if relative == "SOURCE_MANIFEST.json" else self.files[relative])

    def acquire(self):
        with patch.object(bootstrap.urllib.request, "urlopen", side_effect=self.http):
            return bootstrap.acquire(COMMIT, bootstrap.sha(self.raw), self.root)

    def test_pinned_acquisition_hash_sizes_exact_urls_and_commit_anchor(self):
        root = self.acquire()
        self.assertEqual(root, self.root/("source-"+COMMIT))
        for name, expected in self.files.items():
            self.assertEqual((root/name).read_bytes(), expected)
        self.assertEqual((root/"SOURCE_MANIFEST.json").read_bytes(), self.raw)
        anchor = json.loads((root/"PUBLIC_SOURCE_ANCHOR.json").read_text())
        self.assertEqual(anchor["source_commit"], COMMIT)
        self.assertEqual(anchor["manifest_sha256"], bootstrap.sha(self.raw))
        self.assertIn("not retrospective preregistration", anchor["scope"])
        self.assertEqual(set(self.urls), {self.base+name for name in ["SOURCE_MANIFEST.json", *self.files]})

    def test_exact_cache_reuses_only_verified_payload_bytes(self):
        root = self.acquire()
        original = {p: (root/p).stat().st_mtime_ns for p in [*self.files, "SOURCE_MANIFEST.json", "PUBLIC_SOURCE_ANCHOR.json"]}
        self.urls.clear()
        self.assertEqual(self.acquire(), root)
        self.assertEqual(self.urls, [self.base+"SOURCE_MANIFEST.json"])
        self.assertEqual(original, {p: (root/p).stat().st_mtime_ns for p in original})

    def test_corrupt_cache_refused_preserved_and_not_replaced_from_network(self):
        root = self.acquire()
        target = root/"engine/cli.py"
        target.write_bytes(b"corrupted cache evidence")
        self.urls.clear()
        with self.assertRaisesRegex(ValueError, "Source size/hash mismatch"):
            self.acquire()
        self.assertEqual(target.read_bytes(), b"corrupted cache evidence")
        self.assertEqual(self.urls, [self.base+"SOURCE_MANIFEST.json"])

    def test_wrong_manifest_hash_stops_before_payload_download_or_cache_creation(self):
        with patch.object(bootstrap.urllib.request, "urlopen", side_effect=self.http):
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                bootstrap.acquire(COMMIT, "0"*64, self.root)
        self.assertEqual(self.urls, [self.base+"SOURCE_MANIFEST.json"])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_payload_hash_and_bad_size_both_refused(self):
        for defect in ("sha", "size"):
            with self.subTest(defect=defect):
                manifest = json.loads(self.raw)
                if defect == "sha":
                    manifest["files"]["engine/cli.py"] = "0"*64
                else:
                    manifest["sizes"]["engine/cli.py"] += 1
                raw = json.dumps(manifest).encode()
                with patch.object(bootstrap, "download", side_effect=[raw, self.files["engine/cli.py"]]):
                    with self.assertRaisesRegex(ValueError, "Source size/hash mismatch"):
                        bootstrap.acquire(COMMIT, bootstrap.sha(raw), self.root/defect)
                self.assertFalse((self.root/defect/("source-"+COMMIT)/"engine/cli.py").exists())

    def test_traversal_absolute_noncanonical_and_hidden_paths_refused_before_file_fetch(self):
        for path in ("../outside.py", "/outside.py", "engine/../../outside.py", "engine//cli.py", "engine/./cli.py", ".hidden.py", "."):
            with self.subTest(path=path):
                raw = json.dumps({"algorithm": "sha256", "files": {path: bootstrap.sha(b"x")}, "sizes": {path: 1}}).encode()
                with patch.object(bootstrap, "download", return_value=raw) as network:
                    with self.assertRaisesRegex(ValueError, "Unsafe source path"):
                        bootstrap.acquire(COMMIT, bootstrap.sha(raw), self.root)
                self.assertEqual(network.call_count, 1)

    def test_symlink_cache_destination_cannot_escape(self):
        root = self.root/("source-"+COMMIT)
        root.mkdir()
        outside = self.root/"outside"; outside.mkdir()
        (root/"engine").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink cached source"):
            self.acquire()
        self.assertFalse((outside/"cli.py").exists())

    def test_changed_commit_anchor_is_refused_without_rewriting_evidence(self):
        root = self.acquire()
        anchor = root/"PUBLIC_SOURCE_ANCHOR.json"
        anchor.write_text('{"source_commit":"wrong commit"}\n')
        with self.assertRaisesRegex(ValueError, "Cached source changed"):
            self.acquire()
        self.assertEqual(json.loads(anchor.read_text())["source_commit"], "wrong commit")

    def test_mutable_commit_or_missing_pin_rejected_before_http(self):
        with patch.object(bootstrap.urllib.request, "urlopen", side_effect=AssertionError("network accessed")):
            for commit in ("main", "HEAD", "a"*39, "A"*40, None):
                with self.subTest(commit=commit), self.assertRaisesRegex(ValueError, "Immutable source commit"):
                    bootstrap.acquire(commit, bootstrap.sha(self.raw), self.root)
            for digest in (None, "", "a"*63, "A"*64):
                with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "manifest hash"):
                    bootstrap.acquire(COMMIT, digest, self.root)

    def test_download_byte_cap_and_transient_retry_remain_bounded(self):
        with patch.object(bootstrap.urllib.request, "urlopen", return_value=io.BytesIO(b"12345")):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                bootstrap.download("https://example.invalid/fixture", 4)
        failure = urllib.error.HTTPError("https://example.invalid", 429, "fixture", {"Retry-After": "1"}, None)
        with patch.object(bootstrap.urllib.request, "urlopen", side_effect=[failure, io.BytesIO(b"ok")]) as network, \
             patch.object(bootstrap.time, "sleep") as sleep:
            self.assertEqual(bootstrap.download("https://example.invalid/fixture", 4), b"ok")
        self.assertEqual(network.call_count, 2)
        sleep.assert_called_once_with(1.)

    def test_cpu_env_removes_real_collector_keys_and_storage_credentials(self):
        keys = ("PYTHONPATH", "GITHUB_TOKEN", "GH_TOKEN", "E3_V5_BACKUP_KEY", "E3_R3_BACKUP_KEY",
                "E3_R3_GITHUB_TOKEN", "E3_R3_BACKUP_SECRET", "ALPHAVANTAGE_API_KEY", "NASDAQ_API_KEY", "NASDAQ_DATA_LINK_API_KEY")
        original = {k: "PRIVATE_SOFTWARE_FIXTURE" for k in keys}
        with patch.dict(os.environ, original):
            env = bootstrap.cpu_env()
            for key in keys:
                self.assertTrue(key not in env, f"Credential/environment variable not stripped: {key}")
            self.assertEqual(os.environ["NASDAQ_DATA_LINK_API_KEY"], "PRIVATE_SOFTWARE_FIXTURE")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_install_existing_python_verifies_lock_and_pip_check_without_install(self):
        (self.root/"requirements-cpu.lock.txt").write_bytes(self.files["requirements-cpu.lock.txt"])
        python = self.root/"existing-python"
        with patch.object(bootstrap.subprocess, "run") as runner, \
             patch.object(bootstrap.venv, "EnvBuilder", side_effect=AssertionError("Must use supplied interpreter")):
            self.assertEqual(bootstrap.install(self.root, self.root/"cache", existing_python=python), python)
        self.assertEqual(runner.call_count, 2)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(commands[0][:2], [str(python), "-c"])
        self.assertEqual(commands[1][-1], "check")
        self.assertFalse(any("install" in cmd for cmd in commands))
        self.assertTrue(all(call.kwargs["check"] for call in runner.call_args_list))

    def test_new_cpu_install_uses_isolated_lock_specific_environment(self):
        lock = self.root/"requirements-cpu.lock.txt"; lock.write_bytes(self.files["requirements-cpu.lock.txt"])
        with patch.object(bootstrap.subprocess, "run") as runner, patch.object(bootstrap.venv, "EnvBuilder") as builder:
            python = bootstrap.install(self.root, self.root/"cache")
        self.assertIn("venv-cpu-"+bootstrap.sha(lock.read_bytes())[:16], str(python))
        builder.assert_called_once_with(with_pip=False, system_site_packages=False, symlinks=True)
        builder.return_value.create.assert_called_once_with(python.parent.parent)
        install = runner.call_args_list[0]
        self.assertEqual(install.args[0][-3:], ["--disable-pip-version-check", "-r", str(lock)])
        self.assertEqual(install.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(runner.call_count, 3)

    def test_command_credentials_use_stdin_only_not_argv_env_or_shell(self):
        credentials = {"token": "SOFTWARE_FIXTURE_TOKEN/$", "backup_secret": "SOFTWARE_FIXTURE_SECRET"}
        settings = {"python": "/fixture/python", "source_root": str(self.root)}
        process = _ProcessFixture('{"status":"fixture"}\n')
        output = io.StringIO()
        with patch.object(bootstrap.subprocess, "Popen", return_value=process) as runner, redirect_stdout(output):
            completed=bootstrap.command(settings, ["status", "--case", "fixture"], credentials=credentials)
        call = runner.call_args
        self.assertEqual(json.loads(process.stdin.submitted), credentials)
        self.assertEqual(call.args[0][1],"-u")
        self.assertEqual(call.kwargs["stdin"],subprocess.PIPE)
        self.assertEqual(call.kwargs["stdout"],subprocess.PIPE)
        self.assertEqual(call.kwargs["stderr"],subprocess.STDOUT)
        self.assertTrue(call.kwargs["text"])
        self.assertEqual(completed.returncode,0)
        self.assertEqual(completed.stdout,'{"status":"fixture"}\n')
        self.assertNotIn("shell", call.kwargs)
        for secret in credentials.values():
            self.assertNotIn(secret, repr(call.args))
            self.assertNotIn(secret, repr(call.kwargs["env"]))
            self.assertNotIn(secret, output.getvalue())

    def test_blocked_command_is_explicit_and_never_promoted_to_success(self):
        settings = {"python": "/fixture/python", "source_root": str(self.root)}
        for allow in (False, True):
            process = _ProcessFixture('{"status":"blocked_or_stopped"}\n',returncode=2)
            with patch.object(bootstrap.subprocess, "Popen", return_value=process), redirect_stdout(io.StringIO()):
                if allow:
                    self.assertEqual(bootstrap.command(settings, ["run"], allow_blocked=True).returncode, 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, "stage stopped"):
                        bootstrap.command(settings, ["run"])

    def test_progress_is_sanitized_before_display_and_before_next_line(self):
        credentials={"token":"PRIVATE_FIXTURE_TOKEN/$", "backup_secret":"PRIVATE FIXTURE BACKUP/$"}
        settings={"python":"/fixture/python","source_root":str(self.root)}
        output=io.StringIO()
        first="progress "+credentials["token"]+" "+bootstrap.urllib.parse.quote(credentials["backup_secret"],safe="")+"\n"
        process=_ProcessFixture("")

        def lines():
            self.assertTrue(process.stdin.closed)
            yield first
            # The next child line must not be requested before the progress
            # was actually displayed; this detects capture-until-exit behavior.
            self.assertIn("progress [REDACTED] [REDACTED]",output.getvalue())
            self.assertFalse(process.waited)
            yield '{"status":"finished"}\n'

        process.stdout=lines()
        with patch.object(bootstrap.subprocess,"Popen",return_value=process),redirect_stdout(output):
            completed=bootstrap.command(settings,["run"],credentials=credentials)
        self.assertEqual(json.loads(completed.stdout.splitlines()[-1]),{"status":"finished"})
        self.assertNotIn(credentials["token"],completed.stdout)
        self.assertNotIn(credentials["backup_secret"],output.getvalue())
        self.assertNotIn(bootstrap.urllib.parse.quote(credentials["backup_secret"],safe=""),completed.stdout)

    def test_real_private_pipe_roundtrip_merges_stderr_without_retaining_secrets(self):
        package=self.root/"engine";package.mkdir()
        (package/"__init__.py").write_text("")
        (package/"cli.py").write_text(
            "import json,sys,urllib.parse\n"
            "auth=json.load(sys.stdin)\n"
            "print('progress '+auth['token'],flush=True)\n"
            "print('error '+urllib.parse.quote_plus(auth['backup_secret'],safe=''),file=sys.stderr,flush=True)\n"
            "print(json.dumps({'status':'fixture_complete'}),flush=True)\n")
        credentials={"token":"PRIVATE_FIXTURE_TOKEN/$","backup_secret":"PRIVATE FIXTURE BACKUP/$"}
        output=io.StringIO()
        with redirect_stdout(output):
            completed=bootstrap.command({"python":os.sys.executable,"source_root":str(self.root)},["check"],credentials=credentials)
        self.assertEqual(completed.returncode,0)
        self.assertEqual(json.loads(completed.stdout.splitlines()[-1]),{"status":"fixture_complete"})
        self.assertIn("progress [REDACTED]",completed.stdout)
        self.assertIn("error [REDACTED]",completed.stdout)
        self.assertEqual(completed.stdout,output.getvalue())
        for secret in credentials.values():
            self.assertNotIn(secret,completed.stdout)
            self.assertNotIn(bootstrap.urllib.parse.quote_plus(secret,safe=""),completed.stdout)

    def test_child_early_rejection_preserves_structured_output_after_closed_private_pipe(self):
        package=self.root/"engine";package.mkdir()
        (package/"__init__.py").write_text("")
        (package/"cli.py").write_text("import json\nprint(json.dumps({'status':'blocked_before_private_input'}),flush=True)\nraise SystemExit(2)\n")
        output=io.StringIO()
        credentials={"token":"PRIVATE_FIXTURE_TOKEN", "backup_secret":"PRIVATE_BACKUP_"*20000}
        with redirect_stdout(output):
            result=bootstrap.command({"python":os.sys.executable,"source_root":str(self.root)},["check"],
                                     credentials=credentials,allow_blocked=True)
        self.assertEqual(result.returncode,2)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]),{"status":"blocked_before_private_input"})
        self.assertNotIn("PRIVATE_BACKUP",output.getvalue())

    def test_setup_defaults_only_software_tests_and_check_never_market_stage(self):
        output = io.StringIO()
        with patch.object(bootstrap, "acquire", return_value=self.root) as acquire, \
             patch.object(bootstrap, "install", return_value=Path("/fixture/python")), \
             patch.object(bootstrap.subprocess, "run") as runner, redirect_stdout(output):
            settings = bootstrap.setup(COMMIT, bootstrap.sha(self.raw), self.root)
        acquire.assert_called_once_with(COMMIT, bootstrap.sha(self.raw), self.root)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(commands[0][1:], ["-m", "pytest", "-q", "tests"])
        self.assertEqual(commands[1][1:], ["-m", "engine.cli", "--source-root", str(self.root), "check"])
        self.assertEqual(len(commands), 2)
        self.assertTrue(settings["software_tests_run"])
        self.assertEqual(settings["new_market_fits"], 0)
        self.assertFalse(settings["reserved_test_opened"])
        self.assertNotIn("--activate", repr(commands))
        self.assertNotIn("--enable-market-acquisition", repr(commands))

    def test_bootstrap_import_does_not_acquire_install_run_or_train(self):
        spec = importlib.util.spec_from_file_location("independent_bootstrap_import_fixture", bootstrap.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.object(bootstrap.urllib.request, "urlopen", side_effect=AssertionError("Network on import")), \
             patch.object(bootstrap.subprocess, "run", side_effect=AssertionError("Process on import")), \
             patch.object(bootstrap.venv, "EnvBuilder", side_effect=AssertionError("Install on import")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.setup))


if __name__ == "__main__":
    unittest.main()
