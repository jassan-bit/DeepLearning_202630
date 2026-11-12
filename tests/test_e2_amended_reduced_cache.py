"""Reduced-cache safety checks; only synthetic JSON/bytes, stdlib runtime."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_e2_reduced_cache as cache


def fixture():
    trial = {"trial_id": "FCN-fixed:hpo-005", "recipe_id": "FCN-fixed", "learning_rate": .001,
             "weight_decay": 0, "batch_size": 16, "optimization_seeds": [271201, 271202]}
    slot = {"configuration_id": "C01", "parent_trial_id": trial["trial_id"],
            "parent_trial_index": 5, "optimization_seed": 271201, "trial": trial}
    context = {"prepared_signature": "a" * 64, "training_signature": "b" * 64,
               "validation_signature": "c" * 64}
    hardware = {key: 1 for key in cache.RUNTIME_KEYS}
    hardware.update(device="cuda:0", device_name="Tesla T4", torch="2.8.0+cu126", torch_cuda="12.6",
                    dtype="float32", amp=False, tf32=False, deterministic=True, cuda_kernel_probe_passed=True)
    plan = {"source_commit": "d" * 40, "protocol_sha256": "e" * 64, "hardware_binding": hardware,
            "execution": {"dtype": "float32", "amp": False, "tf32": False, "deterministic": True,
                          "cpu_threads": 2, "interop_threads": 1, "device": "cuda:0"}}
    expected = {"model": "FCN-1D", "origin": "2024-01-01", "candidate_id": "hpo:" + trial["trial_id"],
                "payload_sha256": cache.candidate_sha("FCN-1D", trial), "seed": cache.SEED, "stage": "hpo"}
    job = {**expected, "candidate": cache.candidate_payload("FCN-1D", trial), **context,
           "source_commit": plan["source_commit"], "protocol_sha256": plan["protocol_sha256"],
           "hardware_sha256": cache.digest(hardware)}
    start = cache.date(2024, 1, 1) - cache.timedelta(days=90)
    dates = [(start + cache.timedelta(days=i)).isoformat() + "T00:00:00+00:00" for i in range(83)]
    outcome = {"status": "completed", "fatal": False, "prediction_dates": dates,
        "predictions_original": [.01] * 83, "resource_use": {
            "identity": {**expected, "prepared_signature": context["prepared_signature"]},
            "optimizer_steps_unknown": False, "partial_optimizer_update_may_have_occurred": False,
            "optimizer_steps": 20, "phase_records": {"supervised": {"best_epoch": 3, "epochs_completed": 5}}}}
    plan_sha = cache.digest(plan)
    record = {"plan_sha256": plan_sha, "outcome": outcome, "outcome_sha256": cache.digest(outcome),
              "job_payload_sha256": cache.digest(job),
              "job_id": cache.digest({"plan_sha256": plan_sha, "job": job})}
    record["sha256"] = cache.digest(record)
    return record, plan, slot, context, hardware


def domains():
    values = {}
    for model in cache.NEURAL:
        values[model] = [{"configuration_id": "C%02d" % (i + 1), "parent_trial_index": parent,
            "parent_trial_id": model + str(parent), "trial": {"trial_id": model + str(parent)},
            "optimization_seed": cache.SEED, "execution_allowed": model != "TiDE"}
            for i, parent in enumerate(cache.SLOTS)]
    return values


class CacheIdentityTests(unittest.TestCase):
    def inspect(self, record, plan, slot, **kwargs):
        return cache.inspect_record(record, plan, model="FCN-1D", anchor="2024-01-01", slot=slot, **kwargs)

    def test_metadata_without_actual_context_does_not_approve_reuse(self):
        record, plan, slot, _, _ = fixture()
        result = self.inspect(record, plan, slot)
        self.assertFalse(result["reuse_approved"])
        self.assertEqual(result["state"], "requires_prepared_and_runtime_checks")
        self.assertFalse(result["weights_reusable"])
        self.assertFalse(result["old_local_winner_reused"])
        self.assertFalse(result["final_metrics_reused"])

    def test_verified_input_signatures_still_need_current_GPU_runtime(self):
        record, plan, slot, context, _ = fixture()
        result = self.inspect(record, plan, slot, context=context)
        self.assertEqual(result["state"], "reusable_after_runtime_check")
        self.assertFalse(result["reuse_approved"])
        self.assertTrue(result["checks"]["job_payload"])

    def test_exact_runtime_can_close_only_raw_validation_metadata_gate(self):
        record, plan, slot, context, hardware = fixture()
        attested = {"runtime": hardware, "fit_execution_policy_verified": True,
                    "fit_execution_policy": plan["execution"]}
        result = self.inspect(record, plan, slot, context=context, runtime_profile=attested)
        self.assertTrue(result["reuse_approved"])
        self.assertEqual(result["validation_prediction_count"], 83)
        self.assertFalse(result["weights_reusable"])
        changed = {**attested, "runtime": dict(hardware, torch="2.7.0+cu126")}
        result = self.inspect(record, plan, slot, context=context, runtime_profile=changed)
        self.assertFalse(result["reuse_approved"])
        self.assertEqual(result["state"], "runtime_mismatch_not_reusable")

    def test_matrix_probe_and_observed_default_flags_are_not_fit_policy(self):
        record, plan, slot, context, hardware = fixture()
        raw = dict(hardware, amp=True, tf32=True, deterministic=False)
        result = self.inspect(record, plan, slot, context=context, runtime_profile=raw)
        self.assertFalse(result["reuse_approved"])
        self.assertEqual(result["state"], "reusable_after_execution_policy_attestation")
        attested = {"runtime": raw, "fit_execution_policy_verified": True,
                    "fit_execution_policy": plan["execution"]}
        result = self.inspect(record, plan, slot, context=context, runtime_profile=attested)
        self.assertTrue(result["reuse_approved"])

    def test_altered_trial_data_signature_or_record_is_rejected(self):
        record, plan, slot, context, _ = fixture()
        other = deepcopy(slot)
        other["trial"]["learning_rate"] = .0003
        with self.assertRaisesRegex(ValueError, "candidate_identity"):
            self.inspect(record, plan, other)
        with self.assertRaisesRegex(ValueError, "prepared_signature"):
            self.inspect(record, plan, slot, context=dict(context, prepared_signature="f" * 64))
        with self.assertRaisesRegex(ValueError, "full_job_payload"):
            self.inspect(record, plan, slot, context=dict(context, validation_signature="f" * 64))
        changed = deepcopy(record)
        changed["outcome"]["predictions_original"][0] += .001
        with self.assertRaisesRegex(ValueError, "record_hash"):
            self.inspect(changed, plan, slot)

    def test_censored_calendar_and_unknown_attempt_are_not_success(self):
        record, plan, slot, _, _ = fixture()
        changed = deepcopy(record)
        changed["outcome"]["prediction_dates"].pop()
        changed["outcome_sha256"] = cache.digest(changed["outcome"])
        changed["sha256"] = cache.digest({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "calendar_changed"):
            self.inspect(changed, plan, slot)
        changed = deepcopy(record)
        changed["outcome"]["resource_use"]["partial_optimizer_update_may_have_occurred"] = True
        changed["outcome_sha256"] = cache.digest(changed["outcome"])
        changed["sha256"] = cache.digest({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "not_successfully_closed"):
            self.inspect(changed, plan, slot)


class DomainAndDataTests(unittest.TestCase):
    def test_six_shared_blind_slots_and_architecture_gate_are_required(self):
        value = domains()
        self.assertIs(cache.validate_domains(value), value)
        changed = deepcopy(value)
        changed["GRU32"][0]["parent_trial_index"] = 4
        with self.assertRaises(ValueError):
            cache.validate_domains(changed)
        changed = deepcopy(value)
        changed["TiDE"][0]["execution_allowed"] = True
        with self.assertRaisesRegex(ValueError, "architecture_gate"):
            cache.validate_domains(changed)
        changed = deepcopy(value)
        changed["MLP16"] = changed["MLP16"][:5]
        with self.assertRaisesRegex(ValueError, "six_slots"):
            cache.validate_domains(changed)

    def test_data_hash_verification_never_decodes_numerical_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = b"date,target\n2024-01-01,NOT_A_NUMBER\n"
            data = root / "development.csv"
            data.write_bytes(raw)
            expected = {data.name: cache.file_sha(data)}
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"sources": expected}))
            result = cache.verify_development_data(root, expected_sources=expected,
                protocol_path=protocol.name, protocol_sha=cache.file_sha(protocol))
            self.assertFalse(result["numerical_cells_decoded_by_hash_audit"])
            self.assertFalse(result["evaluation_608_dates_opened"])
            data.write_bytes(raw + b"changed")
            with self.assertRaisesRegex(ValueError, "derivative_hash"):
                cache.verify_development_data(root, expected_sources=expected,
                    protocol_path=protocol.name, protocol_sha=cache.file_sha(protocol))

    def test_data_protocol_mismatch_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "development.csv"
            data.write_bytes(b"fixture")
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"sources": {data.name: "a" * 64}}))
            with self.assertRaisesRegex(ValueError, "source_binding"):
                cache.verify_development_data(root, expected_sources={data.name: cache.file_sha(data)},
                    protocol_path=protocol.name, protocol_sha=cache.file_sha(protocol))
            linked = root / "linked.csv"
            linked.symlink_to(data)
            protocol.write_text(json.dumps({"sources": {linked.name: cache.file_sha(data)}}))
            with self.assertRaisesRegex(ValueError, "regular_in_root"):
                cache.verify_development_data(root, expected_sources={linked.name: cache.file_sha(data)},
                    protocol_path=protocol.name, protocol_sha=cache.file_sha(protocol))


if __name__ == "__main__":
    unittest.main()
