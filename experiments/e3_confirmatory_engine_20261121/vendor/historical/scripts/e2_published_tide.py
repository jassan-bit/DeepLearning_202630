"""Source-defined TiDE defaults with an explicit D7 OHLCV-only task adapter.

This is a semantic PyTorch port of Google Research's Apache-2.0 TiDE graph.
All source residual branches, including the r=0 covariate branch, are retained.
The source's zero-bias/ReLU initialization makes part of that empty-covariate
branch permanently inactive. That limitation is disclosed, not repaired by
changing initializers, adding future observations, or pruning the source body.
This prospective recipe does not reuse the historical NAS TiDE observations.
"""
# Derived from TiDE, Copyright 2026 The Google Research Authors, Apache-2.0.
# Source commit and license hashes are verified below before construction.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# A complete copy accompanies experiments/e2_tide_published_r0_20261112/LICENSE.TiDE.
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

RECIPE_ID = "TIDE-D7-PUBLISHED-R0-v2"
AUTHOR_COMMIT = "d3a4bd8f0eceddda553e47085033b537935b283f"
SOURCE_FILES = {
    "tide/models.py": "ad8fd8d498809deeed9c847a169cada42134cbd0351b03e750706132a5d151c7",
    "tide/train.py": "3edd19eed01f92ef50cadca73fbe8a793c701d34edb5ea376c6bcc55e92f0854",
    "tide/scripts/etth1.sh": "55e085f1513af3a4a0ef8f7244512ee78beba5ab09540192b3cd3bb37441981c",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
SOURCE_DEFAULTS = dict(hidden_dims=[256, 256], time_encoder_dims=[64, 4],
    decoder_output_dim=4, final_decoder_hidden=64, transform=False,
    layer_norm=False, dropout_rate=0.0, series_embedding_dim=16)


def verify_tide_sources(source_roots):
    if not isinstance(source_roots, dict) or "TiDE" not in source_roots:
        raise ValueError("An explicit hash-pinned TiDE source root is required")
    root = Path(source_roots["TiDE"]).resolve()
    for name, expected in SOURCE_FILES.items():
        p = root / name
        if not p.is_file() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("TiDE source or license differs: " + name)
    return {"repository": "google-research/google-research", "commit": AUTHOR_COMMIT,
            "files_sha256": deepcopy(SOURCE_FILES), "license": "Apache-2.0",
            "verification_scope": "Four declared files, not the complete repository"}


class SourceResidual(nn.Module):
    """MLPResidual in tide/models.py: Dense-ReLU, Dense + Dense skip."""
    def __init__(self, incoming, hidden, outgoing):
        super().__init__()
        self.lin_a = nn.Linear(incoming, hidden)
        self.lin_b = nn.Linear(hidden, outgoing)
        self.lin_res = nn.Linear(incoming, outgoing)
        for layer in (self.lin_a, self.lin_b, self.lin_res):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.lin_b(F.relu(self.lin_a(x))) + self.lin_res(x)


class SourceTiDEDefaultR0(nn.Module):
    """Complete source-default scalar-series graph, L7/H1/r0/num_ts5.

    The source accepts numeric/categorical covariates shared by the series.
    Here both covariate tensors have zero rows: no invented observed zeros,
    target history, future OHLCV, calendar values or new information is added.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SourceResidual(55, 256, 256)
        self.decoder = SourceResidual(256, 256, 4)
        self.linear = nn.Linear(7, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.time_encoder = SourceResidual(0, 64, 4)
        self.final_decoder = SourceResidual(8, 64, 1)
        self.ts_embs = nn.Embedding(5, 16)
        nn.init.uniform_(self.ts_embs.weight, -.05, .05)

    def forward(self, past_ts, tsidx):
        if past_ts.ndim != 2 or past_ts.shape[1] != 7 or tsidx.shape != (len(past_ts),):
            raise ValueError("Source graph requires [batch,7] and [batch] series IDs")
        batch = len(past_ts)
        # The transpose/flatten order follows the source's fdim x L/H ordering.
        encoded_past = self.time_encoder(past_ts.new_empty((7, 0))).transpose(0, 1)
        encoded_future = self.time_encoder(past_ts.new_empty((1, 0))).transpose(0, 1)
        enc_past = encoded_past.unsqueeze(0).expand(batch, -1, -1).reshape(batch, -1)
        enc_fut = encoded_future.unsqueeze(0).expand(batch, -1, -1)
        enc_future = enc_fut.reshape(batch, -1)
        encoder_input = torch.cat((past_ts, enc_past, enc_future, self.ts_embs(tsidx)), dim=1)
        decoded = self.decoder(self.encoder(encoder_input)).reshape(batch, 4, 1)
        final_input = torch.cat((decoded, enc_fut), dim=1).transpose(1, 2)
        return self.final_decoder(final_input).squeeze(-1) + self.linear(past_ts)


class PublishedTiDED7(nn.Module):
    recipe_id = RECIPE_ID

    def __init__(self, *, source_roots):
        super().__init__()
        source = verify_tide_sources(source_roots)
        self.backbone = SourceTiDEDefaultR0()
        self.readout = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.architecture_metadata = {
            "model": "TiDE", "recipe_id": RECIPE_ID, "training_recipe_id": RECIPE_ID,
            "input_shape": [7, 5], "horizon": 1, "source": source,
            "source_default_preset": deepcopy(SOURCE_DEFAULTS),
            "source_selected_ETTh1_preset_reproduced": False,
            "source_forecast_graph_complete": True,
            "semantic_backend_port": "TensorFlow/Keras to PyTorch; see separate parity receipt",
            "dynamic_covariate_count": 0, "categorical_covariate_count": 0,
            "series_count": 5, "series_ids": "fixed OHLCV positions; not extra market inputs",
            "body_output_shape": [1, 5], "readout": "Linear(5,1) then Softplus",
            "output": "sigma7 / caller-owned positive training-only target scale",
            "no_target_history_added": True, "no_future_ohlcv_added": True,
            "no_calendar_covariates_added": True, "no_moment_features_added": True,
            "r0_inactive_branch_retained": True,
            "inactive_initialization_consequence": "At source zero-bias/ReLU initialization, time_encoder.lin_a bias and lin_b weight have zero task gradients; source graph is retained without pruning or reinitialization",
            "all_parameters_expected_to_have_nonzero_gradients": False,
            "structural_architecture_search": False, "nas": False,
            "restored_published_body": True, "weights_pretrained": False,
            "weights_reused": False, "paper_original_task_replication_claimed": False,
            "dropout": 0.0,
        }

    def channel_predictions(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or tuple(x.shape[1:]) != (7,5)
                or not len(x) or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
            raise ValueError("TiDE D7 requires finite nonempty floating [batch,7,5] OHLCV")
        past = x.transpose(1,2).reshape(-1,7)
        ids = torch.arange(5, device=x.device).repeat(len(x))
        return self.backbone(past, ids).reshape(len(x),5)

    def forward(self, x):
        return F.softplus(self.readout(self.channel_predictions(x))).squeeze(-1)


def registered_trials():
    """New identities; retain only original eighteen optimizer points and seeds."""
    try:
        from . import e2_training_models as historical
    except ImportError:
        import e2_training_models as historical
    keys = ("learning_rate", "weight_decay", "batch_size", "optimization_seeds", "warmup_fraction")
    rows=[]
    for old in historical.registered_trials("TiDE"):
        row = {k: deepcopy(old[k]) for k in keys if k in old}
        row.update(recipe_id=RECIPE_ID, trial_id=RECIPE_ID + ":" + old["trial_id"].rsplit(":",1)[-1], dropout=0.0)
        rows.append(row)
    return rows


def build_tide(*, seed, source_roots, trial=None):
    if type(seed) is not int or seed < 0:
        raise ValueError("An explicit nonnegative initialization seed is required")
    if trial is not None and trial not in registered_trials():
        raise ValueError("Use an exact optimizer point of the new source-defined TiDE recipe")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PublishedTiDED7(source_roots=source_roots)
    model.architecture_metadata.update(initialization_seed=seed,
        actual_parameter_count=sum(p.numel() for p in model.parameters()),
        training_axes_applied=False)
    if trial is not None:
        model.architecture_metadata.update(registered_trial_id=trial["trial_id"], training_hpo_record=deepcopy(trial))
    return model


def select_reduced_tide(ledger, configured_domain, *, final_origin="2025-01-01"):
    """Apply global three-fold criteria after all eighteen run slots are closed.

    A run row binds model_id, configuration_id, fold_origin, seed,
    parent_trial_id, configuration_sha256, status and run_receipt_sha256.
    Every row records the verified common validation_count83. Successful rows
    additionally need rmse_original and best_epoch. SAE rows
    also require phase_best_epochs for ae1, ae2 and supervised. This pure
    function checks metadata and selection arithmetic, not actual artifact bytes.
    """
    # Explicit prospective TiDE criteria; the historical TiDE gate is unchanged.
    from collections.abc import Mapping
    import math
    import statistics
    try:
        from .e2_reduced_protocol import _require, _iso, digest, FOLDS, SLOT_IDS, PARENT_INDICES, HPO_SEED, RESTORED, SHA
    except ImportError:
        from e2_reduced_protocol import _require, _iso, digest, FOLDS, SLOT_IDS, PARENT_INDICES, HPO_SEED, RESTORED, SHA
    model = "TiDE"
    VERSION = "e2-reduced-tide-source-selection-v1"
    _require(_iso(final_origin) >= _iso("2025-01-01"),
             "A global three-fold recipe cannot forecast an earlier outer period")
    if isinstance(configured_domain, Mapping):
        domains = configured_domain.get("models", configured_domain)
        _require(model in domains, "Supply the explicitly registered TiDE extension domain")
        configured_domain = domains[model]
    domain = list(configured_domain)
    _require(len(domain) == 6 and tuple(row.get("configuration_id") for row in domain) == SLOT_IDS,
             "Supply the six ordered, hash-bound configuration rows for this model")
    by_slot = {}
    for row, index in zip(domain, PARENT_INDICES):
        plain = {k: v for k, v in row.items() if k != "configuration_sha256"}
        trial = row.get("trial", {})
        _require(row.get("model_id") == model and row.get("parent_trial_index") == index
                 and row.get("configuration_sha256") == digest(plain)
                 and row.get("optimization_seed") == HPO_SEED and row.get("execution_allowed") is True
                 and row.get("execution_seed_ids") == [HPO_SEED]
                 and row.get("architecture_status") == "published_or_verified_fixed_body"
                 and row.get("parent_trial_id") == trial.get("trial_id")
                 and isinstance(trial.get("trial_id"), str)
                 and trial["trial_id"].endswith(":hpo-%03d" % index)
                 and trial.get("optimization_seeds") == [271201, 271202],
                 "The supplied domain changed its model, parent position, seed or hash")
        _require(trial in registered_trials() and trial.get("recipe_id") == RECIPE_ID
                 and trial.get("dropout") == 0.0
                 and row.get("architecture_extension_id") == "e2-tide-source-default-r0-extension-v1"
                 and row.get("historical_NAS_result_reuse_allowed") is False,
                 "The TiDE row is not the new source-defined recipe or admits old NAS outcomes")
        by_slot[row["configuration_id"]] = row
    rows = list(ledger)
    _require(len(rows) == 18, "All six configurations by three folds require eighteen explicit run rows")
    seen, successes, failures = set(), {}, {}
    for row in rows:
        _require(isinstance(row, Mapping), "Run ledger rows must be mappings")
        slot, fold = row.get("configuration_id"), row.get("fold_origin")
        _require(slot in by_slot and fold in FOLDS, "Foreign configuration or fold in reduced HPO ledger")
        key = (slot, fold)
        _require(key not in seen, "Duplicate reduced HPO configuration/fold")
        seen.add(key)
        configured = by_slot[slot]
        _require(row.get("model_id") == model and row.get("seed") == HPO_SEED
                 and row.get("parent_trial_id") == configured["parent_trial_id"]
                 and row.get("configuration_sha256") == configured["configuration_sha256"],
                 "Run identity differs from the registered reduced configuration")
        _require(type(row.get("validation_count")) is int and row["validation_count"] == 83,
                 "Each fold must retain its verified common83 validation dates")
        _require(isinstance(row.get("run_receipt_sha256"), str)
                 and SHA.fullmatch(row["run_receipt_sha256"]), "Every attempt needs a bound run receipt")
        use = row.get("resource_use", {})
        _require(isinstance(use, Mapping), "Run resource use must be a mapping")
        flags = [source.get(key, False) for source in (row, use) for key in ("fatal", "systematic_failure")]
        _require(all(type(flag) is bool for flag in flags), "Failure flags must be explicit booleans")
        _require(not any(flags), "Fatal or systematic failure blocks reduced selection")
        status = row.get("status")
        if status == "failed":
            _require(isinstance(row.get("reason"), str) and bool(row["reason"].strip()),
                     "Failed attempts remain in the ledger with a reason")
            failures[key] = row["reason"]
        else:
            score, epoch = row.get("rmse_original"), row.get("best_epoch")
            _require(status == "completed" and type(score) in (int, float)
                     and math.isfinite(score) and score >= 0,
                     "Completed attempts need finite nonnegative original-unit RMSE")
            _require(type(epoch) is int and 1 <= epoch <= 200, "Best epoch must lie within1..200")
            phases = row.get("phase_best_epochs", {"supervised": epoch})
            required_phases = {"ae1", "ae2", "supervised"} if model == "LSTM-autoencoder con predictor" else {"supervised"}
            _require(isinstance(phases, Mapping) and set(phases) == required_phases
                     and all(type(value) is int and 1 <= value <= 200 for value in phases.values())
                     and phases["supervised"] == epoch,
                     "Preserve the strict-best epoch of every admitted optimizer phase")
            successes[key] = (float(score), epoch, dict(phases), row["validation_count"])
    expected = {(slot, fold) for slot in SLOT_IDS for fold in FOLDS}
    _require(seen == expected, "The reduced ledger is incomplete")
    candidates = []
    for slot in SLOT_IDS:
        failed = [fold for fold in FOLDS if (slot, fold) in failures]
        if failed:
            candidates.append({"configuration_id": slot, "status": "failed",
                "failed_folds": failed, "replacement_performed": False})
            continue
        pairs = [successes[(slot, fold)] for fold in FOLDS]
        count = sum(p[3] for p in pairs)
        pooled_mse = sum(p[3] * p[0]**2 for p in pairs) / count
        _require(math.isfinite(pooled_mse), "Concatenated validation score overflow")
        candidates.append({"configuration_id": slot, "status": "completed",
            "pooled_cv_RMSE_original": math.sqrt(pooled_mse),
            "mean_fold_RMSE_original_diagnostic": statistics.mean(p[0] for p in pairs),
            "validation_count": count,
            "fixed_epochs": int(math.ceil(statistics.median(p[1] for p in pairs))),
            "fixed_phase_epochs": {phase: int(math.ceil(statistics.median(p[2][phase] for p in pairs)))
                for phase in pairs[0][2]},
            "fold_RMSE_original": {fold: successes[(slot, fold)][0] for fold in FOLDS}})
    complete = [row for row in candidates if row["status"] == "completed"]
    _require(bool(complete), "No complete three-fold configuration remains")
    winner = min(complete, key=lambda row: (row["pooled_cv_RMSE_original"], row["configuration_id"]))
    return {"version": VERSION, "model_id": model, "winner_configuration_id": winner["configuration_id"],
        "winner_parent_trial_id": by_slot[winner["configuration_id"]]["parent_trial_id"],
        "pooled_cv_RMSE": winner["pooled_cv_RMSE_original"],
        "winner_pooled_cv_RMSE_original": winner["pooled_cv_RMSE_original"],
        "winner_mean_fold_RMSE_original_diagnostic": winner["mean_fold_RMSE_original_diagnostic"],
        "validation_count": winner["validation_count"],
        "fixed_epochs": winner["fixed_epochs"], "fixed_phase_epochs": winner["fixed_phase_epochs"],
        "candidate_ledger": candidates,
        "all_eighteen_slots_accounted_for": True, "configuration_count": 6, "fold_count": 3,
        "selection_scope": "one_global_recipe_for_post_HPO_final_cohort",
        "earliest_final_origin": "2025-01-01", "criteria_only": True,
        "run_artifact_bytes_verified": False, "new_training_performed": False}
