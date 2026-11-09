"""Fold-local selection/refit plumbing for dated E2 neural training.

No files, market archives or final payloads are loaded. Callbacks receive copies
of admitted inner/refit tensors. This API is a chronology boundary, not a Python
sandbox against a callback that deliberately closes over another data source.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

try:
    from . import e2_real_executor as dated
except ImportError:
    import e2_real_executor as dated

VERSION = "e2-temporal-training-v1"
DAY = pd.Timedelta(days=1)


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _array_hash(value):
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Portable numerical arrays required")
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    header = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header+b"\0"+array.tobytes()).hexdigest()


def _tensor_hash(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Prepared tensors required")
    return _array_hash(value.detach().cpu().numpy())


def relative_window(raw_windows):
    """Exactly the E1 log-price/last-close and log1p-volume transform."""
    raw = np.asarray(raw_windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1:] != (7, 5):
        raise ValueError("Nonempty raw [N,7,5] windows required")
    if (not np.isfinite(raw).all() or (raw[..., :4] <= 0).any()
            or (raw[..., 4] < 0).any()):
        raise ValueError("Positive finite prices and nonnegative volume required")
    result = np.empty_like(raw)
    logs = np.log(raw[..., :4])
    result[..., :4] = logs-logs[:, -1:, 3:4]
    result[..., 4] = np.log1p(raw[..., 4])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite relative-window transform")
    return result


class DatedPanel:
    """Own copies of an already authorised/synthetic continuous daily calendar."""
    def __init__(self, calendar, features, targets, quality=None):
        self._calendar = dated.dated_index(calendar, consecutive=True).copy()
        self._features = np.array(features, dtype=np.float64, copy=True)
        self._targets = np.array(targets, dtype=np.float64, copy=True)
        self._quality = (np.ones(len(self._calendar), dtype=bool) if quality is None
                         else np.array(quality, copy=True))
        self._support = dated.support_frame(self._calendar, self._features,
                                            self._targets, self._quality)
        self._features.setflags(write=False)
        self._targets.setflags(write=False)
        self._quality.setflags(write=False)

    @property
    def calendar(self):
        return self._calendar.copy()

    @property
    def support(self):
        return self._support.copy(deep=True)

    @property
    def features(self):
        return self._features.copy()

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def quality(self):
        return self._quality.copy()

    def _positions(self, dates):
        index = dated.dated_index(dates)
        positions = self._calendar.get_indexer(index)
        if (positions < 7).any() or (positions >= len(self._calendar)).any():
            raise ValueError("Dates need complete preceding windows in supplied calendar")
        return positions

    def raw_windows(self, dates):
        positions = self._positions(dates)
        return self._features[positions[:, None]-np.arange(7, 0, -1)].copy()

    def targets_at(self, dates):
        return self._targets[self._positions(dates)].copy()


@dataclass(frozen=True)
class FoldScaler:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    target_scale: float
    train_dates: tuple
    training_sha256: str

    def __post_init__(self):
        if (len(self.mean) != 5 or len(self.std) != 5
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or min(self.std) <= 0 or not math.isfinite(self.target_scale)
                or self.target_scale <= 0):
            raise ValueError("Finite train-only channel scales and positive target mean required")

    @classmethod
    def fit(cls, raw_windows, targets, train_dates):
        values = relative_window(raw_windows)
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (len(values),) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Training targets must be finite nonnegative and aligned")
        mean = values.mean(axis=(0, 1))
        std = values.std(axis=(0, 1), ddof=0)
        std = np.where(std == 0, 1., std)
        scale = float(y.mean())
        dates = tuple(dated.dated_index(train_dates))
        if len(dates) != len(values):
            raise ValueError("Scaler training dates differ from windows")
        signature = dated.canonical_hash({"dates": dated.date_strings(dates),
                                         "relative_x": _array_hash(values), "y": _array_hash(y)})
        return cls(tuple(mean), tuple(std), scale, dates, signature)

    def transform(self, raw_windows):
        result = (relative_window(raw_windows)-np.asarray(self.mean))/np.asarray(self.std)
        if not np.isfinite(result).all():
            raise ValueError("Scaled input is not finite")
        return result

    def scale_targets(self, values):
        result = np.asarray(values, dtype=np.float64)/self.target_scale
        if not np.isfinite(result).all():
            raise ValueError("Scaled targets are not finite")
        return result

    def inverse_targets(self, values):
        result = np.asarray(values, dtype=np.float64)*self.target_scale
        if not np.isfinite(result).all():
            raise ValueError("Restored predictions are not finite")
        return result

    def record(self):
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std],
                "target_scale": self.target_scale,
                "train_dates": dated.date_strings(self.train_dates),
                "training_sha256": self.training_sha256}


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    y_original: tuple[float, ...]
    dates: tuple
    scaler: FoldScaler
    signature: str

    def validate(self):
        if (self.x.shape != (len(self.dates), 7, 5) or self.y.shape != (len(self.dates),)
                or len(self.y_original) != len(self.dates) or not len(self.dates)
                or not bool(torch.isfinite(self.x).all()) or not bool(torch.isfinite(self.y).all())):
            raise ValueError("Prepared split tensors and dates differ")
        actual = dated.canonical_hash({"name": self.name, "dates": dated.date_strings(self.dates),
                                      "x": _tensor_hash(self.x), "y": _tensor_hash(self.y),
                                      "y_original": list(self.y_original), "scaler": self.scaler.record()})
        if actual != self.signature:
            raise ValueError("Prepared split changed after sealing")

    def clone(self):
        self.validate()
        return PreparedSplit(self.name, self.x.detach().clone(), self.y.detach().clone(),
                             self.y_original, self.dates, self.scaler, self.signature)

    def record(self):
        self.validate()
        return {"name": self.name, "n": len(self.dates), "dates": dated.date_strings(self.dates),
                "signature": self.signature, "scaler": self.scaler.record()}


@dataclass(frozen=True)
class PreparedB:
    origin: pd.Timestamp
    inner_plan: Any
    outer_plan: Any
    inner_train: PreparedSplit
    inner_validation: PreparedSplit
    refit_train: PreparedSplit
    signature: str

    def validate(self):
        dated.validate_plan(self.inner_plan)
        dated.validate_plan(self.outer_plan)
        for split in (self.inner_train, self.inner_validation, self.refit_train):
            split.validate()
        if (self.inner_plan.cutoff != self.origin-98*DAY
                or self.inner_plan.origin != self.origin-90*DAY
                or self.inner_plan.evaluation_dates[-1] != self.origin-8*DAY
                or self.outer_plan.cutoff != self.origin-8*DAY):
            raise ValueError("B chronology boundaries changed")
        if self.inner_train.scaler is not self.inner_validation.scaler:
            raise ValueError("Inner validation must use its inner-training scaler")
        if self.refit_train.scaler is self.inner_train.scaler:
            raise ValueError("Refit requires a fresh training-only scaler")
        actual = _prepared_signature(self.origin, self.inner_plan, self.outer_plan,
                                     self.inner_train, self.inner_validation, self.refit_train)
        if actual != self.signature:
            raise ValueError("Prepared B provenance changed")


def _prepared_signature(origin, inner, outer, train, valid, refit):
    return dated.canonical_hash({"origin": dated.utc_day(origin).isoformat(),
                                 "inner": inner.record(), "outer": outer.record(),
                                 "inner_train": train.signature, "inner_validation": valid.signature,
                                 "refit_train": refit.signature})


def _split(panel, name, dates, scaler=None):
    dates = tuple(dated.dated_index(dates))
    raw, original = panel.raw_windows(dates), panel.targets_at(dates)
    scaler = FoldScaler.fit(raw, original, dates) if scaler is None else scaler
    x = torch.tensor(scaler.transform(raw), dtype=torch.float32)
    y = torch.tensor(scaler.scale_targets(original), dtype=torch.float32)
    original = tuple(float(v) for v in original)
    signature = dated.canonical_hash({"name": name, "dates": dated.date_strings(dates),
                                     "x": _tensor_hash(x), "y": _tensor_hash(y),
                                     "y_original": list(original), "scaler": scaler.record()})
    result = PreparedSplit(name, x, y, original, dates, scaler, signature)
    result.validate()
    return result


def prepare_b(panel, origin, end):
    """Prepare inner train/validation and a separate fresh refit, never outer y."""
    if not isinstance(panel, DatedPanel):
        raise TypeError("A DatedPanel is required")
    plans = dated.temporal_plan(panel.support, origin, end, scenario="B")
    inner, outer = plans["inner"], plans["outer"]
    train = _split(panel, "inner_train", inner.neural_training_dates)
    valid_dates = tuple(d for d, ok in zip(inner.evaluation_dates, inner.eligible) if ok)
    if not valid_dates:
        raise ValueError("No eligible common validation dates")
    valid = _split(panel, "inner_validation", valid_dates, train.scaler)
    refit = _split(panel, "refit_train", outer.neural_training_dates)
    result = PreparedB(dated.utc_day(origin), inner, outer, train, valid, refit,
                       _prepared_signature(origin, inner, outer, train, valid, refit))
    result.validate()
    return result


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    payload: dict
    payload_sha256: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("Stable nonempty candidate ID required")
        if not isinstance(self.payload, dict):
            raise TypeError("Candidate payload must be explicit JSON metadata")
        copied = _json_copy(self.payload)
        object.__setattr__(self, "payload", copied)
        object.__setattr__(self, "payload_sha256", dated.canonical_hash(copied))

    def validate(self):
        if dated.canonical_hash(self.payload) != self.payload_sha256:
            raise ValueError("Candidate payload changed after registration")

    def clone(self):
        self.validate()
        return CandidateSpec(self.candidate_id, self.payload)


@dataclass
class TrialRun:
    status: str
    run_id: str
    resource_use: dict
    predictions_original: Any = None
    prediction_dates: Any = None
    phase_epochs: dict | None = None
    reason: str | None = None
    fatal: bool = False

    @classmethod
    def failed(cls, reason, resource_use, run_id, *, fatal=False):
        return cls("failed", run_id, resource_use, reason=reason, fatal=fatal)


@dataclass
class RefitRun:
    status: str
    run_id: str
    resource_use: dict
    phase_epochs: dict | None = None
    fresh_initialization: bool = False
    artifact: Any = None
    reason: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    winner: CandidateSpec
    fixed_epochs: dict
    candidate_ledger: dict
    used_run_ids: tuple[str, ...]
    prepared_signature: str
    attempts: int
    selection_rule: str
    tie_candidates: tuple[str, ...]
    delta: float | None
    signature: str

    def validate(self, prepared):
        prepared.validate()
        self.winner.validate()
        actual = _selection_signature(self.winner, self.fixed_epochs, self.candidate_ledger,
                                       self.used_run_ids, self.prepared_signature, self.attempts,
                                       self.selection_rule, self.tie_candidates, self.delta)
        if actual != self.signature or self.prepared_signature != prepared.signature:
            raise ValueError("Selection result changed or belongs to another origin")


class SelectionAborted(RuntimeError):
    def __init__(self, message, ledger):
        super().__init__(message)
        self.candidate_ledger = deepcopy(ledger)


def _selection_signature(winner, epochs, ledger, used, prepared, attempts, rule, ties, delta):
    return dated.canonical_hash({"winner": winner.candidate_id,
                                 "payload": winner.payload_sha256, "epochs": epochs,
                                 "ledger": ledger, "used_run_ids": list(used),
                                 "prepared_signature": prepared, "attempts": attempts,
                                 "selection_rule": rule, "tie_candidates": list(ties), "delta": delta})


def _seeds(seeds):
    values = tuple(seeds)
    if not values or any(type(s) is not int or s < 0 for s in values) or len(set(values)) != len(values):
        raise ValueError("Distinct nonnegative integer seeds required")
    return values


def _epochs(values):
    if (not isinstance(values, dict) or not values
            or any(not isinstance(k, str) or not k or type(v) is not int or v < 1
                   for k, v in values.items())):
        raise ValueError("Positive best epoch for every fitted phase required")
    return dict(values)


def _run_header(run, used):
    if not isinstance(run.run_id, str) or not run.run_id or run.run_id in used:
        raise ValueError("Each fit/refit must have a fresh distinct run ID")
    if not isinstance(run.resource_use, dict) or not run.resource_use:
        raise ValueError("Every attempt needs recorded resource use")
    resources = _json_copy(run.resource_use)
    used.add(run.run_id)
    return resources


def median_phase_epochs(runs):
    if not runs:
        raise ValueError("Completed seed runs required")
    values = [_epochs(run.phase_epochs) for run in runs]
    keys = set(values[0])
    if any(set(v) != keys for v in values):
        raise ValueError("Seed runs have different phase structures")
    return {key: int(math.ceil(float(np.median([v[key] for v in values])))) for key in sorted(keys)}


def select_b(prepared, candidates, seeds, fit_candidate: Callable, *,
             selection_rule="mean_rmse", delta=None):
    """Every supplied candidate/seed is charged; no added attempts or outer y."""
    prepared.validate()
    candidates, seeds = tuple(candidates), _seeds(seeds)
    if selection_rule not in ("mean_rmse", "nas_parsimony"):
        raise ValueError("Use mean_rmse or nas_parsimony selection")
    margin = None
    if selection_rule == "nas_parsimony":
        if type(delta) not in (float, int) or not math.isfinite(delta) or delta < 0:
            raise ValueError("NAS parsimony needs a finite nonnegative registered delta")
        if len(seeds) < 5:
            raise ValueError("NAS confirmation needs at least five distinct seeds")
        margin = float(delta)
    if (not candidates or any(not isinstance(c, CandidateSpec) for c in candidates)
            or len({c.candidate_id for c in candidates}) != len(candidates)):
        raise ValueError("Unique registered candidate specifications required")
    for candidate in candidates:
        candidate.validate()
    used, ledger, successes = set(), {}, []
    successful_runs = {}
    for candidate in candidates:
        candidate_runs, run_rows, failed = [], [], False
        for seed in seeds:
            prepared.validate()
            candidate.validate()
            run = fit_candidate(candidate.clone(), seed, prepared.inner_train.clone(),
                                prepared.inner_validation.clone())
            if not isinstance(run, TrialRun):
                raise TypeError("Fit callback must return TrialRun, including explicit failures")
            resources = _run_header(run, used)
            row = {"seed": seed, "run_id": run.run_id, "status": run.status,
                   "resource_use": resources}
            if run.status == "failed":
                if not isinstance(run.reason, str) or not run.reason:
                    raise ValueError("A failed attempt needs a reason")
                row["reason"] = run.reason
                failed = True
                run_rows.append(row)
                if run.fatal:
                    ledger[candidate.candidate_id] = {"status": "failed", "runs": run_rows}
                    raise SelectionAborted("Systematic failure blocks selection", ledger)
                continue
            if run.status != "completed":
                raise ValueError("Unknown trial status")
            dates = tuple(dated.utc_day(d) for d in run.prediction_dates)
            if dates != prepared.inner_validation.dates:
                raise ValueError("Candidate changed the validation mask or date ordering")
            predictions = np.asarray(run.predictions_original, dtype=np.float64)
            truth = np.asarray(prepared.inner_validation.y_original)
            if predictions.shape != truth.shape or not np.isfinite(predictions).all():
                raise ValueError("Complete finite validation predictions required")
            residual = predictions-truth
            rmse = float(np.sqrt(np.mean(residual*residual)))
            if not math.isfinite(rmse):
                raise ValueError("Selection RMSE is not finite")
            row.update({"RMSE": rmse, "phase_epochs": _epochs(run.phase_epochs)})
            run_rows.append(row)
            candidate_runs.append(TrialRun("completed", run.run_id, resources,
                                           tuple(float(v) for v in predictions), dates,
                                           deepcopy(row["phase_epochs"])))
        record = {"status": "failed" if failed else "completed", "runs": run_rows,
                  "attempts": len(seeds), "payload_sha256": candidate.payload_sha256}
        if not failed:
            record["mean_seed_RMSE"] = float(np.mean([r["RMSE"] for r in run_rows]))
            if selection_rule == "nas_parsimony":
                counts = [r["resource_use"].get("model_parameters") for r in run_rows]
                if (any(type(n) is not int or n < 1 for n in counts)
                        or len(set(counts)) != 1):
                    raise ValueError("NAS seeds must report the same positive predictor parameter count")
                from scipy.stats import t as student_t
                scores = np.array([r["RMSE"] for r in run_rows])
                radius = float(student_t.ppf(.975, len(scores)-1)*scores.std(ddof=1)/math.sqrt(len(scores)))
                if not math.isfinite(radius):
                    raise ValueError("Mean-score interval is not finite")
                mean = record["mean_seed_RMSE"]
                record.update({"model_parameters": counts[0],
                               "mean_RMSE_CI95": [mean-radius, mean+radius],
                               "CI_scope": "nominal t interval across seeds, not equivalence"})
            successes.append((record["mean_seed_RMSE"], candidate.candidate_id, candidate))
            successful_runs[candidate.candidate_id] = candidate_runs
        ledger[candidate.candidate_id] = record
    prepared.validate()
    if not successes:
        raise SelectionAborted("All candidates failed; no replacement search allowed", ledger)
    best_mean, best_id, winner = min(successes, key=lambda v: (v[0], v[1]))
    ties = (best_id,)
    if selection_rule == "nas_parsimony":
        lower, upper = ledger[best_id]["mean_RMSE_CI95"]
        technical = []
        for mean, candidate_id, candidate in successes:
            lo, hi = ledger[candidate_id]["mean_RMSE_CI95"]
            tied = mean-best_mean <= margin or max(lower, lo) <= min(upper, hi)
            ledger[candidate_id]["technical_tie_to_best"] = bool(tied)
            if tied:
                technical.append((ledger[candidate_id]["model_parameters"], mean, candidate_id, candidate))
        _, _, _, winner = min(technical, key=lambda v: (v[0], v[1], v[2]))
        ties = tuple(sorted(row[2] for row in technical))
    epochs = median_phase_epochs(successful_runs[winner.candidate_id])
    winner = winner.clone()
    run_ids = tuple(sorted(used))
    signature = _selection_signature(winner, epochs, ledger, run_ids,
                                     prepared.signature, len(candidates)*len(seeds),
                                     selection_rule, ties, margin)
    result = SelectionResult(winner, epochs, ledger, run_ids, prepared.signature,
                             len(candidates)*len(seeds), selection_rule, ties, margin, signature)
    result.validate(prepared)
    return result


def top_k_with_reference(prepared, selection, candidates, k, reference_id):
    """Freeze distinct phase1 finalists and add the reference exactly once.

    This generic helper is also usable on declared small synthetic fixtures;
    the full protocol separately enforces its registered k20/k40 and 1000 rows.
    A failed reference can still be retained as the predeclared phase2 control.
    """
    selection.validate(prepared)
    if type(k) is not int or k < 1:
        raise ValueError("A positive finalist count is required")
    candidates = tuple(candidates)
    if (any(not isinstance(c, CandidateSpec) for c in candidates)
            or len({c.candidate_id for c in candidates}) != len(candidates)
            or {c.candidate_id for c in candidates} != set(selection.candidate_ledger)):
        raise ValueError("Exact registered phase1 candidate set required")
    by_id = {}
    for candidate in candidates:
        candidate.validate()
        if candidate.payload_sha256 != selection.candidate_ledger[candidate.candidate_id]["payload_sha256"]:
            raise ValueError("Finalist payload differs from phase1 registration")
        by_id[candidate.candidate_id] = candidate
    if reference_id not in by_id:
        raise ValueError("The reference must be a registered phase1 architecture")
    completed = [(row["mean_seed_RMSE"], key) for key, row in selection.candidate_ledger.items()
                 if row["status"] == "completed"]
    if len(completed) < k:
        raise ValueError("Too few completed architectures; do not replace failed attempts")
    ids = [key for _, key in sorted(completed)[:k]]
    if reference_id not in ids:
        ids.append(reference_id)
    return tuple(by_id[key].clone() for key in ids)


def refit_b(prepared, selection, seeds, refit_candidate: Callable):
    """A separate callback receives only fresh refit training data and epochs."""
    selection.validate(prepared)
    seeds = _seeds(seeds)
    selection_seeds = {row["seed"] for candidate in selection.candidate_ledger.values()
                       for row in candidate["runs"]}
    if set(seeds) & selection_seeds:
        raise ValueError("Refit seeds must be distinct from inner-selection seeds")
    used = set(selection.used_run_ids)
    output = []
    for seed in seeds:
        selection.validate(prepared)
        run = refit_candidate(selection.winner.clone(), seed, prepared.refit_train.clone(),
                              deepcopy(selection.fixed_epochs))
        if not isinstance(run, RefitRun):
            raise TypeError("Refit callback must return RefitRun")
        _run_header(run, used)
        if run.status == "completed":
            if not run.fresh_initialization:
                raise ValueError("Refit must initialize a new predictor and optimizer stages")
            if _epochs(run.phase_epochs) != selection.fixed_epochs:
                raise ValueError("Refit changed the epoch budget or used outer-label early stopping")
        elif run.status != "failed" or not isinstance(run.reason, str) or not run.reason:
            raise ValueError("Refit must complete or record an explicit terminal failure")
        output.append(run)
    selection.validate(prepared)
    return tuple(output)

