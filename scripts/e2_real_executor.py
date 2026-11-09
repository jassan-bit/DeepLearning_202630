"""Dated E2 execution primitives. No trainer, search launcher or final-data loader.

The CLI only audits the two registered development derivatives, read-only.
Replay consumes already frozen adapters; tests use deterministic doubles, not
fitted models. A/B are explicit alternatives, never selected using losses.
Historical synthetic-only and final-access gates remain in their own modules.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd

import audit_e2_common_availability as availability

DAY = pd.Timedelta(days=1)
VERSION = "e2-dated-core-v1"


def utc_day(value):
    """ISO dates mean UTC; timestamps must carry an explicit timezone."""
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        value = value + "T00:00:00Z"
    if isinstance(value, (int, float, bool, np.number)):
        raise ValueError("A calendar date is required, not a row number")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must identify an explicit UTC instant")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise ValueError("Use daily UTC midnight, not local midnight or intraday times")
    return stamp


def dated_index(values, *, consecutive=False):
    index = pd.DatetimeIndex([utc_day(v) for v in values])
    if not len(index) or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Dates must be nonempty, unique and increasing")
    if consecutive and not index.equals(pd.date_range(index[0], index[-1], freq="D")):
        raise ValueError("Missing days must remain explicit calendar rows")
    return index


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def date_strings(index):
    return [utc_day(v).strftime("%Y-%m-%d") for v in index]


def support_frame(calendar, features, targets, quality=None):
    """Freeze availability before selection, independently of candidate errors."""
    index = dated_index(calendar, consecutive=True)
    masks = availability.support_masks(index, features, targets, quality)
    return pd.DataFrame(masks, index=index)


def _support(support):
    if not isinstance(support, pd.DataFrame):
        raise ValueError("A dated support table is required")
    index = dated_index(support.index, consecutive=True)
    required = {"D7_window", "target_sigma7_quality", "D7_target", "common",
                "lag1_target", "lag7_target", "lag14_target", "lag28_target"}
    if set(support.columns) != required or any(t != bool for t in support.dtypes):
        raise ValueError("Complete boolean support schema required")
    if not support["D7_target"].equals(support.D7_window & support.target_sigma7_quality):
        raise ValueError("Inconsistent D7 training support")
    target = support.target_sigma7_quality.to_numpy()
    for lag in (1, 7, 14, 28):
        if not np.array_equal(support[f"lag{lag}_target"], availability.lag_mask(target, lag)):
            raise ValueError("Lag support must use true calendar offsets")
    common = support.D7_target.copy()
    for lag in (1, 7, 14, 28):
        common &= support[f"lag{lag}_target"]
    if not common.equals(support.common):
        raise ValueError("Common support must include every registered lag")
    return index


@dataclass(frozen=True)
class BlockPlan:
    name: str
    role: str
    origin: pd.Timestamp
    cutoff: pd.Timestamp
    training_calendar: tuple
    neural_training_dates: tuple
    target_training_dates: tuple
    bridge_dates: tuple
    evaluation_dates: tuple
    eligible: tuple
    support_sha256: str

    def record(self):
        return {"name": self.name, "role": self.role, "origin": self.origin.isoformat(),
                "cutoff": self.cutoff.isoformat(),
                "training_calendar": date_strings(self.training_calendar),
                "neural_training_dates": date_strings(self.neural_training_dates),
                "target_training_dates": date_strings(self.target_training_dates),
                "bridge_dates": date_strings(self.bridge_dates),
                "evaluation_dates": date_strings(self.evaluation_dates),
                "eligible": list(self.eligible), "support_sha256": self.support_sha256}


def block_plan(support, origin, end, *, name, role="outer"):
    index = _support(support)
    origin, end = utc_day(origin), utc_day(end)
    cutoff = origin - 8 * DAY
    if (not isinstance(name, str) or not name or role not in ("outer", "inner")
            or end < origin or cutoff < index[0]):
        raise ValueError("Invalid block or insufficient dated training history")
    if origin < index[0] or end > index[-1]:
        raise ValueError("Block coverage is not certified by supplied development support")
    training = index[index <= cutoff]
    evaluation = pd.date_range(origin, end, freq="D")
    selected = support.loc[index[0]:end]
    signature = {"dates": date_strings(selected.index),
                 "masks": {c: selected[c].tolist() for c in sorted(selected.columns)}}
    return BlockPlan(name, role, origin, cutoff, tuple(training),
                     tuple(training[support.loc[training, "D7_target"]]),
                     tuple(training[support.loc[training, "target_sigma7_quality"]]),
                     tuple(pd.date_range(cutoff + DAY, origin - DAY, freq="D")),
                     tuple(evaluation), tuple(bool(v) for v in support.loc[evaluation, "common"]),
                     canonical_hash(signature))


def temporal_plan(support, origin, end, *, scenario):
    """A: outer CV only. B: 83-day inner validation plus a fresh outer refit.

    This builds dates and support, not models or hyperparameter trials.
    """
    if scenario not in ("A", "B"):
        raise ValueError("Choose A or B explicitly before any selection")
    origin = utc_day(origin)
    outer = block_plan(support, origin, end, name="outer:" + str(origin.date()))
    inner = (block_plan(support, origin - 90 * DAY, origin - 8 * DAY,
                        name="inner:" + str(origin.date()), role="inner")
             if scenario == "B" else None)
    return {"scenario": scenario, "outer": outer, "inner": inner,
            "selection_kind": "global_development_CV_not_nested_OOF" if scenario == "A"
            else "inner_selection_then_fresh_refit_per_origin"}


def validate_plan(plan):
    """Validate stored plans before using any zip/slice over their dates."""
    if not isinstance(plan, BlockPlan) or not plan.name or plan.role not in ("outer", "inner"):
        raise ValueError("A complete dated block plan is required")
    origin, cutoff = utc_day(plan.origin), utc_day(plan.cutoff)
    training = dated_index(plan.training_calendar, consecutive=True)
    evaluation = dated_index(plan.evaluation_dates, consecutive=True)
    if (cutoff != origin - 8 * DAY or training[-1] != cutoff or evaluation[0] != origin
            or len(plan.eligible) != len(evaluation) or any(type(v) is not bool for v in plan.eligible)):
        raise ValueError("Plan cutoffs, evaluation dates and full boolean mask must align")
    if tuple(plan.bridge_dates) != tuple(pd.date_range(cutoff + DAY, origin - DAY, freq="D")):
        raise ValueError("Plan must preserve the seven bridge calendar dates")
    if plan.role == "inner" and len(evaluation) != 83:
        raise ValueError("B inner validation must span 83 calendar days")
    for dates in (plan.neural_training_dates, plan.target_training_dates):
        if dates and not set(dated_index(dates)).issubset(set(training)):
            raise ValueError("Training support falls outside the dated training calendar")
    if not set(plan.neural_training_dates).issubset(set(plan.target_training_dates)):
        raise ValueError("Neural training requires a valid target")
    if not isinstance(plan.support_sha256, str) or not re.fullmatch("[0-9a-f]{64}", plan.support_sha256):
        raise ValueError("Support seal is missing")


def mase_scale(training_targets, cutoff):
    """Common train-only lag-1 denominator; never bridge missing calendar dates."""
    if not isinstance(training_targets, pd.Series):
        raise ValueError("MASE requires a dated target Series")
    index = dated_index(training_targets.index)
    cutoff = utc_day(cutoff)
    # Slice before numeric conversion: values after cutoff are not inspected.
    selected = index <= cutoff
    y = pd.Series(training_targets.iloc[np.flatnonzero(selected)].to_numpy(dtype=float), index=index[selected])
    if y.empty:
        return {"value": None, "pairs": 0, "reason": "no_consecutive_training_pairs"}
    y = y.where(np.isfinite(y) & (y >= 0))
    y = y.reindex(pd.date_range(y.index[0], y.index[-1], freq="D"))
    differences = y.diff().abs().dropna()
    count = len(differences)
    value = float(differences.mean()) if count else None
    reason = "no_consecutive_training_pairs" if not count else "zero_training_scale" if value == 0 else None
    if value is not None and not np.isfinite(value):
        raise ValueError("Nonfinite MASE training scale")
    return {"value": value if value and value > 0 else None, "pairs": count, "reason": reason}


def metrics(truth, forecasts, scale):
    truth, forecasts = np.asarray(truth, float), np.asarray(forecasts, float)
    if (truth.ndim != 1 or truth.shape != forecasts.shape or not len(truth)
            or not np.isfinite(truth).all() or not np.isfinite(forecasts).all()
            or (truth < 0).any()):
        raise ValueError("Scoring requires complete finite predictions on fixed common support")
    errors = forecasts - truth
    mse, mae = float(np.mean(errors ** 2)), float(np.mean(np.abs(errors)))
    if not np.isfinite([mse, mae]).all():
        raise ValueError("Numerical overflow in scoring")
    value = scale["value"]
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError("MASE scale must be positive or explicitly unavailable")
    mase = mae / float(value) if value is not None else None
    if mase is not None and not np.isfinite(mase):
        raise ValueError("Numerical overflow in MASE")
    return {"n": len(truth), "RMSE": float(np.sqrt(mse)), "MAE": mae,
            "MASE": mase,
            "MASE_scale": dict(scale)}


def select_candidate(candidate_order, plans, outcomes, *, scenario):
    """Rank completed synthetic/future authorised trials, never launch HPO.

    Every candidate reports every fold or an explicit terminal failure. The
    common mask cannot be changed per candidate. Ties follow declared order.
    Multiple plans mean scenario A concatenation, not nested OOF evaluation.
    """
    candidates, plans = tuple(candidate_order), tuple(plans)
    if scenario not in ("A", "B"):
        raise ValueError("Selection scenario must be fixed before comparing candidates")
    if not candidates or len(set(candidates)) != len(candidates) or set(outcomes) != set(candidates):
        raise ValueError("Unique ordered candidates and complete outcome ledger required")
    if not plans or len({p.name for p in plans}) != len(plans):
        raise ValueError("Distinct dated selection folds required")
    for plan in plans:
        validate_plan(plan)
    if ((scenario == "A" and any(p.role != "outer" for p in plans))
            or (scenario == "B" and (len(plans) != 1 or plans[0].role != "inner"))):
        raise ValueError("Do not mix global A folds with per-origin B inner selection")
    dates = [d for p in plans for d, ok in zip(p.evaluation_dates, p.eligible) if ok]
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("Selection dates must be nonempty and nonoverlapping")
    ledger, successes = {}, []
    for candidate in candidates:
        outcome = outcomes[candidate]
        if outcome.get("status") == "failed":
            if not outcome.get("reason") or "resource_use" not in outcome:
                raise ValueError("Failures require a reason and recorded resource use")
            ledger[candidate] = dict(outcome)
            continue
        if outcome.get("status") != "completed" or set(outcome.get("folds", {})) != {p.name for p in plans}:
            raise ValueError("Completed candidate must contain every declared fold")
        squared = []
        for plan in plans:
            fold = outcome["folds"][plan.name]
            expected = [d for d, ok in zip(plan.evaluation_dates, plan.eligible) if ok]
            if (fold.get("support_sha256") != plan.support_sha256
                    or fold.get("plan_sha256") != canonical_hash(plan.record())
                    or tuple(utc_day(d) for d in fold.get("dates", [])) != tuple(expected)):
                raise ValueError("Candidate changed the dated common support")
            errors = np.asarray(fold.get("squared_errors"), dtype=float)
            if errors.shape != (len(expected),) or not np.isfinite(errors).all() or (errors < 0).any():
                raise ValueError("Incomplete or nonfinite candidate errors")
            squared.extend(errors.tolist())
        rmse = float(np.sqrt(np.mean(squared)))
        if not np.isfinite(rmse):
            raise ValueError("Nonfinite selection metric")
        ledger[candidate] = {"status": "completed", "n": len(squared), "RMSE": rmse}
        successes.append((rmse, candidates.index(candidate), candidate))
    if not successes:
        raise ValueError("All candidates failed; do not expand search or use the final test")
    return {"winner": min(successes)[2], "scenario": scenario, "primary_metric": "pooled_dated_RMSE",
            "tie_rule": "declared_candidate_order", "candidate_ledger": ledger,
            "dates": date_strings(dates)}


class DatedReplay:
    """Causal view over an already authorised daily panel, with quality gating.

    This class opens no files. Predictors receive only historical X or their
    own filtered state. It is an API boundary, not a sandbox against callbacks
    that deliberately close over an external array.
    """
    def __init__(self, calendar, features, targets, *, quality=None):
        self.calendar = dated_index(calendar, consecutive=True)
        self._features = np.array(features, dtype=float, copy=True)
        self._targets = np.array(targets, dtype=float, copy=True)
        self._support = support_frame(self.calendar, self._features, self._targets, quality)
        self._features.setflags(write=False)
        self._targets.setflags(write=False)
        self._next = None
        self.revealed = []

    @property
    def support(self):
        return self._support.copy(deep=True)

    def begin(self, cutoff):
        if self._next is not None:
            raise ValueError("Replay is single-use; a fresh refit uses a fresh replay")
        cutoff = utc_day(cutoff)
        if cutoff not in self.calendar:
            raise ValueError("Training cutoff is outside supplied calendar")
        self._next = cutoff + DAY
        positions = self.calendar <= cutoff
        values = self._targets[positions].copy()
        values[~self._support.loc[positions, "target_sigma7_quality"].to_numpy()] = np.nan
        return pd.Series(values, index=self.calendar[positions])

    def window(self, target):
        target = utc_day(target)
        if target != self._next:
            raise ValueError("Only current-origin covariates may be requested")
        wanted = pd.date_range(target - 7 * DAY, target - DAY, freq="D")
        positions = self.calendar.get_indexer(wanted)
        if (positions < 0).any() or not self._support.at[target, "D7_window"]:
            return None
        return self._features[positions].copy()

    def reveal(self, target):
        target = utc_day(target)
        if target != self._next or target not in self.calendar:
            raise ValueError("Observations must advance once per calendar day")
        position = self.calendar.get_loc(target)
        result = float(self._targets[position]) if self._support.at[target, "target_sigma7_quality"] else None
        self._next += DAY
        self.revealed.append(target)
        return result


@dataclass(frozen=True)
class FrozenSystem:
    """Already fitted immutable-parameter adapter, or an untrained naive rule.

    kind=state must be initialised at cutoff and forecasts on EVERY day; observe
    advances one calendar day, including None observations. fingerprint covers
    fixed parameters/scales, not the evolving causal filter. kind=window sees
    only D7 OHLCV. No fit method is called or accepted by the executor.
    """
    name: str
    kind: str
    predict: Callable | None = None
    observe: Callable | None = None
    fingerprint: Callable | None = None
    lag: int | None = None
    clip_nonnegative: bool = False

    def validate(self):
        if not isinstance(self.name, str) or not self.name or self.kind not in ("window", "state", "naive"):
            raise ValueError("Named window/state/naive adapter required")
        if self.kind == "naive":
            if type(self.lag) is not int or self.lag not in (1, 7, 14, 28):
                raise ValueError("Declared calendar lag required")
            if any(x is not None for x in (self.predict, self.observe, self.fingerprint)):
                raise ValueError("Naive is a fixed rule, not an arbitrary callback")
        elif not callable(self.predict) or not callable(self.fingerprint) or self.lag is not None:
            raise ValueError("Frozen predictor and parameter fingerprint required")
        if self.kind == "state" and not callable(self.observe):
            raise ValueError("State adapter must advance on every day")
        if self.kind == "window" and self.observe is not None:
            raise ValueError("Window predictors cannot update during evaluation")


class ExecutionFailure(RuntimeError):
    pass


def frozen_arima_system(model, plan, *, name="ARIMA"):
    """Bridge the existing FrozenARIMA API to UTC-aware daily replay, without fit.

    The legacy adapter stores naive timestamps representing UTC. Convert at
    this boundary only; public plans and journals retain explicit UTC offsets.
    """
    import e2_temporal_controls as controls

    if not isinstance(model, controls.FrozenARIMA):
        raise ValueError("An already fitted FrozenARIMA instance is required")
    if (model.current != plan.cutoff.tz_localize(None)
            or model.origin != plan.origin.tz_localize(None) or model._pending is not None
            or model._failure is not None or model.receipt.get("status") != "fitted"):
        raise ValueError("ARIMA must be fresh, active and frozen at this block's cutoff")

    def fingerprint():
        parameters = np.asarray(model._result.params, dtype=float)
        if not np.isfinite(parameters).all() or not np.isfinite(model.scale) or model.scale <= 0:
            raise ExecutionFailure("Invalid fixed ARIMA parameters or scale")
        return canonical_hash({"parameters": parameters.tolist(), "scale": float(model.scale)})

    def predict(target, unused):
        return model.forecast(utc_day(target).tz_localize(None)).raw

    def observe(target, value):
        model.observe(utc_day(target).tz_localize(None), value)

    return FrozenSystem(name, "state", predict, observe, fingerprint, clip_nonnegative=True)


class DatedJournal:
    """Exclusive, local hash-chained fsync journal; not remote tamper proof."""
    def __init__(self, directory):
        raw = Path(directory).absolute()
        if any(p.is_symlink() for p in (raw, *raw.parents)):
            raise ValueError("Symlink output paths are not permitted")
        if not raw.parent.is_dir():
            raise ValueError("Output parent must exist")
        raw.mkdir(exist_ok=False)
        self.file = (raw / "events.jsonl").open("x", encoding="utf-8")
        self.sequence, self.previous = 0, "0" * 64
        self.failed = False
        for path in (raw.parent, raw):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def append(self, kind, **payload):
        if self.failed:
            raise ExecutionFailure("Journal stopped after write failure")
        event = {"sequence": self.sequence, "previous_sha256": self.previous, "kind": kind, **payload}
        token = canonical_hash(event)
        event["event_sha256"] = token
        try:
            self.file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            self.failed = True
            raise
        self.sequence += 1
        self.previous = token
        return token

    def close(self):
        self.file.close()


def run_frozen_block(plan, replay, systems, journal):
    """Replay frozen adapters day by day. Does not fit, choose or load a model.

    All declared systems retain their rows, including unsupported predictions.
    ARIMA-like state adapters forecast even on excluded days before observing.
    A model or write failure stops the whole block without shrinking its mask.
    """
    systems = tuple(systems)
    validate_plan(plan)
    if not systems or len({s.name for s in systems}) != len(systems):
        raise ValueError("A nonempty unique system roster is required")
    for system in systems:
        system.validate()
    expected = block_plan(replay.support, plan.origin, plan.evaluation_dates[-1], name=plan.name, role=plan.role)
    if plan != expected:
        raise ValueError("Plan/support mismatch; never change the frozen mask")
    if not any(plan.eligible):
        raise ValueError("No common evaluation dates; cannot score")
    # Hash values immediately; never alias a mutable fingerprint dictionary.
    signatures = {s.name: canonical_hash(s.fingerprint()) for s in systems if s.fingerprint}

    def check_frozen():
        for system in systems:
            if system.fingerprint and canonical_hash(system.fingerprint()) != signatures[system.name]:
                raise ExecutionFailure("Frozen parameters or scales changed during replay")

    truth, predictions = [], {s.name: [] for s in systems}
    try:
        journal.append("block_plan", version=VERSION, plan=plan.record(), systems=[s.name for s in systems])
        past = replay.begin(plan.cutoff)
        scale = mase_scale(past, plan.cutoff)
        for target in plan.bridge_dates:
            value = replay.reveal(target)
            journal.append("bridge_observation", date=target.isoformat(), value=value)
            for system in systems:
                if system.kind == "state":
                    system.observe(target, value)
                    check_frozen()
            past.loc[target] = np.nan if value is None else value
        for target, eligible in zip(plan.evaluation_dates, plan.eligible):
            check_frozen()
            window = replay.window(target)
            values = {}
            for system in systems:
                reason = None
                if system.kind == "naive":
                    raw = past.get(target - system.lag * DAY, np.nan)
                    if not np.isfinite(raw):
                        reason = "required_calendar_lag_unavailable"
                elif system.kind == "window" and window is None:
                    raw, reason = None, "D7_window_unavailable"
                else:
                    raw = system.predict(target, window.copy() if system.kind == "window" else None)
                    check_frozen()
                if reason:
                    values[system.name] = {"raw": None, "value": None, "reason": reason, "truncated": False}
                else:
                    raw = float(raw)
                    if not np.isfinite(raw):
                        raise ExecutionFailure(f"Nonfinite forecast: {system.name} at {target}")
                    values[system.name] = {"raw": raw,
                        "value": max(0.0, raw) if system.clip_nonnegative else raw,
                        "truncated": bool(system.clip_nonnegative and raw < 0), "reason": None}
            if eligible and any(p["value"] is None for p in values.values()):
                raise ExecutionFailure("Unsupported system on fixed eligible date")
            token = journal.append("predictions", date=target.isoformat(), eligible=eligible, systems=values)
            # No observation accessor call before the completed durable append.
            observed = replay.reveal(target)
            journal.append("observation", date=target.isoformat(), value=observed,
                           prediction_event_sha256=token)
            for system in systems:
                if system.kind == "state":
                    system.observe(target, observed)
            past.loc[target] = np.nan if observed is None else observed
            check_frozen()
            if eligible:
                if observed is None:
                    raise ExecutionFailure("Frozen truth availability changed")
                truth.append(observed)
                for name in predictions:
                    predictions[name].append(values[name]["value"])
        scores = {name: metrics(truth, values, scale) for name, values in predictions.items()}
        result = {"version": VERSION, "status": "frozen_adapter_replay_completed",
                  "model_fits": 0, "HPO_trials": 0, "NAS_trials": 0,
                  "calendar_days": len(plan.evaluation_dates), "scored_days": len(truth),
                  "excluded_dates": date_strings([d for d, ok in zip(plan.evaluation_dates, plan.eligible) if not ok]),
                  "metrics": scores, "plan_sha256": canonical_hash(plan.record())}
        journal.append("block_completed", result=result)
        return result
    except Exception as exc:
        try:
            journal.append("block_failed", error_type=type(exc).__name__, error=str(exc),
                           mask_changed=False, retry_or_fallback=False)
        except Exception:
            pass  # Preserve the original error; a failed journal cannot certify failure bytes.
        raise


def development_preflight(root):
    """Read only the two registered 2023–2024 derivatives, with source hashes.

    No final metadata/payload, weights, scaler, forecast or metric is opened.
    Receipts expose support counts/dates, never market values or statistics.
    """
    raw_root = Path(root).absolute()
    if any(p.is_symlink() for p in (raw_root, *raw_root.parents)):
        raise ValueError("Registered input root cannot be a symlink")
    root = raw_root.resolve()
    names = (availability.PROTOCOL, *availability.SOURCES)
    paths = [root / name for name in names]
    for path in paths:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Registered inputs cannot be symlinks")
    registration_sha256 = availability.digest(paths[0])
    registration = json.loads(paths[0].read_text(encoding="utf-8"))["sources"]
    hashes = {name: availability.digest(root / name) for name in availability.SOURCES}
    if any(hashes[name] != registration[name] for name in availability.SOURCES):
        raise ValueError("Registered development SHA256 mismatch")
    frames = [pd.read_csv(path) for path in paths[1:]]
    calendar = pd.date_range("2023-01-01", "2024-12-31", tz="UTC")
    for frame in frames:
        if not dated_index(frame.market_day_utc, consecutive=True).equals(calendar):
            raise ValueError("Only the registered 2023–2024 calendar is permitted")
    support = support_frame(calendar, frames[0][list(availability.FEATURES)], frames[1].sigma_7)
    scenarios = {}
    for scenario in ("A", "B"):
        blocks = []
        for month in range(1, 11):
            origin = utc_day(f"2024-{month:02d}-01")
            end = origin + pd.offsets.MonthEnd(0)
            plan = temporal_plan(support, origin, end, scenario=scenario)
            blocks.append({key: value.record() if isinstance(value, BlockPlan) else value
                           for key, value in plan.items()})
        scenarios[scenario] = blocks
    final_cutoff = utc_day("2024-12-24")
    final_inner_cutoff = utc_day("2024-09-25")
    result = {"version": VERSION, "status": "development_preflight_only",
              "scenario_selected": None, "scenarios_are_alternatives": True,
              "model_fits": 0, "scaler_fits": 0, "HPO_trials": 0, "NAS_trials": 0,
              "forecasts": 0, "predictive_metrics": 0, "final_files_opened": 0,
              "source_sha256": hashes, "registration_sha256": registration_sha256,
              "development_calendar_days": len(calendar),
              "support_counts": {c: int(support[c].sum()) for c in support},
              "scenarios": scenarios,
              "final_training_support_only": {
                  "cutoff": str(final_cutoff.date()),
                  "neural_count": int(support.loc[:final_cutoff, "D7_target"].sum()),
                  "inner_cutoff": str(final_inner_cutoff.date()),
                  "inner_neural_count": int(support.loc[:final_inner_cutoff, "D7_target"].sum()),
                  "validation_common_count": int(support.loc["2024-10-03":"2024-12-24", "common"].sum())},
              "final_support": {"reference_count": 608, "common_count": None,
                                "status": "not_certified", "missing_dates": None},
              "limits": ["No real training, model comparison, HPO/NAS, pilot or final scoring was executed.",
                         "A/B choice, model-specific adapters, full HPO/NAS budget and independent environment reproduction remain gates before market use.",
                         "Daily derivatives retain quality exclusions; minute completeness was not independently reconstructed."]}
    if any(availability.digest(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Development sources changed during audit")
    if availability.digest(paths[0]) != registration_sha256:
        raise ValueError("Registration changed during audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-preflight", action="store_true", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_output = args.output.absolute()
    if any(p.is_symlink() for p in (raw_output, *raw_output.parents)):
        raise ValueError("Audit output cannot use symlinks")
    if args.output.resolve().is_relative_to(args.project_root.resolve()):
        raise ValueError("Audit output must stay outside the source repository")
    if args.output.exists():
        raise FileExistsError("Never replace an audit receipt")
    result = development_preflight(args.project_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "model_fits": 0,
                      "final_support": result["final_support"]}))


if __name__ == "__main__":
    main()
