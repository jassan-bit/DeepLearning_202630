"""Metadata-only data-role and prospective-freeze guards for amended E2.

This module never reads market payloads or fits a model. Hashes in a freeze
attestation bind declared manifests/bundles; their format does not prove that
private checkpoints exist, that a calibration was performed, or that a human
never inspected data. The campaign verifier must check those artifacts and the
actual protocol registration independently.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import re

try:
    from .e2_required_catalogue import REQUIRED
except ImportError:
    from e2_required_catalogue import REQUIRED

VERSION = "e2-amended-data-guard-v1"
FREEZE_VERSION = "e2-global-freeze-attestation-v1"
BINDING_VERSION = "e2-prospective-test-binding-v1"
DEVELOPMENT_START = date(2023, 1, 1)
DEVELOPMENT_END = date(2024, 12, 31)
HPO_NAS_DATA_END = date(2024, 9, 23)
POWER_PILOT_DATA_END = date(2024, 12, 30)
HISTORICAL_START = date(2025, 1, 1)
HISTORICAL_608_END = date(2026, 8, 31)
KNOWN_SOURCE_END = date(2026, 10, 30)
PROSPECTIVE_TRAINING_END = date(2026, 8, 24)
CALIBRATION_START = date(2026, 9, 1)
CALIBRATION_END = KNOWN_SOURCE_END
HISTORICAL_TRAINING_END = date(2024, 12, 24)
TEST_CALENDAR_DAYS = 100
FINAL_SEED_POOL = tuple(range(271801, 271831))
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value, label):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(label + " must be one lowercase SHA256")
    return value


def _instant(value, label):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(label + " must be an explicit UTC instant") from error
    if (not isinstance(value, datetime) or value.tzinfo is None
            or value.utcoffset() != timedelta(0)):
        raise ValueError(label + " must be an explicit UTC instant")
    return value.astimezone(timezone.utc)


def _now(value):
    return _instant(datetime.now(timezone.utc) if value is None else value, "now_utc")


def _day(value, label="date"):
    if isinstance(value, datetime):
        value = _instant(value, label)
        if value.timetz().replace(tzinfo=None) != time(0):
            raise ValueError(label + " must denote a UTC calendar midnight")
        return value.date()
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError(label + " must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(label + " must be an ISO calendar date") from error


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Freeze and binding metadata must be finite JSON values") from error


def _metadata_sha256(value):
    return sha256(_canonical(value)).hexdigest()


def _validate_freeze(metadata, protocol_sha256, now):
    _hash(protocol_sha256, "protocol_sha256")
    if not isinstance(metadata, dict) or metadata.get("version") != FREEZE_VERSION:
        raise ValueError("Use the complete versioned global-freeze attestation")
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise ValueError("The freeze belongs to a different registered protocol")
    frozen = _instant(metadata.get("frozen_at_utc"), "frozen_at_utc")
    registered = _instant(metadata.get("amendment_registered_at_utc"), "amendment_registered_at_utc")
    selected = _instant(metadata.get("selection_closed_at_utc"), "selection_closed_at_utc")
    if not registered <= frozen <= now or selected > frozen:
        raise ValueError("Registration, selection closure and global freeze must precede binding")
    for field in ("selection_closed", "all_systems_closed", "calibrators_closed"):
        if metadata.get(field) is not True:
            raise ValueError("The global freeze requires " + field)
    if not DEVELOPMENT_START <= _day(metadata.get("hpo_nas_data_end"), "hpo_nas_data_end") <= HPO_NAS_DATA_END:
        raise ValueError("HPO/NAS cannot use labels after 2024-09-23 under this amendment")
    if not DEVELOPMENT_START <= _day(metadata.get("power_pilot_data_end"), "power_pilot_data_end") <= POWER_PILOT_DATA_END:
        raise ValueError("The power pilot cannot use labels after 2024-12-30")
    repetitions = metadata.get("R")
    if type(repetitions) is not int or not 6 <= repetitions <= 30:
        raise ValueError("A power-reviewed common R between 6 and 30 must be frozen")
    if metadata.get("final_seed_ids") != list(FINAL_SEED_POOL[:repetitions]):
        raise ValueError("Freeze the exact common fresh final-seed prefix for R")
    for field in ("power_result_sha256", "analysis_plan_sha256"):
        _hash(metadata.get(field), field)
    calibration = metadata.get("calibration")
    if calibration != {"start": "2026-09-01", "end_inclusive": "2026-10-30", "calendar_days": 60}:
        raise ValueError("Freeze the known sixty-day calibration calendar exactly")
    if frozen.date() <= CALIBRATION_END:
        raise ValueError("Calibration must have matured before the global freeze")
    systems = metadata.get("systems")
    if not isinstance(systems, list) or len(systems) != len(REQUIRED):
        raise ValueError("Every one of the sixteen mandatory systems must be closed")
    identities = [row.get("model_id") if isinstance(row, dict) else None for row in systems]
    if (any(not isinstance(identity, str) for identity in identities)
            or len(set(identities)) != len(REQUIRED) or set(identities) != set(REQUIRED)):
        raise ValueError("The sixteen frozen model IDs must match the mandatory catalogue exactly")
    for row in systems:
        for field in ("recipe_sha256", "weights_sha256", "preprocessing_sha256", "calibration_sha256"):
            _hash(row.get(field), row["model_id"] + ":" + field)
        cutoff = _day(row.get("training_cutoff"), row["model_id"] + ":training_cutoff")
        if not DEVELOPMENT_START <= cutoff <= PROSPECTIVE_TRAINING_END:
            raise ValueError("Prospective refits must stop by 2026-08-24 before calibration embargo")
    _canonical(metadata)
    return registered, frozen


def bind_prospective_test(freeze_metadata, protocol_sha256, *, now_utc=None):
    """Bind a genuinely future 100-day calendar from a complete global freeze.

    Late first-time binding is refused. After the calendar starts, use
    validate_prospective_binding() to read the previously registered binding;
    never manufacture a new prospective reservation after observing outcomes.
    """
    now = _now(now_utc)
    registered, frozen = _validate_freeze(freeze_metadata, protocol_sha256, now)
    start = max(registered.date(), frozen.date()) + timedelta(days=1)
    if start <= now.date():
        raise ValueError("Late binding is prohibited: the independent calendar has already started")
    end = start + timedelta(days=TEST_CALENDAR_DAYS - 1)
    result = {"version": BINDING_VERSION, "protocol_sha256": protocol_sha256,
              "bound_at_utc": now.isoformat(), "start": start.isoformat(),
              "end_inclusive": end.isoformat(), "calendar_days": TEST_CALENDAR_DAYS,
              "minimum_eligible_cases_for_error_analysis": 100,
              "automatic_extension_allowed": False,
              "freeze_attestation": deepcopy(freeze_metadata),
              "freeze_attestation_sha256": _metadata_sha256(freeze_metadata),
              "verification_scope": "Metadata attestation and calendar checks; not private artifact existence or prior human-exposure certification"}
    result["binding_sha256"] = _metadata_sha256(result)
    return result


def bind_new_cohort(freeze_metadata, protocol_sha256, *, now_utc=None):
    """Alias naming the prospective cohort, not an additional binding operation."""
    return bind_prospective_test(freeze_metadata, protocol_sha256, now_utc=now_utc)


def validate_prospective_binding(binding, protocol_sha256, *, now_utc=None):
    """Validate a previously registered binding even after its dates mature."""
    now = _now(now_utc)
    if not isinstance(binding, dict) or binding.get("version") != BINDING_VERSION:
        raise ValueError("Independent test access requires a previously registered binding")
    value = deepcopy(binding)
    seal = value.pop("binding_sha256", None)
    if _hash(seal, "binding_sha256") != _metadata_sha256(value):
        raise ValueError("The prospective binding changed after sealing")
    if value.get("protocol_sha256") != _hash(protocol_sha256, "protocol_sha256"):
        raise ValueError("The binding belongs to a different registered protocol")
    bound = _instant(value.get("bound_at_utc"), "bound_at_utc")
    if bound > now:
        raise ValueError("A future binding receipt cannot establish prior registration")
    freeze = value.get("freeze_attestation")
    registered, frozen = _validate_freeze(freeze, protocol_sha256, bound)
    if value.get("freeze_attestation_sha256") != _metadata_sha256(freeze):
        raise ValueError("The bound global-freeze attestation changed")
    expected_start = max(registered.date(), frozen.date()) + timedelta(days=1)
    expected_end = expected_start + timedelta(days=TEST_CALENDAR_DAYS - 1)
    if (value.get("start") != expected_start.isoformat()
            or value.get("end_inclusive") != expected_end.isoformat()
            or value.get("calendar_days") != TEST_CALENDAR_DAYS
            or bound.date() >= expected_start
            or value.get("minimum_eligible_cases_for_error_analysis") != 100
            or value.get("automatic_extension_allowed") is not False):
        raise ValueError("The independent calendar must be bound before its first day and stay fixed")
    return deepcopy(binding)


def _dates(values):
    if isinstance(values, (str, bytes, date, datetime)):
        raise ValueError("Supply a nonempty chronological sequence of calendar dates")
    try:
        values = tuple(_day(value) for value in values)
    except TypeError as error:
        raise ValueError("Supply a chronological sequence of calendar dates") from error
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError("Dates must be nonempty, unique and chronological")
    return values


def require_data_role(dates, role, *, binding=None, protocol_sha256=None,
                      model_training_cutoff=None, now_utc=None):
    """Check date roles without reading values or asserting data availability.

    historical_exploratory classifies already-exposed data; it makes no
    out-of-sample claim. historical_replay additionally requires the distinct
    Jan-2025 study's training cutoff, preventing prospective 2026 refits from
    being relabelled as an out-of-sample 608-date replay.
    """
    values = _dates(dates)
    ranges = {
        "development": (DEVELOPMENT_START, DEVELOPMENT_END),
        "hpo_nas": (DEVELOPMENT_START, HPO_NAS_DATA_END),
        "power_pilot": (DEVELOPMENT_START, POWER_PILOT_DATA_END),
        "prospective_training": (DEVELOPMENT_START, PROSPECTIVE_TRAINING_END),
        "calibration": (CALIBRATION_START, CALIBRATION_END),
        "calibration_bridge": (date(2026, 8, 25), date(2026, 8, 31)),
        "historical_exploratory": (HISTORICAL_START, KNOWN_SOURCE_END),
        "historical_replay": (HISTORICAL_START, KNOWN_SOURCE_END),
    }
    if role in ("independent_test", "inference_bridge"):
        if not isinstance(binding, dict):
            raise ValueError("Independent test dates cannot be admitted before global freeze and binding")
        expected_protocol = protocol_sha256 if protocol_sha256 is not None else binding.get("protocol_sha256")
        validated = validate_prospective_binding(binding, expected_protocol, now_utc=now_utc)
        if role == "inference_bridge":
            lower, upper = KNOWN_SOURCE_END + timedelta(days=1), _day(validated["start"]) - timedelta(days=1)
            if values[-1] >= _now(now_utc).date():
                raise ValueError("Inference bridge observations must be fully closed UTC days")
        else:
            lower, upper = _day(validated["start"]), _day(validated["end_inclusive"])
        if role == "independent_test" and any(day <= KNOWN_SOURCE_END for day in values):
            raise ValueError("Already-exposed historical dates cannot become an independent test")
    elif role in ranges:
        lower, upper = ranges[role]
        if role == "historical_replay":
            if model_training_cutoff is None or _day(model_training_cutoff) > HISTORICAL_TRAINING_END:
                raise ValueError("Historical replay requires its own weights/scalers fitted no later than 2024-12-24")
    else:
        raise ValueError("Unknown amended data role")
    if values[0] < lower or values[-1] > upper:
        raise ValueError("Dates cross the frozen boundary for data role " + str(role))
    return tuple(day.isoformat() for day in values)


def audit_prospective_coverage(eligible_dates, binding, protocol_sha256, *, now_utc=None):
    """Report missing quality cases after all 100 fixed calendar days have closed."""
    now = _now(now_utc)
    validated = validate_prospective_binding(binding, protocol_sha256, now_utc=now)
    end = _day(validated["end_inclusive"])
    if now.date() <= end:
        raise ValueError("Final coverage cannot be closed before the entire fixed calendar matures")
    supplied = tuple(eligible_dates)
    admitted = require_data_role(supplied, "independent_test", binding=validated,
                                protocol_sha256=protocol_sha256, now_utc=now) if supplied else ()
    start = _day(validated["start"])
    calendar = tuple((start + timedelta(days=i)).isoformat() for i in range(TEST_CALENDAR_DAYS))
    missing = [day for day in calendar if day not in set(admitted)]
    enough = len(admitted) >= 100
    return {"version": VERSION, "binding_sha256": validated["binding_sha256"],
            "calendar_days": TEST_CALENDAR_DAYS, "eligible_cases": len(admitted),
            "missing_or_ineligible_dates": missing, "status": "complete" if enough else "incomplete_quality_coverage",
            "error_analysis_requirement_met": enough, "automatic_extension_allowed": False,
            "fixed_end_inclusive": end.isoformat(),
            "interpretation": "Coverage only; does not certify performance, power or full guide compliance"}
