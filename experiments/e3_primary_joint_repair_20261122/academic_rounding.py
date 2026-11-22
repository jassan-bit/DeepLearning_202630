"""Apply the author's reported teaching criterion without changing the study."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path


def assess(report_path: Path) -> dict:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report["status"] != "completed" or report["completed_replicates"] != 4608:
        raise ValueError("Only the completed, preserved V1 study can be interpreted")
    rows = []
    for scenario in report["scenarios"]:
        for name in ("CI99_rmse_difference", "CI99_relative_rmse_gain"):
            item = scenario["coverage"][name]
            bound = Decimal(str(item["lower_simultaneous_MC"]))
            percent = bound * Decimal(100)
            rounded = percent.quantize(Decimal(1), rounding=ROUND_HALF_UP)
            rows.append({
                "scenario": scenario["id"], "interval": name,
                "nominal_interval_confidence": "0.99",
                "covered": item["covered"], "trials": item["n"],
                "exact_bound_as_recorded": str(bound),
                "exact_percent_as_recorded": str(percent),
                "display_two_decimal_percent": str(percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
                "rounded_whole_percent": int(rounded),
                "original_exact_coverage_criterion_pass": bound >= Decimal("0.95"),
                "academic_coverage_criterion_pass_under_reported_rounding": rounded >= 95,
            })
    other = all(s["global_criterion_pass"] for s in report["scenarios"])
    return {
        "schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "post_result_academic_interpretation_reported_by_author",
        "teacher_criterion_reported_by_user": "el profe dijo que se podia redondear el 94,85 a 95%",
        "teacher_statement_independently_authenticated": False,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "sourceP": report["source_commit"], "protocol_sha256": report["protocol_sha256"],
        "policy": "Nearest whole percentage using ROUND_HALF_UP, applied consistently to all 18 CI99 coverage lower bounds only",
        "policy_was_received_after_results": True,
        "coverage_bounds_checked": len(rows),
        "coverage_bounds_passing_original_exact_rule": sum(r["original_exact_coverage_criterion_pass"] for r in rows),
        "coverage_bounds_passing_reported_academic_rounding": sum(r["academic_coverage_criterion_pass_under_reported_rounding"] for r in rows),
        "other_89_MC_bounds_not_rounded": True,
        "other_original_criteria_pass": other,
        "academic_method_criterion_pass_under_reported_rounding": bool(other and all(r["academic_coverage_criterion_pass_under_reported_rounding"] for r in rows)),
        "scientific_preregistered_acceptance": bool(report["accepted"]),
        "exact_results_or_original_protocol_modified": False,
        "mathematical_lower_bound_at_least_0_95_claimed_from_rounding": False,
        "rows": rows,
        "additional_V2_formal_replicates_executed": 0,
        "V2_status": "Implemented and software verified, archived without a new Monte Carlo study",
        "R1_academic_acceptance": None, "R2_academic_acceptance": None,
        "R3_accredited": False, "official_grade": None, "new_grade_assigned": False,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = assess(root / "results/VALIDATION_REPORT_V1.json")
    output = root / "results/ACADEMIC_ROUNDING_DECISION.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "coverage_bounds_checked", "coverage_bounds_passing_original_exact_rule",
        "coverage_bounds_passing_reported_academic_rounding",
        "academic_method_criterion_pass_under_reported_rounding",
        "scientific_preregistered_acceptance", "R3_accredited", "new_grade_assigned",
    )}, ensure_ascii=False))
