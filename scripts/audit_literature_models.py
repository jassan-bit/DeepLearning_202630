"""Reproduce documentary family counts and conditional parameter arithmetic.

Only reads the extraction CSV. No neural framework, paper code, market data,
predictions or test files are imported or executed. Counts refer to the expanded
exploratory full-text cohort S1/S2/S5/S6/S7, with original and date-sensitivity
strata. They do not describe the entire bibliography or literature.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "model_extraction_review_20261103.csv"
RESULT = "model_complexity_review_20261103.csv"
SUMMARY = "model_frequency_review_20261103.json"
FREQUENCIES = "family_frequencies.csv"


def parameters(spec: dict) -> int | None:
    """Count declared layers and biases analytically; no model instantiation."""
    kind = spec["kind"]
    if kind in {"lstm", "gru"}:
        p, h, q = (spec[k] for k in ("p", "H", "q"))
        directions, biases = spec["directions"], spec["bias_vectors"]
        gates = 4 if kind == "lstm" else 3
        sizes = spec.get("hidden_sizes", [h])
        assert len(sizes) == spec["layers"]
        recurrent = 0
        for hidden_size in sizes:
            recurrent += directions * gates * hidden_size * (p + hidden_size + biases)
            p = directions * hidden_size
        h = sizes[-1]
        width, hidden = directions * h, spec["head_hidden"]
        head = width*q + q if hidden is None else width*hidden + hidden + hidden*q + q
        return recurrent + head
    if kind == "tcn":
        assert spec["stacks"] == 1 and spec["convolutions_per_block"] == 2
        assert spec["bias"] and spec["normalization"] == "none"
        p, f, k, b, q = (spec[x] for x in ("p", "F", "kernel", "blocks", "q"))
        first = k*p*f+f + k*f*f+f
        projection = p*f+f if spec["residual_projection"] and p != f else 0
        remaining = (b-1)*(2*k*f*f+2*f)
        return first + projection + remaining + f*q+q
    if kind == "transformer":
        p, d, m, q = (spec[x] for x in ("p", "d", "M", "q"))
        assert d % spec["heads"] == 0 and spec["pooling"] == "last_and_mean"
        ff = spec["feedforward"]
        encoder = spec["layers"] * (4*d*d+2*d*ff+9*d+ff)
        hidden = spec["head_hidden"]
        head = 2*d*hidden + hidden + hidden*q+q
        return (p+1)*d + m*d + encoder + head
    if kind == "linear":
        return (spec["p_flat"]+1)*spec["q"]
    if kind == "transformer_partial":
        # Missing feedforward/readout/positions cannot be filled with defaults.
        return None
    raise ValueError(f"Unsupported documentary architecture: {kind}")


def render(source: Path) -> dict[str, bytes]:
    rows = list(csv.DictReader(source.open()))
    families: dict[str, set[str]] = {}
    variants: dict[str, set[str]] = {}
    studies = sorted({r["study_id"] for r in rows if r["included_full_text_study"] == "True"})
    for row in rows:
        if row["count_in_baseline_family_frequency"] != "True":
            continue
        families.setdefault(row["family"], set()).add(row["study_id"])
        if row["family"] == "LSTM":
            key = "BiLSTM" if "bidirectional" in row["variant"] and "unidirectional" not in row["variant"] else "LSTM_unidirectional"
            variants.setdefault(key, set()).add(row["study_id"])
    result = []
    source_checks = []
    for row in rows:
        if not row["architecture_spec_json"]:
            continue
        spec = json.loads(row["architecture_spec_json"])
        derived = parameters(spec)
        if derived is not None:
            assert derived == int(row["parameter_count_analytical"]), row
        else:
            assert not row["parameter_count_analytical"], row
        source_check = {"study_id": row["study_id"], "family": row["family"],
                        "parameters": derived, "status": "analytical, not publication-reported or instantiated" if derived is not None
                        else "not identifiable from source; missing feedforward/readout/positions"}
        if spec.get("p_sensitivity"):
            source_check["conditional_input_dimension_counts"] = [
                {"input_features": spec["p"], "parameters": derived}]
            for alternative in spec["p_sensitivity"]:
                other = copy.deepcopy(spec)
                other["p"] = alternative
                source_check["conditional_input_dimension_counts"].append(
                    {"input_features": alternative, "parameters": parameters(other)})
        source_checks.append(source_check)
        if row["count_in_baseline_family_frequency"] != "True":
            continue
        common = copy.deepcopy(spec)
        common.update(p=5, q=1)
        label = row["variant"]
        if common["kind"] in {"lstm", "gru"} and row["study_id"] in {"S1", "S5", "S7"}:
            # Separate one/two-bias conventions for the supplied Keras networks.
            one, two = copy.deepcopy(common), copy.deepcopy(common)
            one["bias_vectors"], two["bias_vectors"] = 1, 2
            counts = parameters(one), parameters(two)
        elif common["kind"] in {"lstm", "gru"}:
            counts = "not applicable", parameters(common)
        elif common["kind"] == "transformer_partial":
            counts = "not identifiable from source", "not applicable"
        else:
            counts = parameters(common), "not applicable"
        width = common.get("H", common.get("F", common.get("d")))
        def append(specification, scenario_label, bias_counts, history):
            result.append({"study_id": row["study_id"], "family": row["family"],
                "variant": scenario_label, "input_features": 5, "outputs": 1,
                "history_days": history, "source_width": width, "source_depth": row["depth"],
                "analytical_parameters_one_bias": bias_counts[0],
                "analytical_parameters_two_bias": bias_counts[1],
                "architecture_spec_json": json.dumps(specification, separators=(",", ":")),
                "assumptions": row["parameter_count_assumptions"] + "; common dimensions p=5,q=1; source width/depth retained; this is not a source-model total",
                "configuration_status": "Analytical scenario; not a selected, implemented or trained project model",
                "own_experiment": False})
        append(common, label, counts, "7/14/21/28")
        if common["kind"] == "transformer":
            for d in (7, 14, 21, 28):
                shorter = copy.deepcopy(common)
                shorter["M"] = d
                append(shorter, label+" positional-capacity sensitivity M=D", (parameters(shorter), "not applicable"), d)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(result[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(result)
    counts = {name: {"studies": sorted(ids), "count": len(ids), "denominator": len(studies)}
              for name, ids in sorted(families.items(), key=lambda item: (-len(item[1]), item[0]))}
    sensitivities = {}
    for label, subset in {
        "original_full_text_cohort": {"S1", "S2", "S5"},
        "expanded_without_undated_S6": set(studies)-{"S6"},
        "expanded_without_explicit_grey_S2_S6_S7": set(studies)-{"S2", "S6", "S7"},
    }.items():
        included = sorted(set(studies) & subset)
        sensitivities[label] = {
            "studies": included, "denominator": len(included),
            "family_counts": {family: len(ids & subset) for family, ids in families.items()},
            "note": "Subset sensitivity, not proof of indexing, Q1 or peer review of remaining studies"}
    exclusions = [{"study_id": r["study_id"], "family": r["family"], "role": r["role"]}
                  for r in rows if r["count_in_baseline_family_frequency"] != "True"]
    summary = {
        "review_date": "2026-11-03", "cohort": "expanded_exploratory_S1_S2_S5_S6_S7",
        "scope": "Documentary arithmetic for reviewer-included full texts; no search or automatic eligibility decision performed by this script",
        "counting_unit": "one included full-text study per independently evaluated comparative DL family",
        "studies": studies, "denominator": len(studies), "family_frequency": counts,
        "variant_studies": {k: sorted(v) for k, v in variants.items()},
        "sensitivity_strata": sensitivities,
        "excluded_from_baseline_family_counts": exclusions,
        "source_parameter_checks": source_checks, "common_dimension_scenarios": len(result),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hyperparameters_corrected": "S1 Appendix C Tables5-6 PDF pages10-11 report winners; they are not missing",
        "limitations": [
            "Frequencies describe five accessible exploratory full texts, not global prevalence or the number of bibliographic citations.",
            "S2/S6/S7 are grey literature; S6 is undated and date-sensitive; declared editorial metadata of S5 were not independently corroborated.",
            "S6 has no exact parameter total; S7 declares12 inputs while supplied code lists8, so its totals are conditional alternatives.",
            "Parameter values are analytical under recorded conventions; TCN library version is unpinned.",
            "Common-dimension arithmetic retains source widths/depths and does not select a project architecture.",
            "The selected-family frequency, simplicity and diversity requirements are not jointly closed by this audit."],
        "neural_models_executed": 0, "market_files_opened": 0, "test_files_opened": 0,
    }
    frequency_rows = []
    for family, detail, eligible, supplementary in [
        ("LSTM", "Unidirectional S1/S5/S7; BiLSTM S2; variants retained", "Candidate; compare source configurations and preserve variants", "S3 author code, not counted"),
        ("GRU", "Standalone GRU in S1/S2", "Candidate; no automatic exclusion for sharing recurrence with LSTM", ""),
        ("TCN", "Single-branch causal dilated baseline S1; not a shallow generic CNN1D", "Candidate; published winner large; no claim of minimum complexity", ""),
        ("Transformer encoder", "S2 encoder comparator and S6 pure-OHLCV Model1; not nine studies", "Candidate; both supporting studies grey; S6 date/configuration incomplete", ""),
        ("MLP no lineal", "No independent nonlinear MLP verified; S7 Dense16 is a head inside LSTM", "Not verified as independent baseline in this cohort", ""),
        ("CNN1D simple", "Do not equate with TCN or CNN-LSTM hybrid", "Not verified as independent baseline in this cohort", ""),
        ("Fully-connected lineal", "S5 Flatten->Dense1, no nonlinear hidden layer", "Context only; not a DL baseline slot", ""),
    ]:
        ids = families.get(family, set())
        if family == "Fully-connected lineal":
            ids = {r["study_id"] for r in rows if r["family"] == "Fully-connected linear"}
        frequency_rows.append({"family": family, "variant_details": detail,
            "verified_full_text_studies": ";".join(sorted(ids)), "count": len(ids),
            "denominator": len(studies), "proportion": len(ids)/len(studies),
            "supplementary_author_code": supplementary, "eligible_for_final_trio": eligible})
    frequency_buffer = io.StringIO(newline="")
    frequency_writer = csv.DictWriter(frequency_buffer, fieldnames=list(frequency_rows[0]), lineterminator="\n")
    frequency_writer.writeheader()
    frequency_writer.writerows(frequency_rows)
    return {RESULT: buffer.getvalue().encode(),
            SUMMARY: (json.dumps(summary, ensure_ascii=False, indent=2)+"\n").encode(),
            FREQUENCIES: frequency_buffer.getvalue().encode()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing artefacts without writing")
    args = parser.parse_args()
    folder = ROOT/"references/literature_e1"
    values = render(folder/NAME)
    for filename, data in values.items():
        for parent in (folder, ROOT/"book/_static/literature_e1"):
            path = parent/filename
            if args.check:
                assert path.read_bytes() == data, path
            else:
                path.write_bytes(data)
    print(json.dumps({"check_only": args.check, "files": list(values),
                      "models_executed": 0, "market_and_test_files_opened": 0}))


if __name__ == "__main__":
    main()
