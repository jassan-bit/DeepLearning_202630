"""Document an algebraic CPU IUT bound from existing cost-only results.

No p-value is estimated, no predictive gain is supplied, and no IUT is executed.
The bound applies only to a compatible CPU hypothesis using this cost component.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "book_e2/_static/e2_corrections_20261115/cpu_diagnostic"
OUTPUT = ROOT / "book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer")
VERSION = "e2-rq3-necessary-component-v1"


def require(condition, reason):
    if not condition: raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_cost_rows(rows):
    require(len(rows) == 10, "Exactly ten existing cost contrasts required")
    indexed = {}
    for row in rows:
        candidate = row["candidate"]
        require(candidate in MODELS and candidate not in indexed, "Unknown or duplicate candidate")
        require(row["reference"] == "GRU32", "Wrong necessary-component reference")
        value = float(row["p_cost_lower_raw"])
        require(math.isfinite(value) and 0 <= value <= 1, "Missing/nonfinite/invalid cost p-value")
        require(int(row["blocks"]) == 100 and int(row["seeds_per_block"]) == 6,
                "Cost design differs from the recorded CPU experiment")
        indexed[candidate] = value
    require(set(indexed) == set(MODELS), "Missing candidate")
    return indexed


def derive_bounds(rows, *, alpha=.05):
    indexed = validate_cost_rows(rows)
    return [{"candidate": model, "reference": "GRU32", "p_cost_CPU_existing": indexed[model],
             "CPU_predictive_gain_p": None, "IUT_lower_bound": indexed[model],
             "lower_bound_reaches_one": indexed[model] == 1.,
             "compatible_CPU_IUT_rejection_excluded_by_cost": indexed[model] > alpha,
             "decision": "rejection_excluded_for_compatible_CPU_IUT" if indexed[model] > alpha
                         else "undetermined_from_necessary_component",
             "full_CPU_IUT_executed": False, "original_714_IUT_executed": False,
             "IUT_p_value_estimated": False, "GPU_predictive_gain_used": False,
             "scope": "conditional algebraic bound for this CPU cost hypothesis and any valid compatible CPU gain component"}
            for model in MODELS]


def verified_inputs():
    paths = {
        "cost_results": CPU / "statistics/results/CPU_cost_only_Holm10.csv",
        "statistics_manifest": CPU / "statistics/results/manifest.json",
        "statistics_receipt": CPU / "statistics/results/execution_receipt.json",
        "statistics_plan": CPU / "statistics/statistical_plan_v3.json",
        "inference_plan": CPU / "plan_v3.json",
        "latency_report": CPU / "latency/latency_report.json",
        "raw_CPU_latency": CPU / "latency/raw_cpu_latency.csv",
    }
    require(all(p.is_file() for p in paths.values()), "Existing CPU evidence is incomplete")
    receipt = json.loads(paths["statistics_receipt"].read_text())
    plan = json.loads(paths["statistics_plan"].read_text())
    inference = json.loads(paths["inference_plan"].read_text())
    latency = json.loads(paths["latency_report"].read_text())
    manifest = json.loads(paths["statistics_manifest"].read_text())
    require(receipt["statistical_plan_sha256"] == sha(paths["statistics_plan"])
            and receipt["inference_plan_sha256"] == sha(paths["inference_plan"])
            and plan["inference_plan_sha256"] == sha(paths["inference_plan"]), "Plan linkage changed")
    require(sha(paths["cost_results"]) == manifest["files"]["CPU_cost_only_Holm10.csv"]["sha256"], "Published cost result changed")
    require(receipt["latency_report_sha256"] == sha(paths["latency_report"]), "Timing report changed")
    require(latency["status"] == "completed_CPU_latency_diagnostic" and latency["states_unchanged"]
            and latency["original_parity_gate_passed"] is False, "Wrong timing scope or state")
    require(latency["statistical_plan_sha256"] == sha(paths["statistics_plan"])
            and latency["inference_plan_sha256"] == sha(paths["inference_plan"]), "Timing plan linkage changed")
    raw_record = latency["all_output_files"]["raw_cpu_latency.csv"]
    require(sha(paths["raw_CPU_latency"]) == raw_record["sha256"]
            and paths["raw_CPU_latency"].stat().st_size == raw_record["bytes"], "Raw timing bytes changed")
    require(plan["cost_analysis"]["reference"] == "GRU32"
            and tuple(plan["cost_analysis"]["candidates"]) == MODELS
            and plan["cost_analysis"]["blocks"] == 100
            and plan["cost_analysis"]["replicates"] == 20000, "Different necessary cost component")
    require(latency["raw_timed_operations"] == 12000 and latency["paired_observations"] == 6000,
            "Incomplete timing experiment")
    records = {name: {"repository_path": path.relative_to(ROOT).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in paths.items()}
    rows = list(csv.DictReader(paths["cost_results"].open()))
    return rows, records, plan, inference


def execute(output):
    require(not output.exists(), "Use a new evidence directory; do not overwrite")
    started = datetime.now(timezone.utc).isoformat()
    rows, sources, plan, inference = verified_inputs()
    bounds = derive_bounds(rows)
    output.mkdir(parents=True)
    with (output / "necessary_component_bounds_10.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bounds[0]))
        writer.writeheader()
        writer.writerows(bounds)
    proof = {
        "version": VERSION, "scope": "Algebraic necessary-component deduction from the specific measured CPU experiment, not a completed IUT",
        "existing_cost_H0": "mean_CPU_latency_candidate-minus-GRU>=0",
        "existing_cost_H1": "mean_CPU_latency_candidate-minus-GRU<0",
        "compatible_joint_H1": "CPU_predictive_gain>delta AND the same_CPU_latency_difference<0",
        "identity": "For every valid u in[0,1], max(u,p_cost_CPU)>=p_cost_CPU",
        "all_ten_lower_bounds_equal_one": all(row["lower_bound_reaches_one"] for row in bounds),
        "observed_special_case": ("All ten existing raw cost p-values equal1; therefore each compatible IUT lower bound equals1"
                                  if all(row["lower_bound_reaches_one"] for row in bounds)
                                  else "Subunit cost p-values retain their actual lower bounds; no unit-bound or completed-IUT claim follows"),
        "multiplicity": "Conditional implication only: a family with all compatible IUT lower bounds1 cannot reject at0.05 or after Holm10; no adjusted IUT p-values estimated",
        "CPU_predictive_gain_p_values": None, "IUT_p_values_estimated": False,
        "full_CPU_IUT_executed": False, "original_714_IUT_executed": False,
        "original_RQ3_integral_method_completed": False, "GPU_predictive_gain_used": False,
        "new_statistical_tests": 0, "new_bootstrap_draws": 0, "new_fits": 0, "new_model_inference": False,
        "cost_design": plan["cost_analysis"], "CPU_measurement_policy": inference["latency_policy"],
        "source_records": sources, "bounds": bounds,
        "answer": ("Joint improvement is not established in the measured CPU environment: its necessary reduction in forward cost is not supported."
                   if all(row["compatible_CPU_IUT_rejection_excluded_by_cost"] for row in bounds)
                   else "The necessary cost component alone does not determine all joint decisions; the integral IUT remains unexecuted."),
        "limitations": ["The statement is conditional on validity of the existing approximate CPU cost test and on the same null, reference, inputs and measurement design.",
                        "It is not a claim of equivalence, impossible improvement, or performance on another hardware/backend.",
                        "No missing CPU predictive component is invented or replaced with a GPU score.",
                        "The integral original IUT remains unexecuted; closing this conditional answer does not close that methodological execution."]}
    write_json(output / "necessary_component_proof.json", proof)
    write_json(output / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
               "python": platform.python_version(), "sources": sources,
               "operation": "validate existing hashes/domains and apply max-bound identity; no new test or model computation"})
    (output / "README.md").write_text("""# RQ3: respuesta condicionada por el componente necesario de coste

No se establece mejora conjunta en el entorno CPU medido: la reducción de coste
necesaria no está respaldada. Los diez p de coste existentes son exactamente1.
Para cualquier p predictivo CPU válido u, max(u,p_coste)≥p_coste=1. Por ello una
IUT compatible con ESTE componente CPU no podría rechazar, tampoco con Holm10.

El CSV publica el límite inferior, no un p de IUT estimado. El componente
predictivo permanece ausente; el JSON lo conserva como null. No se usa precisión
GPU, no se calculan nuevos contrastes y no se ejecuta la IUT original714.
La deducción sólo aplica a las hipótesis, referencia GRU32, pesos, entradas de
desarrollo y mediciones CPU FP32/batch1 de las fuentes enlazadas por hash.
No establece equivalencia ni imposibilidad de mejora en otros entornos.

`necessary_component_proof.json` contiene la identidad, diseños y procedencia;
`necessary_component_bounds_10.csv` conserva las diez cotas y sus estados.
Cerrar esta respuesta condicionada no equivale a completar la ejecución integral
del método original de RQ3. Repetición desde el repositorio con Python estándar:

```sh
python scripts/e2_rq3_necessary_component.py --output /tmp/e2-rq3-necessary-repeat
```
""", encoding="utf-8")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", {"version": VERSION, "files": files, "new_tests": 0, "original_IUT_executed": False})
    return {"version": VERSION, "cost_rows_validated": len(bounds),
            "lower_bounds_equal_one": sum(row["lower_bound_reaches_one"] for row in bounds),
            "full_IUT_executed": False, "IUT_p_value_estimated": False, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
