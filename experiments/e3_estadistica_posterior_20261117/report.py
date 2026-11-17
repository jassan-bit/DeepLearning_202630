"""Export readable book tables from already computed, immutable results."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys


def export(folder, destination):
    from run import csv_output, save, sha
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = json.loads((folder/"result.json").read_bytes())
    mcs = json.loads((folder/"mcs.json").read_bytes())
    effects = {r["rival"]: r for r in result["effects"]}
    original = "CNN-LSTM-atencion-aditiva E3"
    ordered = sorted(result["metrics"], key=lambda row: row["RMSE_mean"])
    rows = []
    for row in ordered:
        rows.append({"Modelo": "CNN-LSTM atención + salida aditiva (E3)" if row["model"] == original else row["model"],
            "R": row["R"], "RMSE media": f"{row['RMSE_mean']:.9f}",
            "DE entre semillas": "No estimable (R=1)" if row["RMSE_sample_SD"] is None else f"{row['RMSE_sample_SD']:.9f}",
            "MAE media": f"{row['MAE_mean']:.9f}"})
    csv_output(destination/"ranking_table.csv", rows)
    rows = []
    methods = ("paired_circular_block_bootstrap", "mean_RMSE_influence_Bartlett_HAC", "paired_seed_t_fixed_calendar")
    for verdict in result["per_rival_verdicts"]:
        model = verdict["rival"]
        tests = {r["method"]: r for r in result["primary_contrasts"] if r["rival"] == model}
        p = [tests[name]["p_holm"] for name in methods]
        effect = effects[model]; lo, hi = effect["gain_ci95_temporal"]
        rows.append({"Rival": model, "Ganancia RMSE": f"{effect['gain_RMSE']:.9f}",
            "IC95% temporal marginal": f"[{lo:.9f}; {hi:.9f}]",
            "p Holm bootstrap": f"{p[0]:.6g}", "p Holm HAC": f"{p[1]:.6g}",
            "p Holm t semillas": f"{p[2]:.6g}",
            "Tres rechazos exploratorios": "Sí" if verdict["all_three_Holm48_reject"] else "No"})
    csv_output(destination/"contrasts_table.csv", rows)
    csv_output(destination/"sensitivity_table.csv", [{"Bloque / rezagos": 14,
        "Rechazos de 48": sum(r["reject"] for r in result["primary_contrasts"]),
        "Máximo p Holm": max(r["p_holm"] for r in result["primary_contrasts"]), "Uso": "Principal"}] +
        [{"Bloque / rezagos": s["block_days"], "Rechazos de 48": sum(r["reject"] for r in s["contrasts"]),
          "Máximo p Holm": max(r["p_holm"] for r in s["contrasts"]), "Uso": "Sensibilidad; no reemplaza principal"}
         for s in result["calendar_sensitivities"]])
    csv_output(destination/"mcs_table.csv", [{"Modelo": name, "MSE diario medio": f"{mcs['mean_loss'][name]:.10g}",
        "MCS90%": "Permanece" if name in mcs["included_models"] else "Excluido",
        "p secuencial de exclusión": "" if name in mcs["included_models"] else f"{mcs['excluded_pvalues'][name]:.6g}"}
        for name in sorted(mcs["initial_models"], key=lambda n: mcs["mean_loss"][n])])
    summary = {"primary_reject": sum(r["reject"] for r in result["primary_contrasts"]),
        "primary_family_size": 48, "all16_all3": result["all_16_all_three_reject"],
        "maximum_primary_p_Holm": max(r["p_holm"] for r in result["primary_contrasts"]),
        "MAE_reject": sum(r["reject"] for r in result["MAE_secondary_contrasts"]),
        "MAE_family_size": 16, "MCS_included": mcs["included_models"], "MCS_unique_winner": False,
        "R3_verified": False, "not_confirmatory_preregistration": True,
        "source_result_sha256": sha(folder/"result.json"), "source_mcs_sha256": sha(folder/"mcs.json"),
        "export_code_sha256": sha(Path(__file__)), "plan_commit": result["execution"]["plan_commit"]}
    save(destination/"publication_summary.json", summary)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
