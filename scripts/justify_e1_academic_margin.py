"""Contextualize E1's fixed academic margin using public development only.

This is a post-execution explanation, not a new preregistration or an estimate
of financial utility. It never fits a model, reads reserve/test observations,
changes delta/R, performs a new hypothesis test or selects a better threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STAGE = "e1_academic_margin_20261106"
EXPECTED_SEEDS = tuple(range(4101, 4111))
FIXED_DELTA = 0.00025772842664981106
INPUTS = (
    "results/gru_lstm_tcn_20261105/run_metrics_cv.csv",
    "results/gru_lstm_tcn_20261105/monthly_summary_cv.csv",
    "results/gru_lstm_tcn_20261105/cost_summary_cv.csv",
    "results/gru_lstm_tcn_20261105/future_margin.json",
    "results/eda_development/daily_volatility_development.csv",
    "results/gru_lstm_tcn_planned_20261105/selected_recipes.json",
)
SOURCE = {
    "author": "Daniël Lakens", "year": 2022,
    "title": "Sample Size Justification", "doi": "10.1525/collabra.33267",
    "primary_url": "https://online.ucpress.edu/collabra/article/8/1/33267/120491/Sample-Size-Justification",
    "access_level": "primary_relevant_sections_read",
    "sections": ["Six Ways to Evaluate Which Effect Sizes are Interesting",
                 "What is the Smallest Effect Size of Interest?",
                 "The Minimal Statistically Detectable Effect"],
    "contribution": "Distinguishes a theoretically/practically interesting effect from detectability; allows an explicitly reasoned, context-specific interest threshold.",
    "does_not_support": "The author does not prescribe 5% RMSE for cryptocurrency forecasting.",
    "original_bytes_sha256": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_csv(root, name):
    if name not in INPUTS or "_test" in name or "reserved" in name:
        raise ValueError("Only explicit public development inputs are permitted")
    return pd.read_csv(root / name, float_precision="round_trip")


def seed_panel(frame):
    selected = frame.loc[frame.history_days.eq(7)].copy()
    neural = selected.loc[selected.model.isin(("GRU", "LSTM", "TCN"))]
    if (neural.duplicated(["model", "seed"]).any()
            or not neural.n_dates.eq(366).all()
            or not np.isfinite(neural.rmse).all() or neural.rmse.lt(0).any()):
        raise ValueError("Need unique finite D7 seed metrics on all 366 CV dates")
    panel = neural.pivot(index="seed", columns="model", values="rmse").sort_index()
    if (set(panel.columns) != {"GRU", "LSTM", "TCN"}
            or tuple(panel.index) != EXPECTED_SEEDS or panel.isna().any().any()):
        raise ValueError("Need paired seeds 4101–4110 for every family")
    persistence = selected.loc[selected.model.eq("Persistence")]
    if len(persistence) != 1 or not persistence.n_dates.eq(366).all():
        raise ValueError("Need one deterministic persistence comparator")
    return panel, float(persistence.iloc[0].rmse)


def target_profiles(frame):
    days = pd.to_datetime(frame.market_day_utc, utc=True)
    if not days.dt.year.isin((2023, 2024)).all():
        raise ValueError("Target profiling may use only development 2023–2024")
    profiles = {}
    for year in (2023, 2024):
        values = frame.loc[days.dt.year.eq(year), "sigma_7"].dropna().to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid public sigma7 development profile")
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        profiles[str(year)] = {"n_valid": len(values), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)), "median": float(median),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25)}
    if profiles["2024"]["n_valid"] != 366:
        raise ValueError("Expected exactly the 366 fixed CV target dates")
    return profiles


def paired_context(panel, persistence, delta):
    rows = []
    for reference in ("LSTM", "TCN", "Persistence"):
        other = panel[reference].to_numpy(float) if reference != "Persistence" else persistence
        differences = other - panel.GRU.to_numpy(float)
        n = len(differences)
        sd = float(differences.std(ddof=1))
        if sd <= 0:
            raise ValueError("A paired difference has zero variability")
        mean = float(differences.mean())
        half = float(stats.t.ppf(.975, n - 1) * sd / np.sqrt(n))
        rows.append({"reference": reference, "model": "GRU/D7", "n_seed_pairs": n,
            "difference_reference_minus_gru": mean,
            "difference_sample_sd": sd, "marginal_ci95_lower": mean - half,
            "marginal_ci95_upper": mean + half, "descriptive_d_z": mean / sd,
            "fixed_delta_over_difference_sd": delta / sd,
            "new_hypothesis_test": False,
            "scope": "Initialization-conditional context on the same 366 CV2024 dates; no temporal or familywise CI."})
    return rows


def build(root=ROOT, output=None, mirror=True):
    root = Path(root).resolve()
    default_output = root / "results" / STAGE
    output = Path(output).resolve() if output else default_output
    if output.is_relative_to(root) and output != default_output:
        raise ValueError("A custom output must be outside the checkout; preserve all existing studies")
    margin = json.loads((root / INPUTS[3]).read_text())
    anchor = float(margin["new_baseline"]["cv_mean_rmse"])
    delta = float(margin["absolute_delta"])
    if delta != FIXED_DELTA or float(margin["relative_margin"]) != .05:
        raise ValueError("The recorded academic margin must remain unchanged")
    panel, persistence = seed_panel(read_csv(root, INPUTS[0]))
    if not np.isclose(panel.GRU.mean(), anchor, rtol=0, atol=1e-15):
        raise ValueError("Stored development anchor disagrees with public seed metrics")
    profiles = target_profiles(read_csv(root, INPUTS[4]))
    baseline_sd = float(panel.GRU.std(ddof=1))
    if baseline_sd <= 0:
        raise ValueError("Baseline variability must be positive")
    ci_half = float(stats.t.ppf(.975, len(panel) - 1) * baseline_sd / np.sqrt(len(panel)))
    months = read_csv(root, INPUTS[1])
    months = months.loc[months.history_days.eq(7) & months.model.eq("GRU") & months.metric.eq("rmse")]
    if len(months) != 12 or months.fold_id.nunique() != 12:
        raise ValueError("Expected the twelve fixed 2024 monthly summaries")
    context = []
    for relative in (.025, .05, .1):
        absolute = delta if relative == .05 else relative * anchor
        context.append({"relative_margin": relative, "absolute_delta": absolute,
            "volatility_percentage_points": absolute * 100,
            "delta_over_baseline_seed_sd": absolute / baseline_sd,
            "delta_over_baseline_seed_ci95_halfwidth": absolute / ci_half,
            "delta_over_target_2024_sd": absolute / profiles["2024"]["sample_sd"],
            "delta_over_target_2024_iqr": absolute / profiles["2024"]["iqr"],
            "registered_choice": relative == .05,
            "interpretation": "Scale sensitivity only; no threshold selected from these diagnostics."})
    costs = read_csv(root, INPUTS[2])
    costs = costs.loc[costs.history_days.eq(7), ["model", "parameters", "training_seconds_mean", "inference_seconds_mean", "timing_scope"]]
    final_recipes = json.loads((root / INPUTS[5]).read_text())
    if final_recipes["baseline_model"] != "GRU":
        raise ValueError("The development-selected final baseline must remain GRU")
    final_parameters = {model: int(recipe["parameters"])
                        for model, recipe in final_recipes["recipes"].items()}
    pairs = paired_context(panel, persistence, delta)
    summary = {
        "schema_version": 1, "analysis_date_local": "2026-11-06",
        "timezone_for_date": "America/Bogota",
        "source_commit_before_supplement": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "purpose": "Explain the already fixed 5% threshold for academic model comparison; retrospective contextualization, not a new preregistration.",
        "application_context_confirmed_by_author": "Academic comparison of models for E1",
        "development_only": True, "new_model_fits": 0, "new_forecasts": 0,
        "reserve_or_test_observations_read": False,
        "delta_changed": False, "R_changed": False,
        "new_hypothesis_tests": 0, "financial_utility_claimed": False,
        "reference": {"model": "GRU", "history_days": 7, "phase": "Initial exploratory CV2024",
            "n_seeds": len(panel), "n_dates": 366, "mean_rmse_anchor": anchor,
            "seed_sd_rmse": baseline_sd, "seed_ci95_halfwidth_rmse": ci_half,
            "seed_ci95_scope": "Marginal t interval conditional on fixed dates; not a market uncertainty interval.",
            "persistence_rmse_same_dates": persistence},
        "registered_relative_margin": .05, "registered_absolute_delta": delta,
        "target_profiles": profiles,
        "monthly_gru_rmse_context": {"n_months": 12,
            "minimum": float(months["mean"].min()), "maximum": float(months["mean"].max()),
            "sample_sd_of_monthly_means": float(months["mean"].std(ddof=1)),
            "scope": "Descriptive calendar heterogeneity, unequal month lengths; months are not independent markets and this is not an IC."},
        "initial_recipe_cost_context": costs.to_dict(orient="records"),
        "final_recipe_parameter_context": final_parameters,
        "final_recipe_complexity_scope": "Selected with development January–October2024; these are configuration metadata, not reserve metrics. They are not paired with the earlier 366-day CV errors as if they were the same experiment.",
        "cost_interpretation": "Parameters and wall times describe the initial recipes on the common CV; wall times reflect documented concurrency and are not causal architecture speed comparisons or a cost-derived optimal delta.",
        "academic_acceptance_rationale": [
            "Require a non-negligible reduction of predictive error before calling added architectural complexity an academic improvement.",
            "The existing 5% rule is a transparent intermediate demand between the 2.5% and 10% sensitivity scenarios; it is not inferred as a uniquely correct value from these data.",
            "Only interval-supported effects and a separately justified model-complexity budget can support a future advancement claim; any future comparison needs its own registered paired design and calendar uncertainty assessment.",
            "The registered absolute delta for the completed neural study and relative gains against each separate reference remain different quantities."],
        "interpretation_limits": [
            "Observed label spread is not measurement noise or irreducible prediction error.",
            "The seed SD and CI contextualize scale but do not define relevance or prove that delta exceeds temporal uncertainty.",
            "The paired differences are descriptive initialization-conditional context; no p-values, equivalence claim or new winner selection is added.",
            "No reported 2025–2026 evaluation metrics or individual evaluation observations are used.",
            "This text supplies a posterior contextual argument; it cannot be attributed to the original preregistration."],
        "methodological_source": SOURCE,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    readme = """# Contexto académico del margen fijo de E1

El autor confirmó que el uso es la comparación académica de modelos. Este suplemento
explica el margen ya fijado con agregados y etiquetas públicas de desarrollo 2023–2024.
Es una fundamentación posterior: no modifica δ, R, las recetas, el baseline o los
resultados de los estudios cerrados y no utiliza observaciones de reserva/test.

`margin_context.csv` normaliza 2,5/5/10% respecto del ancla original GRU/D7-CV, su
variabilidad de inicialización y la dispersión observada de σ7 en 2024. Esa dispersión
no estima ruido de medición ni error irreducible. `paired_development.csv` contextualiza
diferencias de las recetas iniciales sobre las mismas 366 fechas; sus IC son marginales
y condicionados a semillas, sin pruebas nuevas ni certificación de estabilidad temporal.

La elección del 5% es normativa y académica: exige una reducción no trivial del error
para considerar una mayor complejidad como avance, entre escenarios menor y mayor.
Las razones y escalas permiten deliberar esa decisión, pero no prueban un 5% óptimo.
Los costes son los ya publicados para las recetas iniciales, bajo concurrencia CPU;
no se convierten en precios ni se usan para ajustar un umbral al resultado conocido.

La fuente metodológica consultada es Lakens (2022), *Sample Size Justification*,
https://doi.org/10.1525/collabra.33267, apartados sobre efecto de interés y detectabilidad.
Distingue interés sustantivo y detección; no prescribe el 5% para este problema.

Reproducción desde la raíz, con `requirements-e1-supplement.lock.txt`:

```bash
python scripts/justify_e1_academic_margin.py
```

Las entradas admitidas se enumeran de forma cerrada en el script y sus hashes constan
en `source_manifest.json`. No hay entrenamiento, lectura de archivos de mercado
reservados, nuevas predicciones, búsqueda de umbrales ni pruebas de hipótesis.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_bytes(json_bytes(summary))
    pd.DataFrame(context).to_csv(output / "margin_context.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_development.csv", index=False)
    (output / "README.md").write_text(readme, encoding="utf-8")
    public_names = ("README.md", "summary.json", "margin_context.csv", "paired_development.csv")
    manifest = {"schema_version": 1,
        "input_files": {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size}
                        for name in INPUTS},
        "generator": {"path": "scripts/justify_e1_academic_margin.py",
                      "sha256": sha(Path(__file__))},
        "files": {name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
                  for name in public_names}}
    (output / "source_manifest.json").write_bytes(json_bytes(manifest))
    if mirror:
        destination = root / "book/_static" / STAGE
        destination.mkdir(parents=True, exist_ok=True)
        for name in (*public_names, "source_manifest.json"):
            shutil.copyfile(output / name, destination / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    s = build(output=args.output_dir, mirror=not args.no_mirror)
    print(json.dumps({"delta": s["registered_absolute_delta"],
        "development_only": s["development_only"], "new_fits": s["new_model_fits"]}))


if __name__ == "__main__":
    main()
