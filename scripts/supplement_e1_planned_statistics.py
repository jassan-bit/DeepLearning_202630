"""Post-execution diagnostic supplement for E1's frozen D7 study.

Consumes only the published seed-level RMSE table. It never trains a network,
loads a checkpoint, reads market archives or generates predictions. The global
rank test, Q-Q plots and signed-rank sensitivities were not preregistered; they
do not replace the original t/Holm family, temporal analysis or selected model.

Run from the repository root::

    python scripts/supplement_e1_planned_statistics.py
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e1-supplement-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e1-supplement-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STUDY = "gru_lstm_tcn_planned_20261105"
SUPPLEMENT = "e1_statistics_supplement_20261106"
MODELS = ("GRU", "LSTM", "TCN")
PAIRS = tuple(combinations(MODELS, 2))
EXPECTED_SEEDS = tuple(range(8101, 8566))
PUBLIC_FILES = frozenset({"README.md", "summary.json", "friedman_global.json",
    "pairwise_sensitivity.csv", "qq_diagnostics.csv", "qq_coordinates.csv",
    "qq_differences.png", "artifact_availability.json", "source_manifest.json"})
SCOPE = ("Post-execution exploratory supplement; initialization uncertainty "
         "conditional on the same 60 historical dates and frozen recipes. "
         "Not preregistered, not independent temporal confirmation.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def paired_panel(frame):
    """Reject incomplete or silently unpaired inputs before any inference."""
    required = {"model", "seed", "rmse", "n_dates"}
    if not required <= set(frame):
        raise ValueError("Expected model, seed, rmse and n_dates columns")
    if set(frame.model) != set(MODELS) | {"Persistence"}:
        raise ValueError("Unexpected model family in source metrics")
    if frame.model.eq("Persistence").sum() != 1:
        raise ValueError("Expected one deterministic persistence row")
    selected = frame.loc[frame.model.isin(MODELS)].copy()
    if (selected.duplicated(["model", "seed"]).any()
            or not np.isfinite(selected.rmse.to_numpy(float)).all()
            or selected.rmse.lt(0).any()
            or not selected.n_dates.eq(60).all()):
        raise ValueError("Need unique finite nonnegative RMSEs on 60 dates")
    seeds = selected.seed.to_numpy(float)
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Seed identifiers must be finite integers")
    selected["seed"] = selected.seed.astype(int)
    panel = selected.pivot(index="seed", columns="model", values="rmse")
    panel = panel.reindex(columns=MODELS).sort_index()
    if panel.isna().any().any() or tuple(panel.index) != EXPECTED_SEEDS:
        raise ValueError("Every family must have exactly seeds 8101 through 8565")
    return panel


def holm(pvalues):
    """Step-down Holm for one explicitly declared family."""
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p-value")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def registered_integrity(root):
    source = root / "results" / STUDY / "source_manifest.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    mappings = (record["registered_scientific_source_sha256"],
                record["canonical_public_records_sha256"])
    checked = {}
    for mapping in mappings:
        for name, expected in mapping.items():
            actual = digest(root / name)
            if actual != expected:
                raise ValueError("Frozen source differs from recorded hash: " + name)
            checked[name] = actual
    return checked


def analyze(panel):
    n, k = panel.shape
    ranks = stats.rankdata(panel.to_numpy(float), axis=1, method="average")
    global_test = stats.friedmanchisquare(*(panel[m].to_numpy() for m in MODELS))
    global_result = {
        "test": "Friedman chi-square approximation with within-seed blocks",
        "n_blocks": n, "n_models": k, "degrees_of_freedom": k - 1,
        "chi_square": float(global_test.statistic),
        "p_value": float(global_test.pvalue),
        "kendall_W": float(global_test.statistic / (n * (k - 1))),
        "mean_ranks_lower_rmse_better": dict(zip(MODELS, ranks.mean(axis=0).tolist())),
        "blocks_with_any_exact_rmse_tie": int(sum(len(set(row)) < k for row in panel.to_numpy())),
        "null_and_assumptions": ("Exchangeability of model labels within independent seed blocks "
            "under the global null. A rejection establishes a distributional/rank difference; "
            "a median or location-only interpretation needs comparable distribution shapes. "
            "These are seed blocks, not independently sampled dates or markets."),
        "scope": SCOPE,
    }
    pair_rows, diagnostics, coordinates = [], [], []
    for a, b in PAIRS:
        differences = (panel[a] - panel[b]).to_numpy(float)
        nonzero = differences[differences != 0]
        if not len(nonzero):
            raise ValueError("All differences are zero: " + a + "/" + b)
        absolute_ranks = stats.rankdata(abs(nonzero), method="average")
        positive = float(absolute_ranks[nonzero > 0].sum())
        negative = float(absolute_ranks[nonzero < 0].sum())
        signed = stats.wilcoxon(differences, zero_method="wilcox", correction=False,
                               alternative="two-sided", method="approx")
        shapiro = stats.shapiro(differences)
        (theoretical, ordered), (slope, intercept, correlation) = stats.probplot(
            differences, dist="norm", fit=True)
        pair_rows.append({
            "model_a": a, "model_b": b, "n_paired": n,
            "n_nonzero": len(nonzero), "n_zero": n - len(nonzero),
            "mean_difference_a_minus_b": float(differences.mean()),
            "median_difference_a_minus_b": float(np.median(differences)),
            "positive_rank_sum": positive, "negative_rank_sum": negative,
            "wilcoxon_statistic_min_rank_sum": float(signed.statistic),
            "wilcoxon_z_statistic": float(signed.zstatistic),
            "p_value_two_sided_asymptotic": float(signed.pvalue),
            "matched_pairs_rank_biserial_a_minus_b": (positive - negative) / (positive + negative),
            "zero_method": "wilcox", "continuity_correction": False,
            "method": "normal approximation; average ranks for exact absolute-value ties",
            "rounding_before_ranking": "none; published CSV values used as stored",
            "scope": SCOPE,
        })
        diagnostics.append({
            "model_a": a, "model_b": b, "n_paired": n,
            "mean_difference_a_minus_b": float(differences.mean()),
            "sample_sd_difference": float(differences.std(ddof=1)),
            "median_difference_a_minus_b": float(np.median(differences)),
            "sample_skewness_bias_corrected": float(stats.skew(differences, bias=False)),
            "sample_excess_kurtosis_bias_corrected": float(stats.kurtosis(differences, bias=False)),
            "shapiro_W": float(shapiro.statistic), "shapiro_p": float(shapiro.pvalue),
            "qq_fitted_slope": float(slope), "qq_fitted_intercept": float(intercept),
            "qq_linear_correlation_descriptive": float(correlation),
        })
        coordinates.extend({"model_a": a, "model_b": b, "order": i + 1,
            "normal_theoretical_quantile": float(x),
            "ordered_difference_a_minus_b": float(y),
            "fitted_line_value": float(slope * x + intercept)}
            for i, (x, y) in enumerate(zip(theoretical, ordered)))
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_value_holm3_supplement_only"] = holm(pairwise.p_value_two_sided_asymptotic)
    pairwise["reject_holm3_at_0_05"] = pairwise.p_value_holm3_supplement_only.lt(.05)
    return global_result, pairwise, pd.DataFrame(diagnostics), pd.DataFrame(coordinates)


def figure(coordinates, destination):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for ax, (a, b) in zip(axes, PAIRS):
        values = coordinates.loc[(coordinates.model_a == a) & (coordinates.model_b == b)]
        ax.scatter(values.normal_theoretical_quantile,
                   values.ordered_difference_a_minus_b * 1000, s=11, alpha=.68,
                   color="#215f88", label="Diferencias observadas")
        ax.plot(values.normal_theoretical_quantile, values.fitted_line_value * 1000,
                color="#bb5436", linewidth=1.6, label="Recta ajustada")
        ax.set(title=f"{a} − {b}; 465 pares", xlabel="Cuantil normal teórico",
               ylabel="Diferencia RMSE × 10³")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Q–Q de diferencias entre semillas · suplemento posterior de fase II", fontsize=12)
    fig.savefig(destination, dpi=170, metadata={"Software": "E1 post-execution supplement"})
    plt.close(fig)


def availability(private_root):
    families = {}
    for model in MODELS:
        families[model] = {name: sum((private_root / "final" / f"{model}_s{seed}" / name).is_file()
                                    for seed in EXPECTED_SEEDS)
                           for name in ("history_inner.csv", "history_refit.csv", "model.pt",
                                        "preprocessing.json")}
    return {
        "inspection_date": "2026-11-06",
        "declared_default_private_root": "/workspace/e1_new_study_20261105/run",
        "declared_root_exists_in_current_workspace": private_root.is_dir(),
        "score_predictions_csv_present": (private_root / "score" / "predictions.csv").is_file(),
        "expected_final_trajectories_per_family": 465,
        "present_files_per_family": families,
        "inspection_method": "Filesystem existence checks only; no payload loaded or exported",
        "interpretation": ("Absence from this workspace does not demonstrate non-execution or "
            "loss from the author's retained evidence. Aggregate RMSEs do not reconstruct "
            "learning curves, daily predictions, weights or scalers. Temporal tests remain "
            "unreproduced from their original daily vectors in this supplement."),
    }


README = """# Suplemento estadístico posterior de E1: fase II D7

Este análisis del 06/11/2026 consume las métricas RMSE por semilla ya publicadas.
No fue prerregistrado, no cambia las pruebas primarias, el baseline GRU, las
recetas congeladas o el dimensionamiento. No entrena ni produce pronósticos.

- `friedman_global.json`: comparación global bloqueada por semilla, rangos y W de Kendall.
- `pairwise_sensitivity.csv`: tres Wilcoxon bilaterales asintóticos y Holm dentro de esta
  familia suplementaria; sumas de rangos y efecto biserial pareado con signo A−B.
- `qq_diagnostics.csv`, `qq_coordinates.csv`, `qq_differences.png`: normalidad y Q–Q
  reproducibles de las mismas 465 diferencias por par.
- `artifact_availability.json`: comprobación de disponibilidad local de evidencia privada;
  sólo existencia, sin cargar o exportar sus contenidos.
- `summary.json`, `source_manifest.json`: alcance, versiones, entradas y SHA-256.

Friedman prueba diferencias de distribuciones/rangos bajo intercambio de etiquetas
de modelo dentro de bloques independientes. Una interpretación sólo de medianas o
localización requiere formas comparables. W cuantifica concordancia del orden entre
semillas, no ganancia en RMSE ni relevancia económica.

Wilcoxon usa las diferencias como están guardadas, sin redondearlas, descarta ceros
exactos y asigna rango medio a empates absolutos. Emplea aproximación normal sin
corrección de continuidad, no p exacta. Su interpretación de localización requiere
simetría de la distribución de diferencias; no soluciona automáticamente el rechazo
de normalidad. Las asimetrías y Q–Q permanecen visibles para juzgarlo. La magnitud
biserial no es la d_z de la prueba primaria ni su misma estimación de efecto.

Los p suplementarios y los pares primarios reutilizan los mismos datos. No son
corroboraciones independientes; Holm3 sólo cubre los tres Wilcoxon de este archivo,
no todos los contrastes del informe. La comparación global es una prueba posterior
individual. No se selecciona el método con menor p ni se modifica el ranking fijado
con desarrollo. Ningún resultado demuestra independencia temporal, equivalencia,
potencia temporal o mejora práctica superior a δ.

Reproducción desde la raíz del repositorio:

```bash
python scripts/supplement_e1_planned_statistics.py
```

Dependencias numéricas y de gráficos efectivamente usadas constan en `summary.json`;
son el entorno de este suplemento posterior, distinto del entorno de entrenamiento
registrado. Sólo se necesita la tabla pública y las fuentes de identidad verificadas.
Para instalar este entorno: `python -m pip install -r requirements-e1-supplement.lock.txt`.
"""


def run(root, output, mirror=None, private_root=Path("/workspace/e1_new_study_20261105/run")):
    for destination in (output, mirror):
        if destination is not None and destination.exists():
            if any(p.is_dir() or p.name not in PUBLIC_FILES for p in destination.iterdir()):
                raise ValueError("Output must be empty or contain only this supplement's public files")
    source = root / "results" / STUDY / "run_metrics.csv"
    frozen = registered_integrity(root)
    panel = paired_panel(pd.read_csv(source))
    global_result, pairwise, diagnostics, coordinates = analyze(panel)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "friedman_global.json", global_result)
    pairwise.to_csv(output / "pairwise_sensitivity.csv", index=False)
    diagnostics.to_csv(output / "qq_diagnostics.csv", index=False)
    coordinates.to_csv(output / "qq_coordinates.csv", index=False)
    figure(coordinates, output / "qq_differences.png")
    write_json(output / "artifact_availability.json", availability(private_root))
    summary = {
        "analysis_date": "2026-11-06", "scope": SCOPE, "preregistered": False,
        "source_rows": int(len(pd.read_csv(source))), "neural_seed_blocks": len(panel),
        "same_historical_dates_per_seed": 60, "new_model_fits": 0,
        "new_predictions": 0, "baseline_changes": 0,
        "registered_sources_and_public_records_verified": len(frozen),
        "friedman_chi_square": global_result["chi_square"],
        "friedman_p_value": global_result["p_value"], "kendall_W": global_result["kendall_W"],
        "wilcoxon_rejections_holm3": int(pairwise.reject_holm3_at_0_05.sum()),
        "normality_rejections_at_0_05": int(diagnostics.shapiro_p.lt(.05).sum()),
        "seed_inference_does_not_establish_temporal_superiority": True,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(README, encoding="utf-8")
    inputs = {source.relative_to(root).as_posix(): digest(source),
              f"results/{STUDY}/source_manifest.json": digest(source.parent / "source_manifest.json"),
              "requirements-e1-supplement.lock.txt": digest(root / "requirements-e1-supplement.lock.txt")}
    files = {p.name: digest(p) for p in sorted(output.iterdir())
             if p.is_file() and p.name != "source_manifest.json"}
    write_json(output / "source_manifest.json", {
        "scope": SCOPE, "input_sha256": inputs,
        "supplement_script_sha256": digest(root / "scripts" / "supplement_e1_planned_statistics.py"),
        "registered_sources_and_records_verified_sha256": frozen,
        "files_sha256": files,
        "hash_identity_does_not_prove_training_or_temporal_reproduction": True,
    })
    if mirror is not None:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in (*files, "source_manifest.json"):
            shutil.copy2(output / name, mirror / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root / "results" / SUPPLEMENT
    mirror = None if args.no_mirror else root / "book" / "_static" / SUPPLEMENT
    print(json.dumps(run(root, output, mirror), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
