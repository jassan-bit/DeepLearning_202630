"""Draw two aggregate-only P0 development panels; never fit or read predictions."""
from pathlib import Path
import csv
from hashlib import sha256
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "book_e2/_static/remediation_pilot_results"


def main():
    sources = {name: (OUTPUT / name).read_bytes() for name in
               ("aggregate_metrics.csv", "gap.csv", "summary.json")}
    with (OUTPUT / "aggregate_metrics.csv").open() as stream:
        metrics = [r for r in csv.DictReader(stream) if r["phase"] == "oof" and r["block_days"] == "7"]
    with (OUTPUT / "gap.csv").open() as stream:
        gaps = [r for r in csv.DictReader(stream) if r["phase"] == "oof" and r["block_days"] == "7"]
    assert [r["model"] for r in metrics] == ["TimeMixer", "iTransformer", "PatchTST", "GRU"]
    assert [r["model"] for r in gaps] == ["TimeMixer", "iTransformer", "PatchTST"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    for axis, records, field in [(axes[0], metrics, "mean_seed_RMSE"),
                                  (axes[1], gaps, "effect_RMSE_high_minus_rest")]:
        values = [float(r[field]) for r in records]
        bounds = [json.loads(r["ci95"]) for r in records]
        axis.errorbar(range(len(records)), values,
                      yerr=[[v-b[0] for v, b in zip(values, bounds)],
                            [b[1]-v for v, b in zip(values, bounds)]],
                      fmt="o", capsize=5, color="#155e75", markersize=6)
        axis.set_xticks(range(len(records)), [r["model"] for r in records], rotation=15)
        axis.set_ylabel("Unidades de volatilidad diaria σ₇")
        axis.grid(axis="y", alpha=.25)
        axis.spines[["right", "top"]].set_visible(False)
        axis.set_xlim(-.5, len(records)-.5)
    axes[0].set_title("RMSE media por semilla · 305 fechas OOF")
    axes[1].set_title("Brecha RMSE: alta volatilidad − resto")
    axes[1].axhline(float(gaps[0]["delta"]), color="#b45309", linestyle="--", label="Margen δ heredado")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Piloto Bitcoin: recetas fijas de desarrollo, sin HPO\nIC95 nominales por bloques de 7 días; no benchmark final", fontsize=12)
    destination = OUTPUT / "pilot_summary.png"
    fig.savefig(destination, dpi=180, metadata={"Description": "Aggregate-only registered P0 OOF diagnostic"})
    plt.close(fig)
    receipt = {"generator": "scripts/plot_e2_remediation_results.py",
               "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_sha256": {name: sha256(data).hexdigest() for name, data in sources.items()},
               "output_sha256": {destination.name: sha256(destination.read_bytes()).hexdigest()},
               "market_files_read": 0, "individual_predictions_read": 0, "model_fits": 0}
    (OUTPUT / "figure_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
