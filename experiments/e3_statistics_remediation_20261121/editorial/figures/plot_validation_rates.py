"""Plot every declared controlled case from saved summaries; no inference rerun.

AI assistance declared. The figure is a post-result display of the reported
individual Monte Carlo intervals. It does not create bootstrap draws, change
acceptance rules, certify a market method or select favorable scenarios.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VARIANTS = [
    ("MCS_legacy_frozen_control_same_draws", "v1 congelado", "#5b6573", -0.18),
    ("MCS_fixed_studentizer_development_control", "Covarianza nueva, escala fija", "#c57919", 0.0),
    ("MCS_bootstrap_t", "Propuesta bootstrap-t", "#087f8c", 0.18),
]
CASE_NAMES = {
    "EPA_fresh_nominal": "EPA ordinaria",
    "EPA_fresh_strong_dependence": "EPA: dependencia fuerte",
    "EPA_fresh_seed_only": "EPA: variación de semillas",
    "fresh_alternative": "Alternativa",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(plan_path: Path, summary_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    summary = json.loads(summary_path.read_text())
    declared = plan["cases"]
    if len(declared) != 4 or len({c["id"] for c in declared}) != 4:
        raise ValueError("This display expects all four declared scenarios")
    actual = {c["id"]: c for c in summary["cases"]}
    if len(actual) != len(summary["cases"]) or set(actual) - {c["id"] for c in declared}:
        raise ValueError("Duplicate or undeclared scenario in saved summary")
    if out.exists():
        raise ValueError("Use a new figure-output directory; do not overwrite a display")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.hashsalt": "e3-mcs-fresh-controlled-20261121"})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharey=True)
    plotted = []
    for ax, case in zip(axes.ravel(), declared):
        observed = actual.get(case["id"])
        completed = 0 if observed is None else observed["completed_replicates"]
        if observed is not None and (observed["planned_replicates"] != case["replicates"]
                                     or observed["shape"] != [case["R"], case["T"], case["M"]]):
            raise ValueError("Saved scenario does not match the declared plan")
        equal_null = case["theta"] == "equal"
        ax.set_title(CASE_NAMES.get(case["id"], case["id"]) +
                     f"\nn={completed}/{case['replicates']}; R={case['R']}, T={case['T']}, M={case['M']}")
        ax.grid(axis="y", color="#d9dfe4", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if equal_null:
            ax.axhline(0.10, color="#9b2948", linewidth=1.2, linestyle="--")
            ax.axhline(0.13, color="#a6630b", linewidth=1.0, linestyle=":")
        else:
            ax.text(.02, .05, "EPA falsa: rechazo bajo alternativa\nNo es tamaño ni potencia de tres pruebas",
                    transform=ax.transAxes, fontsize=8, va="bottom", color="#434b52")
        if completed == 0:
            ax.text(.5, .5, "Sin réplicas completadas", ha="center", va="center", transform=ax.transAxes)
        else:
            if observed["global_EPA_null_true"] != equal_null:
                raise ValueError("Saved null label differs from the plan")
            for key, label, color, offset in VARIANTS:
                entry = observed[key]["initial_global_rejection"]
                k, n = entry["events"], entry["replicates"]
                low, high = entry["CI95"]
                if n != completed or not 0 <= k <= n or not 0 <= low <= entry["rate"] <= high <= 1:
                    raise ValueError("Invalid saved Monte Carlo interval/count")
                rate = entry["rate"]
                ax.errorbar([1 + offset], [rate], yerr=[[rate-low], [high-rate]],
                            fmt="o", color=color, capsize=5, markersize=6, linewidth=1.6)
                ax.annotate(f"{k}/{n}", (1 + offset, high), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=8, color=color)
                plotted.append({"case": case["id"], "variant": key, "events": k,
                                "replicates": n, "rate": rate, "CI95": [low, high],
                                "global_EPA_null_true": equal_null,
                                "reported_screen": entry.get("screen"),
                                "unsupported_replicates": observed[key]["unsupported_replicates"]})
        ax.set_xlim(.55, 1.45)
        ax.set_xticks([])
        ax.set_ylim(-.015, 1.08)
        ax.set_ylabel("Proporción de rechazos globales")
    handles = [Line2D([0], [0], color=c, marker="o", linestyle="", label=n)
               for _, n, c, _ in VARIANTS]
    handles += [Line2D([0], [0], color="#9b2948", linestyle="--", label="Nominal EPA: 0,10"),
                Line2D([0], [0], color="#a6630b", linestyle=":", label="Cribado: límite superior IC ≤0,13")]
    fig.suptitle("MCS funcional: validación controlada nueva y controles sobre los mismos sorteos", fontsize=13)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .015,
             "Barras: IC95 % binomiales individuales reportados; no simultáneos. "
             "Las variantes no son tres procedimientos primarios distintos.",
             ha="center", fontsize=8, color="#434b52")
    fig.tight_layout(rect=(0, .15, 1, .93))
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(out / ("mcs_fresh_global_rates." + extension), dpi=170, **kwargs)
    plt.close(fig)
    record = {"kind": "post_result_display_of_reported_MC_intervals_not_new_statistical_analysis",
              "AI_assistance_declared": True, "plan_SHA256": sha(plan_path),
              "summary_SHA256": sha(summary_path), "all_declared_cases_displayed": len(declared),
              "plot_source_SHA256": sha(Path(__file__)),
              "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
              "declared_case_order_preserved": True, "plotted_records": plotted,
              "new_model_fits": 0, "new_market_predictions": 0,
              "new_Monte_Carlo_replicates": 0, "new_bootstrap_draws": 0,
              "individual_not_simultaneous_intervals": True,
              "third_primary_method_accredited": False, "R3_accredited": False,
              "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                        for p in sorted(out.iterdir()) if p.is_file()]}
    (out / "FIGURE_RECEIPT.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = draw(args.plan, args.summary, args.out)
    print(json.dumps({"figures": 1, "files": len(result["files"]),
                      "new_Monte_Carlo_replicates": 0, "R3_accredited": False}))
