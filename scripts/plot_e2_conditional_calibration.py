"""Render saved conditional-coverage diagnostics; no model inference."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP3=("LSTM-autoencoder con predictor","GRU32","ARIMA")
NAMES={TOP3[0]:"SAE",TOP3[1]:"GRU32",TOP3[2]:"ARIMA"}
POLICIES=("original_frozen","rolling53_fixed_alpha","ACI53_gamma005")
LABELS=("Original congelado","Móvil 53, α=0,10","ACI 53, γ=0,005")
COLORS=("#285a8e","#b5672a","#21856b")


def render(folder):
    folder=Path(folder)
    models=pd.read_csv(folder/"model_metrics.csv")
    ci=pd.read_csv(folder/"conditional_coverage_CI.csv")
    ci=ci[ci.block_days.eq(14)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    x=np.arange(3)
    for index,(policy,label,color) in enumerate(zip(POLICIES,LABELS,COLORS)):
        rows=models[models.policy.eq(policy)].set_index("model").loc[list(TOP3)]
        intervals=ci[ci.policy.eq(policy)&ci.group.eq("global")].set_index("model").loc[list(TOP3)]
        at=x+(index-1)*.24
        axes[0].bar(at,100*rows.coverage,.22,label=label,color=color)
        axes[0].vlines(at,100*intervals.CI95_low,100*intervals.CI95_high,color="#222",lw=.9)
        axes[1].bar(at,rows.mean_width,.22,color=color)
    axes[0].axhline(90,color="#222",ls="--",lw=1)
    axes[0].set_ylabel("Cobertura empírica (%) e IC temporal marginal")
    axes[0].set_ylim(75,100)
    axes[1].set_ylabel("Longitud media en unidades de σ₇")
    axes[0].set_title("Cobertura global: calendario compartido")
    axes[1].set_title("Eficiencia: longitud media del intervalo")
    for ax in axes:
        ax.set_xticks(x,[NAMES[m] for m in TOP3]);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.text(.5,.01,"Métodos posteriores; pesos fijos. IC por bloques14 (B=10.000), sin garantía de cobertura condicional.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.90))
    for ext in ("svg","png"):fig.savefig(folder/("top3_global_coverage_width."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    for ax,model in zip(axes,TOP3):
        for index,(policy,label,color) in enumerate(zip(POLICIES,LABELS,COLORS)):
            rows=ci[ci.model.eq(model)&ci.policy.eq(policy)&ci.group.str.startswith("prediction_")].set_index("group")
            rows=rows.loc[["prediction_low","prediction_middle","prediction_high"]]
            at=x+(index-1)*.045
            ax.plot(at,100*rows.coverage,marker="o",lw=1.25,color=color,label=label)
            ax.vlines(at,100*rows.CI95_low,100*rows.CI95_high,color=color,lw=1)
        ax.axhline(90,color="#222",ls="--",lw=1)
        ax.set_title(NAMES[model]);ax.set_xticks(x,["Bajo","Medio","Alto"])
        ax.set_xlabel("Rango del pronóstico, cortes de calibración")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Cobertura por rango (%) e IC marginal")
    axes[0].set_ylim(65,101)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.text(.5,.01,"Las frecuencias globales próximas al90 % no acreditan cobertura del90 % en cada rango.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.90))
    for ext in ("svg","png"):fig.savefig(folder/("top3_prediction_range_coverage."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    receipt={"method":"matplotlib saved CSV only","new_model_fits":0,"new_model_inference":False,
        "plot_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs":{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ("model_metrics.csv","conditional_coverage_CI.csv")},
        "figures":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in (".svg",".png")}}
    (folder/"figure_manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--folder",type=Path,required=True)
    render(parser.parse_args().folder)
