"""Describe stored CV stopping epochs without trying another training policy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e1_audit_improvements_20261104/traceability"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sources = [ROOT / "results/dl_r2_cv_20261104/fold_metrics.csv",
               ROOT / "results/filipe_multihistory_20261104/fold_metrics.csv"]
    old, new = [pd.read_csv(path) for path in sources]
    old = old.loc[old.variant.eq("relative_window") & old.history_days.eq(7)]
    frame = pd.concat([old,new],ignore_index=True)
    if len(frame) != 1440 or frame.duplicated(["history_days","model","seed","fold_id"]).any():
        raise ValueError("The complete four-history primary CV is required")
    monthly = frame.groupby(["history_days","model","fold_id"],sort=True).best_epoch.agg(
        n_seeds="count",mean_epoch="mean",median_epoch="median",min_epoch="min",max_epoch="max").reset_index()
    records=[]
    for (history,model),part in frame.groupby(["history_days","model"],sort=True):
        december=part.loc[part.fold_id.eq("2024-12"),"best_epoch"]
        records.append({"history_days":int(history),"model":model,"CV_months":12,
            "seed_month_records":len(part),"all_months_mean_epoch":float(part.best_epoch.mean()),
            "all_months_median_epoch":float(part.best_epoch.median()),
            "all_months_min_epoch":int(part.best_epoch.min()),"all_months_max_epoch":int(part.best_epoch.max()),
            "fraction_epoch_le5":float(part.best_epoch.le(5).mean()),
            "december_median_epoch":float(december.median()),"december_min_epoch":int(december.min()),
            "december_max_epoch":int(december.max()),
            "scope":"Stored early-stopping metadata only; no alternative policy fitted or selected using test"})
    OUT.mkdir(parents=True,exist_ok=True)
    monthly.to_csv(OUT/"epoch_summary_by_month.csv",index=False,lineterminator="\n")
    pd.DataFrame(records).to_csv(OUT/"epoch_policy_summary.csv",index=False,lineterminator="\n")
    (OUT/"README.md").write_text("""# Duración del ajuste final: trazabilidad

Se resumen las épocas elegidas en los doce cortes de CV ya ejecutados para cada historia y familia. Cada fila mensual describe diez semillas; los 120 registros semilla–mes por familia/historia no son 120 mercados independientes.

El ajuste final utilizó, por protocolo congelado antes del test, la época de diciembre de 2024 de su misma familia, historia y semilla. La tabla permite examinar cuánto difiere ese mes del resto del desarrollo. No se entrena una política alternativa ni se demuestra que otra regla habría mejorado el test. Una regla agregada o una validación interna final son posibilidades para un experimento futuro registrado con evaluación independiente.
""")
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
        "code_sha256":digest(__file__),"CV_records":1440,"monthly_rows":len(monthly),"configuration_rows":len(records),
        "new_model_fits":0,"checkpoint_forward_calls":0,"raw_market_archives_opened":0,
        "output_sha256":{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!="summary.json"}}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"stored_epoch_policy_described","monthly_rows":len(monthly),"configurations":len(records)}))


if __name__ == "__main__":main()
