"""Export explicit public aggregates from the completed new-seed pilot."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NAME = "e1_filipe_pilot_20261105"
SOURCE = ROOT / "results" / NAME
DEST = ROOT / "book/_static" / NAME


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def num(value, digits=7):
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "No identificable"


def local(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo("America/Bogota")).strftime("%d/%m/%Y %H:%M:%S")


def main():
    audit = ROOT / "references/audits/revision_filipe_pilot_20261105.json"
    verified = read(audit)
    if verified.get("status") != "PASS":
        raise ValueError("Complete the independent pilot audit before public export")
    execution = read(SOURCE / "execution_summary.json")
    analysis = read(SOURCE / "analysis/summary.json")
    plan = read(SOURCE / "protocol.json")
    context = read(SOURCE / "code_manifest_before_fitting.json")
    if (execution["trajectories"] != 60 or execution["project_fits"] != 1440
            or execution["unique_validation_predictions"] != 21960
            or execution["test_files_opened"] != 0 or analysis["achieved_power_claim"]):
        raise ValueError("Unexpected pilot execution or inferential claim")
    names = ["protocol.json", "runtime.json", "code_manifest_before_fitting.json",
             "preregistration_git_receipt.json", "run_started.json", "execution_summary.json",
             "runs.csv", "fold_metrics.csv", "analysis/pilot_metrics.csv",
             "analysis/paired_dimensioning.csv", "analysis/summary.json",
             "analysis/qq_difference_panels.png", "analysis/pilot_metrics.png", "analysis/README.md"]
    for name in names:
        source, target = SOURCE / name, DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (DEST / "verification").mkdir(exist_ok=True)
    for path in [audit, audit.with_suffix(".md")]:
        shutil.copyfile(path, DEST / "verification" / path.name)
    for name, target in [("private_filipe_pilot_delivery_20261105.json", "private_delivery_receipt.json"),
                         ("replica_filipe_pilot_20261105.json", "replica_receipt.json")]:
        shutil.copyfile(ROOT / "references/audits" / name, DEST / "verification" / target)
    # Read configuration metadata only. Never copy scalers, checkpoints or forecasts.
    catalog = []
    for run in rows(SOURCE / "runs.csv"):
        directory = SOURCE / "runs" / run["run_id"]
        for path in sorted(directory.glob("*/configuration.json")):
            config = read(path)
            catalog.append({"history_days": int(run["history_days"]), "model": run["model"],
                "trajectory_seed": int(run["seed"]), "fold_id": path.parent.name,
                "width": config["width"], "learning_rate": config["learning_rate"],
                "selected_epochs": config["best_epoch"], "parameters": int(run["parameters"]),
                "inner_initialization_seed": config["inner_initialization_seed"],
                "refit_initialization_seed": config["refit_initialization_seed"],
                "configuration_sha256": sha(path)})
    if len(catalog) != 720:
        raise ValueError("Expected720 registered model-month configurations")
    with (DEST / "configuration_catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(catalog)
    tables = DEST / "tables"
    tables.mkdir(exist_ok=True)
    text = "| D | Modelo | Semillas | RMSE: media ± DE | CV (%) | IC marginal 95 % |\n|---:|---|---:|---:|---:|---|\n"
    for row in rows(SOURCE / "analysis/pilot_metrics.csv"):
        text += (f"| {row['history_days']} | {row['model']} | {row['n_seeds']} | "
                 f"{num(row['rmse_mean'])} ± {num(row['rmse_sample_sd'])} | "
                 f"{num(row['rmse_coefficient_variation_percent'],5)} | "
                 f"[{num(row['rmse_ci95_lower'])}; {num(row['rmse_ci95_upper'])}] |\n")
    text += "\nCada observación es una RMSE anual de 366 fechas por semilla. La DE y el IC describen aleatoriedad de entrenamiento condicionada a 2024; no cinco mercados independientes.\n"
    (tables / "pilot_metrics.md").write_text(text)
    text = "| D | Par A−B | DE de diferencias | δ/DE | R normal | R t puntual | R t con DE superior | MDE con R=5 |\n|---:|---|---:|---:|---:|---:|---:|---:|\n"
    for row in rows(SOURCE / "analysis/paired_dimensioning.csv"):
        text += (f"| {row['history_days']} | {row['model_a']}−{row['model_b']} | "
                 f"{num(row['paired_difference_sample_sd'])} | {num(row['standardized_planning_delta_over_sd'],5)} | "
                 f"{num(row['R_normal_approximation_ceiling'],5)} | {num(row['R_exact_noncentral_t_point_SD'],5)} | "
                 f"{num(row['R_exact_noncentral_t_upper_SD'],5)} | {num(row['MDE_at_pilot_R5_point_SD'])} |\n")
    point, upper = analysis["common_point_SD_R"], analysis["common_recommended_upper_SD_R"]
    text += (f"\nMáximo común con DE puntual: **R={point}**. Recomendación condicional usando límites superiores marginales de DE: **R={upper} semillas nuevas por modelo e historia**; las cinco del piloto no se incluyen. "
             f"Se planifica una diferencia absoluta δ={num(plan['dimensioning']['delta_absolute'],12)} frente a cero, con α=0,05/12 y potencia marginal objetivo 0,80. "
             "No es potencia alcanzada ni conjunta, ni una prueba de superar la frontera avanzada del 5 %. Los límites de DE no son simultáneos y dependen de normalidad.\n")
    text += (f"\nShapiro–Wilk p<0,05 en {analysis['normality_warning_pairs_p_lt_005']} de los 12 pares. Cinco diferencias no acreditan normalidad; la incertidumbre piloto y las alertas deben conservarse en la planificación. "
             "La tabla CSV completa conserva MDE con DE puntual/superior al R común y los supuestos.\n")
    (tables / "paired_dimensioning.md").write_text(text)
    receipt = read(SOURCE / "preregistration_git_receipt.json")
    text = "| Hito | Fecha y hora de Bogotá (UTC−5) | Evidencia |\n|---|---|---|\n"
    text += f"| Registro del protocolo | {local(plan['registered_at_utc'])} | SHA-256 `{context['protocol_sha256']}` |\n"
    text += f"| Código y protocolo publicados antes de iniciar | Antes de las nuevas trayectorias | [Commit de preregistro](https://github.com/jassan-bit/DeepLearning_202630/commit/{receipt['preregistration_commit']}) |\n"
    text += f"| Primera trayectoria nueva | {local(execution['first_new_trajectory_started_at_utc'])} | Recibos de inicio individuales conservados |\n"
    text += f"| Finalización de las 60 trayectorias | {local(execution['completed_at_utc'])} | 720 segmentos, 1.440 ajustes y 21.960 predicciones de validación |\n"
    text += f"| Análisis bajo el plan congelado | {local(analysis['analyzed_at_utc'])} | Código analítico incluido en el registro anterior |\n"
    text += "\nEl calendario ya era conocido. Estas fechas acreditan registro previo a los nuevos ajustes piloto, sin atribuirlo al benchmark histórico ni a una evaluación temporal independiente.\n"
    (tables / "chronology.md").write_text(text)
    private = {p.relative_to(SOURCE).as_posix(): sha(p) for p in sorted((SOURCE / "runs").rglob("*")) if p.is_file()}
    (DEST / "retained_private_artifacts.json").write_text(json.dumps({
        "publication_scope": "Hashes only; underlying checkpoints, scalers, histories and individual forecasts remain private",
        "files_sha256": private}, ensure_ascii=False, indent=2)+"\n")
    (DEST / "README.md").write_text(
        "# Piloto nuevo registrado del procedimiento de Filipe\n\n"
        "Cuatro historias, tres modelos y cinco semillas nuevas:60 trayectorias anuales sobre validación2024, con1.440 ajustes. "
        "Protocolo y código se registraron antes de estos entrenamientos. El calendario ya había sido examinado; no es reparación retrospectiva del benchmark ni evaluación nueva de mercado.\n\n"
        "Las tablas y gráficos resumen RMSE, dispersión y dimensionamiento de semillas futuras. La potencia0,80 es un objetivo condicional por contraste; no una potencia lograda o conjunta. "
        "El R recomendado excluye las semillas piloto. No prueba una ganancia avanzada superior al5%.\n\n"
        "Este directorio usa una lista explícita de metadatos y agregados. No publica checkpoints, escaladores, derivados de mercado o predicciones individuales. "
        "Los hashes permiten identificar la evidencia privada. El análisis y la auditoría no vuelven a entrenar ni abren el test histórico.\n")
    manifest = {"exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "publication_scope": "Explicit aggregates, code identity and hashes only",
                "script_sha256": sha(__file__), "individual_forecasts_published": False,
                "checkpoints_or_scalers_published": False,
                "files_sha256": {p.relative_to(DEST).as_posix(): sha(p) for p in sorted(DEST.rglob("*"))
                                 if p.is_file() and p.name != "public_manifest.json"}}
    (DEST / "public_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"public_files": len(manifest["files_sha256"])+1, "configuration_records": len(catalog),
                      "individual_forecasts_published": False, "new_fits_in_export": 0}))


if __name__ == "__main__":
    main()
