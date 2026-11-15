#!/usr/bin/env python3
"""Independent D10 paper package: saved-result reconstruction and opt-in refit.

Copy this file to PACKAGE/run.py. The default requires only Python's standard
library; --train/--score use the bundled unchanged original CPU driver and the
explicitly installed numerical environment. No website, Git checkout or token
is used. Independent training never overwrites the historical result files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace


SOURCE_COMMIT = "204a8543234aac1614e0b76914bd914e28567eb9"
DATA_SHA = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
SOURCES = {
    "replication_seed_metrics.csv": ("book_e2/_static/paper_guide_results/replication_seed_metrics.csv", "18aa5654ddf871365a7cac56fe228e19c103276d51102665e7a5b7827579213c"),
    "replication_summary.json": ("book_e2/_static/paper_guide_results/replication_summary.json", "19f7fd9a0df667f85498ce2f88c27ed478679a6c37a605ea5d47df27a229a165"),
    "training_curve_summary.csv": ("book_e2/_static/paper_guide_evidence/training_curve_summary.csv", "7d06431c852dce744f48faff064848681e5f38f79c0b9350fd5f16ca1fe6bc5c"),
    "pilot_batch_MSE_change.csv": ("book_e2/_static/paper_guide_results/pilot_batch_MSE_change.csv", "f422e431ab19b6024532a62b180ffb26d06f564500a43dd9866d59f8cd7133fd"),
    "pilot_regime_inference.csv": ("book_e2/_static/paper_guide_results/pilot_regime_inference.csv", "54cc7d8ad64f8edf1ecd89d5196f3aedbca7fa875847afccb255b8b326d9d3ff"),
}
ORIGINAL_FILES = {
    "scripts/run_e2_paper_replication.py": "9d0c9ad8f919e77201bc1ea2af0dec13e07577b4dea494645f3a33e178790832",
    "scripts/e2_paper_data.py": "47de2cc18c8fdb956f8a6c22c63ee752d8dc8070df435f90e386d716219a2489",
    "experiments/e2_paper_guide_20261106/design.json": "db278e235bea127820f3cd8bec1407865fc9ac98962e6c4964098719fa567235",
}
T95_DF4 = 2.7764451051977987


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_sources(root):
    for filename, (_, expected) in SOURCES.items():
        require(sha(root / "inputs" / filename) == expected, "Historical input changed: " + filename)
    return {name: expected for name, (_, expected) in SOURCES.items()}


def svg(rows):
    left, top, width, height = 90, 45, 600, 250
    values = [float(r[k]) for r in rows for k in ("train_loss_mean", "validation_loss_mean")]
    low, high = min(values) * .9, max(values) * 1.05
    def points(key):
        return " ".join(f"{left+i*width/max(1,len(rows)-1):.3f},{top+height-(float(r[key])-low)/(high-low)*height:.3f}" for i, r in enumerate(rows))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="370" viewBox="0 0 760 370" role="img" aria-label="TimeMixer ETTh1: media de cinco curvas publicadas">',
             '<rect width="760" height="370" fill="white"/>',
             '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">TimeMixer / ETTh1: curvas guardadas de cinco semillas</text>',
             f'<path d="M {left} {top} V {top+height} H {left+width}" stroke="#333" fill="none"/>']
    for i in range(5):
        v = low + i*(high-low)/4; y = top+height-i*height/4
        parts.append(f'<text x="80" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="12">{v:.3f}</text>')
    for i, row in enumerate(rows):
        x = left+i*width/max(1,len(rows)-1)
        parts.append(f'<text x="{x}" y="316" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(row["epoch"])}</text>')
    for key, color, label, x in (("train_loss_mean", "#1769aa", "Train", 110), ("validation_loss_mean", "#bd4b00", "Validación", 260)):
        parts.extend((f'<polyline points="{points(key)}" stroke="{color}" fill="none" stroke-width="2.5"/>',
                      f'<text x="{x}" y="350" font-family="sans-serif" fill="{color}">{label}</text>'))
    parts.append('<text x="620" y="350" font-family="sans-serif">Época</text></svg>')
    return "\n".join(parts)


def table(rows):
    if not rows:
        return "<p>Sin filas.</p>"
    columns = list(rows[0])
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in columns) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,"")))+'</td>' for k in columns)+'</tr>' for r in rows) + '</tbody></table></div>'


def reconstruct(root, output):
    verified = check_sources(root)
    rows = read_csv(root / "inputs/replication_seed_metrics.csv")
    saved = json.loads((root / "inputs/replication_summary.json").read_text(encoding="utf-8"))
    curves = read_csv(root / "inputs/training_curve_summary.csv")
    require(len(rows) == 5 and {int(r["seed"]) for r in rows} == set(range(2019, 2024)), "Five original paper seeds required")
    require(all(int(r["retained_test_origins"]) == 2688 and int(r["nominal_test_origins"]) == 2785 and int(r["omitted_tail_origins"]) == 97 for r in rows), "Original retained test scope changed")
    reconstructed = []
    max_difference = 0.0
    for metric in ("MSE", "MAE"):
        values = [float(r[metric]) for r in rows]
        require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid original metrics")
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        half = T95_DF4 * sd/math.sqrt(5)
        ci = [mean-half, mean+half]
        reference = float(saved["reference"][metric])
        record = {"metric": metric, "n_seeds": 5, "mean": mean, "sample_SD": sd,
                  "CI95_lower": ci[0], "CI95_upper": ci[1], "paper_reference": reference,
                  "difference_from_paper": mean-reference,
                  "relative_difference_percent": 100*(mean-reference)/reference}
        max_difference = max(max_difference, abs(mean-saved[metric]["mean"]), abs(sd-saved[metric]["sample_sd"]), *(abs(a-b) for a,b in zip(ci,saved[metric]["ci95"])))
        reconstructed.append(record)
    require(max_difference <= 2e-14, "Reconstructed seed summary differs from original")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "replication_reconstructed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(reconstructed[0]),lineterminator="\n");writer.writeheader();writer.writerows(reconstructed)
    for name in SOURCES:
        shutil.copyfile(root/"inputs"/name,output/name)
    (output / "learning_curves.svg").write_text(svg(curves), encoding="utf-8")
    margin = float(saved["registered_success_rule"]["absolute_margin"])
    mse = reconstructed[0]
    compatible = mse["CI95_lower"]-mse["paper_reference"] >= -margin and mse["CI95_upper"]-mse["paper_reference"] <= margin
    require(compatible == saved["registered_success_rule"]["numerical_compatibility"], "Original compatibility decision changed")
    result = {"version":"e2-d10-standalone-paper-v1","status":"verified","source_commit":SOURCE_COMMIT,
              "five_seed_summary_reconstructed":True,"max_abs_summary_difference":max_difference,
              "registered_numerical_compatibility":compatible,"source_hashes":verified,
              "operations":{"model_fits":0,"model_inference_calls":0,"new_bootstrap_draws":0,"new_hypotheses":0},
              "scope":"Reconstruction of the original five seed-level aggregates and saved learning curves; not regeneration of private per-origin NPZ files or new trained checkpoints.",
              "retained_test_origins":2688,"nominal_test_origins":2785,"omitted_tail_origins":97,
              "temporal_intervals_and_pilot_tests":"Copied original published outputs; not recomputed from the aggregate CSV.",
              "new_training":"Optional --train runs the unchanged original CPU fitting functions on bundled ETTh1/author files into an independent local directory. It is not executed by default."}
    write_json(output / "resultado.json", result)
    pilot = read_csv(root / "inputs/pilot_regime_inference.csv")
    batch = read_csv(root / "inputs/pilot_batch_MSE_change.csv")
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D10 · Paper guía: reproducción independiente</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:0 20px;color:#173047}h1,h2{color:#124a70}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #b9c9d5;text-align:left}th{background:#e6f0f7}.note{border-left:4px solid #b05b12;padding:12px;background:#fff7e8}code{background:#edf2f5;padding:3px}img{max-width:100%}a{color:#064f88}</style><h1>D10. Paper guía: TimeMixer / ETTh1</h1><p>Contexto 96, horizonte 96, siete canales. Esta ejecución reconstruye los resultados agregados de cinco semillas con los archivos incluidos; funciona sin abrir el sitio web, sin Git y sin dependencias externas.</p>'''
    page += '<p>Artículo guía: <em>TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting</em>, ICLR 2024, DOI 10.48550/arXiv.2405.14616. Referencia fija: tabla 13, página física 17, ETTh1 multivariado L96/H96. La incertidumbre de la fila original no está disponible.</p>'
    page += '<h2>Resumen reconstruido</h2>'+table(reconstructed)
    page += f'<p>Discrepancia máxima respecto al resumen guardado: <strong>{max_difference:.3g}</strong>. Compatibilidad numérica según el margen registrado: <strong>{"cumplida" if compatible else "no cumplida"}</strong>.</p>'
    page += '<div class="note">La réplica es parcial: 2.688 orígenes retenidos de 2.785 nominales; 97 se omiten por el batch drop-last original. El IC entre semillas describe optimización en ese calendario. No se recuperan los NPZ privados por origen desde una tabla agregada ni se prueba superioridad en Bitcoin.</div>'
    page += '<h2>Curvas medias guardadas</h2><img src="learning_curves.svg" alt="Curvas de train y validación de cinco réplicas"><p>La figura se regenera desde el CSV incluido; no entrena nuevos modelos ni inventa pérdidas.</p>'
    page += '<h2>Piloto de limitación: resultados originales</h2><p>Los IC y p siguientes se conservan, sin volver a remuestrear. La sensibilidad al batch y la hipótesis predictiva por régimen tienen alcances distintos; la segunda no confirma una limitación predictiva transferible a Bitcoin.</p>'+table(batch)+table(pilot)
    page += '<h2>Archivos y ejecución</h2><p><a href="replication_reconstructed.csv">Resumen recalculado CSV</a> · <a href="replication_seed_metrics.csv">Cinco métricas originales</a> · <a href="replication_summary.json">Análisis original completo JSON</a> · <a href="resultado.json">Resultado y alcance JSON</a></p><p>Para volver a entrenar, instalar <code>requirements-training.txt</code>, ejecutar <code>python run.py --validate-training</code> y después <code>python run.py --train</code>. El refit es opcional, consume CPU y genera checkpoints nuevos en una carpeta distinta; no se atribuye a la campaña histórica.</p>'
    document=root/"inputs/paper_guide.md"
    if document.is_file():
        page += '<details><summary>Informe fuente completo del paper guía, incluido en el paquete</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(document.read_text(encoding="utf-8"))+'</pre></details>'
    page += '</html>'
    (output / "informe.html").write_text(page,encoding="utf-8")
    return result


def validate_training_inputs(root):
    code = root / "code"
    for path, expected in ORIGINAL_FILES.items():
        require(sha(code/path) == expected, "Bundled original training source changed: "+path)
    require(sha(root/"inputs/ETTh1.csv") == DATA_SHA,"Bundled ETTh1 differs from the original dataset")
    plan=json.loads((code/"experiments/e2_paper_guide_20261106/design.json").read_text(encoding="utf-8"))
    rows=read_csv(root/"inputs/ETTh1.csv")
    require(len(rows)==17420 and list(rows[0])==["date","HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"],"ETTh1 geometry differs")
    dates=[datetime.fromisoformat(r["date"]) for r in rows]
    require(all(b-a==timedelta(hours=1) for a,b in zip(dates,dates[1:])),"ETTh1 hourly calendar changed")
    require(all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k!="date"),"Invalid ETTh1 numeric value")
    author_count=0
    for name,spec in plan["sources"].items():
        for item in spec["files"]:
            path=root/"authors"/name/item["path"]
            require(path.is_file() and path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],"Pinned paper author source changed: "+name+"/"+item["path"])
            author_count += 1
    return {"status":"verified","data_rows":len(rows),"channels":7,"author_files":author_count,
            "original_driver_and_helper_verified":True,"model_fits":0,"model_inference_calls":0,
            "train_rows":[0,8640],"validation_label_rows":[8640,11520],"test_label_rows":[11520,14400],
            "scope":"Hashes and hourly dataset geometry; validation does not fit a scaler or model."},plan


def independent_training(root, output, seed=None, score_only=False):
    validation,plan=validate_training_inputs(root)
    sys.path.insert(0,str(root/"code/scripts"))
    try:
        import run_e2_paper_replication as driver
    except ImportError as error:
        raise ValueError("Install requirements-training.txt first: "+str(error)) from error
    design=root/"code/experiments/e2_paper_guide_20261106/design.json"
    source=root/"authors/TimeMixer";control=root/"authors/controls";data=root/"inputs/ETTh1.csv"
    driver.read_design(design);driver.model_arguments(plan);driver.training_arguments(plan)
    inputs=driver.registration_payload(design,plan,data,source,control)
    output.mkdir(parents=True,exist_ok=True)
    registration=output/"registration.json"
    if not registration.exists():
        require(not any(output.iterdir()),"Independent training needs a fresh output directory")
        write_json(registration,{"registered_at_utc":datetime.now(timezone.utc).isoformat(),"source_commit":SOURCE_COMMIT,
                    "immutable_inputs":inputs,"dataset_fits_before_registration":0,
                    "registration_kind":"new independent downloaded-package run; not historical preregistration",
                    "historical_preregistration_claimed":False,
                    "scope":"ETTh1 CPU reproduction using the unchanged bundled driver; not the Bitcoin benchmark"})
    else:
        registered=driver.verify_registration(output,inputs)
        require(registered.get("historical_preregistration_claimed") is False,"This command only resumes its own independent package run")
    args=SimpleNamespace(design=design,data=data,source_root=source,control_source_root=control,
                         output_dir=output,model="TimeMixer",seed=seed)
    fits=[]
    if not score_only:
        seeds=[seed] if seed is not None else list(plan["replication"]["seeds"])
        for value in seeds:
            args.seed=value;fits.append(driver.fit_one(args,plan,inputs,pilot=False))
    result={"status":"completed","registration_kind":"new independent package run","validation":validation,
            "fits_completed_or_verified":len(fits),"output_directory":str(output),"historical_results_overwritten":False}
    if score_only or seed is None:
        result["scoring"]=driver.score(args,plan,inputs)
    else:
        result["scoring"]="One seed fitted; score is available after all five registered seed checkpoints exist."
    write_json(output/"independent_run.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output",type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-training",action="store_true",help="Validate bundled raw ETTh1 and original source bytes without fitting")
    modes.add_argument("--train",action="store_true",help="Opt in to five real CPU fits with the unchanged original driver (may take hours)")
    modes.add_argument("--score",action="store_true",help="Score all five completed independent local fits")
    parser.add_argument("--seed",type=int,choices=range(2019,2024),help="Fit only one seed; default --train fits all five then scores")
    args=parser.parse_args(argv);root=args.package_root.resolve()
    try:
        require(args.seed is None or args.train,"--seed is only meaningful with --train")
        if args.validate_training:
            result,_=validate_training_inputs(root)
            out=(args.output or root/"output").resolve();out.mkdir(parents=True,exist_ok=True)
            write_json(out/"training_input_validation.json",result)
        elif args.train or args.score:
            result=independent_training(root,(args.output or root/"independent_training").resolve(),args.seed,args.score)
        else:
            require(args.seed is None,"--seed is only meaningful with --train")
            result=reconstruct(root,(args.output or root/"output").resolve())
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,ImportError) as error:
        print("D10: "+str(error),file=sys.stderr);return 1


if __name__=="__main__":
    raise SystemExit(main())
