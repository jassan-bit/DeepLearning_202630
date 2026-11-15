"""Independent D06/D08/D12 replay from included frozen forecasts.

Default execution needs only Python's standard library. Original scientific
programs for an optional full statistical rerun are bundled separately; the
default never trains, downloads data, loads weights or invents missing ICs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

BASE = Path(__file__).resolve().parent
FROZEN_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
SAE = "LSTM-autoencoder con predictor"
CLASSICS = {"Persistence", "SeasonalNaive", "ARIMA"}
CSS = """body{font:16px/1.55 system-ui,sans-serif;color:#172639;background:#f4f7fb;margin:0}main{max-width:1140px;margin:auto;padding:32px;background:white}h1,h2,h3{color:#164675}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:16px 0}td,th{border:1px solid #d5deea;padding:8px;text-align:left}th{background:#eaf1fb}tr:nth-child(even){background:#f8fafc}.scroll{overflow:auto}a{color:#145b9c}pre{white-space:pre-wrap;background:#f4f7fb;padding:14px}img{max-width:100%}.note{padding:14px;background:#eaf1fb;border-left:4px solid #286cac}.muted{color:#526477}code{background:#edf1f7;padding:2px 5px}@media print{main{padding:0}a{color:inherit}.scroll{overflow:visible}}"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def table(rows, columns=None):
    if not rows:
        return "<p>Sin filas.</p>"
    columns = columns or list(rows[0])
    def cell(value):
        if value is None:
            return "No estimable / no publicado"
        if isinstance(value, float):
            return format(value, ".10g")
        return str(value)
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(cell(r.get(c)))+'</td>' for c in columns)+'</tr>' for r in rows)+'</tbody></table></div>'


def page(title, body):
    return '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><main><h1>'+html.escape(title)+'</h1>'+body+'</main></html>'


def verify_files():
    manifest = read_json(BASE / "source_manifest.json")
    for row in manifest["files"]:
        path = (BASE / row["path"]).resolve()
        require(path.is_relative_to(BASE) and path.is_file() and not path.is_symlink(), "Archivo de paquete inválido: "+row["path"])
        data = path.read_bytes()
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Archivo alterado: "+row["path"])
    return len(manifest["files"])


def panel_metrics(payload):
    expected_dates = [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)]
    require(payload["dates"] == expected_dates and payload["mask"] == [True]*608, "Cambió el calendario común")
    y = payload["y_true"]; n = len(y)
    mean_y = math.fsum(y)/n
    denominator = math.fsum((v-mean_y)**2 for v in y)
    grouped, individual, keys = defaultdict(list), [], set()
    for source in payload["predictors"]:
        model, seed = source["model"], str(source["seed"])
        require((model, seed) not in keys, "Predictor duplicado")
        keys.add((model, seed))
        p = source["prediction_original"]; q = float(source["calibrator_q"])
        require(len(p) == n and all(math.isfinite(v) for v in p), "Pronósticos inválidos")
        require(len(source["calibration_dates"]) == 53, "Calibración incompleta")
        residuals = sorted(abs(a-b) for a,b in zip(source["calibration_predictions"], source["calibration_targets"]))
        require(abs(residuals[48]-q) < 5e-14, "El radio no coincide con el estadístico de orden 49")
        sq = math.fsum((a-b)**2 for a,b in zip(p,y))
        covered, widths, score = 0, [], []
        for prediction, target in zip(p,y):
            lo, hi = max(0., prediction-q), prediction+q
            covered += lo <= target <= hi
            widths.append(hi-lo)
            score.append(hi-lo+20*max(lo-target, 0.)+20*max(target-hi, 0.))
        row = {"model": model, "seed": seed, "RMSE": math.sqrt(sq/n),
               "MAE": math.fsum(abs(a-b) for a,b in zip(p,y))/n,
               "sMAPE": math.fsum(2*abs(a-b)/max(abs(a)+abs(b), 1e-15) for a,b in zip(p,y))/n,
               "R2": 1-sq/denominator,
               "coverage90": covered/n, "mean_interval_width90": math.fsum(widths)/n,
               "interval_score90": math.fsum(score)/n, "n_dates": n}
        individual.append(row); grouped[model].append(row)
    require(len(keys) == 81 and len(grouped) == 16, "Se requieren 81 predictores y 16 sistemas")
    reference = {r["model"]: r for r in payload["original_analysis"]["table_A2"]}
    aggregate, differences = [], []
    for model, rows in grouped.items():
        require(len(rows) == (1 if model in CLASSICS else 6), "R incorrecto: "+model)
        row = {"model": model, "R": len(rows)}
        for key in ("RMSE", "MAE", "sMAPE", "coverage90", "mean_interval_width90", "interval_score90"):
            row[key] = math.fsum(r[key] for r in rows)/len(rows)
        row["R2_mean"] = math.fsum(r["R2"] for r in rows)/len(rows)
        row["RMSE_seed_SD"] = statistics.stdev(r["RMSE"] for r in rows) if len(rows)>1 else None
        ci = reference[model]["RMSE_seed_t_CI95"]
        row["RMSE_seed_t_CI95_lower"] = ci[0] if ci else None
        row["RMSE_seed_t_CI95_upper"] = ci[1] if ci else None
        for key in ("RMSE", "MAE", "sMAPE", "coverage90", "mean_interval_width90", "R2_mean"):
            d = abs(row[key]-reference[model][key]); differences.append(d)
            require(d < 5e-13, "La reproducción aritmética difiere: "+model+"/"+key)
        aggregate.append(row)
    return sorted(aggregate, key=lambda r:r["RMSE"]), individual, max(differences)


def rmse_svg(rows):
    w, left, right, top, step = 920, 245, 120, 40, 32
    maximum = max(r["RMSE_seed_t_CI95_upper"] or r["RMSE"] for r in rows)*1.12
    width = w-left-right
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{top+step*len(rows)+60}" viewBox="0 0 {w} {top+step*len(rows)+60}" role="img"><title>RMSE media individual e IC t publicados entre semillas</title><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13" fill="#172639">']
    for i,r in enumerate(rows):
        yy = top+i*step
        label = "SAE" if r["model"] == SAE else r["model"]
        x = left+width*r["RMSE"]/maximum
        elements.append(f'<text x="{left-12}" y="{yy+5}" text-anchor="end">{html.escape(label)}</text><line x1="{left}" x2="{x:.3f}" y1="{yy}" y2="{yy}" stroke="#86b4d9" stroke-width="14"/><circle cx="{x:.3f}" cy="{yy}" r="4" fill="#164675"/>')
        if r["RMSE_seed_t_CI95_lower"] is not None:
            a = left+width*r["RMSE_seed_t_CI95_lower"]/maximum
            b = left+width*r["RMSE_seed_t_CI95_upper"]/maximum
            elements.append(f'<path d="M{a:.3f} {yy-5}V{yy+5}M{a:.3f} {yy}H{b:.3f}M{b:.3f} {yy-5}V{yy+5}" stroke="#172639" fill="none"/>')
        elements.append(f'<text x="{w-right+15}" y="{yy+5}">{r["RMSE"]:.9f}</text>')
    elements.append(f'<text x="{left}" y="{top+step*len(rows)+25}">Menor = mejor. IC t: seis semillas, condicionados al calendario; clásicos R=1 sin IC entre semillas.</text></g></svg>')
    return ''.join(elements)


def regenerate_error_labels(payload, out):
    protocol = read_json(BASE / "inputs/taxonomy_protocol.json")
    low = protocol["thresholds"]["absolute_daily_change_q50"]
    high = protocol["thresholds"]["absolute_daily_change_q90"]
    originals = csv_rows(BASE / "inputs/all_6080_date_model_labels.csv")
    source = {(r["model"],r["date"]): r for r in originals}
    groups = defaultdict(list)
    for row in payload["predictors"]:
        if row["model"] in protocol["models"]:
            groups[row["model"]].append(row["prediction_original"])
    rebuilt = []
    for model, trajectories in groups.items():
        for i,(day,target) in enumerate(zip(payload["dates"],payload["y_true"])):
            error = math.fsum(p[i] for p in trajectories)/len(trajectories)-target
            category = "small_reference_band" if abs(error)<=low else ("moderate_" if abs(error)<=high else "large_")+("underestimate" if error<0 else "overestimate")
            expected = source[(model,day)]
            require(category == expected["category"], "Categoría de error distinta: "+model+"/"+day)
            rebuilt.append({"model":model,"date":day,"signed_error_mean_forecast":error,"category":category})
    require(len(rebuilt)==6080, "Taxonomía incompleta")
    write_csv(out / "categorias_reconstruidas.csv", rebuilt)
    counts = Counter((r["model"],r["category"]) for r in rebuilt)
    frequencies = [{"model":m,"category":c,"count":n,"frequency":n/608} for (m,c),n in sorted(counts.items())]
    write_csv(out / "frecuencias_reconstruidas.csv", frequencies)
    return {"rows":6080,"categories_match_published_rows":True,"threshold_low":low,"threshold_high":high}, frequencies


def reconstruct_ceiling(payload, rows, out):
    weights={r["model"]:float(r["weight"]) for r in csv_rows(BASE/"inputs/weights.csv")}
    require(len(weights)==16 and abs(math.fsum(weights.values())-1)<1e-12,"Pesos convexos incompletos")
    trajectories=defaultdict(list)
    for row in payload["predictors"]:
        trajectories[row["model"]].append(row["prediction_original"])
    scores=[];rebuilt=[]
    for seed in range(6):
        predictions=[math.fsum(weights[m]*v[0 if len(v)==1 else seed][t] for m,v in trajectories.items()) for t in range(608)]
        scores.append(math.sqrt(math.fsum((p-y)**2 for p,y in zip(predictions,payload["y_true"]))/608))
        rebuilt.extend({"seed":271801+seed,"date":d,"prediction":p,"observed":y} for d,p,y in zip(payload["dates"],predictions,payload["y_true"]))
    point=math.fsum(scores)/6
    values={r["model"]:r["RMSE"] for r in rows}
    gamma=(values[SAE]-point)/(values["Persistence"]-point)
    saved=read_json(BASE/"inputs/summary.json")
    require(abs(point-saved["test_mean_individual_seed_RMSE"]["ensemble"])<5e-13,"El proxy no reproduce el valor publicado")
    require(abs(gamma-saved["gamma"]["raw_algebraic_gamma"])<1e-10,"Gamma distinta")
    interval=next(r["algebraic_gamma_CI95"] for r in saved["bootstrap"] if r["block_days"]==14)
    write_csv(out/"ensemble_reconstruido.csv",rebuilt)
    result={"proxy_RMSE":point,"gamma":gamma,"published_algebraic_CI95_copied":interval,
            "weights_reestimated":False,"model_inference_performed":False,"matches_published_point":True,
            "is_Bayes_or_optimal_ceiling":False,"CI_regenerated_in_default":False}
    write_json(out/"techo_reconstruido.json",result)
    return result


def full_primary(payload, out, draws):
    import numpy as np
    sys.path.insert(0,str(BASE / "code/scripts"))
    from e2_reduced_final_analysis import analyze_benchmark, residual_calibrator
    predictions, calibrators = defaultdict(list), defaultdict(list)
    for row in payload["predictors"]:
        predictions[row["model"]].append(row["prediction_original"])
        calibrators[row["model"]].append(residual_calibrator(row["calibration_predictions"],row["calibration_targets"],dates=row["calibration_dates"],predictor_sha256=row["checkpoint_sha256"]))
    result = analyze_benchmark(payload["y_true"],{m:np.asarray(v) for m,v in predictions.items()},dates=payload["dates"],mask=payload["mask"],calibrators=calibrators,draws=draws)
    write_json(out / "analisis_primario_recalculado.json", result)
    return {"draws":draws,"output":"analisis_primario_recalculado.json","models_retrained":False,"original_program_used":True}


def full_uncertainty(out):
    scripts = BASE / "code/scripts"
    cmds = [
        [sys.executable,str(scripts/"e2_conditional_calibration.py"),"--input",str(BASE/"inputs/e2_f2a_input.json"),"--development",str(BASE/"inputs/daily_volatility_development.csv"),"--plan",str(BASE/"plans/calibration.json"),"--output",str(out/"calibracion_completa")],
        [sys.executable,str(scripts/"e2_error_ceiling_final_audit.py"),"--input",str(BASE/"inputs/e2_f2a_input.json"),"--development",str(BASE/"inputs/daily_volatility_development.csv"),"--plan",str(BASE/"plans/errors.json"),"--draws","10000","--output",str(out/"errores_completos")],
    ]
    for cmd in cmds:
        subprocess.run(cmd,cwd=BASE,check=True)
    return {"original_programs_executed":2,"models_retrained":False,"output_folders":["calibracion_completa","errores_completos"]}


def slides(rows, payload, plot):
    best = rows[0]
    contrasts = payload["original_analysis"]["primary_contrasts"]
    gain = next(r for r in contrasts if r["model"]==SAE and r["baseline"]=="Persistence")
    top = [{"modelo":"SAE" if r["model"]==SAE else r["model"],"RMSE":r["RMSE"],"DE entre semillas":r["RMSE_seed_SD"]} for r in rows[:5]]
    contents = [
        ("Pronóstico diario de volatilidad de Bitcoin", "<p>Entregable 2 · Deep Learning 2026-30 · Jassan Arteta</p><p>D7×5 OHLCV → σ₇ poblacional, horizonte de un día. Ruta F2a: series de tiempo.</p>"),
        ("Datos y disponibilidad", "<p>Entrenamiento: 655 objetivos hasta 31/10/2024. Embargo: 01–07/11. Calibración: 53 fechas. Prueba: 608 fechas entre 01/01/2025 y 31/08/2026.</p><p>Las entradas terminan en t−1; σ₇(t) requiere el cierre de t. Se conservan calendario y máscaras.</p>"),
        ("EDA que determina decisiones", "<p>Un día incompleto de desarrollo: no imputar con futuro ni comprimir huecos. Dependencia residual Ridge ACF(1)=0,632228 justifica F2a.</p><p>La historia observable compartida entre ventanas no equivale automáticamente a fuga. La auditoría integral sigue parcial.</p>"),
        ("Catálogo comparado", "<p>Diez candidatos de cuatro familias; MLP16, LSTM32 y GRU32 de E1; Persistence, SeasonalNaive y ARIMA.</p><p>13 redes × seis semillas + tres clásicos deterministas = 81 predictores. Las arquitecturas están definidas: NAS de 1.000 diseños no aplica.</p>"),
        ("Selección y entrenamiento", "<p>Seis configuraciones por red y tres orígenes internos: 234 oportunidades, 231 válidas, dos fallos conocidos y uno desconocido.</p><p>Épocas finales fijadas desde validación. Calibración no se usa para early stopping final. Una misma cantidad de plazas no prueba igualdad de coste completo.</p>"),
        ("Resultado principal", table(top)+"<p>La primaria es la media de RMSE individuales, no la RMSE de un ensemble.</p>"),
        ("Comparación del catálogo", '<img alt="RMSE reconstruida e IC t guardados" src="'+plot+'"><p>IC entre semillas condicionados a esta cronología; no representan nuevos mercados.</p>'),
        ("Inferencia y relevancia práctica", "<p>δ=0,00025772842664981106. Veinte pares contra GRU32 y Persistence; Holm20.</p><p>SAE frente a Persistence: ganancia "+format(gain["gain"],".9f")+"; IC95 % original "+html.escape(str(gain["CI95"]))+".</p><p>Los veinte p ajustados son 1. No se demuestra superioridad práctica ni equivalencia.</p>"),
        ("Incertidumbre y errores", "<p>Intervalos originales calibrados con 53 fechas; políticas móvil53 y ACI53 posteriores con identidad propia.</p><p>ACI global: SAE 89,94 %, GRU32 90,46 %, ARIMA 90,30 %. Persiste infracobertura por grupos altos. Cobertura global no garantiza cobertura condicional.</p><p>Taxonomía automática: 100 fechas/600 casos por principal; sin atribuir causalidad económica.</p>"),
        ("Robustez, explicaciones y ablaciones", "<p>Cuatro perturbaciones por tres severidades. Fidelidad y estabilidad del Top 3 sobre 24 fechas, con límites de validación.</p><p>18 ajustes de ablación SAE. Retirar dropout empeora RMSE un 13,90 %, p Holm3=0,00059994; las otras diferencias son no concluyentes.</p>"),
        ("Coste y Pareto", "<p>5.765,09 segundos instrumentados de fases finales; no coste HPO completo.</p><p>Frontera RMSE–tiempo: LSTM32, MLP16, GRU32, SAE. Parámetros/latencia/FLOPs core: MLP16, GRU32, SAE.</p><p>Los costes clásicos, energía, dinero y memoria aislada por modelo no se inventan.</p>"),
        ("Reproducción y archivos independientes", "<p>81 reproducciones en checkout y entorno limpio: discrepancia máxima de predicción 0. Son otras ejecuciones sobre el mismo calendario.</p><p>Estos paquetes permiten ejecutar código y reconstruir resultados con sus datos incluidos. El replay desde pronósticos no se presenta como un reentrenamiento.</p>"),
        ("Límites que se mantienen", "<p>Test heredado conocido; una serie y una cronología. Potencia piloto 0,123 frente a objetivo 0,80. Auditoría integral de fuga y prueba formal comparable de δg pendientes.</p><p>Proxy de techo incierto; no es riesgo Bayes. No hay TOST/ROPE que acredite equivalencia.</p>"),
        ("Conclusiones y defensa", "<p>SAE tiene la menor RMSE observada; no se confirma mejora mayor que δ. Persistence mantiene una referencia exigente por MAE.</p><p>Separar inicialización, dependencia temporal, relevancia práctica y coste. E3 requiere una hipótesis propia y evidencia: la limitación de batch de ETTh1 no confirma transferencia predictiva a Bitcoin.</p><p>Preguntas: ¿qué unidad se remuestrea?, ¿qué queda sin acreditar?, ¿qué coste mide cada cifra?</p>"),
    ]
    sections = ''.join('<section class="slide"><h1>'+html.escape(title)+'</h1>'+body+'</section>' for title,body in contents)
    style = CSS+".slide{display:none;min-height:76vh;padding:35px;box-sizing:border-box;font-size:clamp(18px,2.2vw,30px)}.slide.active{display:block}.slide h1{font-size:1.5em}.slide table{font-size:.7em}.slide img{max-height:58vh}.controls{position:fixed;bottom:0;left:0;right:0;background:#153b62;color:white;padding:10px;display:flex;justify-content:space-between;gap:20px}.controls button{font-size:18px;padding:7px 18px;cursor:pointer}.counter{align-self:center}@media print{.slide{display:block;break-after:page;min-height:0;font-size:17px;padding:20px}.controls{display:none}}"
    script = "const s=[...document.querySelectorAll('.slide')];let i=0;function show(v){i=Math.max(0,Math.min(s.length-1,v));s.forEach((x,k)=>x.classList.toggle('active',k===i));document.querySelector('.counter').textContent=(i+1)+' / '+s.length;}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();show(i+1)}if(e.key==='ArrowLeft')show(i-1)});show(0);"
    deck='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Defensa E2</title><style>'+style+'</style>'+sections+'<nav class="controls"><button id="prev">Anterior</button><span class="counter"></span><button id="next">Siguiente</button></nav><script>'+script+'</script></html>'
    notes = "# Notas de defensa E2\n\nPresentación preparada; no acredita que la defensa oral ya haya ocurrido. Flechas para navegar; Ctrl+P para imprimir o guardar PDF.\n\n"+'\n\n'.join('## '+str(i+1)+'. '+t+'\n\n'+__import__('re').sub('<[^>]+>',' ',b) for i,(t,b) in enumerate(contents))
    return deck, notes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--full-primary",action="store_true",help="Repetir los tests originales; requiere requirements_analysis.txt")
    parser.add_argument("--draws",type=int,default=4999)
    parser.add_argument("--full-uncertainty",action="store_true",help="Repetir calibración/taxonomía con sus programas y planes originales")
    args=parser.parse_args()
    config=read_json(BASE/"package.json")
    verified=verify_files()
    source=BASE/"inputs/e2_f2a_input.json"
    require(hashlib.sha256(source.read_bytes()).hexdigest()==FROZEN_SHA,"La entrada congelada no es la original")
    payload=read_json(source)
    rows,individual,difference=panel_metrics(payload)
    out=(args.output or BASE/"output").resolve();out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"benchmark_reconstruido.csv",rows)
    write_csv(out/"metricas_por_semilla.csv",individual)
    (out/"comparacion_rmse.svg").write_text(rmse_svg(rows),encoding="utf-8")
    result={"id":config["id"],"status":"verified","source_commit":config["source_commit"],"files_verified":verified,"systems":16,"predictors":81,"dates":608,"max_metric_difference_from_published":difference,"new_model_fits":0,"weights_loaded":False,"default_reconstructs_saved_forecast_arithmetic":True,"original_scientific_files_modified":False}
    body='<p class="note">Ejecución independiente desde los 81 pronósticos incluidos. Se reconstruyen métricas e intervalos originales sin red, pesos o GPU. Los IC t se copian del análisis publicado; no se inventa incertidumbre temporal. Los p guardados no se recalculan salvo opción explícita.</p><p><a href="benchmark_reconstruido.csv">Benchmark CSV</a> · <a href="metricas_por_semilla.csv">Métricas por semilla</a> · <a href="resultado.json">Registro de ejecución</a></p><img src="comparacion_rmse.svg" alt="RMSE e IC t entre semillas"><h2>Tabla reconstruida</h2>'+table(rows)
    body+='<h2>Veinte contrastes originales publicados</h2>'+table(payload["original_analysis"]["primary_contrasts"],["model","baseline","gain","CI95","p_one_sided","p_Holm","reject_at_FWER05"])
    if config["id"]=="D08":
        extra,freq=regenerate_error_labels(payload,out);result["taxonomy"]=extra
        ceiling=reconstruct_ceiling(payload,rows,out);result["ceiling"]=ceiling
        body+='<h2>Techo: referencia empírica reconstruida</h2><p>Se combinan los pronósticos incluidos con los pesos ya publicados, sin volver a ajustarlos. Esta referencia no identifica riesgo Bayes ni un techo óptimo.</p>'+table([ceiling])+ '<p><a href="ensemble_reconstruido.csv">Trayectorias combinadas</a> · <a href="techo_reconstruido.json">Cálculo y alcance de γ</a>. El IC algebraico se conserva del análisis original; no se remuestrea en el modo predeterminado.</p>'
        body+='<h2>Frecuencias reconstruidas</h2><p>Las 6.080 categorías coinciden con las publicadas. Son etiquetas automáticas de error del pronóstico medio; la RMSE primaria conserva métricas individuales.</p>'+table(freq)
        for filename,title in [("conditional_metrics.csv","Cobertura condicional publicada"),("model_metrics.csv","Políticas temporales publicadas")]:
            f=BASE/"inputs"/filename
            if f.is_file():
                stored=csv_rows(f)
                selected=[r for r in stored if r.get("model") in (SAE,"GRU32","ARIMA")]
                body+='<h2>'+title+'</h2><p>Valores guardados, no otra garantía de cobertura. Los intervalos por grupo y supuestos se conservan en los archivos fuente.</p>'+table(selected[:150],[c for c in ["model","policy","group","coverage","mean_width","mean_interval_score"] if c in stored[0]])
    if config["id"]=="D12":
        deck,notes=slides(rows,payload,"comparacion_rmse.svg")
        (out/"presentacion.html").write_text(deck,encoding="utf-8")
        (out/"notas_orador.md").write_text(notes,encoding="utf-8")
        body='<p><a href="presentacion.html">Abrir las 14 diapositivas</a> · <a href="notas_orador.md">Notas del orador</a>. Flechas para navegar; imprimir para guardar PDF. La preparación del archivo no acredita una defensa oral ya realizada.</p>'+body
        result["slides_prepared"]=14;result["oral_defense_performed"]=False
    if args.full_primary:
        result["full_primary"]=full_primary(payload,out,args.draws)
    if args.full_uncertainty:
        result["full_uncertainty"]=full_uncertainty(out)
    for document in config.get("documents",[]):
        path=BASE/document["path"]
        body+='<details><summary>'+html.escape(document["title"])+': texto fuente incluido</summary><pre>'+html.escape(path.read_text(encoding="utf-8"))+'</pre></details>'
    write_json(out/"resultado.json",result)
    (out/"informe.html").write_text(page(config["title"],body),encoding="utf-8")
    print(json.dumps({"status":"verified","id":config["id"],"output":str(out),"systems":16,"predictors":81,"max_difference":difference},ensure_ascii=False))


if __name__=="__main__":
    main()
