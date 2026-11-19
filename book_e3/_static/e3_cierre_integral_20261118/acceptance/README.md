# Aceptación Render V2

Cliente preparado con asistencia de IA. **La CLI V2 completó un preflight local real contra Gunicorn: tres rutas, cinco predicciones e intervalos, siete oclusiones por predictor y tres huellas de recursos. No se ejecutó una aceptación pública de Render desde esta carpeta.** El recibo está en `local_preflight.json`, con ambas banderas públicas en falso, capturas y registros conservados.

El primer intento se conserva en `local_attempt_01/`: el puerto 8054 estaba ocupado por un proceso anterior sin commit en el healthcheck y el cliente lo rechazó antes de abrir el navegador. Después de detener ese proceso con su responsable, se inició Gunicorn con `APP_SOURCE_COMMIT`. `local_execution_context.json` registra el HEAD real del checkout y aclara que el valor de entorno usado fue una revisión base declarada por el operador para comprobar el contrato, sin afirmar identidad exacta de toda la fuente. El recorrido local general previo de la interfaz está documentado por separado en `../serving/browser/acceptance.json`.

`smoke_render_v2.py` importa dinámicamente el guion histórico `experiments/e3_bloques_fg_20261117/smoke_render.py` y reutiliza su validación del origen Render, tolerancia de pronósticos, comparación de orígenes, reloj, argumentos y rechazo de redirecciones. El archivo histórico permanece intacto. La nueva rutina de navegador admite la extensión de cinco predictores y la etiqueta específica del intervalo posterior E3. Playwright sólo se importa al ejecutar esa rutina; no se importan Torch, ONNX Runtime ni la aplicación.

## Qué comprueba

- Las tres rutas `/`, `/eda` y `/modelos`, los seis gráficos de EDA y el filtro sin resultados.
- Los cinco predictores ONNX registrados: los cuatro históricos y TSMixer. Usa el caso fijo `evaluation-0`, objetivo **2025-01-01**, conservando la referencia de cada predictor y la misma ventana de siete barras.
- SHA-256 de metadatos y pesos locales, coherencia del caso TSMixer y vínculo del perfil posterior con los pesos, metadatos, checkpoint, semilla, escala y unidades del predictor E3.
- Pronóstico mostrado, intervalo numérico y siete oclusiones finitas etiquetadas t−7 a t−1 por predictor; los límites causales y temporales deben aparecer en la interfaz.
- Huellas de la ficha del modelo, declaración de datos y CSS servido, sin redirecciones ni solicitudes fuera del origen autorizado.
- `/healthz`: servicio `e3-dashboard`, catálogo de 17 sistemas y commit reportado por `RENDER_GIT_COMMIT` o `APP_SOURCE_COMMIT`. Para el origen público debe coincidir con `--deployment-commit`, ambos con 40 dígitos hexadecimales.

La igualdad de commits es **identidad reportada por el propio servicio**, no una atestación criptográfica ni una prueba completa del código remoto. El recibo mantiene `remote_commit_identity_verified: false` y registra esa igualdad por separado en `deployment_identity.service_reported_commit_matches_operator`. La prueba tampoco demuestra cobertura futura, causalidad, estabilidad de explicaciones, disponibilidad continua, arranque realmente frío, R1 o R3.

En el caso fijo E3 se exige `INTERVALO POSTERIOR NOMINAL 90%`, q histórico **0.005402463770651868**, y la declaración de coberturas históricas de 94,08 % en 608 fechas y 85,56 % en 2026Q1. No se ajusta un calibrador nuevo. Los pronósticos puntuales originales permanecen como referencias.

## Ejecución

Desde la raíz de un checkout fijado al commit que debe servir Render, en un entorno CPU con Playwright y Chromium instalados:

```bash
python experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py \
  --url https://deeplearning-e3-dashboard.onrender.com \
  --deployment-commit COMMIT_COMPLETO_DE_40_DIGITOS \
  --output /content/e3-acceptance/render/public_smoke.json \
  --screenshots-dir /content/e3-acceptance/render/screenshots \
  --wake-seconds 180
```

El commit se toma del despliegue que se desea aceptar. Si no coincide con el que responde, se conserva un recibo fallido; no se sustituye silenciosamente por el valor remoto. La salida de proceso es cero sólo cuando `status` es `passed`. `render_public_smoke_passed` y su alias de orquestación `public_deployment_smoke_passed` sólo pueden ser verdaderos para un origen público de Render que completa el recorrido. Un loopback requiere `--allow-local` y siempre queda clasificado como preflight local.

Para orquestación Python, importar el módulo mediante `importlib`, obtener `args = parser().parse_args([...])` y llamar `run(args)`. Devuelve el mismo diccionario que guarda en `args.output`. Una URL o un commit de entrada inválido generan `ValueError` antes de iniciar la medición; los fallos de referencias, salud y navegador se guardan como `status: failed` con sus errores.

## Pruebas sin navegador

```bash
python -B -m unittest discover \
  -s experiments/e3_cierre_integral_20261118/acceptance \
  -p test_smoke_render_v2.py -v
```

`verification.json` y `tests.log` registran únicamente las comprobaciones realmente ejecutadas. Los servicios y recorridos simulados no se presentan como evidencia de despliegue público.
