# 10 Bloque F Despliegue en Render

**Estado: servicio Render creado, con URL asignada y construcción observada; disponibilidad y humo públicos por verificar.** Las capturas aportadas por el operador muestran `deeplearning-e3-dashboard`, runtime Docker, plan Free y la construcción del commit `ccefa68`. La dirección asignada es [https://deeplearning-e3-dashboard.onrender.com](https://deeplearning-e3-dashboard.onrender.com). El estado observado **Building** no acredita que el servicio haya alcanzado Live ni que responda correctamente. GitHub Pages sigue publicando este informe y no ejecuta Dash.

El servicio se creó desde el [blueprint de este repositorio](https://render.com/deploy?repo=https://github.com/jassan-bit/DeepLearning_202630). Ese enlace abre la configuración de Render y requiere la cuenta del propietario; **es el panel de creación, distinto de la URL de la aplicación**. El archivo solicita el plan `free`, runtime Docker y `autoDeploy: false`; las capturas muestran Docker y Free. No se declara contratación de un plan de pago ni capacidad remota medida.

El objetivo de cierre es verificar que esa URL permita recorrer Contexto, EDA y Modelos sin instalación local, con cuatro ONNX en CPU y trece sistemas precalculados. El [capítulo 9](09_dashboard_predictivo.md) conserva la implementación y sus límites. El {download}`deployment_readiness.json <../_static/e3_bloques_fg_20261117/deployment_readiness.json>` mantiene sus bytes y el estado anterior a la creación; la {download}`observación posterior del servicio <../_static/e3_cierre_guia_20261118/render_service_observation.json>` registra la URL y la evidencia limitada de las capturas. Ninguno de esos documentos es un recibo de humo público aprobado.

## 10.1 Pasos de despliegue

### 1. Conservar la aplicación local verificada

`python app.py` inicia la aplicación local en `127.0.0.1:8050`, con `use_pages=True`; `server = app.server` es la entrada WSGI de Gunicorn. El capítulo 9 conserva 25 casos unitarios del dashboard y seis recorridos reales de navegador, incluidos filtros, entradas inválidas, navegación y predicción. No son una certificación de Render ni una reproducción humana independiente de la campaña científica.

El nuevo cliente de humo también recorrió las tres páginas contra Gunicorn y contra el contenedor local, y ejecutó una predicción con explicación para **cada uno de los cuatro ONNX**, sin entrenamiento ni recalibración. Los recibos locales permanecen identificados como tales:

- {download}`smoke_local_preflight.json <../_static/e3_bloques_fg_20261117/smoke_local_preflight.json>`.
- {download}`smoke_container_preflight.json <../_static/e3_bloques_fg_20261117/smoke_container_preflight.json>`.

### 2. Mantener Python, dependencias y contenedor reproducibles

La vía implementada y probada es **Docker**, con Python **3.12.14** y la imagen base fijada por digest en `Dockerfile`. `requirements.txt` incluye las 40 versiones exactas del lock de ejecución; entre ellas Gunicorn 23.0.0 y ONNX Runtime 1.23.2. El runtime excluye Torch, TensorFlow, JAX, Playwright y las herramientas de entrenamiento; el cliente de humo utiliza el lock de desarrollo separado.

El contenedor copia únicamente código de servicio, datos admitidos y los ONNX verificados; ejecuta el proceso como UID 10001, sin privilegios. Su comando efectivo es:

```text
gunicorn --bind "0.0.0.0:${PORT:-8050}" --workers 1 --threads 1 --timeout 120 app:server
```

Render suministra `PORT`; Gunicorn escucha en `0.0.0.0`. El proceso usa un worker y un hilo, `CPUExecutionProvider`, ejecución serial y una única sesión de modelo residente bajo demanda. El cambio de modelo desaloja la sesión anterior. No existe una sesión ONNX global creada al importar la aplicación.

La prueba de la imagen conservada en {download}`docker_verification.json <../_static/e3_dashboard_20261117/docker_verification.json>` utilizó un límite local de 512 MiB y un CPU, sin OOM, y observó un pico próximo a 186 MiB durante ese recorrido. **Ese perfil local no demuestra la memoria, concurrencia o capacidad del plan vigente de Render.** Python 3.11.9 y los dos hilos del ejemplo de la guía no son el perfil medido aquí.

### 3. Versionar código, artefactos y metadatos

Los cuatro ONNX conservan los pesos publicados de la semilla representativa 271801. Sus archivos pesan entre 3.319 y 23.138 bytes, por lo que estos cuatro no requieren Git LFS. Los metadatos documentan preprocesamiento, semilla, escala, versiones, integridad y prueba de ida y vuelta. La paridad completa sobre 608 fechas usa tolerancia absoluta de `1e-6` en unidades originales de σ₇; los resultados del nuevo humo sobre un ejemplo no sustituyen esa prueba.

La aplicación verifica hashes y usa un registro permitido de modelos. No acepta rutas, URLs o pesos del usuario, ni reajusta normalización o calibradores con el CSV. Los trece sistemas sin exportación de servicio validada conservan su modo precalculado. Si se añaden pesos mayores, su almacenamiento inmutable, descarga, licencias e integridad deberán documentarse antes de habilitarlos. Las condiciones de redistribución de datos y la licencia propia mantienen su estado en las fichas; no se inventa una licencia al desplegar.

### 4. Conservar la configuración del Web Service creado

La creación ya se observó en la cuenta del operador. Para reconstruir o revisar su configuración se conserva [el enlace al blueprint](https://render.com/deploy?repo=https://github.com/jassan-bit/DeepLearning_202630), con el repositorio y la rama `main`. La configuración versionada solicita:

```yaml
services:
  - type: web
    name: deeplearning-e3-dashboard
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    healthCheckPath: /healthz
    autoDeploy: false
    envVars:
      - key: PYTHONUNBUFFERED
        value: '1'
```

Con runtime Docker, Render construye `Dockerfile` y usa su `CMD`; no se añade a la vez un `buildCommand` de Python nativo. `autoDeploy: false` mantiene los despliegues bajo una acción explícita del propietario. Cada despliegue deberá conservar el SHA efectivo mostrado por Render y su log de construcción/arranque. La configuración por sí sola no crea la cuenta o el servicio.

Antes de confirmar se deben consultar las condiciones actuales de [servicios gratuitos](https://render.com/docs/free), [Docker](https://render.com/docs/docker), [blueprints](https://render.com/docs/blueprint-spec) y [healthchecks](https://render.com/docs/health-checks). La selección del plan del archivo no acredita disponibilidad, gratuidad o límites de una cuenta concreta. Esta revisión no activó recursos de pago.

### 5. Verificar construcción, arranque y URL pública

El log debe mostrar instalación correcta, arranque WSGI y el healthcheck `/healthz` con `status=ok`, `service=e3-dashboard` y 17 sistemas. Se conservarán URL HTTPS, identificador del servicio, commit, fecha y configuración efectiva. La espera hasta que el servicio responda debe medirse; sólo se llamará «arranque frío» si se acredita el estado previo de inactividad.

Se comprobarán navegación y callbacks, disponibilidad de artefactos, recuperación ante errores, carga y desalojo de modelos, latencia completa y coste de explicaciones. Una entrada inválida no debe detener la aplicación, revelar trazas internas o producir un valor cero de sustitución. La aceptación local del capítulo 9 ya cubre esas entradas; su funcionamiento público deberá contrastarse sobre la versión desplegada.

El ejemplo Softmax de la guía corresponde a clasificación. Este problema pronostica **σ₇ continua con h=1**: el servicio aplica la escala guardada exactamente una vez y no convierte la salida en probabilidades. MLP16, LSTM32 y GRU32 usan su intervalo nominal heredado del 90 %; E3 declara **sin intervalo calibrado**. La dispersión entre semillas no sustituye ese intervalo. Cada sensibilidad por oclusión es un contraste local y no una explicación causal o una descomposición aditiva garantizada.

### 6. Ejecutar el guion de humo contra la URL real

El cliente ya implementado se conserva como {download}`smoke_render.py <../_static/e3_bloques_fg_20261117/smoke_render.py>` y su procedimiento como {download}`README_smoke.md <../_static/e3_bloques_fg_20261117/README_smoke.md>`. Puede ejecutarse desde un entorno de desarrollo o mediante el workflow manual **Verificar cliente de humo E3 y despliegue público solicitado**. Las versiones de Playwright y Chromium pertenecen al cliente de comprobación, nunca al servicio.

La URL siguiente ya está asignada al servicio. El SHA completo resuelve el prefijo de construcción observado y debe cotejarse con el log efectivo de Render antes de ejecutar el humo. Si se despliega otro commit, se registrará ese valor real:

```bash
python experiments/e3_bloques_fg_20261117/smoke_render.py \
  --url https://deeplearning-e3-dashboard.onrender.com \
  --deployment-commit ccefa68111cc29e9f75eb724c15a94655d63ca55 \
  --output work/render-public-smoke/report.json \
  --screenshots-dir work/render-public-smoke/captures
```

| Comprobación automatizada | Criterio fijado antes de medir |
|---|---|
| Disponibilidad inicial | Healthcheck correcto antes de 180 s; conservar intentos y estados transitorios. |
| Tres páginas y callbacks | HTTP 200; navegación y componentes completos antes de 30 s por página; EDA vacía las seis figuras con la misma máscara. |
| Registro visible | Diecisiete filas en Modelos y ejecución de los cuatro identificadores habilitados. |
| Integridad visible | Ficha del modelo, declaración de datos y CSS iguales por SHA-256 al checkout del cliente. |
| Cada ONNX en vivo | Callback real HTTP 200, salida finita y no negativa, caso fijo `evaluation-0` y diferencia ≤ `1e-6` frente a su predictor guardado. |
| Incertidumbre y explicación | Nominal del 90 % en referencias; ausencia calibrada explícita en E3; siete sensibilidades finitas y límites de interpretación visibles. |
| Respuesta completa | Antes de 30 s por predictor, incluyendo red, explicación y renderizado; registrar aparte la latencia CPU sin explicación. |
| Errores tras estar listo | Sin HTTP 5xx internos, errores de consola ni excepciones no capturadas de navegador. |

El caso fijado corresponde al objetivo **2025-01-01**, elegido por posición cronológica y no por rendimiento. En el contenedor local las cuatro consultas completas, incluida explicación, tardaron respectivamente **1.792, 326, 250 y 256 ms** para MLP16, LSTM32, GRU32 y E3. La mayor diferencia frente al pronóstico guardado fue menor que `3.86e-7`; la interfaz muestra seis decimales. Son observaciones de una consulta por modelo y de ese entorno local: no constituyen percentiles de latencia, métricas nuevas, evaluación remota o una prueba de toda la distribución de entradas.

El guion devuelve código de salida distinto de cero ante fallo y conserva un JSON con tiempos, estados HTTP, errores, huellas y resultados. La admisión de URL exige un origen HTTPS `nombre.onrender.com`; loopback requiere `--allow-local` y **nunca** puede producir `render_public_smoke_passed=true`. Los seis contratos de ese cliente se verifican sin red en CI. La prueba pública sólo se ejecuta al invocar manualmente el workflow con una URL existente y el SHA declarado.

**La identidad de versión tiene un límite explícito:** el healthcheck actual no expone un commit. El SHA introducido queda como declaración del operador y `remote_commit_identity_verified` sigue en `false`; debe cotejarse y anexarse el log de Render. Las huellas de recursos y la paridad son comprobaciones adicionales y no se presentan como una atestación completa del código remoto.

La observación actual registra `render_service_url=https://deeplearning-e3-dashboard.onrender.com`, creación observada y último estado aportado **Building**. El humo público sigue **NO EJECUTADO / SIN RECIBO APROBADO**; las capturas muestran la creación y construcción, no una aceptación del servicio. No se dispone aquí de un log de arranque ni vídeo que permita certificar su funcionamiento. Los recibos locales conservan `render_public_smoke_passed=false`. Cuando Render indique Live, se podrá ejecutar [el workflow manual de humo](https://github.com/jassan-bit/DeepLearning_202630/actions/workflows/verify-e3-render-smoke.yml), en `main`, con la URL asignada y el SHA cotejado con el log.

### Recursos del plan y contingencia

La suspensión por inactividad, el tiempo de espera, la memoria/CPU efectiva y la persistencia deben contrastarse con el plan actual y medirse en el servicio real. El cliente de navegador no puede deducir memoria pico remota o concurrencia. Si un modelo excede los recursos disponibles, se documentará su modalidad precalculada o una compresión validada con paridad y pérdida de métrica frente al margen heredado; no se reentrena dentro del servidor ni se activa automáticamente un plan mayor.

Antes de la sustentación se abrirá la URL con antelación, se repetirá el humo y se grabará un vídeo breve de esa versión, con fecha y capturas. Si falla el servicio durante la exposición, se explicará el incidente y el alcance del respaldo. Las capturas locales del capítulo 9 y un vídeo futuro no sustituyen una URL Render funcional ni un recibo público aprobado. El [capítulo 12](12_lista_entregables.md) debe conservar esos pendientes y el [capítulo 13](13_sustentacion.md) distingue la demostración real de la contingencia.
