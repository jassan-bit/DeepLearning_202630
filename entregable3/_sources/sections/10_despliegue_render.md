# 10 Bloque F Despliegue en Render

**Estado actual: Render respondió HTTP200 en la comprobación pública del20de noviembre; el perfil servido no informa su SHA y no se aprobó el humo de cinco modelos.** La dirección asignada es [https://deeplearning-e3-dashboard.onrender.com](https://deeplearning-e3-dashboard.onrender.com). Las capturas anteriores muestran `deeplearning-e3-dashboard`, runtime Docker, plan Free y construcción del commit `ccefa68`; ese corte **Building** se conserva como antecedente. El informe posterior **Live** no acredita por sí solo el SHA servido, las cinco predicciones o los callbacks. GitHub Pages sigue publicando este informe y no ejecuta Dash.

El servicio se creó desde el [blueprint de este repositorio](https://render.com/deploy?repo=https://github.com/jassan-bit/DeepLearning_202630). Ese enlace abre la configuración de Render y requiere la cuenta del propietario; **es el panel de creación, distinto de la URL de la aplicación**. El archivo solicita el plan `free`, runtime Docker y `autoDeploy: false`; las capturas muestran Docker y Free. No se declara contratación de un plan de pago ni capacidad remota medida.

El objetivo de cierre es verificar esa URL con Contexto, EDA y Modelos, **cinco ONNX en CPU y doce sistemas precalculados**. El [capítulo 9](09_dashboard_predictivo.md) conserva la implementación y sus límites. El {download}`deployment_readiness.json histórico <../_static/e3_bloques_fg_20261117/deployment_readiness.json>` mantiene sus bytes y el estado anterior a la creación; la {download}`observación de creación y construcción <../_static/e3_cierre_guia_20261118/render_service_observation.json>` conserva el corte de las capturas. Ninguno acredita el funcionamiento de la actualización actual. Como `autoDeploy: false`, antes de probarla se requiere **Manual Deploy → Deploy latest commit** y conservar el SHA efectivo mostrado por Render.

## 10.1 Pasos de despliegue

### 1. Conservar la aplicación local verificada

`python app.py` inicia la aplicación local en `127.0.0.1:8050`, con `use_pages=True`; `server = app.server` es la entrada WSGI de Gunicorn. El perfil vigente pasó **29 pruebas de software**, **seis recorridos de navegador con 36 comprobaciones** y **cinco predicciones dentro del contenedor**, con intervalos y explicaciones. Se publican el {download}`log unitario <../_static/e3_cierre_integral_20261118/serving/unit_tests.log>`, la {download}`aceptación del navegador <../_static/e3_cierre_integral_20261118/serving/browser/acceptance.json>` y el {download}`recibo del contenedor <../_static/e3_cierre_integral_20261118/serving/docker_runtime_verification.json>`. No son una certificación de Render ni una reproducción integral independiente de la campaña científica.

El cliente de humo histórico recorrió las tres páginas contra Gunicorn y el contenedor local, y ejecutó una predicción con explicación para **cada uno de los cuatro ONNX de aquel perfil**, sin entrenamiento ni recalibración. Sus recibos conservan ese alcance y sus bytes; la aceptación de cinco modelos usa la versión V2 descrita abajo:

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

La actualización de cinco modelos conserva por separado su {download}`construcción Docker <../_static/e3_cierre_integral_20261118/serving/docker_build.log>` y su {download}`verificación de ejecución <../_static/e3_cierre_integral_20261118/serving/docker_runtime_verification.json>`: usuario 10001, sistema de archivos de sólo lectura, sin red y con capacidades adicionales retiradas. Pasaron cinco inferencias con el CSV fijo, intervalos y siete oclusiones por predictor, sin Torch. El máximo error frente a las salidas congeladas fue **4,8394×10⁻⁹ σ₇**. Esta prueba nueva no midió memoria pico ni concurrencia; no reutiliza como propia la cifra del perfil anterior.

### 3. Versionar código, artefactos y metadatos

Los cinco ONNX conservan los pesos publicados de la semilla representativa 271801. Sus archivos pesan entre **3319 y 165263 bytes** y no requieren Git LFS. TSMixer se añade mediante `artifacts/e3_serving_extensions_v2/`, sin alterar los cuatro artefactos históricos. Los metadatos documentan preprocesamiento, semilla, escala, versiones, integridad y prueba de ida y vuelta. La paridad sobre 608 fechas usa tolerancia absoluta de `1e-6` en unidades originales de σ₇; TSMixer también se comprobó en 53 ventanas de calibración. El humo sobre un ejemplo no sustituye esas pruebas.

La aplicación verifica hashes y usa un registro permitido de modelos. No acepta rutas, URLs o pesos del usuario, ni reajusta normalización o calibradores con el CSV. Los doce sistemas precalculados se distinguen en la {download}`matriz de capacidades <../_static/e3_cierre_integral_20261118/serving/export_capabilities/capability_matrix.csv>`: **nueve redes ya exportadas y verificadas offline, pendientes de habilitación en este perfil** y **tres clásicos justificados por necesitar más historia o estado que siete barras**. No se atribuye a las nueve redes un exceso de recursos sin medirlo. Si se añaden pesos mayores, su almacenamiento inmutable, descarga, licencias e integridad deberán documentarse antes de habilitarlos. Las condiciones de redistribución de datos y la licencia propia mantienen su estado en las fichas; no se inventa una licencia al desplegar.

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

Con runtime Docker, Render construye `Dockerfile` y usa su `CMD`; no se añade a la vez un `buildCommand` de Python nativo. `autoDeploy: false` exige una acción explícita del propietario para incorporar los cambios. Después de publicarlos en GitHub se debe elegir **Manual Deploy → Deploy latest commit**, esperar **Live** y anotar el SHA efectivo de 40 caracteres y su log de construcción/arranque. Sólo después se ejecuta el cliente CPU de comprobación desde Colab o un entorno de desarrollo. No necesita GPU ni vuelve a entrenar.

Antes de confirmar se deben consultar las condiciones actuales de [servicios gratuitos](https://render.com/docs/free), [Docker](https://render.com/docs/docker), [blueprints](https://render.com/docs/blueprint-spec) y [healthchecks](https://render.com/docs/health-checks). La selección del plan del archivo no acredita disponibilidad, gratuidad o límites de una cuenta concreta. Esta revisión no activó recursos de pago.

### 5. Verificar construcción, arranque y URL pública

El log debe mostrar instalación correcta, arranque WSGI y el healthcheck `/healthz` con `status=ok`, `service=e3-dashboard` y 17 sistemas. La actualización añade `source_commit` y `source_commit_source`: expone únicamente un SHA válido de 40 caracteres de `RENDER_GIT_COMMIT`, o de `APP_SOURCE_COMMIT` en verificaciones controladas; si no hay uno válido, devuelve `null`. Se conservarán URL HTTPS, identificador del servicio, SHA efectivo, fecha y configuración. La espera hasta que el servicio responda debe medirse; sólo se llamará «arranque frío» si se acredita el estado previo de inactividad.

Se comprobarán navegación y callbacks, disponibilidad de artefactos, recuperación ante errores, carga y desalojo de modelos, latencia completa y coste de explicaciones. Una entrada inválida no debe detener la aplicación, revelar trazas internas o producir un valor cero de sustitución. La aceptación local del capítulo 9 ya cubre esas entradas; su funcionamiento público deberá contrastarse sobre la versión desplegada.

El ejemplo Softmax de la guía corresponde a clasificación. Este problema pronostica **σ₇ continua con h=1**: el servicio aplica la escala guardada exactamente una vez y no convierte la salida en probabilidades. MLP16, LSTM32, GRU32 y TSMixer conservan su intervalo nominal heredado del 90 %. E3 adopta el **intervalo posterior nominal del 90 %** del mismo predictor, con q verificado y aplicabilidad después del 30 de diciembre de 2024. La interfaz identifica su cobertura individual histórica **94,08 %** y la de **2026Q1: 85,56 %**, sin garantía temporal o por subgrupo. El [capítulo 9](09_dashboard_predictivo.md) y las {download}`notas de adopción <../_static/e3_cierre_integral_20261118/serving/README.md>` delimitan esa calibración; no se atribuye una media de semillas a un intervalo individual. Cada oclusión es un contraste local y no una explicación causal o una descomposición aditiva garantizada.

### 6. Ejecutar el guion de humo contra la URL real

El cliente vigente es `experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py`: verifica los **cinco modelos**, la calibración posterior E3, las huellas visibles y el commit declarado por el servicio. El cliente anterior se conserva sin modificar como {download}`smoke_render.py histórico <../_static/e3_bloques_fg_20261117/smoke_render.py>` y su {download}`procedimiento histórico <../_static/e3_bloques_fg_20261117/README_smoke.md>`; sus cuatro referencias no describen la aceptación actual. Playwright y Chromium pertenecen al cliente CPU de comprobación, nunca al servicio.

V2 también pasó una {download}`ejecución local real contra Gunicorn <../_static/e3_cierre_integral_20261118/acceptance/local_preflight.json>`: tres rutas, cinco predicciones con sus intervalos, siete oclusiones por modelo y tres huellas de recursos, sin errores. Su {download}`contexto de ejecución <../_static/e3_cierre_integral_20261118/acceptance/local_execution_context.json>` distingue el SHA suministrado al servidor para comprobar el contrato del estado real del checkout. Esta prueba no demuestra que aquel SHA esté desplegado en Render y mantiene ambos indicadores de aceptación pública en `false`.

El cliente y sus referencias deben proceder del mismo checkout publicado que se acaba de desplegar. Define `SHA_RENDER` con los **40 caracteres hexadecimales del despliegue efectivo**, cotejados con el log de Render; no reutilices el SHA antiguo de las capturas. Tras el despliegue manual y el estado Live, ejecuta:

```bash
python experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py \
  --url https://deeplearning-e3-dashboard.onrender.com \
  --deployment-commit "$SHA_RENDER" \
  --output work/render-public-smoke/report.json \
  --screenshots-dir work/render-public-smoke/captures
```

| Comprobación automatizada | Criterio fijado antes de medir |
|---|---|
| Disponibilidad inicial | Healthcheck correcto antes de 180 s; conservar intentos y estados transitorios. |
| Tres páginas y callbacks | HTTP 200; navegación y componentes completos antes de 30 s por página; EDA vacía las seis figuras con la misma máscara. |
| Registro visible | Diecisiete filas en Modelos y ejecución de los cinco identificadores habilitados. |
| Integridad visible | Ficha del modelo, declaración de datos y CSS iguales por SHA-256 al checkout del cliente. |
| Versión declarada | `source_commit` de `/healthz` válido y coincidente con `--deployment-commit`; conservar también su campo de procedencia. |
| Cada ONNX en vivo | Callback real HTTP 200, salida finita y no negativa, caso fijo `evaluation-0` y diferencia ≤ `1e-6` frente a su predictor guardado. |
| Incertidumbre y explicación | Nominal del 90 % en los cuatro E2; etiqueta posterior E3, q y límites correctos para el caso; siete sensibilidades finitas y límites de interpretación visibles. |
| Respuesta completa | Antes de 30 s por predictor, incluyendo red, explicación y renderizado; registrar aparte la latencia CPU sin explicación. |
| Errores tras estar listo | Sin HTTP 5xx internos, errores de consola ni excepciones no capturadas de navegador. |

El caso fijado corresponde al objetivo **2025-01-01**, elegido por posición cronológica y no por rendimiento. En el contenedor del **perfil histórico de cuatro ONNX**, las consultas completas incluida explicación tardaron respectivamente **1.792, 326, 250 y 256 ms** para MLP16, LSTM32, GRU32 y E3. Su mayor diferencia frente al pronóstico guardado fue menor que `3.86e-7`; la interfaz muestra seis decimales. Estas observaciones conservan ese entorno y esos cuatro modelos: no son percentiles, mediciones de TSMixer ni resultados de Render. El recibo nuevo de cinco inferencias citado arriba tampoco reemplaza la medición pública del V2.

El guion devuelve código de salida distinto de cero ante fallo y conserva un JSON con tiempos, estados HTTP, errores, huellas y resultados. La admisión de URL exige un origen HTTPS `nombre.onrender.com`; loopback requiere `--allow-local` y **nunca** puede producir `render_public_smoke_passed=true`. El estado Live y el SHA introducido por el operador no sustituyen ejecutar las comprobaciones y conservar el recibo aprobado.

**La identidad de versión conserva un límite explícito:** V2 exige que el SHA autodeclarado por `/healthz` coincida con el del operador y contrasta las huellas de ficha, datos y CSS con las referencias locales. Esa concordancia refuerza la trazabilidad, pero **no es una atestación criptográfica de todo el código remoto**; `remote_commit_identity_verified` sigue en `false`. Deben anexarse el log de Render, las huellas y el recibo completo, sin confundir esos niveles de evidencia.

La URL sigue siendo `https://deeplearning-e3-dashboard.onrender.com` y el operador comunica **Live**. El estado de aceptación de la actualización permanece **SIN RECIBO PÚBLICO APROBADO**. Los recibos locales conservan `render_public_smoke_passed=false`; no se dispone de una verificación pública de esta versión que pueda sustituirlos. El paso pendiente es desplegar manualmente el último commit, cotejar su SHA y ejecutar el cliente CPU V2 sobre esa URL. Un error o falta de concordancia se conserva como tal y no se convierte en aprobación por declarar Live.

### Recursos del plan y contingencia

La suspensión por inactividad, el tiempo de espera, la memoria/CPU efectiva y la persistencia deben contrastarse con el plan actual y medirse en el servicio real. El cliente de navegador no puede deducir memoria pico remota o concurrencia. Si un modelo excede los recursos disponibles, se documentará su modalidad precalculada o una compresión validada con paridad y pérdida de métrica frente al margen heredado; no se reentrena dentro del servidor ni se activa automáticamente un plan mayor.

Antes de la sustentación se abrirá la URL con antelación, se repetirá el humo y se grabará un vídeo breve de esa versión, con fecha y capturas. Si falla el servicio durante la exposición, se explicará el incidente y el alcance del respaldo. Las capturas locales del capítulo 9 y un vídeo futuro no sustituyen una URL Render funcional ni un recibo público aprobado. El [capítulo 12](12_lista_entregables.md) debe conservar esos pendientes y el [capítulo 13](13_sustentacion.md) distingue la demostración real de la contingencia.


(e3-current-public-acceptance-handoff)=
## 10.2 Comprobación externa y aceptación preparada para el cierre

La comprobación del 19 de noviembre de 2026, hora de Bogotá, verificó las referencias locales del cliente V2 existente: **17 archivos para cinco ONNX**, metadatos y el intervalo posterior E3, sin inferencia. Sus blobs se cotejaron con la fuente publicada. La {download}`sonda de disponibilidad <../_static/e3_evidence_close_20261120/release/render_readiness_attempt.json>` ejecutó el helper `wake_service` de ese cliente durante diez segundos. No ejecutó el navegador ni obtuvo HTTP del servidor de origen; no sustituye el recorrido público completo de 180/30/30 segundos.

Las {download}`consultas públicas <../_static/e3_evidence_close_20261120/release/network_observations.json>` fallaron en el túnel del proxy del entorno con 403. Ese fallo corresponde al acceso del cliente y no permite afirmar que Render esté caído. Se mantiene **aceptación pública sin acreditar**, junto con la observación anterior Live comunicada por el operador. Tampoco se dispone del SHA efectivo actual confirmado en su log; no se adivinó a partir de `main` ni de la autodeclaración de `/healthz`.

Queda preparado el [helper de aceptación CPU para Colab](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_evidence_close_20261120/release/render_public_acceptance_colab.py) y su {download}`procedimiento completo <../_static/e3_evidence_close_20261120/release/README.md>`. Tras **Manual Deploy**, el propietario copia el SHA de 40 caracteres del despliegue y lo introduce en el helper. Éste obtiene las referencias de esa versión, verifica los clientes revisados y ejecuta el V2 original, con sus tres páginas, cinco predicciones, intervalos, explicaciones y huellas. El entorno de Playwright es separado, no requiere GPU y no reentrena ni recalibra.

El ZIP de salida conserva recibo, contexto y capturas; debe persistirse antes de cerrar Colab. Sólo su aprobación permite registrar el humo público. La concordancia de SHA autodeclarado conserva el límite de no ser una atestación completa del código remoto. El plan del servicio y los pesos/calibradores permanecen como estaban; este cierre no contrata recursos ni convierte el catálogo histórico en el de la auditoría nueva.



La [celda CPU fijada al código revisado](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_cierre_publico_cpu_colab.ipynb) solicita el SHA del log de despliegue y verifica el helper antes de ejecutarlo. También se conserva su {download}`código de arranque <../_static/e3_evidence_close_20261120/release/colab_cell.py>` para pegarlo directamente en Colab. Guarda el ZIP de salida y súbelo a GitHub antes de cerrar la sesión; el helper no promete un guardado remoto que no realiza.


(e3-completion-public-release)=
## 10.8 Comprobación pública ejecutada y paso del propietario

La [ejecución externa en GitHub Actions](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44754550086), ligada al commit `6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe`, obtuvo salud **HTTP 200** de Render tras once intentos dentro del presupuesto de 180 segundos. El JSON observado declara 17 sistemas y estado `ok`, pero carece de `source_commit` y `source_commit_source`. No se sustituyó ese dato por el SHA del libro ni por una declaración ficticia del operador. El cliente V2 no inició las predicciones: `not_started_identity_unavailable`, aceptación pública `false`.

El {download}`reporte público completo <../_static/e3_completion_20261120/release/github_actions_public_report.json>` distingue origen real de los bloqueos previos del proxy. No demuestra que el servicio caiga ni que sus cinco predictores coincidan con la versión local. Conserva respuestas, fecha, cliente de referencia, hashes y alcance. Los pesos/calibradores del perfil operativo siguen intactos; las nueve exportaciones offline y V3 experimental no se activan por publicar este informe.

La versión fuente actual conserva el contrato de salud que informa `RENDER_GIT_COMMIT` y las cinco referencias del cliente. El propietario debe seleccionar **Manual Deploy → Deploy a specific commit**, usar el SHA de 40 caracteres de esta publicación, esperar `Live` y ejecutar el [workflow público](https://github.com/jassan-bit/DeepLearning_202630/actions/workflows/verify-e3-public-release.yml) mediante **Run workflow**. Debe conservar el SHA realmente observado y el recibo V2: páginas, callbacks, cinco predicciones, intervalos aplicables, siete oclusiones por modelo, OOD, errores y huellas. Ese despliegue no se ejecutó desde esta auditoría. Están preparados los {download}`pasos mínimos <../_static/e3_completion_20261120/release/AUTHOR_ACTIONS.md>` y una {download}`celda CPU con preservación de evidencia <../_static/e3_completion_20261120/release/render_public_acceptance_colab_cell.py>`; preparación no significa ejecución.

La misma ejecución consultó el registro público de Zenodo, su API y el resolvedor: los tres devolvieron **HTTP 404 desde origen**; API: «The persistent identifier is not registered», resolvedor: «DOI Not Found». En ese momento no había publicación verificable de `10.5281/zenodo.24356217`. La reserva y el borrador privado aportados antes permanecen como antecedentes; no se confundieron con una publicación ni se solicitó acceso a sus credenciales.
