# Prueba de humo del despliegue E3

El guion `smoke_render.py` visita una aplicación existente mediante Chromium.
No crea un servicio, no instala nada en Render, no importa el servidor ni
reentrena, reajusta escaladores o recalibra modelos. La dependencia Playwright
pertenece al cliente de desarrollo, no al contenedor del dashboard.

## Antes del despliegue público

La aplicación y el Dockerfile ya tienen la evidencia local conservada en el
capítulo 9. El nuevo cliente se probó contra Gunicorn y contra ese contenedor
local. Sus recibos usan `scope=local_preflight_not_public_deployment` y
`render_public_smoke_passed=false`, incluso cuando `status=passed`.

```bash
python -m venv work/render-smoke-venv
work/render-smoke-venv/bin/python -m pip install -r requirements-dashboard-dev.lock.txt
work/render-smoke-venv/bin/python -m playwright install --with-deps chromium
work/render-smoke-venv/bin/python -m unittest discover -s tests -p test_render_smoke_contracts.py -v
```

El cliente CI usa Python 3.12.14 y Ubuntu 24.04, compatible con Playwright 1.56.
El lock de desarrollo incluye el lock de inferencia; ninguna de esas versiones
se instala adicionalmente en el servicio cuando se ejecuta esta comprobación.

## Sobre una URL de Render ya desplegada

Consultar primero el log del despliegue para anotar el SHA de 40 caracteres.
Sustituir los valores del ejemplo por el origen HTTPS y ese SHA reales; el
ejemplo siguiente no identifica un servicio existente:

```bash
work/render-smoke-venv/bin/python experiments/e3_bloques_fg_20261117/smoke_render.py \
  --url https://NOMBRE-REAL.onrender.com \
  --deployment-commit SHA-REAL-DE-40-CARACTERES \
  --output work/render-public-smoke/report.json \
  --screenshots-dir work/render-public-smoke/captures
```

También se puede ejecutar manualmente el workflow
**Verificar cliente de humo E3 y despliegue público solicitado**, introduciendo
la URL y el SHA. En cada push que afecte al cliente sólo se prueban sus seis
contratos sin red; la prueba pública depende de `workflow_dispatch`. El workflow
conserva el JSON y las capturas como artefacto incluso cuando la comprobación
falle. No hay una ejecución pública aprobada mientras no exista ese recibo.

El cliente rechaza HTTP público, destinos privados, credenciales en la URL,
consultas y hosts que no sean `nombre.onrender.com`. Para una comprobación
local se requiere `--allow-local` y un origen loopback. Ese modo no puede
certificar Render. Los dominios propios requieren un cambio explícito y
revisado de la admisión. El healthcheck y la lectura de recursos no siguen
redirecciones; el navegador bloquea solicitudes iniciales a otros orígenes y
exige que la navegación termine en el origen configurado. Un cambio de origen
no puede producir una aprobación de humo público de Render.

## Qué se comprueba

1. `/healthz` debe identificar `e3-dashboard`, estado `ok` y 17 sistemas. El
   presupuesto inicial de disponibilidad es 180 s; se registran los intentos
   y sus estados. Esa espera no demuestra que hubiera un arranque frío.
2. Contexto, EDA y Modelos responden HTTP 200 y completan sus componentes en
   30 s por página. EDA actualiza las seis figuras con la misma máscara vacía
   al retirar las particiones; Modelos mantiene las 17 filas.
3. La ficha del modelo, declaración de datos y CSS local coinciden por SHA-256
   con el checkout del cliente. Esto comprueba esos recursos, no toda la fuente
   remota ni el proveedor de alojamiento por sí solo.
4. MLP16, LSTM32, GRU32 y CNN-LSTM aditiva E3 ejecutan una consulta real mediante
   el botón de la interfaz y el callback HTTP. Se usa únicamente
   `evaluation-0`, objetivo 2025-01-01, fijado por posición y no por rendimiento.
   Cada número mostrado debe ser finito, no negativo y diferir del predictor
   congelado menos de `1e-6` en unidades originales de σ₇. La presentación tiene
   seis decimales; este contraste no sustituye la paridad completa de 608 fechas.
5. Cada consulta devuelve las siete sensibilidades por oclusión finitas y el
   aviso de interpretación no causal. Las tres referencias muestran su
   intervalo nominal del 90 %; E3 declara la ausencia de intervalo calibrado.
   El límite por consulta completa es 30 s, incluyendo red, explicación y
   renderizado. Se registra aparte la latencia CPU que muestra la aplicación,
   que excluye explicación. Los umbrales se registran antes de medir y pueden
   cambiarse por argumentos; el informe debe conservar los valores usados.
6. Después de quedar listo, no se aceptan HTTP 5xx internos, errores de consola
   o excepciones no capturadas de navegador. Un fallo devuelve código de salida
   distinto de cero y conserva el recibo con `status=failed`.

## Identidad de versión y límites

El healthcheck actual no expone un SHA de despliegue. Por ello,
`operator_declared_deployment_commit` documenta lo anotado por el operador y
`remote_commit_identity_verified` sigue siendo `false`. Para cerrar la
trazabilidad se debe anexar el log de Render con servicio, commit, fecha y
configuración; no basta con copiar un SHA en la línea de comandos. Los hashes
de recursos, la paridad y las capturas son comprobaciones adicionales de la
aplicación observada.

El guion no mide memoria pico remota, concurrencia, disponibilidad prolongada,
frontera de coste científica ni arranque frío demostrado. Tampoco acredita
reproducción humana independiente o las condiciones R1, R2 y R3. Una oclusión
disponible no demuestra fidelidad explicativa ni estabilidad entre semillas.
Los trece sistemas precalculados no se presentan como inferencia en vivo.

## Contingencia de la sustentación

Comprobar la URL con antelación, conservar las capturas de esa versión y grabar
un vídeo breve de la navegación y la predicción. Si el servicio falla durante
la exposición, identificar el incidente y el alcance del respaldo. El vídeo
o una captura local no sustituyen una URL pública funcional ni una prueba de
humo pública aprobada.
