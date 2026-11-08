# Nueva ejecución científica sobre las 608 fechas históricas de E1

Este estudio ejecuta nuevamente GRU, LSTM y MLP sobre el calendario diario UTC
01/01/2025–31/08/2026. Conserva los anchos solicitados: GRU32, LSTM32 y MLP16,
D7, cinco canales OHLCV, horizonte de un día y objetivo de desviación poblacional
(ddof=0) de siete log-retornos diarios sin anualizar. Las clases históricas se
importan sin cambios. La MLP conserva una capa oculta ReLU de 16 unidades,
flatten ordenado de 35 entradas, dropout y salida Softplus: 593 parámetros.

El calendario ya fue observado y evaluado por el proyecto. Esta nueva ejecución
es histórica y exploratoria; publicar el diseño antes de estos nuevos ajustes no
convierte las 608 fechas en una reserva confirmatoria nueva. No repara
retroactivamente el antiguo dimensionamiento ni garantiza reproducir sus métricas.
Los protocolos, pesos, fuentes congeladas y resultados de todos los estudios
anteriores se conservan. TCN permanece en su estudio complementario separado.

Cada familia recibe la misma búsqueda: learning rate .001/.003 por dropout 0/.1,
cuatro candidatos, semillas nuevas 16101/16102 y diez folds mensuales
enero–octubre 2024. Cada trayectoria contiene 305 fechas; cada bloque selecciona
épocas con validación interna causal y realiza un refit desde cero. Son 24
trayectorias, 240 bloques y 480 etapas nuevas, 160 por familia. Se selecciona por
media de RMSE agregado sobre las 305 fechas de cada semilla, tolerancia 1e-12,
menor número de parámetros e identificador. Entre las tres recetas se congela el
baseline con esa misma regla antes de cualquier ajuste del piloto.

El entrenamiento común usa AdamW, weight decay .0001, batch 32, máximo 100 épocas,
patience 12, min delta 1e-5 y clip 1. Cada proceso usa CPU, un hilo torch y un hilo
interop, algoritmos deterministas, float32 y MKLDNN activo; tres workers con
`spawn`. El registro y los workers guardan las versiones, controles efectivos,
configuración torch, afinidad y cuota CPU. La partición final antes de
01/01/2025 usa 709 muestras en refit, 619 internas y 83 de validación, con embargo
D7 y última etiqueta 24/12/2024. Ninguna etiqueta 2025–2026 entra en entrenamiento.

El piloto nuevo usa 17101–17110 para cada familia, ajuste anterior a 01/11/2024 y
pesos fijos para las 60 fechas 01/11–30/12/2024. El cálculo utiliza las tres SD
muestrales de diferencias pareadas de RMSE, su límite superior simultáneo 80%,
δ=.00025772842664981106, α=.05/3 y potencia marginal .80 de t no central.
`R=max(30, R_required)`; el piso 30 es una regla nueva de estabilidad numérica
prerregistrada, no una herencia del antiguo 465. Si se requieren más de 10000,
se conserva el resultado de inviabilidad y se aborta sin truncar ni afirmar
potencia. Las semillas finales comunes empiezan en 18101 y terminan en 18100+R.
La potencia se refiere a aleatoriedad de entrenamiento condicionada al calendario
bajo las hipótesis registradas; no es potencia conjunta ni temporal futura.

El plan derivado del piloto se publica y verifica por GET antes del primer
ajuste final. Se verifican exactamente 3R checkpoints, escaladores, curvas y
configuraciones antes de cualquier lectura de archivos de precios de esta fase.
La única fase que decodifica la reserva es `score`, con los 20 ZIP mensuales
originales y sus hashes. El contexto proviene del desarrollo congelado 2023–2024.
No se descarga ni abre septiembre–octubre 2026. Se exige la cohorte completa de 608
fechas; datos ausentes o inválidos producen fallo sin imputar, comprimir ni
sustituir fechas. Una extracción inicial incompleta exige revisión y no se
reintenta silenciosamente. Los fallos de workers cancelan trabajos pendientes y
conservan el error y los artefactos; no se reemplazan semillas desfavorables.

El análisis queda fijado y firmado antes de cualquier ajuste. Los tres pares
principales usan t pareada por semilla, Holm 3, intervalos marginales y familiares,
efecto pareado y bootstrap de semillas 20261109. Friedman, Wilcoxon de los tres
pares y QQ son secundarios y siempre se reportan, sin cambiar de prueba según
el resultado. Los vectores privados completos permiten DM-HAC/HLN sobre pérdidas
diarias de modelos individuales, bootstrap circular temporal 20261110 con bloques
7 (sensibilidad 3/14), BDS y Ljung–Box. Persistencia es comparador temporal, no una
cuarta arquitectura entrenada. La incertidumbre de semillas permanece separada
de la dependencia temporal y de la relevancia práctica del margen 5%.

## Secuencia y archivos privados

`prepare` verifica las fuentes históricas y firma todas las fuentes efectivas,
datos de desarrollo, manifiestos, estadísticas, exportador, pruebas, este README
y el lock existente antes de escribir `design.json`; no crea modelos ni
optimizadores. Se ejecuta sólo cuando esas dependencias estén completas y
revisadas. Después se publica el commit científico completo y se verifica por
GET. El recibo incluye `sha`, `observed_at_utc`, una `url` que contenga ese commit y
`verified_by` o `method` con `GET`. Se trabaja desde un checkout limpio e inmutable
del commit, con una raíz privada nueva fuera de todo checkout Git:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py prepare
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py register --output-dir /ruta/privada/nueva --published-commit SHA_FUENTE --publication-receipt /ruta/recibo_fuente.json
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py hpo --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py pilot --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py plan --output-dir /ruta/privada/nueva
```

Se publica una copia exacta del `plan.json` privado en un commit separado. Su
recibo GET incluye además `file_sha256`, igual al archivo publicado. Las fuentes
científicas originales no se modifican para incorporar el plan; se continúa
desde su checkout inmutable:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py register-plan --output-dir /ruta/privada/nueva --published-plan-file /ruta/publicada/plan.json --published-commit SHA_PLAN --publication-receipt /ruta/recibo_plan.json
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py fit --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py freeze --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py score --output-dir /ruta/privada/nueva --raw-root /ruta/archivos_originales
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py analyze --output-dir /ruta/privada/nueva
```

Los recibos documentan verificaciones externas realizadas por el responsable;
el ejecutor no consulta una red implícita ni aporta una prueba de tiempo
independiente. Las métricas y predicciones llevan `study_id`, `execution_id`
(firma del registro) y `source_signature` que vincula fuentes, diseño y commit.
Todos los checkpoints, escaladores, observaciones y predicciones diarias quedan
privados fuera de Git. El exportador público separado sólo admite fuentes,
metadatos, hashes, métricas agregadas, inferencia y curvas de pérdida por época;
revisa el cierre completo y excluye los vectores diarios.
