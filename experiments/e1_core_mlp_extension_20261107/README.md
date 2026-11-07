# Nueva ejecución coherente del núcleo GRU–LSTM–MLP

Esta extensión se registra antes de sus nuevos ajustes. El calendario septiembre–
octubre de 2026 ya fue evaluado en el estudio anterior: la extensión es histórica
y exploratoria; no aporta un holdout confirmatorio nuevo. Se conservan intactos
las fuentes, protocolos, pesos y resultados previos. TCN permanece como estudio
complementario separado, fuera de los tres pares principales nuevos.

Se ajustan las tres familias desde cero durante HPO, piloto y ejecución definitiva.
No se mezclan métricas, pesos ni vectores anteriores con los nuevos. Cada fila de
resultados privados conserva `study_id=e1_core_mlp_extension_20261107` y
`execution_id` igual a la firma del nuevo registro; las estadísticas rechazan
procedencias o cohortes mezcladas.

Las tres familias comparten D7, cinco canales OHLCV diarios, objetivo poblacional
de siete log-retornos diarios y horizonte de un día. El preprocesamiento relativo
usa el último cierre ya observado de cada ventana; medias/escalas se aprenden
sólo en entrenamiento. El objetivo se divide por una media positiva de
entrenamiento sin centrarlo. Los modelos producen una salida Softplus no negativa.

GRU y LSTM importan sus clases históricas intactas. MLP importa `SmallMLP`: ventana
ordenada de35 entradas, una capa oculta densa ReLU, dropout y capa de salida densa.
La elección de familia superficial se apoya en los comparadores cripto de
García-Medina/Aguayo-Moreno y Dudek. La variante ReLU se vincula al experimento de
una capa del AppendixB de *Volatility Forecasting with Machine Learning and
Intraday Commonality*, conservando su condición de evidencia de transferencia de
acciones. El objetivo, los inputs, la salida positiva y la rejilla son adaptaciones
del proyecto; no se presenta una réplica exacta de esos artículos.

Cada familia dispone de ocho candidatos: dos anchos, dos learning rates
(.001/.003) y dropout0/.1. GRU/LSTM mantienen anchos16/32. MLP busca32/128, con
1185/4737 parámetros, para abarcar un intervalo comparable al de las recurrentes.
Igual esfuerzo de búsqueda no significa idéntico número de parámetros.

HPO usa semillas6101/6102 y los diez folds enero–octubre de2024, 305 fechas por
trayectoria. Cada candidato/semilla/mes selecciona épocas mediante validación
interna causal y realiza un refit nuevo. Son48 trayectorias,480 bloques y960 etapas
en total,320 etapas por familia. Se selecciona por media de RMSE calculado sobre
las305 fechas de cada semilla, con tolerancia de empate1e-12, luego menor número
de parámetros e identificador. El baseline entre las tres recetas se congela
antes del piloto; ninguna familia se declara ganadora por anticipado.

El piloto nuevo usa7101–7110 para cada familia, fit antes del01/11/2024 y pesos
fijos para las60 fechas01/11–30/12/2024. A partir de sus tres pares se recalcula
el tamaño completo con δ=.00025772842664981106, α=.05/3, potencia marginal.80 y el
límite superior simultáneo de dispersión registrado. `R_final=max(465,R_required)`;
un requerimiento superior al límite numérico10000 produce un fallo explícito,
sin truncamiento oculto ni afirmación de potencia. Las semillas finales comunes
son8101..8100+R. El piso465 no procede de un dimensionamiento antiguo para MLP.

El JSON del plan calculado se publica y verifica antes del primer ajuste final.
Cada ajuste definitivo aprende exclusivamente de2023–2024 con el mismo embargo:
709 ejemplos de entrenamiento,619 de entrenamiento interno y83 de validación.
La congelación verifica exactamente3R checkpoints, escaladores, curvas y
configuraciones antes de permitir lectura de precios de la reserva.

## Secuencia ejecutable

Primero se publica `design.json`, ejecutor, adaptador, estadísticas, exportador y
pruebas. Desde un checkout limpio e inmutable de ese commit, en el entorno fijado:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py register --output-dir /ruta/privada/nueva --published-commit SHA_FUENTE --publication-receipt /ruta/recibo_fuente.json
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py hpo --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py pilot --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py plan --output-dir /ruta/privada/nueva
```

El recibo de fuente registra un GET de GitHub anterior al registro, con `sha` y
`observed_at_utc`. Se publica una copia exacta de `plan.json`; otro GET verifica
su commit y bytes. Su recibo debe incluir `sha`, `observed_at_utc` y
`file_sha256`, igual al SHA256 del archivo publicado. Se continúa desde el mismo
checkout científico original, sin modificar fuentes para incorporar el plan:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py register-plan --output-dir /ruta/privada/nueva --published-plan-file /ruta/publicada/plan.json --published-commit SHA_PLAN --publication-receipt /ruta/recibo_plan.json
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py fit --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py freeze --output-dir /ruta/privada/nueva
```

Los argumentos y recibos documentan lo observado por el responsable; no son una
consulta remota implícita del ejecutor ni una prueba independiente de tiempo. La
fecha del commit sola no acredita su publicación anterior a los ajustes.

HPO/piloto/plan/ajustes no requieren precios de reserva. `score` es una fase
explícita posterior, admite únicamente los32 ZIP originales con sus hashes
congelados: agosto de contexto, septiembre y30 archivos diarios de octubre. Si
faltan datos o hashes válidos, se conserva ese bloqueo sin fabricar métricas ni
reemplazar fechas. Una extracción parcial falla y exige revisión; no se relee
silenciosamente. `analyze` utiliza exclusivamente predicciones privadas guardadas:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py score --output-dir /ruta/privada/nueva --raw-root /ruta/archivos_originales
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py analyze --output-dir /ruta/privada/nueva
```

Los tres pares principales usan métricas por semilla del nuevo núcleo, Holm e
intervalos familiares. El análisis temporal secundario incluye persistencia y
conserva el estimando de media de RMSE por modelo individual; no convierte días
en semillas independientes. Bootstrap de semillas20261107 y temporal20261108
quedan fijados antes del ajuste. Las pérdidas por época, runtime efectivo y todos
los resultados se conservan. El exportador público separado sólo permite código,
metadatos, hashes, pérdidas y agregados; pesos, escaladores, observaciones y
predicciones diarias permanecen en la raíz privada exterior al checkout.
