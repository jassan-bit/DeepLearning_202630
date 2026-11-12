# 48. Protocolo reducido para la entrega del 12 de noviembre

**Paso 1: registro de reglas, datos, caché y presupuesto.** La enmienda conserva los dieciséis sistemas, la métrica y el contrato de E1; reduce las oportunidades HPO antes de nuevas ejecuciones. Este registro no acredita entrenamiento nuevo, aceptación de todos los modelos en GPU ni una modificación de la sesión privada de Colab.

La versión científica anterior y sus recibos permanecen en [§45](45_enmienda_protocolo.md). El [informe consolidado de §47](47_entrega_consolidada.md) conserva los resultados existentes. No se renombran como resultados del protocolo reducido.

## Contrato y alcance conservados

La tarea sigue siendo **D7×5 OHLCV → σ₇ poblacional de siete retornos logarítmicos de Close**, horizonte de un día y sin anualización. Se conserva la **media de las RMSE individuales por semilla en unidades originales** como métrica primaria y **δ=0,00025772842664981106**. No se sustituye por RMSE del ensamble ni se mezclan escalas entre folds.

El catálogo contiene diez candidatos de cuatro familias, las referencias **MLP16, LSTM32 y GRU32**, y **Persistence, SeasonalNaive y ARIMA**: **13 redes + 3 comparadores = 16 sistemas**. El híbrido adicional y Ridge permanecen opcionales/históricos.

## Seis plazas HPO fijadas antes de comparar resultados

Cada red recibe las mismas seis plazas **C01–C06**, una semilla efectiva de optimización, **271201**, y los mismos tres folds temporales internos. Los puntos se extraen de las configuraciones originales de cada receta sin modificar sus valores numéricos.

| Plaza común | Posición en el dominio original de 18 puntos, base cero |
|---|---:|
| C01 | 5 |
| C02 | 7 |
| C03 | 8 |
| C04 | 9 |
| C05 | 10 |
| C06 | 15 |

**Plazas comunes no significa seis tuplas numéricas idénticas entre arquitecturas.** El generador original es específico por receta. Se conserva una regla común, el número de oportunidades y los espacios aplicables; el [CSV de configuraciones](../_static/reduced_protocol_20261112/hpo_configurations.csv) expone LR, weight decay, batch y demás campos por modelo/plaza. Los cuatro cuerpos restaurados mantienen los parámetros estructurales y dropout de su fuente.

La elección usa exclusivamente metadatos HPO: cobertura de batch **16/32/64**, weight decay cero y positivo, y warmup cero; después maximiza la menor cobertura normalizada del intervalo logarítmico de LR entre recetas, con desempates deterministas. No utiliza RMSE, rankings de modelos ni tiempos observados para elegir los seis puntos. El subconjunto no representa cobertura completa del dominio original.

Son **13 × 6 × 1 × 3 = 234 ajustes lógicos HPO**, antes de reutilización comprobada. SAE añade 36 etapas auxiliares: **270 etapas optimizadoras** en selección. Los reajustes finales, pilotos, análisis y cualquier NAS aplicable se contabilizan aparte. Las 18 oportunidades y dos semillas del protocolo anterior se conservan como procedencia; el nuevo ejecutor debe usar únicamente la semilla efectiva registrada.

## Tres folds internos y una receta global para la evaluación posterior

Las fechas de mercado son días UTC del contrato E1. Las marcas de registro también se presentan en hora de Bogotá.

| Origen del fold interno | Entrenamiento interno hasta | Validación interna, inclusive |
|---|---|---|
| 01/01/2024 | 25/09/2023 | 03/10–24/12/2023 |
| 01/07/2024 | 25/03/2024 | 02/04–23/06/2024 |
| 01/10/2024 | 25/06/2024 | 03/07–23/09/2024 |

Se mantienen los desplazamientos **o−98** para entrenamiento y **o−90…o−8** para validación, con embargo de siete días. Cada transformación se ajusta únicamente con su entrenamiento admisible.

La receta de cada red será la plaza con **menor RMSE sobre las predicciones de validación concatenadas**, conservando la agregación del contrato E1. Se calcula `sqrt(sum(n_fold × RMSE_fold²) / sum(n_fold))`: 83 fechas comunes por fold, 249 en total. La media aritmética de las tres RMSE queda sólo como diagnóstico. El empate exacto se resuelve por ID estable. Las épocas de reajuste son las medianas de las tres mejores épocas ganadoras; SAE conserva por separado AE1, AE2 y supervisado. Se mantienen los límites de entrenamiento y el horizonte original del scheduler.

La decisión requiere el dominio completo **seis plazas × tres folds × una semilla** con recibos y estados explícitos. Un fallo consume su oportunidad; no autoriza otra semilla ni una configuración de reemplazo. Un error sistemático bloquea la selección.

Esta selección global sólo se aplica a la evaluación que comienza en enero de 2025, posterior a los tres folds. **No se atribuye esa receta global a pronósticos externos de enero–octubre de 2024.** Los resultados internos siguen siendo resultados de selección.

## Ajuste, calibración y evaluación común

| Etapa | Fechas y regla |
|---|---|
| Entrenamiento de pesos y transformaciones finales | Etiquetas admisibles hasta **31/10/2024**, con el mismo corte para las trece redes y referencias |
| Puente de embargo | **01–07/11/2024** |
| Calibración y piloto de desarrollo | **08/11–30/12/2024**, 53 días calendario; recetas previamente congeladas, sin reajuste HPO |
| Puente posterior | **31/12/2024**, observaciones causales para ventanas D7 y estado de comparadores |
| Evaluación heredada de E1 | **01/01/2025–31/08/2026**, 608 fechas calendario y una máscara común verificable |

Reservar calibración dentro de 2023–2024 es una subpartición del desarrollo; no cambia el periodo externo heredado. Todos los sistemas deben respetar esa separación. Los pesos, escaladores y calibradores utilizados en calibración permanecen iguales durante la evaluación: **no se realiza otro refit con etiquetas de diciembre después de calibrar**.

El 31 de diciembre puede alimentar las entradas históricas y la persistencia; ARIMA puede actualizar su estado con coeficientes congelados cuando corresponda. No puede ajustar de nuevo pesos, hiperparámetros, escaladores o calibradores. Su admisión y el soporte de las 608 fechas deben comprobarse antes de evaluar: el loader de desarrollo limitado al 30 de diciembre no basta para el primer pronóstico de enero.

La guía pide heredar la partición de E1; **no exige cien días prospectivos nuevos**. La exposición previa de E1 queda declarada. Se congela la campaña actual antes de su scoring, se evalúa una sola vez y no se modifican decisiones a partir de esos resultados. No se afirma haber recuperado el cegamiento histórico ni una confirmación prospectiva.

## R, arquitectura y aceptación GPU siguen teniendo condiciones de cierre

**R=6 es una decisión operativa acotada por recursos**, fijada antes del análisis de calibración. No acredita potencia del 80 %. Se requiere el piloto registrado de al menos cinco semillas sobre dos representantes, con potencia y diferencia mínima detectable en la escala de RMSE. Si la potencia no alcanza el objetivo, se informa la limitación con el R disponible; no se aumentan semillas automáticamente ni se presenta la ausencia de significancia como equivalencia.

Las doce redes con cuerpo actualmente definido tienen su identidad de receta. **TiDE permanece bloqueado para la ejecución reducida** hasta verificar una receta publicada compatible y registrar su identidad propia. Las seis filas TiDE heredadas identifican puntos HPO de procedencia; no convierten la variante NAS en un cuerpo publicado. Si la arquitectura continúa requiriendo NAS, el requisito aplicable de la guía permanece y no se elimina por urgencia.

La comprobación local identifica **Python 3.12.14 y Torch 2.8.0+cpu, sin CUDA disponible**. Esto no describe la GPU privada de Colab. El [notebook de una celda para verificarla](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_comprobar_protocolo_reducido.ipynb) sólo genera un diagnóstico; no entrena ni instala paquetes.

La comprobación local identifica el entorno realmente disponible. La GPU de la sesión privada debe verificarse en Colab: disponibilidad, versiones, dispositivo y una pequeña prueba CUDA. Una prueba de matriz no acredita todos los kernels de los modelos ni inicia HPO/NAS.

## Qué acredita la comprobación de caché y presupuesto

**Resultado del paso 1:** 61 resultados HPO coinciden con las nuevas plazas y las firmas de datos preparados; siguen condicionados a verificar el trabajador. Quedan 155 ajustes sin resultados de la receta vigente y 18 plazas TiDE bloqueadas. Hay **cero reutilizaciones aprobadas y cero pesos finales reutilizables** en este corte.

La auditoría conserva la trazabilidad de ambos diarios y comprueba los archivos de desarrollo registrados, los parámetros de cada plaza y los recibos de ajuste. Distingue una **candidatura reutilizable** de una reutilización totalmente verificada: todavía deben coincidir datos preparados, receta, semilla y política de ejecución efectiva.

La campaña histórica guardó **metadatos de selección**, no pesos finales. Un resultado HPO compatible puede evitar repetir esa oportunidad de selección; no sustituye los reajustes finales, sus curvas ni sus checkpoints. Tampoco se reutiliza directamente un ganador elegido entre 18 puntos como ganador del nuevo dominio de seis.

El techo permanece en **64 horas acumuladas**, con todos los cargos anteriores: **222.927,99 segundos consumidos y 7.472,01 segundos de saldo observado**. La siguiente reserva histórica de 7.500 segundos no cabe; este paso no cambia esa reserva. El [recibo de conciliación](../_static/reduced_protocol_20261112/budget_reconciliation.json) fija el corte observado, no un saldo garantizado en vivo. La enmienda no reinicia el presupuesto ni concede más tiempo. La política anterior de reserva completa debe migrarse y validarse antes de admitir ajustes nuevos; reducir el dominio HPO no modifica por sí solo esa política.

## Archivos de este registro

- [Protocolo reducido](../_static/reduced_protocol_20261112/protocol.json) · [configuraciones completas](../_static/reduced_protocol_20261112/configurations.json) · [configuraciones CSV](../_static/reduced_protocol_20261112/hpo_configurations.csv) · [calendario](../_static/reduced_protocol_20261112/calendar.csv).
- [Auditoría de datos y caché](../_static/reduced_protocol_20261112/data_cache_audit.json) · [presupuesto conciliado](../_static/reduced_protocol_20261112/budget_reconciliation.json) · [entorno local observado](../_static/reduced_protocol_20261112/environment_local.json).
- [Recibo de fuentes y hashes](../_static/reduced_protocol_20261112/registration.json) · [registro Git y fecha real](../_static/reduced_protocol_20261112/git_registration.json) · [verificación](../_static/reduced_protocol_20261112/verification.json).

El cierre de este primer paso es un registro verificable de reglas y comprobaciones. Los recibos de ejecución posteriores deberán acreditar GPU, arquitectura pendiente, presupuesto admitido, ajustes y resultados; esta página no los reemplaza.
