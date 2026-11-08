# Decisiones, evidencia y alcance de E1

La selección vigente es **LSTM–MLP–GRU**. TCN conserva su evidencia y resultados como comparación complementaria. El {ref}`capítulo 10 <final-core-mlp-e1>` identifica la ejecución nueva y el [capítulo 5](05_benchmark.md) concilia los antecedentes compatibles del núcleo. Las versiones anteriores conservan su identidad y cronología.

## Literatura: selección y trazabilidad

| Requisito | Evidencia entregada | Alcance |
|---|---|---|
| **§3.1: consulta de WoS, Scopus e IEEE** | {ref}`Bitácora nativa <current-native-execution-e1>`: cadenas comunicadas, fecha confirmada del 06/11/2026 en Colombia, campos, filtros y totales **271/538/32**; exportados **271/538/25** | Consulta acreditada; resultados, exportaciones y textos leídos se cuentan por separado |
| **§3.2: recurrencia de tres modelos base** | {ref}`Selección vigente <core-selection-e1>`: estrato cripto de diez experimentos con reporte suficiente, **LSTM 10, MLP 4, GRU 2, TCN 1**; selección de las tres primeras familias | Frecuencias del corpus examinado, con búsquedas dirigidas y cobertura declaradas |
| **§3.2: simplicidad** | Una capa oculta MLP y una capa recurrente GRU/LSTM; fuentes, diferencias y parámetros explícitos; candidatos MLP **1.185/4.737**, GRU **1.121/3.777**, LSTM **1.489/5.025** | Profundidad, estados y capacidad comprobables; la receta ganadora depende de desarrollo |
| **§3.2: diversidad** | Ventana ordenada sin recurrencia, estado GRU con reset/actualización y memoria LSTM separada; mismo D7 y covariables | Diferencias funcionales justificadas; no se infiere independencia de errores |
| **Originales y fuentes gratuitas** | {ref}`Síntesis de 57 obras <homogeneous-expansion-e1>`, 35 textos primarios completos y una transcripción; catálogo canónico de 70 referencias; autores, DOI y nivel de acceso por fuente | Catorce informes históricos no recuperados conservan disposición y motivo; alternativas verificadas cubren funciones, sin atribuir lecturas inexistentes |

Esta actualización **cambia la selección y la comparación experimental** para hacerlas concordar con el criterio documental. No elimina artículos del flujo para alterar frecuencias, no transforma la cuota de búsqueda en prevalencia ni atribuye las métricas de TCN a MLP. Los cortes de siete estudios, 47 obras y 57 obras permanecen fechados y con sus denominadores propios.

Las sesiones históricas de WoS/Scopus conservan cadenas y horas no recuperadas. La ronda nativa posterior permite documentar consultas reales de las tres bases sin reconstruir esos valores. IEEE exportó 25/32 resultados; los siete restantes son una limitación del universo exportado, no una ausencia de consulta efectiva.

## Procedimiento y evidencia experimental

La tarea mantiene **σ₇ poblacional, D7, horizonte de un día y cinco covariables OHLCV**. Las transformaciones se aprenden únicamente del entrenamiento permitido y los pesos permanecen fijos dentro de la evaluación. Las cuatro entradas D=7/14/21/28 conservan métricas históricas del núcleo con diez semillas y los mismos calendarios.

El nuevo [protocolo público](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) establece HPO, piloto y ajustes nuevos de GRU, LSTM y MLP. No mezcla sus repeticiones con las del estudio anterior. Cada familia tiene ocho candidatos, dos semillas de búsqueda y diez bloques comunes: **320 etapas de ajuste por familia**. La selección y el baseline se congelan con desarrollo antes de puntuar.

El piloto completó diez semillas nuevas por familia, treinta trayectorias y sesenta etapas. Requiere **R=331** para sus tres pares; el piso previo produce **R final=465**, con α*=0,05/3, potencia marginal objetivo 0,80 y cota simultánea del 80 % de la DE. El [plan exacto](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) quedó publicado y verificado antes del primer ajuste definitivo. La cifra histórica R=465 no se presenta como dimensionamiento antiguo de MLP.

El capítulo 10 entrega configuraciones, cronología, pérdidas de entrenamiento/validación y análisis propios: **1.395 modelos, 2.790 etapas definitivas, 83.700 predicciones y 44.201 registros de pérdida**. La ejecución completa comprende 3.810 etapas contando HPO y piloto. Todos los pesos quedaron congelados antes de abrir observaciones de evaluación. Una revisión independiente recalculó las seis métricas con diferencia máxima cero y verificó t/Holm, DM, bootstrap y archivos públicos. Los manifiestos comprueban fuentes, semillas, cohortes y firmas; los artefactos privados se separan de los agregados públicos.

Los resultados se conservan completos: GRU y MLP no rechazan igualdad de medias de RMSE tras Holm (**p=0,243**); ambas superan a LSTM entre inicializaciones. Persistencia tiene el menor error puntual. Ninguno de los seis DM rechaza con Holm y ningún límite temporal familiar acredita mejora relativa superior al 5 %. Los resultados inconclusos o adversos son resultados entregados, con su alcance inferencial explícito.

## Margen académico

**δ=0,00025772842664981106** conserva el ancla del 5 % del RMSE CV del GRU inicial. El autor confirmó como uso la comparación académica de modelos. La [justificación de §1.1.4](01_problema_dataset.md) sitúa su magnitud frente a la variabilidad entre semillas y del objetivo, con escenarios del 2,5 % y 10 %.

Es una elección normativa documentada, posterior al test inicial y anterior a los ajustes definitivos siguientes. No se redefine usando los resultados nuevos ni demuestra utilidad financiera, equivalencia o una tolerancia óptima. Potencia, magnitud práctica e incertidumbre temporal tienen interpretaciones distintas.

## Versiones históricas y reproducibilidad

- **Benchmark inicial:** cuatro entradas y diez semillas; test de 608 fechas ya observado. El [suplemento histórico del núcleo](05_benchmark.md) deriva métricas de archivos reales y distingue los esfuerzos de ajuste originales.
- **Estudio anterior GRU–LSTM–TCN:** el [capítulo 9](09_estudio_planificado.md) conserva HPO, piloto, R=465 y evaluación del 01/09–30/10/2026, con resultados adversos e inconclusos visibles.
- **Curvas reproducidas posteriormente:** {ref}`§9.10 <curvas-finales-reproducidas>` conserva 1.395 trayectorias y 2.790 etapas de desarrollo. La discrepancia de hashes con los archivos históricos se mantiene explícita.
- **Repetibilidad actual:** la {ref}`auditoría numérica <auditoria-numerica-e1>` obtiene 18/18 coincidencias canónicas en dos procesos independientes para seis celdas fijadas. No recupera los archivos privados originales.
- **Nueva ejecución del núcleo:** tiene fuente, registro, curvas y métricas propios. Sus resultados se interpretan con su calendario y no reemplazan silenciosamente los anteriores.

La guía §4.3 exige semillas, declaración de determinismo y un paquete ejecutable; no se presenta la falta de archivos privados históricos idénticos como ausencia de entrenamiento. Al mismo tiempo, las curvas nuevas no prueban identidad numérica con un entrenamiento cuyos tensores originales no están disponibles.

## Límites de interpretación

La evaluación de septiembre–octubre de 2026 **ya es conocida**. La nueva extensión es histórica y exploratoria, aunque su protocolo preceda a sus propios ajustes. No recupera independencia confirmatoria para una hipótesis escogida después de observar ese periodo.

Sesenta días de BTCUSDT no constituyen otro mercado. La planificación entre semillas no demuestra potencia temporal del 80 % ni potencia conjunta del 80 %. Los contrastes t requieren interpretar distribución y dependencia de las diferencias; DM y bootstrap temporal conservan sus supuestos y multiplicidad. Los resultados no permiten afirmar rentabilidad, equivalencia o generalización universal.

E2 debe declarar una versión inmutable de su baseline y registrar una evaluación propia para la propuesta avanzada. El baseline histórico GRU/D7 de 32 unidades y las recetas posteriores no se confunden. Ningún cambio del núcleo de E1 reescribe automáticamente experimentos o configuraciones ya cerrados en E2.
