# 62. Elegibilidad, selección documental y brecha sustentada

**Actualización del 14/11/2026, Bogotá.** Se completa un nuevo escenario documental con **50/50 componentes y diez puntuaciones calculables**, se verifica disponibilidad identificada de datos y código para las nueve fuentes del catálogo y se actualizan las decisiones de 30 objetivos aplicados. La búsqueda recibida, las decisiones de elegibilidad y el puntaje tienen denominadores separados. **El PRISMA global y la atribución prospectiva de la selección siguen abiertos.**

## Qué requisito se resuelve y qué evidencia lo acredita

| Producto evaluable | Estado de esta actualización | Evidencia |
|---|---|---|
| Citas uniformes de las fuentes actuales | **9/9 estudios, 10/10 sistemas**; FCN y ResNet comparten Wang | [Recibo](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/receipt.json) y [registros Scopus](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/records.csv) |
| Tabla explícita C/U/P/A/F | **50/50 componentes calculables en un escenario posterior**; se conserva el escenario P9 y el histórico | [Veinte filas de los dos escenarios](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/conditional_scores.csv) y [fórmulas](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/scenario_protocol.json) |
| Extracción y calidad de las fuentes del catálogo | **9 estudios / 10 sistemas**, con campos conocidos, NR y NV identificados | [Matriz vigente](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_quality.csv), [puente estudio–sistema](../_static/e2_literature_delivery_closure_20261114_v3/quality/model_study_bridge.csv) y disponibilidad abajo |
| Disponibilidad pública identificada | **Datos 9/9 y código de autor 9/9**; no equivale a reproducción | [Evidencia por estudio](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv) y [verificación oficial](../_static/e2_literature_delivery_closure_20261114_v3/quality/verified_urls.json) |
| Extracción y calidad aplicada | **14 inclusiones narrativas**, con población, target, partición, métricas, repeticiones, HPO, código y límites | [Matriz de catorce estudios](../_static/e2_literature_delivery_closure_20261114_v3/quality/quality_extraction_matrix_14.csv) |
| Elegibilidad focal de treinta objetivos | **14 incluidos, 10 excluidos del núcleo aplicado cripto, 4 sin original y 2 con constructo pendiente** | [Registro completo](../_static/e2_literature_delivery_closure_20261114_v3/eligibility.csv) y [flujo](../_static/e2_literature_delivery_closure_20261114_v3/focal_flow.json) |
| PRISMA de todo el universo histórico | **Abierto**: las decisiones focales no sustituyen el cribado y elegibilidad globales | [Conciliación histórica (§59)](59_auditoria_literatura_final.md) |

## Citas y puntuación reproducible de los diez sistemas

La exportación dirigida del usuario contiene **nueve EID únicos**, todos Scopus, con título, autores, año y número de citas. Sus identidades cubren las nueve fuentes originales de los diez sistemas. Wang identifica FCN y ResNet; TSMixer es *An All-MLP Architecture* de Chen et al., distinto del modelo de KDD. En seis registros no hay DOI editorial en la exportación: se conserva vacío y el enlace se apoya en título, autores, fuente y EID. No se suman citas de versiones.

Todos los conteos nuevos proceden del mismo archivo recibido el **14/11/2026**. El CSV no incorpora hora de consulta; el recibo distingue la fecha de recepción, la captura del usuario y los límites de la consulta nativa visible. Se conservan sin cambios los seis C previos y sus cuatro valores desconocidos en el paquete anterior. La tabla nueva usa:

```text
C_nuevo = clip(citas_Scopus / (2026 − año_inicial_del_estudio + 1) / 487, 0, 1)
S = 100 × (0,15 C + 0,15 U + 0,25 P + 0,25 A + 0,20 F)
```

El año inicial incluye la primera versión conocida del estudio; por ejemplo, Wang 2016 frente a editorial 2017 e Inception 2019 frente a editorial 2020. Se conserva la referencia fija 487, la regla de antigüedad y los pesos anteriores. El clip de PatchTST se declara; no se recalibra el denominador para obtener un resultado favorable.

| Sistema | Citas Scopus | C | U | P contextual | A | F | S documental |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN-1D | 2248 | 0.4196 | 0.0000 | 1.0000 | 1.0000 | 0.2500 | 61.29 |
| ResNet-1D | 2248 | 0.4196 | 0.0000 | 0.9362 | 1.0000 | 0.2500 | 59.70 |
| Inception-1D | 1761 | 0.4520 | 0.0000 | 0.0000 | 1.0000 | 0.2500 | 36.78 |
| LSTM-autoencoder con predictor | 321 | 0.0824 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 46.24 |
| TimesNet | 1886 | 0.7745 | 0.4286 | 0.9712 | 1.0000 | 0.2500 | 72.33 |
| TSMixer | 247 | 0.1268 | 0.5714 | 0.7209 | 1.0000 | 0.3333 | 60.16 |
| TiDE | 247 | 0.1268 | 0.5714 | 0.6295 | 1.0000 | 0.3333 | 57.88 |
| TimeMixer | 619 | 0.4237 | 0.7143 | 1.0000 | 1.0000 | 0.3333 | 73.74 |
| PatchTST | 2477 | 1.0000 | 0.4286 | 0.8633 | 1.0000 | 0.5000 | 78.01 |
| iTransformer | 1378 | 0.7074 | 0.5714 | 0.9640 | 1.0000 | 0.5000 | 78.28 |

Las filas se muestran en el orden del catálogo, no como un nuevo ranking global. U, A y F conservan el escenario previo; C procede del nuevo corte. **La aritmética de los diez scores está completa; su interpretación sigue siendo condicional a escalas locales de desempeño que no están calibradas entre tareas.** No acredita igualdad de HPO ni eficacia sobre Bitcoin y no convierte este puntaje posterior en la causa de la selección autorizada el 12/11.

### Desempeño de una red Inception, separado del ensemble

Se verificaron los [resultados numéricos del autor](https://github.com/hfawaz/InceptionTime/tree/952d115e83fb3a66d75c5858c702f8dd5eeb18b7): el archivo de redes individuales contiene **640 filas, 128 datasets UCR2018 × cinco ejecuciones**, y el del ensemble de cinco redes contiene 128 filas para los mismos nombres. El promedio uniforme del error de cada ejecución individual es **0,1639239119738714**; el error medio del ensemble es **0,1564056859794546**. Promediar errores de redes individuales no combina sus predicciones.

El nuevo pool local contiene esas dos variantes publicadas y utiliza `P=(máxima pérdida−pérdida)/(máxima pérdida−mínima pérdida)`. Produce **P=0 para la red individual**: sólo significa la mayor pérdida entre esas dos variantes, no eficacia absoluta nula. Es error medio de clasificación, **no MPCE**, no la tabla UCR85 del paper ni una medición del predictor D7 de E2. No se verificaron los bytes de los splits originales ni igualdad de coste. Se conservan los [CSV, comprobaciones y cálculo](../_static/e2_literature_delivery_closure_20261114_v3/inception_single/candidate.json).

La regla P9 anterior permanece intacta. La extensión de Inception se identifica como **nuevo escenario de artefactos de autor UCR2018/128**, formulado después de ver esos datos. Los otros nueve P siguen sus pools anteriores: UCR44 para FCN/ResNet; RMSE de Beijing para SAE; MSE ETTh1/H96/L96 dentro de las tablas elegidas para los seis predictores restantes. Se publica también el escenario de **49/50 componentes**, que conserva P9 sin esa extensión. Las sensibilidades MAE/SMAPE anteriores no se eliminan; SAE sigue teniendo P=0 con RMSE y P=1 con esas métricas alternativas en su pool.

## Elegibilidad aplicada: cuatro decisiones nuevas, con su etapa real

El denominador permanece en **30 objetivos por DOI**, provenientes de la recuperación focal anterior; no es una muestra aleatoria ni toda la revisión global. Las decisiones cambian de 13/7/8/2 a **14/10/4/2**:

- **ARCH–LSTM**, DOI `10.3390/jrfm19090688`: una nueva lectura focal del cuerpo editorial acredita BTC/ETH, varianza condicional evaluada frente a retornos centrados al cuadrado y pronóstico de un día. Se incluye en narrativa aplicada. La tabla 7 muestra para el propuesto en BTC MSE **−2,88 %** y MAE **+4,82 %** frente a GARCH: mejora una pérdida y empeora la otra. No son RMSE absolutos ni pruebas independientes de significancia. Se conservan diez semillas declaradas para el propuesto, sin atribuir ese presupuesto a todos los comparadores. [Extracción y límites](../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/pair_c_primary_assessments.json).
- Los artículos aplicados de **TimesNet** (`10.1016/j.eswa.2024.124851`), **TSMixer** (`10.1016/j.jfds.2024.100143`) y *Can transformers transform financial forecasting?* (`10.1108/cfri-01-2024-0032`) describen población bursátil, incluidas 80 acciones S&P100. Se excluyen del núcleo aplicado cripto por evidencia positiva de población en resúmenes o fragmentos editoriales. Se conservan como contexto financiero y de mecanismos. **No se afirma lectura completa de esos tres cuerpos ni se extraen cifras finales no verificadas.** La guía permite mecanismos de otros dominios; esta exclusión aplicada no los elimina de toda la revisión. [Dictámenes y etapas](../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/pair_d_adjudications.json).

Los cuatro originales aún no accesibles son `10.1016/j.physa.2022.127158`, `10.1016/j.physa.2021.126613`, `10.1007/s10614-024-10713-2` y `10.1016/j.future.2024.04.043`. Se buscaron editoriales y depósitos de autores. No obtener el cuerpo no demuestra falta de semillas, código o calidad. Se mantienen pendientes en lugar de convertir sus resúmenes en elegibilidad final.

También se releyeron focalmente los dos contextos con dudas de constructo. En Zahid (`10.3390/risks10120237`) no se concilian los cierres diarios declarados con la agregación intradía y los horizontes. En CryptoMamba-SSM (`10.1109/OJCS.2026.3651226`), H.1 enumera tres estimadores sin asignarlos inequívocamente a las tablas; la varianza de incertidumbre predictiva de III.E no es la varianza de retornos de B.1. Se confirma su utilidad narrativa crítica, con efectos y operacionalización pendientes. {download}`Fuentes y localizadores <../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/two_constructs.md>`. Diferir de σ7 no es, por sí solo, motivo de exclusión narrativa.

## Calidad, disponibilidad y brecha que justifica el benchmark

La matriz de las **nueve fuentes actuales** valora el reporte de cada condición seleccionada. El escalado ajustado sólo con entrenamiento está reportado en **2/9**, NR en 6/9 y NV en 1/9. Las repeticiones están reportadas en **4/9**, NR en 3/9, parcialmente en una y NV en otra. La igualdad del presupuesto HPO queda NR en 9/9. NR/NV no prueban fuga, cero ejecuciones ni desigualdad, y tampoco certifican la ausencia de esos problemas.

Una comprobación adicional de fuentes oficiales identifica publicación pública del dataset para **9/9 estudios** y repositorio de autor fijado para **9/9**. Wang se enlaza a UCR44 dentro del legado 2015; Inception al UCR85, y su nuevo escenario numérico a UCR2018/128 por separado. SAE se enlaza a UCI Beijing PM2.5; los seis estudios de pronóstico restantes a ETTh1. La disponibilidad identificada no verifica el snapshot, los datos preparados, sus splits o reproducción. Por ejemplo, UCI publica 43.824 observaciones y SAE informa 43.800 preparadas. [Evidencia y límites por fuente](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv).

La matriz aplicada tiene otro denominador: **14 estudios incluidos narrativamente**, de los cuales **seis declaran conteos de ejecuciones o inicializaciones**, con alcances distintos. No se escribe que los otros ocho carezcan de repeticiones. Las {download}`cuatro brechas sustentadas <../_static/e2_literature_delivery_closure_20261114_v3/quality/synthesis_updated.md>` conectan observaciones concretas con decisiones de E2:

| Brecha observada en las fuentes examinadas | Respuesta verificable de E2 | Límite que permanece |
|---|---|---|
| Objetivos heterogéneos: varianza intradía, desviación a treinta días, volatilidad implícita y condicional | σ7 no anualizada, D7×5, h=1 y las mismas 608 fechas | No permite combinar métricas externas en un solo efecto |
| Folds, HPO, activos y nuevas inicializaciones se reportan con alcances distintos | Separar selección, seis semillas finales y evidencia temporal pareada | Potencia condicional del piloto ≈12,3 %, inferior al 80 % objetivo |
| Alcance del escalado/interpolación y esfuerzo de comparadores incompletamente establecido | Transformaciones de entrenamiento, calendario, embargo y artefactos conservados | El reporte de terceros no demuestra fuga; igual número de pruebas no garantiza igual coste |
| El ganador cambia con pérdida, target y versión | Comparadores simples, resultados adversos, Holm e inferencia temporal con 20.000 réplicas | Test conocido desde E1 y δg no definido previamente |

La afirmación de gap es refutable y acotada: **en las nueve fuentes que sustentan el catálogo no se identificó una evaluación conjunta de esos mecanismos para σ7 de Bitcoin, h=1 y siete días OHLCV, bajo un presupuesto común e inferencia temporal pareada**. Encontrar en cualquiera de esas nueve fuentes un experimento que cumpla ese contrato obliga a corregirla. La síntesis aplicada respalda la necesidad de comparar contratos, no una afirmación de ausencia global ni superioridad de E2. La contribución observada es un contraste retrospectivo auditable.

## Búsqueda recibida y PRISMA: denominadores conservados

La nueva exportación aplicada de Scopus contiene **1.104 EID únicos**, todos los resultados mostrados en la captura y los diez registros de la entrega parcial anterior. Hay **1.047 DOI distintos y 57 registros sin DOI**. Frente al catálogo local, 817 DOI coinciden y 230 están ausentes; 51 candidatos por título requieren resolución de identidad y tres son ambiguos. No se fusionan DOI distintos ni se renumeran los grupos históricos. [Recibo de la nueva ronda](../_static/e2_literature_delivery_closure_20261114_v3/identification/receipt.json), [metadatos](../_static/e2_literature_delivery_closure_20261114_v3/identification/records.csv) y [conciliación](../_static/e2_literature_delivery_closure_20261114_v3/identification/matching_summary.json).

La cadena suministrada, la captura parcial, los años observados 2019–2026 y los hashes se distinguen de una historia nativa completa de consulta. Los **1.104 recibidos no son 1.104 cribados o incluidos**. Esta ronda no demuestra recuperación de los 374 registros ausentes del total Scopus histórico. Tampoco los 14 incluidos focales o los 43 recibos históricos se suman como estudios independientes sin resolver solapamientos. **No hay PRISMA global cerrado**: falta acreditar la cobertura histórica y resolver cribado/elegibilidad globales.

## Auditoría y pendientes concretos

El [manifiesto](../_static/e2_literature_delivery_closure_20261114_v3/manifest.json) y este comando verifican hashes, denominadores, citas, aritmética y preservación del material anterior, sin entrenamiento ni red:

```bash
python scripts/audit_e2_literature_actions.py
```

Quedan dos tareas documentales que esta actualización no puede dar por realizadas: obtener los cuatro originales aplicados pendientes y cerrar la trazabilidad/decisiones del flujo global. Las dudas de etiqueta de los otros dos estudios requieren aclaración verificable del propio estudio; un nuevo PDF idéntico no las resuelve. La selección previa por puntaje tampoco se recupera retrospectivamente.

Se conserva la {download}`síntesis focal anterior <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/focused_synthesis.md>` y el [paquete v2](../_static/e2_literature_delivery_closure_20261114_v2/manifest.json), con sus estados de 13/7/8/2 y 45/50 intactos como antecedente. La revisión actual es asistida por IA, sin doble revisión humana independiente. No hubo nuevos entrenamientos ni evaluación de modelos de mercado.
