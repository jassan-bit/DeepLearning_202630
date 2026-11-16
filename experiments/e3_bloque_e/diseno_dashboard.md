# Diseño del dashboard predictivo — Bloque E

Fecha: 16 de noviembre de 2026. Versión documental: borrador 0.1. Fuente de requisitos: sección 9 de la guía aportada, páginas 30–35.

**Estado: especificación, sin aplicación Dash implementada ni despliegue Render verificado.** Este documento no acredita callbacks, inferencia desde una interfaz, pruebas de aceptación o consumo de RAM de una aplicación. No ejecuta entrenamientos ni altera resultados, recetas o artefactos de los bloques anteriores.

## 1. Propósito y preguntas de la persona usuaria

La aplicación debe permitir comprender el pronóstico de volatilidad de BTCUSDT, explorar datos admisibles y consultar modelos con sus límites. σ7 es la desviación poblacional de siete retornos logarítmicos, pronosticada a un día, sin anualizar. Siete retornos no significan horizonte de siete días; tampoco se predice el precio de Bitcoin.

| Página | Pregunta que resuelve | Componentes previstos |
|---|---|---|
| Contexto `/` | ¿Qué se predice, con qué datos y para qué sirve el estudio? | Portada, cifra de contexto citada, RQ/hipótesis, Data Card, pipeline, limitaciones, autor/versión y referencias. |
| EDA `/eda` | ¿Cómo varía la volatilidad, qué calidad tienen los datos y qué cambia entre periodos? | Filtros globales, distribución continua, series/descomposición, ACF/PACF, diagnósticos temporales, calidad y explorador de variables; interpretación que cambia con cada filtro. |
| Modelos `/modelos` | ¿Qué resultados existen y qué modelos permiten predecir este caso? | `dash_table.DataTable` ordenable/filtrable con todos los sistemas, selector, inferencia admitida, intervalo, aviso de rango, explicación, diagnósticos, MCS/Pareto y ficha del candidato. |

El diseño es multipágina con `use_pages=True`, navegación fija y `server=app.server`. El informe se mantiene en `book_e3/`; no se reemplaza `book/`, que corresponde a E1, ni `book_e2/`.

## 2. Inventario real y modo que deberá mostrar la aplicación

| Grupo | Cantidad | Evidencia existente | Estado de producto que puede afirmarse hoy |
|---|---:|---|---|
| Top 10 de E2 | 10 | Recetas y pronósticos históricos del benchmark. | Resultados precalculados; no inferencia web verificada. |
| MLP16, LSTM32 y GRU32 | 3 | TorchScript CPU, metadata y round-trip de cada modelo en `artifacts/e3_handoff/`. | Artefacto disponible; integración y validación desde OHLCV crudo pendientes. |
| Persistence, SeasonalNaive y ARIMA | 3 | Resultados históricos y recetas de comparadores clásicos. | Precalculados hasta verificar su contrato temporal y su servicio. |
| Candidato original | 1 | Formulación, teoría de la cabeza latente y 35 pruebas artificiales. | Sin entrenamiento/evaluación BTC; no habilitar un botón de predicción con pesos aleatorios. |

Se representarán **17 sistemas**; una limitación de memoria no permite ocultar rivales. Los diez modelos del Top 10, las referencias y los clásicos conservan sus identidades. El candidato tendrá campos ausentes con estado «sin evaluación», nunca métricas cero ni un ranking inventado.

Las tres exportaciones usan la semilla 271801, elegida por el menor ID registrado. Su paridad numérica se verificó sobre 53 fechas históricas de calibración del 08/11 al 30/12/2024, con error máximo cero en la predicción puntual. Esto no verifica la A2 de 608 fechas ni un intervalo servido por una aplicación. La tabla principal de E2 resume la cohorte de semillas; la predicción de un artefacto representante no es el ensemble de esa cohorte.

Fuentes existentes: `experiments/e3_bloques_ab_20261115/master_configurations.csv`, `deployment_master.csv`, `export_roundtrip_summary.csv`, `artifacts/e3_handoff/README.md` y los tres `metadata.json`. El wrapper `predict_exported_reference` de `scripts/e3_export_readiness.py` carga un modelo por llamada, recibe ventanas ya transformadas/estandarizadas y devuelve σ7. No es todavía el preprocesador, caché, detector OOD, servicio de intervalos o generador de explicaciones del dashboard.

## 3. Estructura objetivo y separación de responsabilidades

El árbol siguiente es una **estructura propuesta**; no anuncia la creación de estos módulos en este corte:

~~~text
app.py                         crea Dash multipágina y expone server
pages/contexto.py              contenido y referencias de Contexto
pages/eda.py                   layout y callbacks de EDA
pages/modelos.py               layout y callbacks de Modelos
src/inference.py               registro permitido, CPU, caché y preprocesamiento
src/explain.py                 explicaciones sobre artefactos admitidos
src/data.py                    lectura y validación de datos y tabla maestra
src/plots.py                   figuras, unidades, estilos e interpretaciones
src/original_model.py          módulo del candidato ya existente; no pesos BTC
training/                     scripts/configuraciones de ajuste fuera del servicio
artifacts/                    pesos exportados, metadata y calibradores vinculados
data/                         datos ligeros admisibles y DATA_STATEMENT.md
assets/                       CSS, recursos y ejemplos verificados
experiments/                  protocolos, diseño y evidencia de cada bloque
tests/                        pruebas significativas del servicio y la consistencia
book_e3/                      informe E3; se preservan los otros dos libros
.github/workflows/            pytest, round-trip y consistencia app/informe
Dockerfile                    contenedor reproducible obligatorio
requirements.txt              versiones exactas del entorno de inferencia
requirements-train.txt        versiones exactas de entrenamiento, separadas
render.yaml                   configuración de despliegue aún por verificar
MODEL_CARD.md                 ficha accesible desde Modelos
CITATION.cff y LICENSE         citación y licencia, sin cambiar derechos de los datos
~~~

El servicio no importará `training/`, ajustará pesos, ejecutará HPO ni reajustará escaladores/calibradores. La integración debe conservar los ejecutores y las evidencias existentes, con una separación que pueda comprobarse por sus importaciones y efectos. Los modelos se elegirán por identificadores permitidos, no por rutas o URLs proporcionadas por el usuario. No se aceptan pesos o programas subidos.

## 4. Contrato de datos, filtros y tabla maestra

El EDA consumirá OHLCV y objetivos auténticos con fuente, versión, licencia/condiciones y hashes. Los 655 ejemplos train y 53 de calibración reconstruidos en las auditorías de E2 tienen su alcance propio. Las entradas OHLCV de las 608 fechas de prueba siguen ausentes en el entorno auditado: disponer de etiquetas y pronósticos no permite reconstruirlas. La app no debe inventar precios, ventanas ni valores ausentes para completar gráficos.

Se distinguirán fechas, observaciones crudas, ventanas supervisadas y unidades independientes. El rango de fechas, selector de variables y controles de EDA actualizarán todos los gráficos de esa página y su interpretación, con n efectivo, periodo y estado de datos insuficientes. ACF/PACF y pruebas de estacionariedad necesitan soporte y supuestos apropiados; una ventana de siete filas del formulario no se presenta como un EDA temporal concluyente. El objetivo es continuo: no hay balance de clases que construir artificialmente.

El panel temporal incluirá descomposición cuando el soporte y sus supuestos lo permitan. La calidad mostrará duplicados y casi duplicados, faltantes, anomalías de datos y posibles errores de reconstrucción del objetivo σ7, con la evidencia disponible y su alcance. El cambio de distribución entre particiones se documentará mediante comparaciones admisibles; un gráfico o prueba no ejecutada seguirá indicado como pendiente, sin afirmar ausencia de shift.

La tabla maestra compartida con el informe deberá identificar sistema, dataset/cohorte, partición, receta, semillas, métrica/unidad, media, DE, IC y alcance, calibración, parámetros, FLOPs, backend/hardware, latencias y estados de inferencia/evidencia. Incluirá fuente, versión y hash. La vista de 17 sistemas se generará desde las fuentes versionadas sin editar el CSV histórico de E2 para añadir una fila de resultados inexistentes.

Los filtros de exploración no recalcularán ni cambiarán silenciosamente el ranking congelado de la A2. Una métrica recalculada en un subconjunto se etiquetará con su soporte y carácter exploratorio, y no reemplazará la selección por validación. La app distinguirá TimeMixer, líder entre las 15 validaciones disponibles, del SAE, líder descriptivo de la prueba histórica. Persistence no tiene score CV en el recibo de receta fija. Se usarán las mismas reglas de redondeo, unidades, valores ausentes y procedencia en tabla, gráficos e informe.

## 5. Contrato de inferencia propuesto

El primer contrato de entrada sólo corresponde a MLP16/LSTM32/GRU32. Selector de serie BTCUSDT, origen en UTC, horizonte fijo h=1 y siete días completos hasta t−1 para predecir el objetivo de t. Un selector no ofrecerá horizontes, activos o fuentes para los que no exista validación. Los clásicos requieren otro soporte: incluso σ7(t−1), usado por Persistence, necesita ocho cierres para siete retornos; SeasonalNaive y ARIMA tienen historia propia. No se los alimenta con un tensor neuronal D7×5 por conveniencia.

Propuesta inicial de ingeniería para las tres redes: CSV UTF-8 sin comprimir, como máximo **102400 bytes (100 KiB)**, con siete filas de datos y las columnas `date,open,high,low,close,volume`. El límite es configurable y deberá comprobarse antes de publicar; no es un límite de Render ni una prueba de rendimiento. Un ejemplo precargado debe provenir de datos verificados y tener versión/hash.

La guarda externa validará tipo/tamaño, esquema, fechas UTC únicas y consecutivas en orden, correspondencia con el origen, precios positivos finitos, volumen finito no negativo, `low <= min(open,close) <= max(open,close) <= high` y ausencia de valores futuros o faltantes. Rechazará booleanos, NaN/infinito, desbordamientos al convertir a float32 y cualquier forma distinta de la admitida. Una entrada inválida devuelve un mensaje claro sin romper la app; no se rellena ni reordena silenciosamente para obtener una predicción.

El preprocesamiento reproducirá por artefacto `log(P)-log(último cierre conocido)` para precios y `log1p(V)` para volumen, seguido de medias/desviaciones train guardadas y orden exacto `open,high,low,close,volume`. No se estima nada con la carga del usuario. La forma será `(1,7,5)` float32; se comprobarán los hashes del modelo, metadata, escalador y calibrador vinculados. La salida cruda del TorchScript se multiplica por `target_scale` una sola vez y se muestra como σ7 sin anualización. El helper existente `predict_exported_reference` ya devuelve σ7 tras esa inversión: si se reutiliza, la interfaz no vuelve a multiplicar su resultado. Si se ofrece porcentaje, se explicita la conversión ×100.

Los umbrales de aviso fuera de rango/distribución se estimarán sólo con entrenamiento y se versionarán. Los metadatos actuales los declaran pendientes. Pasar los rangos no demuestra pertenecer a la distribución; tampoco un aviso permite extrapolar con fiabilidad. Mientras el control no esté implementado y verificado no se mostrará un distintivo de entrada «en dominio».

Para un artefacto con su calibrador válido, el intervalo previsto es `[max(0,ŷ−q), ŷ+q]`, con q del mismo modelo/semilla y unidades originales, cobertura nominal del 90 % y enlace a cobertura empírica realmente disponible para ese predictor y cohorte. No se garantiza cobertura exacta bajo dependencia temporal, no se mezcla q de otra semilla o ensemble y no se calibra de nuevo con la entrada. Su servicio y paridad deben verificarse antes de habilitarlos. Si falta incertidumbre válida, se informa su ausencia; no se fabrica con ruido alrededor de ŷ.

La explicación local de una entrada nueva debe calcularse para esa entrada y ese artefacto, por ejemplo atribución por rezago/covariable con Integrated Gradients cuando sea compatible y se verifique su fidelidad/estabilidad. Un mapa de atención no se presenta por sí solo como explicación causal. Las explicaciones precalculadas sólo corresponden a sus ejemplos identificados; si el servicio no puede explicar una carga nueva se muestra «no disponible», sin reutilizar otro mapa.

## 6. Recursos, interfaz, privacidad y accesibilidad

Propuesta a validar: inferencia exclusivamente CPU, un worker, cargas serializadas y caché de como máximo dos modelos por proceso. La admisión/evicción debe impedir que cargas concurrentes excedan el presupuesto; varios workers multiplicarían la memoria. Se medirá el RSS conjunto del intérprete, runtime, datos, modelos y explicaciones, el transitorio de carga, concurrencia, calentamiento y latencia completa. El RSS histórico de cada modelo separado no demuestra cabida conjunta ni memoria incremental, y las latencias forward no incluyen el flujo completo. No se ha verificado un plan concreto de Render.

Usar `dash-bootstrap-components`, tema FLATLY y hasta cinco colores; navbar fija, tipografía/espaciado uniformes y rejilla `dbc.Row`/`dbc.Col` en escritorio/tablet. Plotly tendrá una plantilla común, títulos con unidades, leyendas y hover claros. Cada gráfico llevará una interpretación coherente con sus filtros. Contraste, etiquetas, navegación con teclado, textos alternativos y estados expresados mediante texto además de color se comprobarán en la app implementada.

Los callbacks se organizarán por página; `dcc.Store` sólo contendrá datos acotados y apropiados para ser visibles al cliente, nunca pesos, secretos o todo el dataset por defecto. Los cálculos repetidos y explicaciones de ejemplos se precalcularán cuando proceda; `prevent_initial_call` y `dash.no_update` evitarán trabajo innecesario. La predicción se inicia con una acción explícita y el selector de un modelo precalculado explica por qué no ofrece inferencia.

No registrar entradas del usuario sin consentimiento. Limitar tipos, bytes, filas y vida útil; no ejecutar archivos ni aceptar pickle, pesos, comprimidos o rutas arbitrarias. Los errores no expondrán credenciales, rutas internas ni trazas. El pie incluirá una política de privacidad que describa el tratamiento real. Las licencias/condiciones de código y datos se documentan por separado. La ficha declara uso académico, una sola serie/proveedor, exposición histórica, sesgos y ausencia de utilidad financiera acreditada; no se ocultan límites detrás de un indicador de confianza.

## 7. Evidencia necesaria para cerrar el Bloque E

La construcción posterior deberá demostrar, con pruebas que aún no se han ejecutado:

- Navegación de las tres páginas, filtros que actualizan todas las figuras e interpretaciones y estados de datos insuficientes, vacíos o errores.
- Los 17 sistemas visibles, tabla ordenable/filtrable, estados honestos y coincidencia automática de tabla/figuras con la misma fuente maestra del informe.
- Flujo permitido de entrada válida y rechazo de entradas inválidas; ausencia de importación/ejecución de entrenamiento; preprocesamiento, escala, hashes, intervalo y predicción consistentes con el artefacto de referencia.
- OOD con umbrales train documentados, explicaciones del caso real y disponibilidad explícita, sin mapas prestados o incertidumbre inventada.
- Diagnósticos F2a, comparación global y ficha: valores reales/predichos, residuos/ACF, h=1, cobertura, atribuciones globales, Pareto medido bajo condiciones comparables y MCS con alcance correcto. El MCS histórico de E2 no incluye al candidato; el MCS de 17 sistemas y la comparación original/mejor benchmark siguen pendientes.
- Subsección original con procedencia, diferencias, pseudocódigo, curvas y evidencia disponibles, robustez/hipótesis/p ajustados cuando existan y veredicto explícito «superioridad no establecida» al corte actual. Ficha accesible con uso, datos, métricas, riesgos y límites.
- Caché/concurrencia, límites de carga, errores recuperables, privacidad, accesibilidad, escritorio/tablet y mediciones de memoria/latencia total.
- CI en cada push con dependencias exactas, pytest del servicio y del modelo original, round-trip y consistencia del maestro. Las pruebas existentes de software no acreditan la aplicación futura por sí solas.
- Dockerfile, entornos separados, DATA_STATEMENT, MODEL_CARD, citación/licencias y configuración Render revisados. El despliegue y su prueba de humo pertenecen al Bloque F y no se declaran completos por publicar el libro.

La aplicación debe seguir permitiendo explorar resultados desfavorables y escenarios en los que el candidato no supere a los benchmarks. El estado de R1, defensa académica de R2 y R3 sigue abierto, tal como se documenta en las secciones 6–8.
