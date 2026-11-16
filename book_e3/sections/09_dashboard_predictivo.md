# 9 Bloque E Dashboard predictivo en Dash Plotly

**Estado: diseño documental, sin aplicación construida o desplegada.** Este capítulo aplica la Sección 9 de la guía al proyecto BTCUSDT y especifica la integración futura de datos, artefactos e interpretaciones. Esta actualización no crea `app.py`, las páginas Dash, `Dockerfile` ni `render.yaml`, ni ejecuta una campaña de modelos. La [selección y exportación](05_configuraciones_exportacion.md), el [modelo candidato](06_modelo_original.md) y el [protocolo estadístico en borrador](08_verificacion_estadistica.md) conservan sus propios alcances.

## 9.1 Objetivo

El dashboard permitirá que una persona no experta comprenda qué volatilidad se pronostica, explore la evidencia y examine una predicción con su incertidumbre, explicación y límites. Las tres páginas obligatorias serán **Contexto, EDA y Modelos**. Cada gráfico responderá a una pregunta y mostrará una interpretación ligada a los datos de su vista.

La cantidad objetivo es la desviación poblacional de siete retornos logarítmicos diarios de Close, σ₇, sin anualización. El horizonte es **un día**. En el origen UTC del día t sólo se usarán observaciones disponibles hasta t−1; la etiqueta de t se conoce al finalizar ese día. El uso previsto es la exploración académica de pronósticos de volatilidad y sus limitaciones; los usos operativos exigirían validación adicional.

## 9.2 Alcance y límites de memoria

El catálogo de la vista tendrá **17 filas**: los 16 sistemas heredados de E2 y el candidato original. El registro del candidato indicará «sin evaluación de mercado»; sus métricas no se rellenarán con ceros, resultados de pruebas sintéticas o cifras de otra arquitectura.

| Grupo | Evidencia disponible | Modalidad prevista y condición de habilitación |
|---|---|---|
| MLP16, LSTM32 y GRU32 | Tres TorchScript reales, metadatos y paridad de predicción sobre 53 fechas de calibración de 2024. | Inferencia CPU integrable después de implementar y verificar validación, preprocesamiento, incertidumbre, explicación y control de dominio. Actualmente no hay interfaz en vivo. |
| Diez candidatos del Top 10 | Resultados históricos y artefactos originales de E2, con límites documentados. | Resultados precalculados con fuente y justificación del estado; exportación e integración pendientes. No se presume que hayan excedido memoria sin medirlos. |
| Persistence, SeasonalNaive y ARIMA | Comparadores históricos con su contrato temporal propio. | Precálculos hasta definir y verificar la historia de σ₇ requerida; no se alimentan automáticamente con siete barras OHLCV. |
| Original CNN-LSTM con atención candidata | Código, demostración de cabeza latente y 35 pruebas del módulo con datos artificiales. | Presentación documental y de evidencia de software. No hay pesos BTC ni R1, defensa académica de R2 o R3 acreditados. |

La carga será bajo demanda en CPU, con **un worker propuesto**, cargas serializadas y caché de como máximo dos modelos por proceso. Una caché LRU de tamaño dos en varios workers no limita a dos los modelos del servicio completo. Se reducirá su capacidad si la medición del proceso, la concurrencia y el límite vigente del plan lo requieren.

El pico RSS medido en procesos de exportación incluye Python, PyTorch, checkpoint y cohorte. No acredita memoria del dashboard ni cabida en Render. La selección de modelos en vivo se cerrará tras medir arranque, carga, inferencia, explicaciones y sesiones concurrentes. Los precálculos identificarán qué operación no está habilitada y por qué, sin presentar una captura de resultados como inferencia ejecutada por el usuario.

El [perfil de despliegue de la sección 10](10_despliegue_render.md) exige ONNX Runtime sin el framework de entrenamiento y propone un modelo residente bajo demanda. Los TorchScript actuales requieren PyTorch: son referencias verificadas, pero falta exportar y verificar ONNX/ORT para ese perfil. El máximo de dos modelos de este diseño es un techo; no prueba cabida ni sustituye la política más restrictiva de producción.

## 9.3 Arquitectura y estructura del repositorio

Se propone una aplicación multipágina con `Dash(..., use_pages=True)`, navegación fija y una página por sección. La interfaz, los datos, la inferencia y las explicaciones tendrán módulos separados. El entrenamiento permanecerá fuera del proceso de servicio y nunca se importará desde la aplicación.

El árbol siguiente es una **arquitectura propuesta**, no un inventario de archivos ya creados:

```text
app.py                         # propuesto: app multipágina y server WSGI
pages/
  contexto.py                  # propuesto: narrativa y ficha de datos
  eda.py                       # propuesto: filtros y exploradores
  modelos.py                   # propuesto: catálogo e inferencia
src/
  original_model.py            # existente: candidato sin pesos BTC entrenados
  data.py                      # propuesto: fuentes autorizadas y caché de datos
  plots.py                     # propuesto: figuras y plantillas reutilizables
  inference.py                 # propuesto: contratos, preprocesamiento y carga CPU
  explain.py                   # propuesto: explicaciones y sus validaciones
training/                      # propuesto: ejecución científica separada del servicio
artifacts/e3_handoff/           # existente: tres TorchScript y sus metadatos/reportes
experiments/e3_bloque_e/        # documento de diseño y futuras evidencias de ingeniería
data/                          # existente: insumos de servicio y declaración pendientes
assets/                        # propuesto: estilos y ejemplos con procedencia
tests/                         # pruebas existentes; integración Dash por incorporar
book_e3/                       # existente: informe de esta entrega
Dockerfile                     # pendiente: entorno reproducible del servicio
requirements.txt               # propuesto: versiones exactas de inferencia
requirements-train.txt         # propuesto: entorno científico separado
MODEL_CARD.md                  # pendiente: ficha del original y usos previstos
DATA_STATEMENT.md              # pendiente: declaración de los datos
CITATION.cff y LICENSE          # pendientes: citación y licencia; derechos de datos separados
render.yaml                    # pendiente: despliegue del servicio
.github/workflows/             # existente: cobertura Dash/paridad app-libro pendientes
```

Los scripts históricos de E2 conservarán identidad y procedencia; este esquema no los mueve ni los importa como entrenamiento al arrancar Dash. Los procesos de análisis producen tablas y figuras versionadas antes del servicio. El backend sólo leerá insumos permitidos, validará su esquema e integridad y ejecutará inferencia de artefactos verificados.

La tabla de vista conservará la procedencia de las 16 filas de E2 y añadirá el original con estado pendiente. Campos mínimos: modelo, familia, dataset, partición, receta, semilla o número de trayectorias, estimando, métrica/DE/IC disponibles, cobertura y costes con hardware/backend. No se combinarán como una misma medición perfiles GPU, eager CPU y TorchScript de distintas condiciones.

El flujo de una predicción admitida deberá conservar esta separación de responsabilidades:

1. El registro resuelve un ID de modelo permitido y determina si corresponde a inferencia habilitada, precálculo o evidencia documental del candidato.

2. La carga del ejemplo o CSV se valida contra el origen UTC y el contrato del modelo; las entradas inválidas terminan con un mensaje de campo.

3. Una acción explícita del usuario inicia la predicción. Entrar a una página o cambiar un filtro EDA no dispara una carga o entrenamiento de modelos.

4. El cargador CPU verifica integridad y versión, serializa la carga y gestiona la expulsión de la caché acotada por proceso.

5. El módulo de inferencia aplica las transformaciones guardadas y ejecuta el artefacto en evaluación, sin modificar pesos, escaladores o calibradores.

6. La capa de salida recibe σ₇ en unidades originales, evitando invertir dos veces la escala, y registra una latencia con alcance definido.

7. Los módulos de incertidumbre y explicación usan ese mismo caso y artefacto; si falta una capacidad verificada, muestran su ausencia explícita.

8. La interfaz presenta el resultado y la procedencia permitida. Los módulos de entrenamiento quedan fuera del recorrido del servicio.

## 9.4 Contenido mínimo de cada sección

Las páginas compartirán títulos, unidades, versión y fecha de la evidencia, leyendas de estado y acceso al informe. Una vista filtrada indicará su número de observaciones, periodo, máscara y fuente. Las cifras se obtendrán de la misma tabla maestra que alimenta el libro, con pruebas de igualdad de identificadores, valores y estados.

### 9.4.1 Contexto

La portada explicará la relevancia económica y científica de pronosticar cambios de variabilidad de Bitcoin, el objetivo σ₇ y su horizonte. Toda cifra de contexto tendrá fuente verificable y fecha. Como escala documentada del experimento, el [benchmark de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/06_benchmark_final.html) compara 16 sistemas sobre 608 fechas; no se presenta ese dato como validación prospectiva del candidato.

Se mostrarán las preguntas e hipótesis heredadas de E2 y la hipótesis del mecanismo original, con sus fuentes y estado. La Data Card incluirá Binance Spot BTCUSDT, días UTC, periodo efectivo, licencia/condiciones de uso, observaciones admisibles, OHLCV y objetivo, máscaras de calidad, sesgos y exposición histórica de las particiones. Se vinculará al [contrato vigente de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) y a su [auditoría actual](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html), distinguiendo observaciones crudas, ventanas y cohortes finales.

Un diagrama recorrerá EDA → revisión → benchmark → protocolo temporal → ajuste/sobreajuste → comparación → modelo original → evaluación ampliada. Las etapas pendientes se identificarán junto a las ya ejecutadas. El pie incluirá autores, versión, actualización, límites de generalización, uso previsto y política de privacidad.

### 9.4.2 EDA

Se proponen `dcc.Dropdown` para fuente/partición/variables, `dcc.DatePickerRange` para periodo y `dcc.RangeSlider` para rangos pertinentes. En la fuente actual sólo se ofrecerán series que realmente existan. Los callbacks aplicarán una selección común a **todos los gráficos e interpretaciones de la página EDA**, incluyendo sus conteos y advertencias de muestra insuficiente.

| Pregunta del usuario | Vista temporal prevista | Control de interpretación |
|---|---|---|
| ¿Cómo se distribuye σ₇? | Histograma, cuantiles y serie temporal del objetivo. | Es una regresión: no se inventa un balance de clases. Identificar escala y periodo. |
| ¿Qué dependencia existe? | ACF/PACF, descomposición y pruebas de estacionariedad pertinentes. | Ajustar retardos al soporte disponible, mostrar supuestos y distinguir diagnóstico descriptivo de confirmación. |
| ¿Qué calidad tienen los datos? | Fechas faltantes, duplicados y casi duplicados, barras inválidas, máscaras y posibles errores de reconstrucción de σ₇. | Comprobar definición/soporte del objetivo y conservar la diferencia entre fallo demostrado, control parcial y control no ejecutado; la reconstrucción aritmética no certifica el proxy económico. |
| ¿Qué cambia entre particiones? | Distribuciones y resúmenes comparables de entrenamiento, calibración y prueba disponible. | Mostrar exposición previa y disponibilidad real; no equiparar periodos distintos como réplicas independientes. |
| ¿Qué variables y rezagos quiero comparar? | Explorador de OHLCV, retornos y σ₇, con vistas seleccionables. | Explicitar unidades y disponibilidad temporal; las correlaciones no se presentan como efectos causales. |

La auditoría actual dispone de ventanas reconstruidas train/cal y mantiene incompleta la disponibilidad de OHLCV de las 608 fechas de prueba en ese entorno. Las vistas que requieran esas barras seguirán pendientes hasta obtenerlas y verificarlas. Se podrán mostrar agregados, figuras o trayectorias guardadas sólo cuando estén disponibles y autorizados, con su procedencia; no se reconstruirán datos faltantes de forma ficticia.

Cada gráfico tendrá una frase de interpretación que cambie con el filtro y explique la implicación para el modelado. Si el subconjunto no permite una prueba o ACF válida, se mostrará «datos insuficientes», sin rellenar el resultado con cero. Los filtros EDA no recalcularán silenciosamente el ranking A2: la referencia histórica de 608 fechas conservará su etiqueta. Cualquier ranking de otra cohorte requerirá cálculo y evidencia propios antes de mostrarse.

### 9.4.3 Modelos

La página tendrá una `dash_table.DataTable` ordenable y filtrable con las 17 filas previstas. Mostrar/ocultar una fila es una operación de vista; no cambia la familia científica. Para cada sistema se conservarán familia, primaria media±DE, calibración, parámetros, FLOPs y latencia con unidad y alcance. Una trayectoria determinista no recibirá DE=0 como sustituto de incertidumbre; los campos inexistentes aparecerán como pendientes.

Se distinguirán los líderes documentados: TimeMixer encabeza los quince sistemas con score CV recuperado, mientras que el SAE es líder descriptivo de prueba histórica y pertenece al MCS90 de E2. Esa distinción no permite elegir retrospectivamente una configuración o semilla por su prueba. El original no tiene posición de rendimiento ni comparación explicativa global con esos líderes todavía.

**Contrato propuesto para las tres referencias exportadas.** El usuario elegirá modelo y origen UTC t, con horizonte h=1 fijo, y aportará siete barras de días consecutivos hasta t−1 o un ejemplo autorizado. La propuesta inicial admite sólo CSV UTF-8 sin compresión, siete filas de datos y las seis columnas `date,open,high,low,close,volume`. El límite configurable propuesto es **100 KiB (102.400 bytes)**; es una decisión de ingeniería por validar, no un límite de Render ni una carga ya probada.

Antes de inferir se deberán comprobar esquema y tamaño, fechas UTC ordenadas sin duplicados, ausencia del día objetivo, valores finitos, precios positivos, volumen no negativo y coherencia low≤open/close≤high. Se rechazarán huecos y entradas inválidas con mensajes de campo comprensibles, sin imputación o reordenamiento silenciosos.

El backend transformará los precios como log(P)−log(Ct−1), el volumen con log1p y estandarizará cada canal con media/DE guardadas del entrenamiento. No estimará estadísticas con el archivo del usuario. El helper existente `predict_exported_reference` recibe ventanas **ya estandarizadas** de forma lote×7×5 y verifica hash, forma y finitud. El TorchScript devuelve σ₇/target_scale y ese helper ya invierte la escala una vez: la interfaz no debe volver a multiplicarla. El helper actual tampoco implementa una caché de modelos ni valida OHLCV crudo o dominio; esa integración es trabajo pendiente.

El contrato D7×5 no se extiende automáticamente a los clásicos. Persistence usa historia de σ₇; calcular σ₇(t−1) requiere ocho cierres anteriores. SeasonalNaive y ARIMA tienen sus propios rezagos y soportes. Permanecerán en modo precalculado hasta que cada contrato y su paridad queden verificados.

**Salida e incertidumbre.** La vista mostrará pronóstico σ₇, fecha objetivo, modelo/versión, latencia medida y explicación local. Los metadatos heredados contienen un radio residual q y cobertura **nominal 0,90**. Se propone construir el intervalo en unidades originales como [máx(0, σ̂₇−q), σ̂₇+q] sólo tras implementar y comprobar esa operación y su procedencia. Se mostrará por separado la cobertura empírica realmente disponible para el mismo predictor y cohorte. q no se reajustará con la entrada cargada. La paridad actual comprobó predicciones puntuales sobre 53 fechas de calibración de 2024, con error máximo y ΔRMSE cero; no verificó la métrica A2 de 608 fechas ni constituye una prueba de cobertura de intervalos exportados. La dependencia temporal y el cambio de distribución impiden presentar el nominal como garantía individual.

**Dominio y explicación local.** Se separará el rechazo de un formato inválido del aviso por extrapolación. Los rangos, distancias y umbrales OOD deben estimarse sólo con entrenamiento y guardarse con versión; siguen pendientes. Hasta disponer de ese control no se etiquetará una entrada como «dentro de dominio» sólo por ser numéricamente válida.

La explicación propuesta atribuirá la predicción a rezagos/covariables mediante un método adecuado, como Integrated Gradients u oclusión controlada, con referencia procedente de entrenamiento. Deben verificarse escala, fidelidad y estabilidad, y preservar la plausibilidad de las alteraciones. Los mapas de atención serán un complemento, no la única atribución ni una explicación causal. Los ejemplos precargados podrán tener explicaciones precalculadas enlazadas al mismo modelo/hash; una explicación de otra entrada, aunque use el mismo modelo, no se presentará como la de la carga actual.

**Diagnósticos y comparación.** Para F2a se mostrarán reales frente a predichos, residuos, ACF de residuos y error por origen/horizonte disponible, además de calibración e incertidumbre pertinentes. No se importarán ROC o matrices de confusión de clasificación. La interpretación global del modelo seleccionado y su comparación con el original y el mejor benchmark deberán conservar métodos y cohortes; la comparación del original sigue pendiente.

La tabla MCS futura incluirá al candidato y los 16 rivales bajo el protocolo de la sección 8; el MCS histórico de E2 no lo contiene. La frontera de Pareto distinguirá desempeño y costes medidos en el mismo hardware/backend y condiciones. La ficha del original mostrará procedencia, diferencias, pseudocódigo, pruebas, curvas, efectos/IC/p ajustados y robustez cuando existan, con veredicto explícito, límites y escenarios adversos. Su Model Card accesible incluirá uso previsto, datos, métricas, riesgos y usos no recomendados.

## 9.5 Criterios de calidad de diseño y de ingeniería

| Criterio | Diseño o comprobación necesaria antes de habilitar el servicio |
|---|---|
| Coherencia visual | `dash-bootstrap-components`, tema **FLATLY**, paleta funcional de como máximo cinco colores, tipografía y espaciado uniformes, navegación fija y estados consistentes. |
| Figuras | Plantilla Plotly común, títulos que respondan preguntas, ejes/unidades, leyendas y hover legibles; interpretaciones sincronizadas con la vista. |
| Responsividad | Rejilla `dbc.Row`/`dbc.Col`, revisión en escritorio y tablet, tablas desplazables y controles utilizables sin recortes. |
| Accesibilidad | Contraste suficiente, navegación por teclado/foco, etiquetas de controles y textos alternativos; estados y diferencias legibles sin depender sólo del color. |
| Rendimiento | CPU, un worker propuesto, carga serial y caché ≤2 por proceso; medir arranque, memoria/concurrencia y tiempos completos antes de declarar cabida. Precálculos para ejemplos y políticas de invalidación por versión/hash. |
| Callbacks | Organización por página, funciones reutilizables y uso acotado de `dcc.Store` por sesión para resultados repetidos; `prevent_initial_call` y `dash.no_update` cuando proceda. No guardar cuerpos de archivos o cargas innecesarias en el navegador. |
| Entradas y errores | Límites de bytes/filas/tipos, validación temporal y física, y `dbc.Alert` con mensajes comprensibles; errores del modelo controlados sin detener el servicio ni sustituir pronósticos por cero. |
| Privacidad y seguridad | No registrar entradas sin consentimiento; política visible y retención definida. No ejecutar contenido subido ni aceptar objetos serializados del usuario; ocultar credenciales y rutas internas, sin mostrar trazas técnicas. |
| Organización | Dependencias exactas de inferencia separadas del entrenamiento; rutas/insumos mediante configuración autorizada; funciones de carga, gráficos y explicación reutilizables. |
| CI e integridad | GitHub Actions con pytest en cada push: pruebas del original, validaciones de entrada, integridad/guardado/carga, ida y vuelta de exportaciones y errores de los callbacks. La cobertura Dash es futura; los tests actuales no prueban una app inexistente. |
| Consistencia con el informe | Una tabla maestra versionada y prueba automática de igualdad de IDs, estimandos, métricas, unidades, estados y procedencia entre dashboard y libro. Verificar que filtros de vista no alteren cifras históricas sin recálculo explícito. |

La aceptación posterior tendrá casos y evidencia concretos; ninguno se declara probado por esta especificación:

| Recorrido | Caso a verificar | Evidencia que deberá conservarse |
|---|---|---|
| Navegación | Contexto, EDA y Modelos en escritorio y tablet. | Capturas/registro de navegación, accesibilidad y estados de carga. |
| Filtros | Periodo vacío, muestra corta y selección de variables. | Figuras e insights con el mismo n efectivo; mensajes de insuficiencia y fuente de datos. |
| Predicción | Ejemplo verificado y una carga nueva admisible. | Contrato raw→estandarizado→σ₇, hash, tolerancias y salida comparable a la referencia. |
| Rechazo | Fecha futura, hueco, duplicado, NaN, precio inválido y archivo sobredimensionado. | Mensaje de campo sin ejecución de inferencia ni caída del servicio. |
| Incertidumbre | q del mismo modelo/semilla y unidades. | Verificación de extremos, nominal y origen de cobertura empírica, sin recalibrar con la carga. |
| Explicación | Entrada real y ejemplo precargado. | Atribución correspondiente, referencia de entrenamiento, fidelidad y estabilidad; ausencia explícita cuando proceda. |
| Recursos | Tercer modelo solicitado, expulsión de caché y cargas concurrentes. | Memoria conjunta/transitoria, límite por proceso y latencia completa, incluyendo explicación en vivo. |
| Consistencia | Comparación con el maestro del libro y modo precalculado. | Igualdad automática de IDs, valores, estados, unidades y soporte; ninguna fila pendiente convertida en un resultado. |

La paridad incluirá el preprocesamiento completo y tolerancias declaradas, no sólo que un archivo cargue. La prueba de humo y las evidencias del servicio deberán acompañar su implementación y despliegue posteriores.

El [diseño del Bloque E](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_e/diseno_dashboard.md) y su {download}`copia descargable <../_static/e3_bloque_e/diseno_dashboard.md>` recogerán los contratos y decisiones pendientes. La documentación pública de E3 está disponible; la construcción, pruebas de usuario y despliegue del servicio Dash siguen pendientes y deberán producir evidencia antes de declararse completados.
