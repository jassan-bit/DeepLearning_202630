# 9 Bloque E Dashboard predictivo en Dash Plotly

**Estado: prototipo local verificado, con cuatro modelos ONNX en vivo; servicio Render creado, con funcionamiento y humo públicos por verificar.** Este bloque pasa del diseño documental a una aplicación multipágina con **Contexto, EDA y Modelos**. Integra datos y pronósticos conservados, y su registro de inferencia sólo habilita artefactos exportados con metadatos y comprobaciones de integridad. La publicación de este libro en GitHub Pages es distinta del servicio Dash.

El [Bloque B](05_configuraciones_exportacion.md) conserva el linaje de selección y exportación; el [modelo de referencia](06_modelo_original.md) y el [análisis posterior](08_verificacion_estadistica.md) mantienen sus alcances. Hacer utilizable la evidencia no acredita novedad, defensa del autor o R3 ni convierte un calendario conocido en confirmación prospectiva.

## 9.1 Objetivo

La aplicación permite entender qué volatilidad se pronostica, explorar la información disponible y examinar el error y las capacidades de cada sistema. La cantidad objetivo es **sigma7 poblacional de siete retornos logarítmicos diarios de Close, sin anualización**, con horizonte **un día**. Una salida alta indica más variabilidad esperada; no informa la dirección del precio o una probabilidad de beneficio.

El proyecto documenta **608 fechas históricas comunes**, del 1 de enero de 2025 al 31 de agosto de 2026, para 17 sistemas. Es una cifra del experimento, con [fuente y auditoría](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html); no una validación prospectiva. El dashboard debe comunicar también qué se desconoce, cuándo una entrada extrapola y qué escenarios fueron desfavorables.

## 9.2 Alcance y límites de memoria

El catálogo representa los **diez modelos del Top 10, seis referencias y la CNN-LSTM de E3**. Sus métricas medias proceden de 143 predictores: trece redes E2 con diez semillas, tres clásicos con una trayectoria y diez réplicas E3. El ranking conserva las 608 fechas y la métrica primaria media de RMSE individuales; los filtros EDA no vuelven a seleccionar recetas o semillas.

Las curvas y residuos usan un predictor representativo por sistema: semilla **271801** en las redes, elegida por su menor identificador, y la trayectoria única de cada clásico. Esa curva tiene su propia RMSE; no se presenta como la media de diez ejecuciones o como un ensamble.

| Catálogo | Evidencia histórica | Capacidad del prototipo y límite |
|---|---|---|
| FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer | Métricas y trayectorias conservadas de E2. | Resultados precalculados y diagnósticos; no se presume que excedan memoria sin medirlos. Su exportación e integración de inferencia siguen pendientes. |
| MLP16, LSTM32 y GRU32 | Pesos y calibradores heredados; las tres exportaciones TorchScript anteriores conservan su alcance. | ONNX CPU de la semilla 271801, con ida y vuelta sobre las 608 ventanas de prueba. El intervalo reutiliza el q del mismo predictor, sin recalibrar. |
| Persistence, SeasonalNaive y ARIMA | Una trayectoria por comparador, con contrato temporal propio. | Precálculos: no se alimentan con el formulario D7×5 de las redes ni se multiplican como semillas independientes. |
| CNN-LSTM con atención de contenido y salida aditiva positiva | Receta de 1210 parámetros en diez semillas; métricas posteriores y representante 271801. | ONNX CPU del representante 271801, con paridad sobre las 608 ventanas. **No dispone de intervalo calibrado**; la aplicación muestra esa ausencia. |

La conversión conserva pesos, escalas y receta: **cero nuevos entrenamientos y cero recalibraciones**. La tolerancia declarada de la ida y vuelta es 10⁻⁶ en unidades originales de sigma7. Se comparan las mismas entradas crudas, su preprocesamiento guardado y las salidas del módulo en memoria y del artefacto. Las diferencias frente a los pronósticos históricos guardados se registran aparte.

| Artefacto ONNX, semilla 271801 | Error máximo frente a memoria CPU | RMSE del artefacto, 608 fechas | ΔRMSE ONNX − memoria |
|---|---:|---:|---:|
| MLP16 | 9,992×10⁻⁹ | 0,0044821001 | −9,705×10⁻¹¹ |
| LSTM32 | 9,992×10⁻⁹ | 0,0047289048 | −5,161×10⁻¹⁰ |
| GRU32 | 9,992×10⁻⁹ | 0,0039595558 | −2,993×10⁻¹⁰ |
| CNN-LSTM con atención E3 | 6,410×10⁻¹⁰ | 0,0029102931 | −2,273×10⁻¹³ |

Los cuatro casos cumplen la tolerancia declarada. La {download}`comprobación del runtime sin Torch <../_static/e3_dashboard_20261117/serving_verification.json>` también confronta sus 608 salidas por modelo con los pronósticos congelados, dentro de la tolerancia. Estos valores corresponden a **un predictor**, no reemplazan las métricas medias de diez semillas del capítulo 8. La comprobación de salida e intervalos heredados no demuestra cobertura futura o calibración de E3.

El runtime mantiene **una sesión ONNX residente por proceso**, con acceso serializado y carga bajo demanda. Leer el catálogo no carga redes. Cambiar de modelo libera la sesión anterior antes de crear la siguiente. Esta política acota la residencia; no demuestra que todo el servicio quepa en un plan de Render, cuyo arranque, memoria, concurrencia y coste total requieren medición propia.

La aplicación no importa PyTorch ni los ejecutores científicos. Las dependencias runtime están fijadas en `requirements-dashboard.lock.txt`, incluidas Dash, Plotly y ONNX Runtime CPU. Los picos RSS de la exportación científica, las latencias de distintas plataformas y la memoria del servicio no se mezclan como una misma medición.

## 9.3 Arquitectura y estructura del repositorio

`app.py` crea `Dash(..., use_pages=True)` y expone `server`; cada sección es una página. El paquete `e3_dashboard/` separa datos, figuras, admisión/inferencia y explicación del código científico histórico de `src/`. Los ejecutores anteriores se conservan con su procedencia; no se importan al arrancar la aplicación ni se ejecuta HPO desde un callback.

```text
app.py                              # aplicación multipágina y server WSGI
pages/contexto.py                    # problema, metodología y Data Card
pages/eda.py                         # filtros comunes, gráficos e interpretaciones
pages/modelos.py                     # catálogo, diagnósticos y formulario de predicción
e3_dashboard/data.py                 # bundle local y comprobación de integridad
e3_dashboard/plots.py                # figuras y diagnósticos descriptivos
e3_dashboard/inference.py            # validación, preprocesamiento guardado y ORT CPU
e3_dashboard/explain.py              # contrastes locales del predictor
e3_dashboard/history.py              # historia de entrenamiento conservada y verificada
assets/e3_dashboard.css              # diseño visual y comportamiento responsive
data/e3_dashboard_v1/                # datos derivados, trayectorias y manifiesto
data/e3_dashboard_training_v1/       # historia del representante, con unidades separadas
artifacts/e3_dashboard_v1/           # registro de inferencia y artefactos admitidos
experiments/e3_dashboard_20261117/   # construcción y comprobaciones de esta versión
tests/                              # pruebas de datos, inferencia e integración
requirements.txt                    # entrada al lock exacto del runtime
requirements-dashboard.lock.txt     # dependencias de servicio sin framework de entrenamiento
Dockerfile                          # contenedor de servicio separado del entorno científico
MODEL_CARD.md                       # ficha de la CNN-LSTM de referencia
DATA_STATEMENT.md                   # fuentes, unidades, derechos y privacidad
book_e3/                            # informe publicado, capítulos 1–16 y anexos
```

Este árbol distingue el runtime nuevo del pipeline científico heredado. La consolidación del entorno de entrenamiento y los ficheros de despliegue se tratan en la [sección 10](10_despliegue_render.md); una estructura de carpetas no prueba por sí sola Docker, despliegue o reproducción integral.

Para ejecutar el prototipo desde una copia del repositorio y un entorno Python 3.12:

```bash
python -m pip install -r requirements.txt
python app.py
```

La ejecución sirve las páginas localmente; no publica una URL de Render. El registro de inferencia identifica modelos permitidos y verifica sus hashes. No acepta rutas o modelos serializados arbitrarios del usuario.

## 9.4 Contenido de las tres secciones

### 9.4.1 Contexto

La portada explica el objetivo sigma7 y el horizonte, la diferencia entre variabilidad y dirección, las preguntas del proyecto, el tamaño del experimento y su fuente. El recorrido metodológico conecta datos/EDA, literatura, benchmark, diseño E3, evaluación y confirmación pendiente.

La Data Card distingue fuente Binance Spot BTCUSDT, días UTC, ventana D7×5, variable objetivo y cohortes. La declaración de datos deja explícito que la licencia final de redistribución no está acreditada; no le atribuye una licencia inexistente al dataset. Las limitaciones de un activo, una fuente, dependencia y exposición histórica se muestran con autor, versión y fecha.

La hipótesis de la referencia aditiva no se confunde con la penalización y saturación del primer diseño. La [sección 7](07_diseno_riguroso.md) conserva los resultados adversos y la necesidad de aislar qué aporta cada componente. La aplicación enlaza el informe y permite leer las fichas del modelo y de los datos.

```{figure} ../_static/e3_dashboard_20261117/contexto_desktop.png
:name: e3_dashboard_contexto
:alt: Portada real del prototipo local con el objetivo sigma7, cifra de 608 fechas, preguntas, metodología, ficha de datos y límites.
:width: 100%

Vista Contexto del prototipo local. Las cifras describen las cohortes y la evidencia conservada; la captura no acredita una confirmación prospectiva o un despliegue Render.
```

### 9.4.2 EDA

Los controles `dcc.Dropdown`, `dcc.DatePickerRange` y `dcc.RangeSlider` seleccionan cohortes, fechas objetivo, rango de sigma7 y variable conocida al origen. Un mismo resultado filtrado alimenta **todos los gráficos, interpretaciones, conteos y avisos de esa página**. No hay balance de clases ficticio: es una regresión continua.

| Pregunta | Vista e interpretación |
|---|---|
| ¿Cómo varía el objetivo? | Serie y distribución de sigma7 con cuantiles y número efectivo de observaciones de la selección. |
| ¿Qué información se conocía al pronosticar? | Explorador de OHLCV de la última barra conocida, último retorno conocido y piso observable, distinguiendo fecha t de t−1. |
| ¿Qué cambia entre cohortes? | Resúmenes de distribuciones con la misma máscara, sin presentar diferencias descriptivas como confirmación independiente. |
| ¿Qué dependencia hay entre días? | ACF/PACF sobre el tramo diario continuo más largo de la selección; ADF cuando hay soporte suficiente, con su hipótesis y límites. No se unen purgas o huecos para calcular retardos. |
| ¿Qué calidad tiene esta vista? | Fechas inválidas/duplicadas, campos no finitos, incoherencia de la última barra y saltos del calendario. Los filtros o purgas pueden crear saltos; no se etiquetan automáticamente como pérdida de datos. |

El bundle conserva **1316 ejemplos ventana–objetivo**: 655 de entrenamiento E2, 53 de calibración y 608 de evaluación. No son 1316 barras originales independientes. Cada covariable de la vista procede de t−1 o de historia anterior, mientras que el objetivo se realiza en t. No se reconstruye una serie única pegando ventanas solapadas.

Los subconjuntos vacíos, constantes o demasiado pequeños reciben un estado explicativo, sin p-valores o correlaciones rellenados con cero. Las bandas ACF/PACF son referencias aproximadas de ruido blanco y ADF no certifica estacionariedad global. Casi duplicados, ruido del proxy y auditoría completa siguen teniendo el alcance de la [sección 4](04_correcciones_e2.md).

```{figure} ../_static/e3_dashboard_20261117/eda_desktop.png
:name: e3_dashboard_eda
:alt: Vista EDA real con filtros comunes de cohorte, fecha y objetivo, gráficos y panel temporal con soporte explícito.
:width: 100%

Vista EDA del prototipo local. Las interpretaciones y conteos se calculan con la misma selección; los diagnósticos temporales conservan un soporte continuo explícito.
```

### 9.4.3 Modelos

La tabla `dash_table.DataTable` permite ordenar y filtrar los **17 sistemas** y conserva las cifras de la [tabla ampliada del capítulo 8](../_static/e3_revision_bloque_d_20261117/metrics_extended.csv). Presenta media y DE de RMSE, MAE, sMAPE, R² y cobertura/anchura disponibles. Los metadatos y costes ausentes se identifican como tales; no se rellenan con cero ni se combinan perfiles no comparables.

La selección de un modelo muestra reales frente a pronósticos, residuos, ACF residual y la relación observado–predicho para su representante. La etiqueta identifica semilla y cohorte, para separar el comportamiento de esa curva del promedio de diez semillas del ranking. No se añaden ROC o matrices de confusión de clasificación.

La vista estadística conserva la familia principal de 48 contrastes y el MCS auxiliar de 17 sistemas: **diez supervivientes, incluida E3**. El MCS usa otro estimando y no declara ganador único, equivalencia o ganancia RMSE mayor que δ.

La frontera de Pareto inicial compara **los cuatro ONNX verificados**, con RMSE del representante 271801 sobre las 608 fechas y mediana CPU del mismo perfil: un hilo, lote uno, diez calentamientos y cien llamadas, incluyendo preprocesamiento e inferencia caliente y excluyendo explicación. No usa la media de diez semillas para etiquetar la métrica de ese artefacto ni mezcla estas medidas con GPU o TorchScript anteriores. Los trece sistemas restantes mantienen sus costes de servicio comparables pendientes; esta frontera parcial no declara eficiencia global de toda la familia o cabida en Render.

La subsección E3 enlaza procedencia, formulación/pseudocódigo, pruebas, métricas y límites. La aditiva obtiene menor RMSE media histórica, pero la interpretación conserva los tratamientos distintos de E2 y E3 y los requisitos pendientes de R3. También se conserva el diagnóstico adverso del piloto: 22,50 % más RMSE que M00 proyectado en 18 fechas de objetivo realizado alto; ese subconjunto es retrospectivo y no está disponible al pronosticar.

La curva de épocas del representante 271801 procede de su checkpoint conservado: 40 épocas, con mejor validación en la 38. Se distinguen dos unidades: pérdida MSE normalizada promediada durante las actualizaciones de ajuste y RMSE de validación en las 116 fechas, en sigma7 original. No son riesgos directamente comparables para calcular la brecha de adecuación. La curva no sustituye las Tablas A/B ni el estudio de aprendizaje al variar el tamaño de muestra.

**Predicción de un caso.** La acción explícita del usuario ejecuta sólo modelos habilitados por el registro. Se admiten CSV UTF-8 de hasta **100 KiB**, exactamente siete filas y las columnas, en ese orden, `date,open,high,low,close,volume`. Se exige t−7 a t−1, sin huecos, desorden o duplicados, con precios finitos/positivos, volumen no negativo y coherencia low≤open/close≤high. La fecha objetivo no puede superar el día UTC actual; se usan barras diarias completas. Una entrada inválida produce un mensaje comprensible antes de inferir.

El preprocesamiento aplica transformaciones y estadísticas guardadas de ajuste; no se estiman con la carga del usuario. La salida puntual se muestra en unidades originales de sigma7, junto con latencia de carga/preprocesamiento/inferencia y el alcance de esa medida. La explicación tiene un tiempo separado. El intervalo E2, cuando está disponible para ese predictor, es `[max(0, predicción−q), predicción+q]`, con q congelado y nominal 0,90; la cobertura media de diez semillas no se atribuye al intervalo de un caso individual. E3 comunica la ausencia de intervalo calibrado.

**Fuera de dominio y explicación local.** El aviso OOD compara la ventana transformada con rangos observados sólo en ajuste, por retardo y canal. No excederlos no certifica fiabilidad. La explicación local reemplaza, una por vez, cada barra OHLCV completa por la referencia de ajuste, reescalada al último cierre de la entrada. Se preserva la coherencia interna de cada barra, aunque puede cambiar la dinámica temporal. Los siete contrastes describen sensibilidad del predictor; no suman necesariamente su salida ni identifican causas. La fidelidad explicativa, estabilidad entre semillas y comparación global con el mejor benchmark no quedan acreditadas por este ejercicio.

```{figure} ../_static/e3_dashboard_20261117/modelos_desktop.png
:name: e3_dashboard_modelos
:alt: Vista Modelos real con tabla de 17 sistemas, diagnósticos de un representante, formulario, evidencia estadística y ficha del modelo.
:width: 100%

Vista Modelos del prototipo local. El catálogo representa toda la familia; sólo los cuatro artefactos ONNX admitidos ofrecen inferencia en vivo. La media de diez semillas y los diagnósticos del representante se identifican por separado.
```

## 9.5 Calidad de diseño, ingeniería y aceptación

La imagen Docker también pasó los seis recorridos de navegador con usuario 10001,
sistema de archivos de aplicación de sólo lectura, un worker y un hilo. En esa
ejecución local se impuso un límite de 512 MiB y el cgroup registró un pico de
unos 185,9 MiB, sin OOM. Es una comprobación local de esos recorridos, no una
certificación de concurrencia o recursos de Render. El {download}`recibo del contenedor <../_static/e3_dashboard_20261117/docker_verification.json>` conserva
digest, dependencias, hashes y configuración. La construcción utilizó el modo
de ruedas offline del Dockerfile porque el builder local no resolvía DNS; las
ruedas no se versionan ni forman una capa de la imagen.

El diseño usa `dash-bootstrap-components` con FLATLY, paleta coherente, navegación fija, tipografía y espaciado comunes. Los gráficos Plotly incluyen pregunta/título, unidades, leyenda y hover, y cada vista EDA tiene interpretación dependiente de la selección. Se comprueban rejilla responsive, desplazamiento de tablas, contraste, foco/etiquetas y estados que no dependan sólo del color.

| Criterio de la guía | Implementación y alcance que debe verificar la aceptación |
|---|---|
| CPU y memoria | ORT con CPUExecutionProvider, una sesión residente y cargas serializadas. Medir aparte memoria de arranque/carga y sesiones concurrentes antes de afirmar cabida en Render. |
| Callbacks | Organización por página, funciones reutilizables y ejecución de predicción explícita; cambiar un filtro EDA no entrena, carga todas las redes o altera el ranking histórico. |
| Entradas y errores | CSV hasta 100 KiB, esquema/fechas/coherencia física y errores controlados. El servidor limita además el cuerpo HTTP a 512 KiB y devuelve 413 controlado si se excede; ese tope no amplía el tamaño permitido del CSV. No se ejecuta contenido subido ni se aceptan checkpoints del usuario. |
| Privacidad | Procesamiento de la carga en memoria y sin guardado deliberado del contenido en archivos, bases o logs de aplicación. La declaración distingue los registros técnicos de acceso del contenido de la carga. |
| Accesibilidad | Etiquetas y foco por teclado, texto en estados, contraste y diseño legible en escritorio/tablet. No se certifica una norma completa de accesibilidad por una captura. |
| Separación científica | Runtime sin framework de entrenamiento ni import de los ejecutores; artefactos, datos y exportación tienen procedencia y hashes. |
| Consistencia libro–app | Catálogo derivado de la misma tabla maestra de métricas, con prueba de igualdad de identificadores, valores y estados. Los diagnósticos de un representante conservan otra etiqueta que la media del ranking. |
| CI | Pruebas de datos, admisión, integridad, salida, caché y callbacks según el alcance del workflow; una prueba de software no acredita despliegue Render o reproducción humana independiente. |
| Producto completo | Ficha de modelo y datos accesibles; declarar exportaciones, interpretabilidad, recursos y despliegue todavía no completados. |

Las pruebas de navegador y los recibos de exportación deben indicar versión, entorno, casos ejercitados y límites, incluyendo muestra vacía, entrada inválida, hash corrupto, cambio de modelo y coherencia de los filtros. Las capturas describen la versión probada; no prueban por sí solas una predicción o un servicio público.

### Aceptación local ejecutada

La prueba de navegador usó Chromium 154.0.8037.57 y Python 3.12.14, con vistas de **1440×1000** y **800×1050**. Completó **seis flujos y 35 comprobaciones**, sin fallos, errores de consola, solicitudes fallidas o respuestas HTTP 5xx. Verificó navegación, las 17 filas y su orden/filtro, máscara EDA común de 1316/608/0 casos y borrado de trazas obsoletas, diagnósticos, ficha dentro de la app, Pareto de cuatro exportaciones, predicción CPU y CSV válido/inválido. La salida E3 mostrada a seis decimales coincidió con su representante conservado; su ausencia de intervalo calibrado fue explícita.

El recibo conserva los hashes de la fuente probada y seis capturas de escritorio/tablet. También comprueba el enlace de salto por teclado y ausencia de desbordamiento horizontal del documento en tablet. No es una auditoría completa de accesibilidad, una reproducción humana independiente o una prueba del servicio Render. El {download}`recibo de aceptación en navegador <../_static/e3_dashboard_20261117/browser_verification.json>` delimita los casos ejercitados; las vistas de tablet de [Contexto](../_static/e3_dashboard_20261117/contexto_tablet.png), [EDA](../_static/e3_dashboard_20261117/eda_tablet.png) y [Modelos](../_static/e3_dashboard_20261117/modelos_tablet.png) completan las capturas anteriores.

Las pruebas de software distinguen nueve casos de datos, once de admisión/inferencia, tres de historia y dos de integración de la aplicación. Incluyen rechazo de hashes corruptos, caché acotada, transformación/piso, límites de entrada y separación de imports científicos. La oclusión comprueba contrastes del artefacto, sin acreditar fidelidad explicativa o estabilidad entre semillas. La aceptación local no completa los costes de los trece modelos sin exportar ni los requisitos estadísticos. El [diagnóstico de intervalos posterior del capítulo 8](08_verificacion_estadistica.md) produjo cuantiles E3 separados; **no están activados en este perfil de servicio**, que continúa declarando la ausencia de intervalo E3.

Se conservan la {download}`matriz de requisitos y estados <../_static/e3_dashboard_20261117/dashboard_requirements.csv>`, las {download}`notas de implementación y alcance <../_static/e3_dashboard_20261117/implementation_notes.md>`, el {download}`informe del bundle de datos <../_static/e3_dashboard_20261117/data_verification.json>` y el {download}`informe de exportación e ida y vuelta <../_static/e3_dashboard_20261117/export_verification.json>`. La {download}`ficha del modelo <../_static/e3_dashboard_20261117/MODEL_CARD.md>` y la {download}`declaración de datos y privacidad <../_static/e3_dashboard_20261117/DATA_STATEMENT.md>` también son accesibles desde la aplicación.

El [diseño anterior del Bloque E, versión 0.2](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_e/diseno_dashboard_v2.md) y su {download}`copia descargable <../_static/e3_bloque_e/diseno_dashboard_v2.md>` conservan sus bytes y su alcance histórico, igual que la {download}`versión 0.1 publicada anteriormente <../_static/e3_bloque_e/diseno_dashboard.md>`. Sus estados anteriores de datos, MCS e implementación no describen automáticamente este prototipo. La revisión vigente se apoya en la aplicación y sus evidencias; el despliegue público y su prueba de humo final se cierran en la sección 10.
