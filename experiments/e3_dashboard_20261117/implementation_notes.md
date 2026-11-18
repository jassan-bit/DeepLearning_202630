# Implementación inicial del dashboard E3

Fecha de esta revisión: **17 de noviembre de 2026, America/Bogota**. Fuente
de partida: `b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4`. Se preparó con asistencia
de IA; no es una firma, revisión humana independiente o acreditación R1/R2/R3.

## Producto y separación de responsabilidades

La aplicación usa Dash multipágina (`use_pages=True`) y las rutas Contexto,
EDA y Modelos. `app.py` expone `server`. El paquete `e3_dashboard/` separa
bundle de datos, figuras, admisión/inferencia y explicación. El runtime no
importa PyTorch, TensorFlow, JAX ni los ejecutores de entrenamiento históricos.
No hay optimización, ajuste de pesos o recalibración en callbacks.

El lock exacto del servicio incluye Dash, dash-bootstrap-components, Plotly,
NumPy, ONNX Runtime CPU, Flask y Gunicorn, con sus dependencias. El entorno de
navegador/pruebas es distinto: Playwright no forma parte del runtime. Un
entorno virtual con versiones fijadas no se presenta como una imagen Docker
probada ni como un servicio ya desplegado en Render.

## Datos, métricas y selección de representantes

`build_data.py` deriva un bundle local de cinco archivos de datos/metadatos y
un manifiesto. La verificación registra **1316 ejemplos ventana–objetivo**:
655 de entrenamiento E2, 53 de calibración y 608 de evaluación. El bundle
incluye 17 filas de catálogo, 17 trayectorias representativas, tres ejemplos
de entrada y fuentes/huellas. Su tamaño registrado inicial es **525100 bytes**,
incluido el manifiesto, no el tamaño de todo el repositorio o del runtime.

Las métricas se reutilizan de la tabla ampliada de 17 sistemas y conservan
143 predictores científicos: trece redes E2 por diez semillas, tres clásicos
con una trayectoria y diez réplicas E3. Las curvas usan 271801 para las redes
por ser el menor ID, no por ganar en prueba. Las métricas de esa curva son
distintas de las medias de diez ejecuciones del catálogo.

Las observaciones EDA distinguen fecha objetivo t y última barra conocida t−1.
La vista no pega ventanas solapadas como una serie original independiente.
Los filtros usan una máscara común para todos los gráficos y textos. La ACF,
PACF y el diagnóstico ADF se calculan sobre el segmento diario continuo más
largo que admite la selección; no se unen purgas, huecos o fechas descartadas.

El MCS y la familia de contrastes son resultados guardados de la etapa
estadística, con su pérdida, nivel y alcance. La navegación o un filtro de
tabla no redefine la familia científica. Los IC entre semillas, intervalos
predictivos y variabilidad temporal conservan etiquetas distintas.

La historia de entrenamiento de 271801 se deriva de su checkpoint verificado:
40 épocas, mejor validación en la 38. La pérdida normalizada de ajuste se
promedia durante actualizaciones del optimizador; la validación usa RMSE en
116 fechas y unidades originales. Sus ejes se distinguen y no se usa esa
diferencia como una prueba de adecuación o una curva de tamaño de muestra.

## Exportación y servicio CPU

El registro incorpora ONNX de **MLP16, LSTM32, GRU32 y la CNN-LSTM de E3**,
todos del representante 271801. El informe `export_verification.json` conserva
las 608 fechas, tolerancia de salida **10⁻⁶ en unidades originales de sigma7**,
comparación con memoria CPU, comparación con pronósticos congelados y métricas
de cada artefacto. Los cuatro cumplen la tolerancia: errores máximos frente a
memoria de aproximadamente 9,992×10⁻⁹ en las tres referencias y 6,410×10⁻¹⁰
en E3. No se entrenó ni recalibró durante la exportación; no se usó GPU.

Las latencias del informe son preprocesamiento e inferencia ORT caliente, un
hilo CPU, lote uno, diez calentamientos y cien llamadas; excluyen explicación.
No miden HTTP, arranque de Render, memoria del servicio, múltiples usuarios o
todos los 17 sistemas. El RSS del proceso exportador incluye el framework
científico y no se atribuye a la app.

El servicio mantiene una sesión residente y carga serializada; verifica los
hashes de artefacto y metadatos. El CSV se valida antes de inferir: UTF-8,
máximo 100 KiB, siete filas, seis columnas exactas, precios finitos/positivos,
volumen no negativo, coherencia OHLC, fechas t−7..t−1 y objetivo no posterior
al día UTC actual. No se reordenan, imputan o reparan silenciosamente entradas.

Los rangos OOD se estiman con ajuste, por retardo y canal. El intervalo de las
tres referencias reutiliza el q histórico del mismo predictor. E3 no tiene q
calibrado y devuelve estado pendiente, no un intervalo fabricado de la DE de
semillas. El nominal 90 % no es una garantía bajo dependencia o shift.

Además del límite de 100 KiB del CSV, Flask limita el cuerpo de una solicitud
HTTP a 512 KiB y responde con 413 controlado. Son capas distintas: el tope HTTP
no permite cargar CSV de mayor tamaño ni elimina la validación semántica.

## Explicación, privacidad y límites

La explicación local efectúa siete intervenciones: sustituir una barra completa
por una barra de una referencia real de ajuste, con precios reescalados al
último cierre de la entrada. Los contrastes se obtienen de llamadas reales al
mismo artefacto y se conserva la coherencia de cada barra. Pueden cambiar la
dinámica temporal, no suman necesariamente la salida y no son causas del
mercado. Fidelidad explicativa, estabilidad entre semillas y comparación
global con el mejor benchmark siguen pendientes.

Las entradas del usuario no se usan para entrenar y no se guardan de forma
deliberada en ficheros, bases de datos o logs de aplicación. El contenido se
procesa en memoria; los registros técnicos de acceso tienen otro alcance.
Las rutas de fichas permiten sólo los archivos fijos MODEL_CARD.md y
DATA_STATEMENT.md. El usuario no elige rutas de sistema o checkpoints.

## Evidencia y trabajo restante

Los informes de datos y exportación se conservan con este corte. Las pruebas
de admisión, integridad, bundle, paridad e integración y la aceptación de
navegador deben leerse por su alcance concreto. Sus resultados y capturas se
incorporan a la matriz de requisitos; no se marca una función cumplida por
haberla descrito o mostrado en una maqueta.

La aceptación local final del navegador, registrada a las 00:26:21 UTC del
18 de noviembre (17 de noviembre en America/Bogota), completó seis flujos y
35 comprobaciones sin fallos o errores. Chromium 154.0.8037.57 verificó las
tres páginas en 1440×1000 y 800×1050, máscara EDA compartida y estado vacío,
tabla de 17 filas, predicción CPU, CSV válido/inválido, ficha, Pareto de cuatro
modelos, salto por teclado y ausencia de desbordamiento del documento en
tablet. El informe conserva hashes y seis capturas. No hubo solicitudes
fallidas, errores de consola o respuestas internas 5xx en la repetición final.

Los grupos de software conservan alcances separados: nueve casos de datos,
once de inferencia/admisión, tres de historia y dos de integración. La
verificación de inferencia científica y la aceptación de interfaz no son una
prueba de fidelidad explicativa, una auditoría completa de accesibilidad o una
reproducción humana desde cero.

Siguen pendientes el despliegue Render y su prueba de humo pública, medición
completa de memoria/concurrencia y recursos comparables, exportaciones del
resto del catálogo, calibración E3, interpretabilidad global y estable, cierre
de licencia/derechos y reproducción humana. La novedad, defensa del autor y
la evaluación confirmatoria R3 mantienen los requisitos de los capítulos 6–8.
