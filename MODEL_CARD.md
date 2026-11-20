# Ficha del modelo CNN-LSTM con atención de E3

**Versión documental: 19 de noviembre de 2026, America/Bogota.** Responsable del
proyecto: Jassan Arteta. La preparación de código, cálculos y documentación usa
asistencia de IA declarada; esta ficha no sustituye la revisión y defensa del
autor ni acredita una evaluación académica.

## Identificación y uso previsto

La referencia evaluada es una **CNN causal → LSTM unidireccional → atención
temporal por contenido → salida aditiva positiva**, con **1210 parámetros**.
«Aditiva» describe la cabeza de salida; la atención no es otro modelo entrenado
por separado. No es el primer módulo con penalización de discordancia y
saturación de 1211 parámetros ni la cabeza de momentos de 1236 parámetros.
El candidato V3 de 1262 parámetros completó un piloto de desarrollo de 30
ajustes. Ese estudio se documenta por separado: no sustituye a este predictor
ni hereda sus métricas, calibrador o exportación.

Su uso previsto es explorar pronósticos académicos de volatilidad de
Binance Spot BTCUSDT y sus limitaciones. La salida es la desviación estándar
poblacional de siete retornos logarítmicos diarios de Close, **sigma7 sin
anualización**, con horizonte **un día**. No predice el precio, la dirección del
retorno ni una probabilidad de ganancia. La evidencia disponible no sustenta
ejecución automática de inversiones o decisiones económicas de alto impacto.

## Entrada, salida y disponibilidad temporal

La entrada física corresponde a siete barras OHLCV diarias consecutivas hasta
el día anterior a la fecha objetivo, en UTC. El CSV de servicio usa las columnas
`date,open,high,low,close,volume`; precios finitos y positivos, volumen no
negativo y coherencia `low <= open/close <= high`. El servidor debe rechazar
fechas objetivo dentro de la ventana, huecos, duplicados, desorden, valores no
finitos y formatos o tamaños no admitidos, antes de inferir.

El preprocesamiento y las escalas se conservan desde entrenamiento. El piso
observable de la salida se calcula sólo con los seis retornos conocidos de
esas siete barras; la etiqueta del siguiente día no es una entrada. La salida
puntual está en unidades originales de sigma7. No se vuelve a multiplicar una
escala si el artefacto de servicio ya la invierte.

La receta histórica de E3 no incorporó un calibrador. El **corte de servicio
del 18 de noviembre** adopta por separado un diagnóstico residual posterior
ya calculado para la semilla 271801. El perfil versionado usa
`q = 0.005402463770651868` en unidades originales de sigma7 y el intervalo
`[max(0, predicción−q), predicción+q]`, con nivel nominal del 90 %. Sus 53
fechas objetivo de calibración van del 8 de noviembre al 30 de diciembre de
2024; no intervienen en el ajuste o la selección de época de este predictor.
El runtime sólo comunica el intervalo para fechas objetivo **posteriores al
30 de diciembre de 2024**. Para fechas anteriores o iguales conserva el
pronóstico puntual y devuelve el intervalo ausente.

El perfil se vincula mediante hashes al ONNX, checkpoint, metadatos originales,
escala y semilla. No cambia los pesos, escalador, pronósticos puntuales o cuantil.
Es una adopción posterior de servicio; el plan original y su decisión de no
activar el diagnóstico en el dashboard se conservan como corte histórico.
La dispersión o un IC entre semillas tampoco son un intervalo de predicción
para un día nuevo.

## Artefacto de servicio del representante

Se exportó a ONNX el predictor de la semilla **271801**, con inferencia en
ONNX Runtime CPU y sin importar el framework de entrenamiento en el servicio.
La ida y vuelta sobre las 608 ventanas crudas de prueba conserva las escalas:
error absoluto máximo frente al módulo en memoria CPU **6,40984571×10⁻¹⁰**, con
tolerancia declarada **10⁻⁶** en unidades originales de sigma7. La RMSE ONNX
de ese predictor es **0,0029102931283034364**, con diferencia frente a memoria
de **−2,27307322×10⁻¹³**. No es la RMSE media de diez semillas.

La exportación original no ajustó pesos ni calibró intervalos. La verificación
del servicio actualizado ejecutó 661 ventanas históricas en ONNX Runtime CPU:
53 de calibración y 608 de evaluación. El error puntual máximo frente a las
predicciones conservadas fue **2,58387130×10⁻⁸ sigma7**, dentro de la misma
tolerancia de 10⁻⁶. Con el cuantil conservado, la cobertura del representante
es **94,078947 %** en las 608 fechas y **85,555556 %** en 2026Q1. Su anchura
media histórica es **0,01080287256 sigma7**. Son resultados del predictor
individual; no sustituyen la media de diez semillas ni garantizan cobertura
futura o por subgrupo bajo dependencia temporal.

Una observación de calibración situada en el cuantil cambia de inclusión por
el redondeo de la conversión: la cobertura empírica de calibración es 48/53
(90,566038 %) en ORT, frente a 49/53 (92,452830 %) en el cálculo CPU original.
El error máximo de calibración es 3,01696685×10⁻¹⁰ sigma7. No se retocó el
cuantil para eliminar ese efecto y la cobertura de evaluación se conserva.

El servicio mantiene una sesión residente con carga serializada y verifica
los hashes de artefacto y
metadatos. El control OOD usa rangos de ajuste por retardo/canal. La explicación
local de oclusión reemplaza barras completas por una referencia de ajuste y
reporta sensibilidad; fidelidad explicativa y estabilidad entre semillas
siguen pendientes. La latencia de explicación se registra por separado.

El catálogo actualizado habilita cinco predictores ONNX CPU: MLP16, LSTM32,
GRU32, este E3 y TSMixer. La extensión TSMixer conserva el checkpoint de la
semilla 271801 y su cuantil E2 original `0.007345522909209999`, sin ajuste nuevo.
Su error máximo frente a las 608 predicciones congeladas es
**2,28505873×10⁻⁸ sigma7**, dentro de 10⁻⁶; se contrastaron también las 53
predicciones de calibración. Su cobertura histórica individual es 89,309211 %.
Estos resultados no se atribuyen al modelo E3.

Nueve redes del catálogo histórico siguen pendientes de recuperar/exportar
con sus pesos y recetas originales y comprobar su paridad; no se acredita una
exclusión definitiva por recursos. Los tres controles clásicos se muestran
como resultados históricos: siete cierres no determinan los ocho que necesita
Persistence, los catorce que necesita SeasonalNaive al terminar en t−1, ni el
estado causal fechado de ARIMA. Habilitarlos requiere un contrato adicional
de historia o estado y su verificación; no exige que deban usar ONNX.

## Entrenamiento y selección conservados

| Campo | Evidencia histórica |
|---|---|
| Ajuste E3 | 532 ejemplos, hasta 2024-06-30. |
| Separación ajuste–validación | Siete ejemplos purgados. |
| Validación de desarrollo | 116 fechas, 2024-07-08 a 2024-10-31; seleccionó mejor época y parada. |
| Calibración E3 | No utilizada por la receta de entrenamiento; el corte posterior de servicio adopta el cuantil de 53 fechas sólo para el representante 271801. |
| Receta | MSE normalizada, AdamW, tasa 0,001, lotes 32, hasta 40 épocas, paciencia 8. |
| Réplicas del análisis actual | Diez semillas, 271801–271810; se conservan todas. |
| Semilla histórica separada | 20261116; no integra la media actual de prueba. |
| Predictor representativo del dashboard | 271801, elegido por el menor identificador, no por ganar en prueba. |

La historia del representante conserva 40 épocas y mejor validación en la
38. Su pérdida MSE normalizada durante actualizaciones y la RMSE de validación
se muestran con unidades separadas; no constituyen por sí solas una brecha
train–validación comparable ni una curva de tamaño de muestra.

El candidato se compara con recetas de E2 ajustadas con **655 ejemplos** y
calibradas con **53 fechas**. Diez semillas y fechas comunes no igualan esos
tratamientos ni sus presupuestos. Los pilotos y el esfuerzo de diseño se
conservan en la [bitácora de diseño](experiments/design_log.md), que separa
propuestas, variantes, parametrizaciones y réplicas y declara los límites del
inventario global. Repetir semillas no crea diseños nuevos.

## Evaluación histórica

Se usaron las mismas **608 fechas del 1 de enero de 2025 al 31 de agosto de
2026** para los 17 sistemas. Las métricas siguientes son la media de las diez
métricas individuales, no la métrica de un ensamble ni la del único predictor
representativo de la aplicación.

| Métrica E3 | Valor |
|---|---:|
| RMSE media individual | 0,002939432418821988 |
| DE muestral de RMSE entre semillas | 0,00004712541790127227 |
| MAE media individual | 0,0018928273030167733 |
| sMAPE medio | 11,097711472596185 % |
| R² medio | 0,911778724568431 |
| IC95 % bootstrap de RMSE entre semillas | [0,002913282254796285; 0,0029684039720926795] |
| Cobertura y anchura predictivas de la media de diez semillas | No se incorporan a esta tabla puntual; el perfil de servicio individual se documenta por separado. |

El IC cuantifica variación de inicialización condicionada al calendario
conocido; no es incertidumbre temporal, predictiva o simultánea de la familia.
La comparación posterior conserva 48 rechazos tras Holm global bajo los
supuestos de sus procedimientos. El MCS auxiliar sobre MSE conserva diez
sistemas, incluido E3; no declara un ganador único. Esos resultados no
acreditan automáticamente tres pruebas admisibles de la Tabla 4 o R3.

## Límites, sesgos y escenarios adversos

El calendario de este catálogo ya era conocido y su análisis es retrospectivo.
La [auditoría exploratoria posterior](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/08_verificacion_estadistica.html#e3-final-extended-audit)
completó 796 trabajos: 19 sistemas en D00–D03, con tres archivos adicionales,
perturbaciones, calibración y curvas. Conserva sus propios pesos, datos y
recibos; sus resultados no se atribuyen a los cinco artefactos de este servicio.
La independencia de los dominios, la potencia prospectiva y conjunta, la
adecuación completa y la validación confirmatoria conservan límites abiertos.
La novedad R1, comprensión y defensa del autor para R2 y cierre R3 no se
certifican con este dashboard.

El [piloto posterior V3](https://github.com/jassan-bit/DeepLearning_202630/blob/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z/payload/development_verdict.json)
completó 30 ajustes de desarrollo y evaluó tres semillas finales por receta en
55 ventanas de calendario previamente expuesto. La variante completa
`atom_crps` tuvo 2,70 % más RMSE media que el mejor control `atom_mse_only` y
no superó su heurística previa. No se promueve automáticamente el control
ganador ni se presenta ese resultado como una confirmación de novedad o R3.
V3 no integra el catálogo ni los artefactos servidos.

En el piloto v2, la referencia aditiva tuvo RMSE **22,50 % mayor** que M00
conservado y proyectado en **18 fechas de objetivo realizado alto**. Es un
diagnóstico posterior: ese grupo no está disponible al pronosticar y no
identifica una causa. El primer mecanismo de atención modificado y el piloto
por momentos tampoco superaron sus respectivos controles o heurísticas.

La cota condicional de la cabeza aditiva exige estados latentes acotados,
piso, escala y parámetros fijos. No demuestra estabilidad desde OHLCV, menor
error, calibración o novedad. Un aviso de fuera de rango es una señal de
extrapolación, no una validación de fiabilidad dentro del rango. Las
atribuciones locales, cuando estén habilitadas y verificadas, describen el
comportamiento del artefacto y no explican causalmente el mercado.

## Procedencia y reproducción

- [Diez réplicas E3 conservadas](https://github.com/jassan-bit/DeepLearning_202630/tree/510c3bddfdc9926322e3210ac61ee1f5eeffad2e/runs/estabilidad_aditiva_v1).
- [Tabla descriptiva ampliada y definiciones](https://github.com/jassan-bit/DeepLearning_202630/tree/b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4/experiments/e3_revision_bloque_d_20261117).
- [Informe: diseño y límites](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/07_diseno_riguroso.html).
- [Informe: métodos estadísticos y requisitos pendientes](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/08_verificacion_estadistica.html).
- [Declaración de los datos](DATA_STATEMENT.md).
- [Perfil posterior del representante](artifacts/e3_serving_calibration_v2/e3_additive_271801.json) y [verificación ORT de 661 ventanas](experiments/e3_cierre_integral_20261118/serving/calibration_runtime_verification.json).
- [Extensión TSMixer y paridad](artifacts/e3_serving_extensions_v2/tsmixer/roundtrip.json) y [matriz de capacidades pendientes](experiments/e3_cierre_integral_20261118/serving/export_capabilities/capability_matrix.csv).

Los metadatos de cada artefacto, sus hashes y pruebas de ida y vuelta deben
acompañar la versión de servicio. La aplicación sólo carga artefactos ya
entrenados; no optimiza hiperparámetros, ajusta pesos o recalibra con la entrada
del usuario. El despliegue público en Render y su prueba de humo son una etapa
distinta de la publicación del informe en GitHub Pages. Las verificaciones
locales y del contenedor no acreditan por sí solas la aceptación pública de
Render; ésta debe vincular la URL y el commit que efectivamente sirve.
