# Ficha del modelo CNN-LSTM con atención de E3

**Versión documental: 17 de noviembre de 2026, America/Bogota.** Responsable del
proyecto: Jassan Arteta. La preparación de código, cálculos y documentación usa
asistencia de IA declarada; esta ficha no sustituye la revisión y defensa del
autor ni acredita una evaluación académica.

## Identificación y uso previsto

La referencia evaluada es una **CNN causal → LSTM unidireccional → atención
temporal por contenido → salida aditiva positiva**, con **1210 parámetros**.
«Aditiva» describe la cabeza de salida; la atención no es otro modelo entrenado
por separado. No es el primer módulo con penalización de discordancia y
saturación de 1211 parámetros ni la cabeza de momentos de 1236 parámetros.

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

La receta de E3 **no tiene un calibrador ni intervalos predictivos congelados**.
Por ello no se comunica cobertura nominal de un intervalo E3 inexistente. La
dispersión o un IC entre semillas tampoco son un intervalo de predicción para
un día nuevo. Las capacidades habilitadas del artefacto de servicio se describen
en el informe de exportación y en el capítulo 9, sin trasladar evidencia de
otro modelo o de otra semilla.

## Artefacto de servicio del representante

Se exportó a ONNX el predictor de la semilla **271801**, con inferencia en
ONNX Runtime CPU y sin importar el framework de entrenamiento en el servicio.
La ida y vuelta sobre las 608 ventanas crudas de prueba conserva las escalas:
error absoluto máximo frente al módulo en memoria CPU **6,40984571×10⁻¹⁰**, con
tolerancia declarada **10⁻⁶** en unidades originales de sigma7. La RMSE ONNX
de ese predictor es **0,0029102931283034364**, con diferencia frente a memoria
de **−2,27307322×10⁻¹³**. No es la RMSE media de diez semillas.

La exportación no ajustó pesos ni calibró intervalos. El servicio mantiene una
sesión residente con carga serializada y verifica los hashes de artefacto y
metadatos. El control OOD usa rangos de ajuste por retardo/canal. La explicación
local de oclusión reemplaza barras completas por una referencia de ajuste y
reporta sensibilidad; fidelidad explicativa y estabilidad entre semillas
siguen pendientes. La latencia de explicación se registra por separado.

## Entrenamiento y selección conservados

| Campo | Evidencia histórica |
|---|---|
| Ajuste E3 | 532 ejemplos, hasta 2024-06-30. |
| Separación ajuste–validación | Siete ejemplos purgados. |
| Validación de desarrollo | 116 fechas, 2024-07-08 a 2024-10-31; seleccionó mejor época y parada. |
| Calibración E3 | No utilizada por la receta congelada. |
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
conservan en la bitácora; Nd global sigue pendiente de conciliación.

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
| Cobertura y anchura predictivas de E3 | Pendientes de calibración. |

El IC cuantifica variación de inicialización condicionada al calendario
conocido; no es incertidumbre temporal, predictiva o simultánea de la familia.
La comparación posterior conserva 48 rechazos tras Holm global bajo los
supuestos de sus procedimientos. El MCS auxiliar sobre MSE conserva diez
sistemas, incluido E3; no declara un ganador único. Esos resultados no
acreditan automáticamente tres pruebas admisibles de la Tabla 4 o R3.

## Límites, sesgos y escenarios adversos

El calendario ya era conocido y el análisis es retrospectivo. La potencia
prospectiva y conjunta, comparación plenamente justa, tres conjuntos
adicionales, adecuación completa, calibración y estudios de robustez siguen
pendientes. La novedad R1, comprensión y defensa del autor para R2 y cierre R3
no se certifican con este dashboard.

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

Los metadatos de cada artefacto, sus hashes y pruebas de ida y vuelta deben
acompañar la versión de servicio. La aplicación sólo carga artefactos ya
entrenados; no optimiza hiperparámetros, ajusta pesos o recalibra con la entrada
del usuario. El despliegue público en Render y su prueba de humo son una etapa
distinta de la publicación del informe en GitHub Pages.
