# 12. Incertidumbre, errores y referencias de dificultad

**Estado vigente del 14/11/2026:** existen intervalos y cobertura empírica para los 16 sistemas, diagnósticos F2a, robustez, explicaciones, tres ablaciones entrenadas y una taxonomía de residuos de los diez modelos principales. Los artefactos actuales están en [§§49–59](53_cierre_requisitos.md). Esta página distingue los resultados existentes de las garantías que no están acreditadas.

## 12.1 Intervalos y calibración originales

Los predictores neuronales conservan seis trayectorias y los comparadores una. Antes del scoring se fijó por predictor un cuantil de residuos absolutos sobre **53 fechas de calibración**. Se reportan cobertura nominal del 90 % y anchura de intervalos. El [benchmark](06_benchmark_final.md) y la [auditoría de calibración](49_diagnosticos_f2a.md) mantienen los calibradores y sus resultados adversos.

El uso de una cronología dependiente limita la garantía de un calibrador basado en intercambiabilidad. Cuantificar cobertura observada no demuestra cobertura condicional ni validez para cualquier régimen futuro. Los intervalos de cobertura por bloques reflejan sensibilidad temporal bajo sus propios supuestos; no convierten el procedimiento de calibración original en uno adaptado por bloques. Tampoco se inventa PIT sin una CDF predictiva completa. El suplemento de [§60](60_calibracion_temporal_condicional.md) añade cobertura por condiciones y variantes temporales causales sin modificar los calibradores originales; conserva las infracoberturas observadas y no selecciona una variante usando el test.

## 12.2 Robustez y explicaciones

[§54](54_robustez_interpretabilidad.md) publica cuatro tipos de perturbación, cada uno con tres severidades, sobre los predictores congelados. Se conservan objetivos, pesos y calibradores. Las alteraciones de ventanas independientes son escenarios de estrés de entrada; pueden romper la coherencia de una trayectoria OHLCV y no se describen como intervenciones económicas reales.

SAE, GRU32 y ARIMA, el Top 3 observado por RMSE, tienen diagnósticos de oclusión, borrado/inserción y un orden aleatorio de referencia en 24 fechas. Las redes disponen de seis semillas; ARIMA determinista no tiene variación de optimización por semilla. La fidelidad medida sobre esas fechas no se interpreta como causalidad ni validación independiente del explicador.

Las [18 ablaciones de §55](55_reproduccion_ablaciones.md) retiran tres componentes de SAE, con seis semillas por variante. Retirar el dropout empeora la RMSE con evidencia ajustada; las otras dos comparaciones son no concluyentes. No se vuelve a elegir el modelo a partir de estas pruebas posteriores.

## 12.3 Taxonomía operativa y muestra auditable

[§57](57_errores_techo.md) etiqueta **6.080 fechas-modelo** y publica **100 fechas distintas por modelo**, con sus seis residuos individuales: 600 casos por modelo. Las categorías de magnitud/signo usan umbrales de desarrollo y reglas registradas antes del etiquetado automático, después de conocer el test. No se presentan como anotaciones humanas o causas semánticas.

Las frecuencias y sus IC temporales se calculan con 10.000 remuestreos y bloques de 7/14/28 días. Las categorías no observadas pueden producir IC empíricos [0,0], que no implican probabilidad poblacional cero. Se conservan los pesos de muestreo y la concordancia entre semillas.

## 12.4 Techo y distancia γ

Se calculó entropía ordinal como referencia de complejidad. El ensemble uniforme de los diez modelos resultó peor que SAE y Persistence, por lo que no aporta un techo empírico válido. Frente a la cota matemática de RMSE cero, γ ideal es **0,93094**, con IC temporal **[0,84448; 1,02974]**. Ese cociente no identifica ruido irreducible ni un error mínimo alcanzable; se conserva el resultado negativo sin buscar otra combinación favorable sobre el test.

El nuevo [proxy empírico de §63](63_ensamble_calibracion.md) ajusta una combinación convexa sólo con los 53 pronósticos de calibración guardados, sin reentrenar las redes. Su RMSE puntual es 0,00368953 y γ del proxy 0,28757; el IC algebraico [−1,49210; 2,37128] y el orden favorable en 6.588/10.000 remuestreos muestran que no es una referencia estable. No se estima riesgo Bayes ni se cambia A2.

El alcance de las referencias anteriores está documentado en [§57](57_errores_techo.md). El [Apéndice B actual](61_reproduccion_consolidada.md) mantiene este requisito parcial, junto con el margen δg previo ausente y las limitaciones de calibración temporal.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/12_incertidumbre_errores.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
