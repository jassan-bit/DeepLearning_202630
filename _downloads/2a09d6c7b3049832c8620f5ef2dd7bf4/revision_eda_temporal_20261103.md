# Revisión del EDA específico de regresión temporal — 03/11/2026

Se revisa el bloque C de §2.2 de la guía (pp. 6–7), con evidencia exclusiva de desarrollo 2023–2024. El contrato sigue siendo **tipo 4**, objetivo σ7 diario, **n=7**, **h=1** y entradas OHLCV de **D=7/14/21/28** días anteriores, sin rezagos explícitos del objetivo. Se conserva la exclusión del 24/03/2023, sin imputación ni compresión del calendario.

El usuario autorizó cuatro referencias lineales regularizadas únicamente para diagnosticar residuos en el EDA. No se entrenan las redes DL, no se ejecutan piloto o benchmark y no se consultan archivos de test. Completar este bloque no acredita cumplimiento total de E1.

| ID | Requisito | Estado | Evidencia | Decisión de preprocesamiento | Impacto en arquitectura | Riesgo o límite |
|---|---|---|---|---|---|---|
| C01 | Frecuencia, regularidad y huecos de las covariables | Ejecutado | 731 posiciones diarias UTC de 2023–2024; el 24/03/2023 permanece ausente y se excluyen muestras que lo necesitan sin comprimir fechas. | Exclusión estricta de entradas o etiquetas incompletas; sin imputación. | Tensores de D días de calendario completos, no de las últimas D filas disponibles. | El episodio no identifica MCAR/MAR/MNAR ni autoriza interpolación futura. |
| C02 | Disponibilidad y desfases en el momento de predicción | Ejecutado | Alineación explícita de OHLCV(d−k), k≥1, disponible al inicio de d; σ7(d) se conoce al terminar d y no entra en X. | Conservar marcas UTC y permitir solo días completos anteriores al origen. | Cinco covariables OHLCV y D∈{7,14,21,28}; ningún rezago explícito del objetivo. | OHLCV completo de d, σ7(d) o correlaciones móviles etiquetadas d introducirían información no disponible al inicio de d. |
| C03 | Correlaciones cruzadas con rezagos de covariables | Ampliado con cohorte común | 140 pares variable–rezago, k=1…28, usan las mismas 308 fechas de 2023; Pearson y Spearman. Se distingue el análisis marginal anterior con 328–350 pares. | Cohorte común para describir rezagos; log(OHLC) y log1p(volumen) son transformaciones fijas del diagnóstico. | Se conservan las cinco variables y las cuatro ventanas; los picos no seleccionan D ni un rezago óptimo. | Correlaciones de niveles pueden ser espurias; no prueban causalidad, habilidad predictiva ni significancia tras buscar máximos. |
| C04 | Deriva de relaciones covariable–objetivo | Ejecutado descriptivamente | Pearson móvil de 60 días con mínimo 50 pares; 309 ventanas definidas en 2023 y 366 en 2024 por covariable, con resumen anual. | Mantener calendario y resumir variación temporal sin igualar artificialmente años. | Evaluar después estabilidad de modelos simples antes de aumentar capacidad; este análisis no selecciona arquitectura. | Ventanas y objetivos solapados; cambios de signo no constituyen quiebres independientes ni causas económicas demostradas. |
| C05 | Ajuste de referencia lineal regularizada | Ejecutado bajo autorización específica para EDA | Cuatro Ridge estáticos con α=1 fijo, sin tuning; Nfit=343/322/301/280 según D. Escalado train-only ddof=0 e intercepto libre; objetivo SSE+α\|\|β\|\|², sin factor 1/N. | Ajustar media/escala solo en muestras permitidas de 2023, escala 1 si varianza cero y objetivo sin transformar. | Referencia lineal de 5D entradas para diagnóstico, separada de los tres modelos DL; pesos fijos durante 2024. | α y cadencia estática son decisiones diagnósticas; no congelan E2. Predicciones sin recorte, aunque podrían ser negativas. |
| C06 | Autocorrelación de residuos fuera del ajuste | Ejecutado como diagnóstico exploratorio | 366 residuos de 2024 por D, 1464 en total; 112 ACF a rezagos 1…28 y 12 Ljung–Box a 7/14/28 con model_df=0 y Holm sobre la familia completa. | Conservar errores y fechas pareados, sin seleccionar D ni α por valores p; documentar bandas puntuales aproximadas. | La dependencia residual motiva considerar la estructura temporal en la evaluación; no decide añadir rezagos del objetivo ni reclasificar el proyecto. | Etiquetas consecutivas comparten seis retornos. Solapamiento, límites lineales y deriva pueden explicar dependencia; χ² es una referencia asintótica y Holm no corrige su calibración. |
| C07 | Partición cronológica, embargo y escalado pasado | Verificado para las referencias EDA | Entrenamiento u≤t−D−1; se omiten [t−D,t−1] antes del primer origen 01/01/2024. Índices y hashes coinciden con el manifiesto CV; el ajuste fijo satisface el embargo en todos los orígenes posteriores. | Mantener las máscaras y el embargo existentes; ninguna etiqueta 2024 ajusta pesos o escalas. | Los cuatro modelos diagnósticos no modifican política de reajuste ni protocolo de los futuros modelos DL. | Un embargo de covariables no demuestra independencia de etiquetas; no se fijan longitud de bloques ni inferencia confirmatoria. |
| C08 | Tipo de problema e implicaciones técnicas explícitas | Documentado con evidencia | Tipo 4: regresión temporal; n=7, h=1 y D=7/14/21/28, con cinco covariables históricas y sin rezagos explícitos del objetivo. | Mantener σ7(d) como salida y OHLCV(d−D…d−1) como entrada; sin imputación. | Las redes documentales siguen sin implementar o entrenar en este alcance; las cuatro Ridge no constituyen esos tres baselines DL. | Incorporar rezagos del objetivo exigiría revisar el contrato y la ruta de series de tiempo. Completar este bloque no acredita cumplimiento global de E1. |

## Resultados y verificación

Las 140 correlaciones usan una cohorte común de **308 fechas de 2023**; las correlaciones móviles de 60 días requieren 50 pares y definen 309 ventanas de 2023 y 366 de 2024 por covariable. Son análisis retrospectivos; los máximos observados no seleccionan un rezago o una ventana óptima.

Las referencias Ridge usan α=1 fijo, estandarización con media y desviación poblacional de las muestras de ajuste e intercepto sin penalizar. Se minimiza SSE + α‖β‖², sin dividir SSE por N. La solución se calcula con SciPy/NumPy; no se instala ni se afirma usar sklearn. El objetivo no se transforma y las predicciones no se recortan.

| D | Muestras de ajuste 2023 | Último resultado usado | Residuos 2024 | ACF(1) |
|---|---|---|---|---|
| 7 | 343 | 2023-12-24 | 366 | 0.766480 |
| 14 | 322 | 2023-12-17 | 366 | 0.732170 |
| 21 | 301 | 2023-12-10 | 366 | 0.762465 |
| 28 | 280 | 2023-12-03 | 366 | 0.762320 |

Los cuatro ajustes permanecen fijos durante 2024. Sus índices coinciden con los hashes del manifiesto CV para el primer origen y respetan u≤t−D−1 en todos los orígenes. Los escaladores solo usan las muestras permitidas de 2023. Las predicciones se verificaron mediante una formulación independiente de mínimos cuadrados aumentados: diferencia máxima **2,46×10⁻¹⁴**. No se observaron predicciones negativas, aunque el procedimiento las conservaría.

Se conservan **1464 predicciones/residuos**, **112 valores ACF** y **12 Ljung–Box** a 7/14/28 días con `model_df=0`. La ACF presenta bandas puntuales aproximadas de ruido blanco; no son bandas simultáneas ni robustas a cualquier dependencia. Los doce valores de referencia asintótica quedan por debajo de 0,05 tras Holm; los ajustados oscilan entre **3,85×10⁻²¹¹ y 2,92×10⁻⁷⁷** y se conservan también en logaritmos. Esto no es una prueba confirmatoria ni demuestra que deban añadirse rezagos del objetivo.

Los objetivos consecutivos comparten seis retornos. El solapamiento, los límites de una referencia lineal y la deriva pueden generar dependencia residual. Holm corrige multiplicidad bajo una referencia válida; no corrige su calibración ni convierte las observaciones en independientes. La clasificación continúa determinada por las entradas admitidas.

## Evidencia trazable

- Descriptivos y disponibilidad: `scripts/eda_temporal_specific.py`, `results/eda_temporal_specific/summary.json` y CSV acompañantes.
- Referencia y residuos: `scripts/eda_temporal_reference.py`, `results/eda_temporal_reference/summary.json`, parámetros, muestras, predicciones, ACF y Ljung–Box/Holm.
- Fuentes de mercado, etiquetas, correlaciones anteriores y manifiesto CV: rutas y hashes en el JSON de esta revisión.
- Figuras y artefactos consultables en `book/_static/eda_temporal_specific/` y `book/_static/eda_temporal_reference/`.
- Contadores de este bloque: **4 referencias Ridge, 0 redes DL, 0 piloto, 0 comparaciones de benchmark, 0 archivos de test y 0 ZIP abiertos**. El módulo descriptivo y el EDA general conservan por separado sus contadores de 0 modelos.

Las fuentes y el calendario permanecen intactos. Estas referencias no fijan la política de reajuste o el diseño inferencial de E2, no seleccionan arquitectura, D o α y no sustituyen los tres baselines DL. Continúan pendientes la cobertura bibliográfica obligatoria, la selección sustentada de modelos, la evidencia externa de faltantes/ruido y los productos experimentales reservados por el usuario.
