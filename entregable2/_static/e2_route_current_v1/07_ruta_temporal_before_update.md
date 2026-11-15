(ruta-metodologica-series-de-tiempo-y-controles-causales)=
(e2-ruta-temporal)=
# 7. Ruta Metodológica según el Tipo de Problema

**Lectura vigente:** el protocolo ejecutado es [§48](48_protocolo_reducido.md), los resultados comunes están en [§6](06_benchmark_final.md), la reproducción registrada en [§55](55_reproduccion_ablaciones.md) y el Apéndice B actual en [§61](61_reproduccion_consolidada.md). Las cifras científicas y los recibos históricos conservan su fecha y su alcance.

**Estado:** el benchmark reducido está completado en [§6](06_benchmark_final.md), con el calendario efectivamente ejecutado de [§48](48_protocolo_reducido.md). La auditoría posterior de ajuste, residuos, calibración y estabilidad temporal está en [§49](49_diagnosticos_f2a.md); las comparaciones S8 se documentan en [§50](50_comparaciones_f2a.md), mientras robustez y otros requisitos conservan su estado en [§53](53_cierre_requisitos.md). Los estados y calendarios anteriores son antecedentes, no trabajos finales pendientes.

**Antecedente de planificación del 12/11/2026:** ese corte fue sustituido por la ejecución documentada en [§48](48_protocolo_reducido.md). Su texto exacto se conserva en el archivo enlazado al final; no describe trabajos finales pendientes.

**Calendario ejecutado:** tres folds de selección, con **253/435/527 ejemplos de entrenamiento y 83 de validación en cada fold**; ajuste final de **655 ejemplos**, calibración de **53** y evaluación heredada de **608**, con política de pesos fijos. La [auditoría del EDA vigente](../_static/e2_eda_current_v1/partition_audit.json) distingue IDs y soportes. Los diez folds mensuales del EDA inicial y los diseños de §§40/45 conservan su identidad histórica.



**Guía y aplicación:** los controles F2a actuales están en §§49–57. La NAS es condicional a cuerpos sin arquitectura definida; las seis plazas HPO de §48 no constituyen NAS. Los planes anteriores se conservan como antecedentes y no ordenan nuevas corridas.


La ruta vigente es **F2a, series de tiempo**, decidida el 06/11/2026 tras recibir la guía completa de 65 páginas. El [catálogo obligatorio de 16 sistemas](43_catalogo_obligatorio.md) conserva esa clasificación, el objetivo y las precauciones causales. El protocolo efectivamente ejecutado y sus enmiendas están en §48, y los recibos finales en §§6/55. El híbrido adicional y Ridge son opcionales/históricos, sin superioridad demostrada.

El [registro de decisiones del EDA](../_static/eda_traceability/decisiones_vigentes.csv) conserva la procedencia de la reclasificación y la fila histórica `RESID`. Su incorporación documental el 08/11/2026 no es una decisión experimental retroactiva.

## 7.1 Motivo de la reclasificación y contrato que se conserva

La {ref}`referencia Ridge forward <e2-diagnostico-forward>` dejó ACF(1)=0,632228 en 305 fechas de validación de 2024. Ljung–Box de lag7 dio Q=266,4062 y p≈8,88×10⁻⁵⁴, con sus limitaciones asintóticas y de dependencia. Es evidencia histórica del EDA, no un modelo adicional del catálogo vigente. La regla RT1 (PDF p32) y el diagnóstico del EDA C (p6) determinan seguir F2a; excluir Y rezagada de las entradas no convierte por sí solo la tarea en regresión iid.

Se conservan **D7×5 OHLCV → σ₇, horizonte h=1**, escala, primaria RMSE media por trayectoria y **δ=0,00025772842664981106**. Los periodos ya evaluados mantienen su condición de conocidos. Las referencias obligatorias de E1 son MLP16, LSTM32 y GRU32, adicionales a los diez modelos; Persistence, SeasonalNaive y ARIMA completan los 16 sistemas.

La dependencia entre etiquetas σ₇ consecutivas, con seis retornos compartidos, contribuye a la autocorrelación. Ese diagnóstico no demuestra que un canal nuevo mejore fuera de muestra ni prueba no linealidad. Reclasificar la ruta tampoco añade automáticamente Y rezagada como sexto canal de las diez redes.

## 7.2 Información endógena y alcance de los controles

Los controles temporales actuales se vinculan al protocolo ejecutado de [§48](48_protocolo_reducido.md), a la [admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json) y a los diagnósticos posteriores de [§49](49_diagnosticos_f2a.md). Persistence, SeasonalNaive y ARIMA ya aportan resultados de Bitcoin en la cohorte común. Los controles del diseño sustituido quedan en el [archivo del protocolo](../_static/remediation_protocol/preregistration.json), sin trasladar su catálogo o presupuesto al actual.

El objetivo σ₇ necesita ocho cierres consecutivos para siete retornos. Usar una etiqueta histórica exige auditar todos esos cierres, no solo la fecha nominal de la etiqueta. Una referencia con mayor historia puede ser causal y, aun así, disponer de información distinta de las redes D7×5. Las comparaciones deben informar esa diferencia y usar una cohorte elegible común sin comprimir huecos.

El último objetivo de calibración, 30/12/2024, y el primero de evaluación, 01/01/2025, comparten seis cierres y cinco retornos. Sus IDs son distintos; la historia compartida y el puente del 31/12 son admisibles al avanzar el origen con parámetros congelados. Esa dependencia debe conservarse en la incertidumbre temporal. La [auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json) no declara intersección cero entre todas las etapas.

**Antecedente opcional:** CNN–LSTM–atención conservó el mismo soporte D7×5 y controles causales propuestos. El híbrido y Ridge no amplían los 16 sistemas obligatorios ni aportan superioridad acreditada. Sus ablaciones propuestas no se confunden con las tres ablaciones entrenadas del autoencoder que sí publica [§55](55_reproduccion_ablaciones.md). [Contrato vigente](17_contrato_sincronizado.md).

## 7.3 Política causal de ajuste y actualización

El protocolo ejecutado distingue selección de hiperparámetros, ajuste de pesos, escalado, actualización de estado y scoring. Cada operación identifica las fechas que puede usar. El horizonte h=1 no elimina el embargo heredado ni permite ajustar escaladores con validación o test. Los tres folds HPO preceden a la evaluación externa; su receta global no se atribuye a pronósticos externos de enero–octubre de 2024.

El ajuste final usa etiquetas hasta el **31/10/2024**; el embargo es **01–07/11**, la calibración **08/11–30/12**, y el **31/12** aporta sólo historia causal. Los pesos, escaladores y calibradores permanecen fijos en las **608 fechas de 01/01/2025–31/08/2026**. La incorporación de una observación ya disponible al estado de ARIMA con coeficientes congelados no equivale a reestimar sus parámetros. Las fechas faltantes permanecen en el calendario y la elegibilidad se decide por soporte, no por el error obtenido.

Las restricciones de salida y la gestión de fallos conservan las reglas registradas. Una predicción no finita se registra como fallo; no se sustituye por cero ni se elimina una fecha favorablemente. La auditoría actual consulta registros guardados y no vuelve a certificar mediante ejecución los kernels de los modelos.

## 7.4 EDA temporal heredado: evidencia reutilizada y cobertura

El EDA de E1 se amplía, como exige la guía; no se desecha ni se repite automáticamente. Los siguientes resultados ya existen en el [capítulo EDA de E1](https://jassan-bit.github.io/DeepLearning_202630/sections/02_eda.html). El {ref}`bloque B del EDA actual <e2-eda-series-heredado>` explicita para cada uno la decisión de preprocesamiento, arquitectura, validación y riesgos. Se distinguen de los diagnósticos finales ya ejecutados en §49, posteriores al benchmark.

| Requisito F2a / EDA B | Evidencia ejecutada de E1 | Uso y límite para E2 |
|---|---|---|
| Tendencia y estacionalidad | STL robusto de log-cierres diarios, 25/03–31/12/2023, periodo7; E1 §2.5.1 | Apoya representación causal; periodo impuesto no prueba estacionalidad ni habilita STL centrado como predictor |
| Estacionariedad | ADF con constante/7 rezagos y KPSS constante/rezagos automáticos; σ₇ de 01/04–31/12/2023, n275, pADF≈0,002423 y pKPSS>0,10; E1 §2.5.2 | Compatibilidad de ese tramo con estacionariedad, sin garantía para 2024/2026; no diferencia automáticamente Y |
| Dependencia temporal | ACF/PACF y periodograma de retornos diarios, 26/03–31/12/2023, n281; E1 §2.5.3 | Distingue unidades y retornos frente a σ₇; no selecciona D o temporada por máximo visual |
| Heterocedasticidad | ARCH-LM sobre retornos diarios centrados, siete rezagos, p≈0,341; E1 §2.5.3 | No rechazo exploratorio no prueba homocedasticidad; ARCH-LM de las trayectorias actuales se reporta por separado en §49 |
| IID residual | BDS y Ljung–Box del estudio histórico de E1 por fold y semilla; E1 §2.5.6 | BDS contrasta iid; no atribuir todo rechazo a no linealidad ni trasladar el resultado al catálogo vigente |
| Anomalías y cambios | Máscaras de continuidad, extremos sin borrado; candidato CUSUM 15/10/2023 sin confirmación; E1 §2.5.1/2.5.5 | Preservar eventos legítimos y revisar sensibilidad; el candidato no prueba una ruptura causal |
| Dificultad de pronóstico | Persistencia forward de desarrollo: RMSE0,00474310, MAE0,00292428; E2 §2.2.3 | Referencia informal de predictibilidad; no es ruido irreducible ni techo validado |

Los CSV heredados de [estacionariedad diaria](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/stationarity_daily.csv), [estacionariedad de σ₇ y sensibilidades](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/stationarity_volatility.csv) y el [resumen de E1](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/summary.json) contienen especificaciones, rangos y advertencias; el capítulo fuente conserva sus figuras. No se exige recalcularlos por cambiar el nombre de la ruta. El {ref}`diagnóstico forward de E2 <e2-diagnostico-forward>` amplía esta evidencia con Ridge/persistencia sobre 305 fechas de 2024.

El **bloque C** permanece desarrollado en §2.2.1–2.2.3: estructura diaria y huecos, disponibilidad de cada covariable, asociaciones rezagadas sobre cohorte común, relaciones móviles y diagnóstico residual de referencia. La [trazabilidad actual](../_static/e2_eda_current_v1/decision_traceability.csv) y la [lista de cotejo](../_static/e2_eda_current_v1/guide_checklist.csv) cubren conjuntamente 2.1, B y C. El [perfil actual del objetivo](../_static/e2_eda_current_v1/target_profiles.csv) y su [ECDF](../_static/e2_eda_current_v1/target_ecdf.svg) describen la partición ejecutada sin inferencia iid. La [copia anual de covariables E1](../_static/e2_eda_current_v1/inherited_annual_profiles.csv) conserva variables del mismo día y referencia hasta el 03/12/2024; no representa entradas causales D7×5 ni se mezcla con las distancias actuales.

**Decisión de preprocesamiento:** mantener escalado causal, máscaras y calendario, sin diferenciación automática ni imputación de observaciones. **Impacto en arquitectura y validación:** mantener F2a y las operaciones causales de §48; discutir por separado los residuos finales de §49 y la predictibilidad informal del EDA. **Riesgos:** heredar conclusiones de otra frecuencia/tramo, confundir un contraste iid con prueba exclusiva de no linealidad o tratar el error naive como techo de Bayes. La entropía ordinal posterior de [§57](57_errores_techo.md) y el proxy empírico de [§63](63_ensamble_calibracion.md) conservan sus supuestos: ni identifican ruido externo de σ₇ ni cierran un techo de desempeño.

## 7.5 Verificación y estado

El módulo histórico `scripts/e2_temporal_baselines.py` y sus pruebas sintéticas se conservan sin cambios. Los resultados actuales de Persistence, SeasonalNaive y ARIMA están en §6, con su selección y congelación en §48; la evidencia sintética anterior mantiene un papel distinto de esas ejecuciones de Bitcoin.

Estado: **ruta F2a definida, benchmark de 16 sistemas completado y partición ejecutada comprobada mediante registros guardados**. La [auditoría actual](../_static/e2_eda_current_v1/partition_audit.json) distingue el soporte compartido legítimo de las etiquetas de ajuste. [§49](49_diagnosticos_f2a.md) publica diagnósticos de las 81 trayectorias finales; no se presentan como análisis anteriores al entrenamiento. El [P0](09_protocolo_registrado.md) conserva su identidad histórica. La herencia del test de E1 se acepta con exposición previa declarada; las limitaciones de ruido externo, cegamiento y otros pendientes de [§61](61_reproduccion_consolidada.md) impiden atribuir cierre integral a esta actualización.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/07_ruta_temporal.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.
