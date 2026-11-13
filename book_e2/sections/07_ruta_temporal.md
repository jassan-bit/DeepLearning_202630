(e2-ruta-temporal)=
# 7. Ruta metodológica: series de tiempo y controles causales

**Estado vigente al 13/11/2026:** el benchmark reducido está completado en [§6](06_benchmark_final.md), con el calendario efectivamente ejecutado de [§48](48_protocolo_reducido.md). La auditoría posterior de ajuste, residuos, calibración y estabilidad temporal está en [§49](49_diagnosticos_f2a.md); la batería comparativa S8 sigue parcial. Los estados y calendarios del 11–12/11 conservados debajo son antecedentes, no el estado actual de ejecución.

**Vigencia documental (12/11/2026):** la selección HPO/NAS ejecutada parcialmente corresponde al plan histórico de catorce redes. El catálogo obligatorio vigente es de **16 sistemas**, con las identidades de [§43](43_catalogo_obligatorio.md), las arquitecturas de [§44](44_arquitecturas_publicadas.md) y el protocolo registrado de [§45](45_enmienda_protocolo.md). Esos registros históricos no cierran el benchmark vigente ni acreditan las cuatro recetas restauradas. R y el cierre inferencial siguen pendientes. [Corte verificable de resultados (§41)](41_avance_optimizacion_11_origenes.md).

**Calendario registrado al 12/11/2026:** se conserva F2a. [§45](45_enmienda_protocolo.md) registra tres anclas HPO y diez orígenes mensuales de evaluación con reajuste; su ejecución vigente sigue pendiente. El [plan de once búsquedas de §40](40_calendario_original.md) es histórico. Se conservan embargo, disponibilidad y carácter conocido del test.



**Guía original cotejada:** [correcciones y pendientes por página](20_contraste_guia.md). La cuadrícula reducida no cubre el espacio mínimo; controles F2a, NAS condicional y batería estadística requieren completar el plan antes de entrenar.


La ruta vigente es **F2a, series de tiempo**, decidida el 06/11/2026 tras recibir la guía completa de 65 páginas. El [catálogo obligatorio de 16 sistemas](43_catalogo_obligatorio.md) conserva esa clasificación, el objetivo y las precauciones causales. La enmienda del protocolo está registrada en §45; registrarla no acredita ejecución. El híbrido adicional y Ridge son opcionales/históricos, sin superioridad demostrada.

El [registro de decisiones del EDA](../_static/eda_traceability/decisiones_vigentes.csv) conserva la procedencia de la reclasificación y la fila histórica `RESID`. Su incorporación documental el 08/11/2026 no es una decisión experimental retroactiva.

## 7.1 Motivo de la reclasificación y contrato que se conserva

La {ref}`referencia Ridge forward <e2-diagnostico-forward>` dejó ACF(1)=0,632228 en 305 fechas de validación de 2024. Ljung–Box de lag7 dio Q=266,4062 y p≈8,88×10⁻⁵⁴, con sus limitaciones asintóticas y de dependencia. Es evidencia histórica del EDA, no un modelo adicional del catálogo vigente. La regla RT1 (PDF p32) y el diagnóstico del EDA C (p6) determinan seguir F2a; excluir Y rezagada de las entradas no convierte por sí solo la tarea en regresión iid.

Se conservan **D7×5 OHLCV → σ₇, horizonte h=1**, escala, primaria RMSE media por trayectoria y **δ=0,00025772842664981106**. Los periodos ya evaluados mantienen su condición de conocidos. Las referencias obligatorias de E1 son MLP16, LSTM32 y GRU32, adicionales a los diez modelos; Persistence, SeasonalNaive y ARIMA completan los 16 sistemas.

La dependencia entre etiquetas σ₇ consecutivas, con seis retornos compartidos, contribuye a la autocorrelación. Ese diagnóstico no demuestra que un canal nuevo mejore fuera de muestra ni prueba no linealidad. Reclasificar la ruta tampoco añade automáticamente Y rezagada como sexto canal de las diez redes.

## 7.2 Información endógena y alcance de los controles

Las exigencias de controles temporales se vinculan al protocolo registrado de §45; las verificaciones no ejecutadas siguen pendientes. Su papel de referencia o ablación no amplía el catálogo de modelos seleccionado. Los controles y resultados del diseño sustituido quedan en el [archivo del protocolo](../_static/remediation_protocol/preregistration.json), sin trasladar su lista o presupuesto a este diseño.

El objetivo σ₇ necesita ocho cierres consecutivos para siete retornos. Usar una etiqueta histórica exige auditar todos esos cierres, no solo la fecha nominal de la etiqueta. Una referencia con mayor historia puede ser causal y, aun así, disponer de información distinta de las redes D7×5. Las comparaciones deben informar esa diferencia y usar una cohorte elegible común sin comprimir huecos.

**Antecedente opcional: el candidato 11 usa el mismo soporte D7×5 de E1.** La CNN opera causalmente, la LSTM es unidireccional y la atención pondera sólo los siete estados observados. Sus ablaciones propuestas aíslan convolución, memoria y atención; no añaden candidatos al catálogo ni acreditan mejora. [Contrato vigente](17_contrato_sincronizado.md).

## 7.3 Política causal de ajuste y actualización

La nueva enmienda deberá distinguir selección de hiperparámetros, ajuste de pesos, escalado, actualización de estado y scoring. Cada operación debe identificar las fechas de datos que puede usar. El horizonte h=1 no autoriza reducir unilateralmente el embargo heredado ni ajustar escaladores con validación o test.

Un predictor mensual conserva su receta, pesos y escalador durante el fold, salvo que se registre otra política causal antes de comparar. La incorporación de una observación ya disponible al estado de un modelo no equivale a reestimar sus parámetros. Las fechas faltantes permanecen en el calendario y la elegibilidad se decide por soporte, no por el error obtenido.

Las restricciones de salida, la gestión de fallos y cualquier truncamiento de pronósticos deben quedar predefinidos. Una predicción no finita se registra como fallo; no se sustituye por cero ni se elimina una fecha de forma favorable a un modelo. Para el híbrido debe verificarse que convoluciones, estados recurrentes y atención sólo usen las siete posiciones observadas, sin etiquetas ni estados del futuro.

## 7.4 EDA temporal heredado: evidencia reutilizada y cobertura

El EDA de E1 se amplía, como exige la guía; no se desecha ni se repite automáticamente. Los siguientes resultados ya existen en el [capítulo EDA de E1](https://jassan-bit.github.io/DeepLearning_202630/sections/02_eda.html), y se separan de los diagnósticos pendientes del catálogo vigente.

| Requisito F2a / EDA B | Evidencia ejecutada de E1 | Uso y límite para E2 |
|---|---|---|
| Tendencia y estacionalidad | STL robusto de log-cierres diarios, 25/03–31/12/2023, periodo7; E1 §2.5.1 | Apoya representación causal; periodo impuesto no prueba estacionalidad ni habilita STL centrado como predictor |
| Estacionariedad | ADF con constante/7 rezagos y KPSS constante/rezagos automáticos; σ₇ de 01/04–31/12/2023, n275, pADF≈0,002423 y pKPSS>0,10; E1 §2.5.2 | Compatibilidad de ese tramo con estacionariedad, sin garantía para 2024/2026; no diferencia automáticamente Y |
| Dependencia temporal | ACF/PACF y periodograma de retornos diarios, 26/03–31/12/2023, n281; E1 §2.5.3 | Distingue unidades y retornos frente a σ₇; no selecciona D o temporada por máximo visual |
| Heterocedasticidad | ARCH-LM sobre retornos diarios centrados, siete rezagos, p≈0,341; E1 §2.5.3 | No rechazo exploratorio no prueba homocedasticidad ni sustituye el ARCH-LM de residuos de nuevas redes |
| IID residual | BDS y Ljung–Box del estudio histórico de E1 por fold y semilla; E1 §2.5.6 | BDS contrasta iid; no atribuir todo rechazo a no linealidad ni trasladar el resultado al catálogo vigente |
| Anomalías y cambios | Máscaras de continuidad, extremos sin borrado; candidato CUSUM 15/10/2023 sin confirmación; E1 §2.5.1/2.5.5 | Preservar eventos legítimos y revisar sensibilidad; el candidato no prueba una ruptura causal |
| Dificultad de pronóstico | Persistencia forward de desarrollo: RMSE0,00474310, MAE0,00292428; E2 §2.2.3 | Referencia informal de predictibilidad; no es ruido irreducible ni techo validado |

Los CSV heredados `results/eda_development/stationarity_daily.csv`, `stationarity_volatility.csv` y `summary.json` contienen especificaciones, rangos y advertencias; sus figuras están en `book/_static/eda/`. E1 ya permite reutilizar STL, ADF/KPSS, PACF y ARCH con esos límites. No se exige volver a calcularlos solo porque cambie el nombre de la ruta. El {ref}`diagnóstico forward de E2 <e2-diagnostico-forward>` amplía esta evidencia con Ridge/persistencia sobre 2024.

**Decisión de preprocesamiento:** mantener escalado causal, máscaras y calendario, sin diferenciación automática ni imputación de observaciones. **Impacto en arquitectura y validación:** mantener F2a y validar por origen con políticas explícitas en la nueva enmienda; realizar diagnósticos de residuos de los modelos efectivamente ejecutados. **Riesgos:** heredar conclusiones de otra frecuencia/tramo, confundir un contraste iid con prueba exclusiva de no linealidad y tratar el error naive como techo de Bayes. La referencia condicional de predictibilidad de §6 requiere una definición compatible con el nuevo diseño; una ley empírica de innovaciones aprendida solo en entrenamiento y sus supuestos de transferencia no constituyen un techo de Bayes universal.

## 7.5 Verificación y estado

El módulo histórico `scripts/e2_temporal_baselines.py` y sus pruebas sintéticas se conservan sin cambios para reproducir sus controles de calendario, embargo y actualización. Su existencia no constituye la implementación del catálogo vigente ni una ejecución de Bitcoin del nuevo diseño.

Estado: **ruta temporal definida; soportes, controles e implementación del nuevo catálogo pendientes de la nueva enmienda**. El [P0 ejecutado](09_protocolo_registrado.md) mantiene su identidad histórica. No se hereda su presupuesto, no se declara un ganador y no se certifica el cierre de todos los requisitos F2a.
