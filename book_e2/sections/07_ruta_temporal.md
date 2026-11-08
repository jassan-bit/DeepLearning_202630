(e2-ruta-temporal)=
# 7. Ruta metodológica: series de tiempo y referencias causales

**Decisión del 06/11/2026:** se adopta la ruta **F2a, series de tiempo**, tras recibir la guía completa de 65 páginas. El diagnóstico de desarrollo activa la regla RT1 de la página 32: la autocorrelación residual fuerte de una referencia requiere reclasificar el problema. Esta decisión cierra la clasificación metodológica; el benchmark Bitcoin continúa pendiente. La resolución no modifica E1 ni convierte la reserva conocida en una prueba confirmatoria.

El [registro de decisiones vigentes del EDA](../_static/eda_traceability/decisiones_vigentes.csv) enlaza esta resolución con la fila histórica `RESID`, conservada sin cambios. La fecha de decisión es el 06/11/2026; su incorporación al registro complementario el 08/11/2026 es una aclaración documental, no una nueva reclasificación ni un resultado experimental.

## 7.1 Motivo de la reclasificación y contrato que se conserva

La {ref}`referencia Ridge forward <e2-diagnostico-forward>` dejó ACF(1)=0,632228 en 305 fechas de validación de 2024. Ljung–Box de lag7 dio Q=266,4062 y p≈8,88×10⁻⁵⁴, con sus limitaciones asintóticas y de dependencia. No se clasifica la tarea solo por excluir Y rezagada de las entradas. La regla específica RT1 (PDF p32) y el diagnóstico del EDA C (p6) determinan seguir F2a; la instrucción general de p40 no rebaja ese requisito. La guía exige resolver la ruta antes de implementar el benchmark (p20).

Se conservan **D7×5 OHLCV → σ₇, horizonte h=1**, escala, primaria RMSE media por trayectoria, **δ=0,00025772842664981106**, unidad y partición heredadas. Las tres recetas de E1 continúan como anclas fijas. Reclasificar la ruta no añade automáticamente Y rezagada como sexto canal de las redes ni cambia su historia. Una ampliación de información futura necesitaría su propia decisión y comparación de desarrollo, sin sustituir la referencia heredada.

La dependencia de etiquetas σ₇ consecutivas —seis retornos compartidos— contribuye a la autocorrelación. El diagnóstico no demuestra que un canal nuevo mejoraría fuera de muestra, ni prueba no linealidad. La decisión sí exige las referencias y controles propios de F2a, con soportes explícitos.

## 7.2 Referencias endógenas y panel ampliado

F2a S5 (PDF p30) requiere naive, naive estacional y **ARIMA o ETS**. Se incorporan **persistencia, referencia semanal y ARIMA**; ETS queda como alternativa posible, sin ser necesario añadir ambos para esa disyunción. El panel actualizado tiene veinte configuraciones: diez Top10, tres variantes GRU/LSTM/TCN ajustadas en E2, las tres anclas fijas de E1, Ridge y estas tres referencias endógenas. Las veinte filas representan modelos/configuraciones, no veinte observaciones independientes ni veinte resultados ya obtenidos.

| Referencia | Pronóstico emitido al comenzar t | Información efectiva | Alcance respecto al origen |
|---|---|---|---|
| Naive/persistencia | Ŷₜ=σ₇(t−1) | Ocho cierres de t−8 a t−1 | Ocho días hacia atrás |
| Naive semanal | Ŷₜ=σ₇(t−7) | Ocho cierres de t−14 a t−7 | Catorce días hacia atrás |
| ARIMA | E[Yₜ\|información observada hasta t−1] bajo el modelo ajustado | Historia de etiquetas hasta t−1; parámetros aprendidos solo hasta el corte registrado | Historia expansiva declarada, superior a D7 |

El alcance de catorce días de la referencia semanal **no significa catorce retornos ni catorce cierres usados para calcular esa σ₇**: su etiqueta requiere ocho cierres, y los seis días posteriores no entran en ese valor. La semana es un comparador de calendario registrado, no una conclusión de estacionalidad significativa obtenida del STL.

Estas referencias utilizan información endógena y longitudes distintas de D7×5. Se comparan sus errores sobre las mismas fechas elegibles, pero no se atribuye toda diferencia a arquitectura bajo información idéntica. El soporte de la referencia semanal y la historia ARIMA se incorporarán a la auditoría del panel común antes de ejecutar; no se comprime el calendario ni se decide la cohorte según los errores observados.

## 7.3 Implementación de las referencias y política de reajuste

El módulo `scripts/e2_temporal_baselines.py` recibe etiquetas diarias σ₇ ya verificadas y fechas UTC; no descarga datos, no abre archivos de mercado y no escoge un ganador. `naive_forecast` accede exclusivamente a la etiqueta de lag1 o lag7 de calendario. Si falta, devuelve una observación inelegible, sin sustituirla por la fila anterior disponible.

`FixedParameterARIMA` usa statsmodels: la verificación sintética inicial empleó **0.14.6** (`requirements-eda.lock.txt`); el entorno nuevo de esta etapa fija **0.15.0** (`requirements-e2-remediation.lock.txt`). Se registra la versión por ejecución y no se confunden esos entornos. Ajusta una vez por origen externo registrado: para el comienzo t₀ del fold, las etiquetas usadas para estimar parámetros y escala terminan como máximo en t₀−8, manteniendo el embargo de siete días. La escala de Y es su media solo de entrenamiento; una serie totalmente nula usa escala1. El estado se actualiza con observaciones efectivamente conocidas hasta t−1 y **`refit=False`**; ese filtrado no reestima parámetros. Un objeto nuevo corresponde a un nuevo fold. Los orígenes deben crecer estrictamente y no se admiten pronósticos retrospectivos sobre un estado que ya vio fechas posteriores.

La observación más reciente disponible cambia con cada día; no se introducen covariables futuras ni Yₜ. Los huecos se mantienen como `NaN` en su día dentro del modelo de estado: la transición del estado no crea una etiqueta observada imputada. Si falta Yₜ₋₁, el módulo marca el origen como inelegible para el panel de información observada. Las políticas de actualización y reajuste deben figurar por familia en el protocolo común; no se concede un refit diario oculto a ARIMA.

La cuadrícula local contiene doce órdenes: p∈{0,1,2}, d∈{0,1}, q∈{0,1}; usa constante con d=0 y sin tendencia determinista con d=1, restricciones de estacionariedad/invertibilidad y máximo100 iteraciones. Se registra la no convergencia como fallo; no se sustituye silenciosamente por otro orden ni se usa test para escogerlo. El protocolo común `experiments/e2_remediation_20261106/hpo.json` asigna **T=12 a cada una de quince familias ajustables y 172.800 CPU-s por familia**. La cuadrícula local y esa asignación no son una búsqueda ejecutada, un orden ganador ni prueba de consumo equivalente medido.

El accessor `FixedParameterARIMA.predict` conserva el pronóstico original para auditoría. Antes de observar resultados del benchmark, `architecture_decisions.json` fija para ARIMA y Ridge el enlace puntual **max(0, pronóstico original)**. El wrapper explícito `benchmark_nonnegative` aplica esa misma regla, conserva `raw_value`, motivo y elegibilidad, y rechaza valores no finitos: un fallo no se convierte en cero ni se elimina una fecha por una predicción negativa. Las observaciones inelegibles mantienen valor puntuable `None`. El panel del benchmark consume el valor posprocesado; el valor original sigue disponible para auditar el efecto de la regla.

F2a (p31) exige declarar la política común, no impone un refit diario en todos los casos. Se conserva el calendario y el embargo heredados; h=1 no es razón para reducir unilateralmente el embargo D7. El análisis del test histórico seguirá siendo exploratorio.

## 7.4 EDA temporal heredado: evidencia reutilizada y cobertura

El EDA de E1 se amplía, como exige la guía; no se desecha ni se repite automáticamente. Los siguientes resultados ya existen en el [capítulo EDA de E1](https://jassan-bit.github.io/DeepLearning_202630/sections/02_eda.html), y se separan de los diagnósticos futuros del Top10 Bitcoin.

| Requisito F2a / EDA B | Evidencia ejecutada de E1 | Uso y límite para E2 |
|---|---|---|
| Tendencia y estacionalidad | STL robusto de log-cierres diarios, 25/03–31/12/2023, periodo7; E1 §2.5.1 | Apoya representación causal; periodo impuesto no prueba estacionalidad ni habilita STL centrado como predictor |
| Estacionariedad | ADF con constante/7 rezagos y KPSS constante/rezagos automáticos; σ₇ de 01/04–31/12/2023, n275, pADF≈0,002423 y pKPSS>0,10; E1 §2.5.2 | Compatibilidad de ese tramo con estacionariedad, sin garantía para 2024/2026; no diferencia automáticamente Y |
| Dependencia temporal | ACF/PACF y periodograma de retornos diarios, 26/03–31/12/2023, n281; E1 §2.5.3 | Distingue unidades y retornos frente a σ₇; no selecciona D o temporada por máximo visual |
| Heterocedasticidad | ARCH-LM sobre retornos diarios centrados, siete rezagos, p≈0,341; E1 §2.5.3 | No rechazo exploratorio no prueba homocedasticidad ni sustituye el ARCH-LM de residuos de nuevas redes |
| IID residual | BDS y Ljung–Box de GRU/LSTM/TCN de E1 por fold y semilla; E1 §2.5.6 | BDS contrasta iid; no atribuir todo rechazo a no linealidad ni trasladar el resultado al Top10 |
| Anomalías y cambios | Máscaras de continuidad, extremos sin borrado; candidato CUSUM 15/10/2023 sin confirmación; E1 §2.5.1/2.5.5 | Preservar eventos legítimos y revisar sensibilidad; el candidato no prueba una ruptura causal |
| Dificultad de pronóstico | Persistencia forward de desarrollo: RMSE0,00474310, MAE0,00292428; E2 §2.2.3 | Referencia informal de predictibilidad; no es ruido irreducible ni techo validado |

Los CSV heredados `results/eda_development/stationarity_daily.csv`, `stationarity_volatility.csv` y `summary.json` contienen especificaciones, rangos y advertencias; sus figuras están en `book/_static/eda/`. E1 ya permite reutilizar STL, ADF/KPSS, PACF y ARCH con esos límites. No se exige volver a calcularlos solo porque cambie el nombre de la ruta. El {ref}`diagnóstico forward de E2 <e2-diagnostico-forward>` amplía esta evidencia con Ridge/persistencia sobre 2024.

**Decisión de preprocesamiento:** mantener escalado causal, máscaras y calendario, sin diferenciación automática ni imputación de observaciones. **Impacto en arquitectura y validación:** adoptar F2a, incorporar sus referencias y validar por origen con políticas registradas; realizar diagnósticos de residuos de los modelos efectivamente ejecutados. **Riesgos:** heredar conclusiones de otra frecuencia/tramo, confundir un contraste iid con prueba exclusiva de no linealidad y tratar el error naive como techo de Bayes. La referencia condicional de predictibilidad y γ de §6 ya dispone de código específico, pero requiere ejecución; su ley empírica de innovación train-only y supuestos iid/transferencia no constituyen un techo de Bayes universal.

## 7.5 Verificación y estado

Se verificó el módulo con doce pruebas **sintéticas**, usando el entorno de `requirements-eda.lock.txt`:

```bash
python -m unittest discover -s tests -p test_e2_temporal_baselines.py -v
```

Las pruebas cubren lag de calendario frente a filas retenidas, alcance8/14, exclusión de valores futuros, embargo, huecos conservados, ausencia de refit, orden temporal de llamadas y fallos declarados. Verifican también que el enlace registrado conserva el negativo original, no cambia la elegibilidad y rechaza `NaN`/infinito. Los pequeños ajustes ARIMA de estas pruebas son sintéticos: **no son corridas del benchmark Bitcoin**, no usan sus precios ni aportan resultados al ranking.

La decisión reproducible está en `experiments/e2_remediation_20261106/route.json`. Estado: **ruta cerrada, referencias implementadas y verificadas sintéticamente; presupuesto HPO asignado por el protocolo común, sin ejecuciones Bitcoin acreditadas por esta sección**. El registro previo y el piloto de desarrollo se documentan en la [sección 9](09_protocolo_registrado.md); HPO y dimensionamiento final siguen pendientes. El cierre de esta sección no permite afirmar superioridad o cumplimiento de todos los requisitos de F2a.
