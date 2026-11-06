# Decisiones, revisión y pendientes de E1

El estudio compara **GRU–LSTM–TCN** mediante índices forward chaining de Filipe, con n=7, h=1 y covariables OHLCV diarias. El libro conserva **dos fases**, sin cambiar retrospectivamente la cronología o los resultados de ninguna.

## Decisiones incorporadas

| Elemento | Fase I exploratoria | Fase II D7 planificada |
|---|---|---|
| Entradas | D=7/14/21/28 | D=7, seleccionado previamente |
| CV | Doce bloques mensuales de 2024 | Diez bloques enero–octubre de 2024 para HPO |
| Recetas | Anchos GRU/LSTM 32, TCN 16; valores fijos | Ocho candidatos por familia; seleccionados GRU/LSTM 16 y TCN 8 |
| Esfuerzo HPO | Sin búsqueda nueva comparable | Dos semillas y el mismo método/presupuesto: 320 ajustes por familia, 960 en total |
| Piloto | No precedió al benchmark | Diez semillas nuevas por familia; treinta trayectorias, sesenta ajustes y 60 días de evaluación |
| Repeticiones definitivas | Diez por familia e historia, presupuesto fijo | R=465 por familia, calculado antes del ajuste definitivo |
| Baseline por desarrollo | **GRU/D7 de 32 unidades**, 3.777 parámetros | **GRU/D7 de 16 unidades**, 1.121 parámetros; candidato c03 |
| Test / reserva | 01/01/2025–31/08/2026, 608 fechas ya observadas | 01/09–30/10/2026, 60 fechas no utilizadas anteriormente para resultados |
| Inferencia | Posterior y exploratoria | Comparaciones, margen y métodos registrados antes de la reserva |
| Potencia | No se acredita 0,80 por tener diez semillas | Planificación marginal por semillas con objetivo 0,80 y supuestos explícitos |
| Otros mercados | No evaluados | No evaluados; dos meses del mismo BTCUSDT no son otro mercado |

El contrato conserva σ₇ poblacional, `ddof=0`, sin anualizar, entrada anterior al día objetivo y ausencia de rezagos explícitos de la volatilidad. Las transformaciones se ajustan solo a entrenamiento permitido; no se imputan días incompletos ni se comprime el calendario. Los pesos quedan fijos dentro del periodo de evaluación correspondiente.

Los porcentajes **27,26/27,33/45,41** siguen describiendo los 365/366/608 días del calendario de fase I. La nueva reserva de 60 días se asigna mediante un protocolo adicional, sin recalcular esos porcentajes como si fueran proporciones nativas del método de Filipe.

## Orden y alcance del nuevo estudio

El [diseño y código de fase II](https://github.com/jassan-bit/DeepLearning_202630/commit/ac7dd7f67c0dec2397b519897cf6da8056b238f5) se publicaron antes del HPO. Se congelaron las recetas antes del piloto. El [dimensionamiento](https://github.com/jassan-bit/DeepLearning_202630/commit/5a0acfbc2ff582e252f0e6fb18aaf652963221f8) quedó publicado antes de sus ajustes definitivos. La cronología de cierre de pesos, acceso a la reserva y análisis se conserva en el [capítulo 9](09_estudio_planificado.md) y su [registro público](../_static/gru_lstm_tcn_planned_20261105/chronology.json).

La adquisición previa había recorrido los CSV para auditar integridad, timestamps y esquema. Según los registros conservados, la nueva reserva no se había usado para calcular objetivos o evaluar modelos. Es una evaluación preespecificada sobre una reserva histórica, no una afirmación de ZIP nunca abiertos, predicciones en vivo o confirmación en otros mercados.

La potencia se dimensiona con t no central, α*=0,05/3 y límite superior simultáneo del 80 % de las DE pareadas del piloto. **R=465** satisface el objetivo marginal bajo normalidad y transferencia de dispersión. No acredita potencia conjunta del 80 %, potencia temporal de 60 días ni la dispersión de un futuro modelo avanzado. Los análisis temporales secundarios tienen sus propios supuestos e intervalos.

## Revisión de literatura

El **corte histórico del 05/11/2026** conserva **40 referencias de antecedentes y siete estudios cripto con texto completo verificado**. Los siete incluyen BTC y uno incluye otras diecinueve criptomonedas: los subconjuntos se solapan. Las frecuencias de comparadores independientes de ese corte son **GRU 3/7, LSTM 5/7 y TCN 1/7**. El denominador no significa siete artículos de revistas indexadas.

Akbiyik et al. (2023) compara las tres arquitecturas para volatilidad de Bitcoin y Miura et al. (2019) añade respaldo publicado directo para GRU/LSTM {cite:p}`akbiyik2023,miura2019`. Los [niveles de acceso y procedencia](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv) y la [matriz](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.json) permanecen separados de las decisiones del experimento nuevo.

El **06/11/2026** se incorporan **cuatro exportaciones nativas**: dos de Scopus, con 19 y 158 registros, y dos de WoS, con 8 y 59. Las **244 ocurrencias** se agrupan provisionalmente en **182 grupos por DOI/título**. El {download}`registro de importación <../_static/literature_e1/native_search_20261106/README.md>`, el [manifiesto de fuentes](../_static/literature_e1/native_search_20261106/source_manifest.json), los [registros](../_static/literature_e1/native_search_20261106/records.csv) y el [cribado](../_static/literature_e1/native_search_20261106/screening.csv) dejan trazabilidad de esta ampliación. El recuento de grupos no equivale al de estudios elegibles: hay que comprobar duplicados ambiguos, objetivos, papel de cada arquitectura y calidad del diseño en el original.

WoS y Scopus cuentan ahora con exportaciones nativas para E1. El autor confirmó las familias TCN/GRU–LSTM y el filtro 2019–2026; las cadenas exactas, totales y rangos no se reconstruyen automáticamente. La trazabilidad IEEE permanece pendiente. Los registros de precio, retorno, dirección o índices de volatilidad se distinguen del objetivo σ₇; los de año nominal 2027 requieren verificar Early Access o fecha efectiva anterior al corte. La ampliación no se declara una revisión cerrada ni sustituye el denominador histórico antes de completar la lectura y documentar sus decisiones. Sus conclusiones deben determinar el respaldo de las familias, sin forzar los nuevos estudios a confirmar la terna o el resultado del benchmark.

Se completó la lectura de **dos originales adicionales**, con {download}`fichas y matriz de evidencia <../_static/literature_e1/primary_reviews_20261106/README.md>`: Nguyen et al. (2023), TCN/LSTM para CVI de volatilidad implícita, y Zahid et al. (2022), redes como componentes híbridos GARCH {cite:p}`nguyen2023cvi,zahid2022hybrid`. La bibliografía actual pasa de 40 a **42 referencias**; los nuevos textos conservan objetivos y roles separados del corte N=7. PSO–GRU y Direct Neuro conservan evidencia de resumen; CryptoMamba tiene secciones primarias parciales. Ninguno se eleva a texto completo o comparador integralmente verificado.

## Matriz de cierre derivada de la auditoría

“Ejecutado” o “registrado” no equivale a superioridad demostrada. Los requisitos y límites se valoran por la evidencia real, incluso cuando un contraste o resultado sea inconcluso.

| Prioridad | Requisito | Estado por fase | Evidencia y límite |
|---|---|---|---|
| **A01 · Alta** | Extraer tres modelos base de literatura aplicada | **Parcialmente resuelto; ampliación nativa en cribado y dos lecturas nuevas** | Corte 05/11: GRU–LSTM–TCN con comparadores independientes. El 06/11 añade cuatro exportaciones WoS/Scopus y originales de CVI/híbridos GARCH en estratos separados; falta verificar los restantes originales, completar trazabilidad IEEE y valorar recurrencia y mecanismos. Los resúmenes y el experimento no aumentan las frecuencias históricas. |
| **A02 · Alta** | Margen y planificación | **Registrados antes de fase II; posteriores a fase I** | δ=0,00025772842664981106, ancla académica del 5 % del GRU CV inicial. Se distingue diferencia frente a cero, relevancia superior al 5 % y utilidad financiera. |
| **A03 · Alta** | Comparación formal | **Dos alcances explícitos** | Fase I exploratoria; fase II D7 con tres pares primarios, Holm-3, IC y efectos preespecificados. Análisis temporal secundario y resultados en [§9.6](09_estudio_planificado.md). |
| **A04 · Alta** | Piloto y tamaño de muestra | **Resuelto para la nueva comparación D7** | Diez semillas piloto por familia y R=465 calculado antes del ajuste definitivo, con incertidumbre de DE y multiplicidad. No se atribuye a D14/21/28, a fase I o a un futuro avanzado. |
| **A05 · Alta** | Código, configuración y reproducción | **Separados por versión y ejecución** | Registros y hashes previos, scripts y salidas independientes por fase. La reproducción no crea otro mercado o una evaluación desconocida. |
| **A06 · Alta** | Equidad del ajuste de hiperparámetros | **Resuelta en el espacio registrado de fase II D7** | Ocho candidatos, dos semillas y diez bloques comunes por familia; todos los resultados se conservan. No se afirma óptimo global o coste idéntico de las arquitecturas. |

## Mejoras concretas

- La búsqueda nueva comparte amplitud, evaluaciones, semillas y cortes entre las tres familias. Las recetas seleccionadas tienen 1.121/1.489/1.185 parámetros, distintos de los 3.777/5.025/4.289 de fase I.
- El piloto precede a los ajustes definitivos y no se cuenta como repetición definitiva. Las semillas HPO 6101–6102, piloto 7101–7110 y definitivas 8101–8565 no se solapan.
- El dimensionamiento conserva DE puntual, cota simultánea del 80 % y sensibilidades del 90/95 %. No se escoge el R menor después de examinar la reserva ni se detiene por significación.
- La nueva reserva se asigna explícitamente antes de construir sus objetivos. Todos los pesos y escaladores se cierran antes de puntuar; los análisis usan después las predicciones guardadas.
- Los resultados y firmas de fase I se conservan. Su R² 0,8542 pertenece al GRU de 32 unidades y a un test de 608 días; no se atribuye a la nueva receta de 16 unidades o al nuevo periodo de 60 días.
- Persistencia permanece visible en ambas evaluaciones. Una mejora neuronal en RMSE no implica mejora en todas las métricas.

## Resultado de la evaluación nueva

La reserva conserva resultados adversos e inconclusos: persistencia obtiene la menor RMSE puntual y TCN la menor entre redes; GRU sigue siendo el baseline registrado por desarrollo. Las tres t pareadas rechazan tras Holm bajo su referencia asintótica, pero Shapiro–Wilk rechaza normalidad de las diferencias. Ninguna de las seis comparaciones DM rechaza tras Holm y el bootstrap temporal familiar no acredita una ganancia superior al 5 %. No se reelige receta, semilla, prueba o baseline según esos resultados. Véanse las tablas e interpretaciones completas en [§9.6](09_estudio_planificado.md).

## Límites que permanecen abiertos

- **Literatura:** completar el cribado y lectura de originales de la ampliación nativa, documentar objetivos y comparadores autónomos, resolver duplicados ambiguos y fechas efectivas, y completar trazabilidad IEEE. Valorar el respaldo de TCN y la selección conjunta de GRU/LSTM con esos resultados, sin inferir Q1 o indexación individual por editorial ni confundir grupos provisionales con estudios incluidos.
- **Temporalidad:** 60 días y un mercado limitan las conclusiones; la potencia entre semillas no sustituye el tamaño efectivo temporal o la incertidumbre de otros regímenes.
- **Normalidad y transferencia del piloto:** las diferencias definitivas rechazan normalidad; la referencia t y sus IC son aproximaciones a R=465. La dispersión y la distribución de 2024 pueden cambiar en 2026. El límite superior de DE y las sensibilidades hacen explícito ese supuesto, sin garantizarlo.
- **Fase I:** su carácter exploratorio y ausencia de piloto previo permanecen como hechos históricos; no se reparan retrospectivamente.
- **Otras entradas:** D14/21/28 son resultados exploratorios iniciales. El piloto y HPO nuevos solo dimensionan D7.
- **Equivalencia y utilidad:** no se afirman. TOST/ROPE requieren otro planteamiento si se quisiera sostener equivalencia; un margen académico no demuestra utilidad financiera.

## Proyección a E2

E2 deberá fijar explícitamente una **versión inmutable del baseline**, incluyendo receta, semilla, selección y protocolo: GRU/D7 de 32 unidades de fase I y GRU/D7-c03 de 16 unidades de fase II no son el mismo modelo entrenado. La propuesta avanzada requiere su propio piloto o dispersión pareada, planificación y datos que no hayan sido utilizados para ajustarla o escogerla. Los R=465 y el nuevo periodo ahora evaluado no se transfieren automáticamente a ese estudio.

No se entrenó aquí un modelo avanzado ni se inventan requisitos de documentos completos de E2/E3 que no se han aportado.

La DE definitiva de GRU–TCN supera aproximadamente 1,64 veces su cota de planificación obtenida con el piloto. Junto con el rechazo de normalidad, esto limita la transferencia de los supuestos: el R se calculó antes de la ejecución, pero no se afirma potencia alcanzada o demostrada.
