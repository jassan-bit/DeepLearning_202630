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

WoS y Scopus cuentan con estas exportaciones nativas para E1. El autor confirmó las familias TCN/GRU–LSTM y el filtro 2019–2026; esa confirmación no acredita las cadenas literales, fechas de consulta, totales o rangos de exportación. La {download}`recuperación de trazabilidad por Internet <../_static/literature_e1/search_recovery_20261106/README.md>` registra **cuatro intentos de acceso nativo bloqueados**, cero consultas nativas recuperadas y **doce consultas propuestas que no se ejecutaron**. Los totales de esas consultas siguen desconocidos (`null`), no son cero. No se acredita consulta efectiva de IEEE Xplore ni se presenta una búsqueda web como búsqueda nativa. **§3.1 permanece abierto** hasta conservar cadena, campo, colección/ediciones, filtros, fecha, total y rango de exportación de la sesión correspondiente. Las nuevas propuestas sirven para otra ronda reproducible; no reconstruyen el historial privado anterior.

Se completó la lectura de **dos originales adicionales**, con {download}`fichas y matriz de evidencia <../_static/literature_e1/primary_reviews_20261106/README.md>`: Nguyen et al. (2023), TCN/LSTM para CVI de volatilidad implícita, y Zahid et al. (2022), redes como componentes híbridos GARCH {cite:p}`nguyen2023cvi,zahid2022hybrid`. Esa primera ampliación llevó la bibliografía de 40 a **42 referencias**; sus textos conservan objetivos y roles separados del corte N=7. En ese corte, PSO–GRU y Direct Neuro quedaron a nivel de resumen y CryptoMamba-SSM con secciones primarias parciales; ese estado histórico no se presenta como lectura completa. La recuperación posterior de CryptoMamba-SSM se documenta entre los casos revisados a continuación, conservando sus límites de pertinencia.

La actualización añade {download}`decisiones editoriales por título/resumen <../_static/literature_e1/editorial_screening_20261106/README.md>` y una {download}`matriz de siete familias candidatas <../_static/literature_e1/selection_synthesis_20261106/README.md>`. El cribado posterior tiene su propio responsable, evidencia y motivos; no se reescribe la primera pasada automática. Las exclusiones de ese nivel y los candidatos pendientes de original no constituyen un corpus final nuevo. La selección GRU–LSTM–TCN se explica como elección operativa condicionada, con objeciones y alternativas explícitas, sin equipararla a las tres familias más frecuentes de una revisión sistemática cerrada.

El **primer corte editorial** de los 182 grupos registró 145 exclusiones del núcleo, 11 contextuales, **22 pendientes de original** y cuatro fichas completas ya documentadas. Se conserva como fotografía histórica: esas cuatro fichas corresponden a S1/S8 del corpus histórico y CVI/GARCH–DL de la ampliación, no a cuatro estudios nuevos.

El corte de la {download}`revisión posterior de los 22 casos <../_static/literature_e1/internet_closure_20261106/README.md>` registró **cuatro originales completos leídos** (E1N053/075/118/163) y **una transcripción completa** (E1N091), distinguida del PDF editorial. Entonces se cerraron **cinco decisiones estrictas de exclusión/contexto** (E1N053/075/091/163/170) y **17 casos conservaron elegibilidad pendiente**, incluido E1N118 por el suplemento. Revisar un texto no equivale a incluirlo: se separan precio, dirección, volatilidad implícita, componentes híbridos y comparadores autónomos del objetivo σ₇. Las fuentes complementarias X001–X004 quedan fuera del recuento nativo de 244 ocurrencias/182 grupos. Las decisiones y la [matriz de respaldo de modelos](../_static/literature_e1/internet_closure_20261106/model_evidence.csv) documentan el análisis de recurrencia, simplicidad y diversidad. Esta ronda no añade inclusiones estrictas ni aumenta las frecuencias verificadas; §3.2 conserva un cierre parcial.

La petición posterior del autor establece una ruta de **sustitución por texto completo gratuito**. La {download}`etapa nueva <../_static/literature_e1/open_access_replacement_20261107/README.md>` revisa los originales de E1N057/E1N138 y reevalúa E1N118. Los tres quedan como contexto por insuficiencias comprobadas de objetivo, arquitectura o protocolo; la disponibilidad del PDF no obliga a incluirlo. La cola original tiene ahora **ocho decisiones verificadas y 14 informes no recuperados al corte**. Estos últimos conservan identidad, DOI y motivo de acceso informado; no se califican como estudios malos ni como originales leídos. El [mapa por tema y función](../_static/literature_e1/open_access_replacement_20261107/replacement_map.csv) indica qué evidencia abierta cubre cada necesidad, con fuentes reutilizadas cuando corresponde y sin exigir catorce nuevos artículos. Las comparaciones o tablas no verificadas de las alternativas mantienen sus límites.

En los cinco casos de año nominal 2027, la [comprobación de fechas primarias](../_static/literature_e1/search_recovery_20261106/publication_dates.json) recupera dos fechas First Online de capítulos en 2026 y dos fechas de volúmenes relacionados. Estas últimas no prueban por sí solas la fecha efectiva del registro individual. Tres identidades conservan su fecha efectiva exacta sin resolver; no se excluyen automáticamente por el año nominal ni se infiere que el filtro original se aplicó correctamente.

## Matriz de cierre derivada de la auditoría

“Ejecutado” o “registrado” no equivale a superioridad demostrada. Los requisitos y límites se valoran por la evidencia real, incluso cuando un contraste o resultado sea inconcluso.

| Prioridad | Requisito | Estado por fase | Evidencia y límite |
|---|---|---|---|
| **A01 · Alta** | Extraer tres modelos base de literatura aplicada | **Fuentes de acceso gratuito integradas; trazabilidad nativa parcial** | La cola de 22 conserva ocho decisiones verificadas y 14 informes no recuperados al corte. §3.8 y sus fichas documentan alternativas por función, nivel de lectura y estrato. Los 182 grupos y N7 histórico no cambian. El historial nativo e IEEE siguen abiertos: cuatro accesos bloqueados y doce propuestas no ejecutadas. |
| **A02 · Alta** | Margen y planificación | **Registro conservado; justificación académica ampliada después** | δ=0,00025772842664981106 no cambia. El uso confirmado es comparación académica; la [justificación de §1.1.4](01_problema_dataset.md) sitúa el 5 % frente a escalas de σ₇, variación entre semillas y escenarios del 2,5/10 %. La argumentación es posterior, normativa y separada de detectabilidad, incertidumbre temporal y utilidad financiera. |
| **A03 · Alta** | Comparación formal | **Análisis registrado y suplemento posterior separados** | Fase I exploratoria; fase II D7 con tres pares primarios, Holm-3, IC y efectos preespecificados. Q–Q, Friedman/Kendall W y Wilcoxon/Holm posteriores desde RMSE publicados completan diagnósticos sin sustituir DM ni bootstrap temporal. Véase el [capítulo 9](09_estudio_planificado.md). |
| **A04 · Alta** | Piloto y tamaño de muestra | **Resuelto para la nueva comparación D7** | Diez semillas piloto por familia y R=465 calculado antes del ajuste definitivo, con incertidumbre de DE y multiplicidad. No se atribuye a D14/21/28, a fase I o a un futuro avanzado. |
| **A05 · Alta** | Código, configuración y reproducción | **Curvas nuevas públicas; reproducción exacta y originales pendientes** | {ref}`§9.10 <curvas-finales-reproducidas>` publica 1.395 trayectorias nuevas, 2.790 etapas y 59.210 registros de pérdidas sólo de desarrollo. Coinciden 4.185 hashes de configuración/preprocesamiento/partición, ninguno de 2.790 CSV de curvas o 1.395 checkpoints. Se conserva toda discrepancia; no se recuperan los originales ni se añade una evaluación. |
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

## Reproducción posterior de curvas del modelo final

El [protocolo publicado](https://github.com/jassan-bit/DeepLearning_202630/commit/fe41b8ad1eaa1425aad900f6c3cd966843d600c7) precedió al registro y comienzo de estos ajustes nuevos. La ejecución usa las recetas congeladas, sus 465 semillas por familia y sólo desarrollo 2023–2024. Las {ref}`curvas de entrenamiento/validación y reajuste <curvas-finales-reproducidas>` y el [resumen público](../_static/e1_curve_reproduction_20261106/summary.json) permiten observar el aprendizaje de esta reproducción posterior. No se puntúa de nuevo la reserva ni se cambian las métricas históricas, el baseline o δ.

Las épocas coinciden en 1.366/1.395 trayectorias: 465 GRU, 465 LSTM y 436 TCN; difieren 29 TCN. Ningún CSV de curvas ni checkpoint serializado coincide byte a byte con su hash histórico. **No se acredita reproducción exacta del entrenamiento**, aunque sí identidad de configuración, preprocesamiento y particiones. Sin los originales no se cuantifica la diferencia numérica de pérdidas o pesos ni se determina su causa. La publicación de nuevas curvas no recupera los archivos privados ni reproduce los contrastes temporales originales desde observaciones.

## Límites que permanecen abiertos

- **Reproducción histórica:** recuperar curvas, checkpoints, escaladores y vectores diarios originales para comparar discrepancias numéricas y reproducir los análisis temporales desde sus observaciones; los nuevos ajustes son una ejecución separada.
- **Literatura:** completar el historial nativo de WoS/Scopus e IEEE y las identidades/fechas efectivas aún no resueltas. La ruta actual usa alternativas de texto completo gratuito y conserva 14 informes no recuperados; obtenerlos en el futuro permitiría reabrir su elegibilidad, sin ser condición para citar las alternativas verificadas. Mantener el balance de recurrencia, simplicidad y diversidad según §3.2, distinguiendo activos, objetivos, componentes y versiones. No inferir Q1 o indexación individual por editorial ni confundir registros con estudios incluidos.
- **Temporalidad:** 60 días y un mercado limitan las conclusiones; la potencia entre semillas no sustituye el tamaño efectivo temporal o la incertidumbre de otros regímenes.
- **Normalidad y transferencia del piloto:** las diferencias definitivas rechazan normalidad; la referencia t y sus IC son aproximaciones a R=465. La dispersión y la distribución de 2024 pueden cambiar en 2026. El límite superior de DE y las sensibilidades hacen explícito ese supuesto, sin garantizarlo.
- **Fase I:** su carácter exploratorio y ausencia de piloto previo permanecen como hechos históricos; no se reparan retrospectivamente.
- **Otras entradas:** D14/21/28 son resultados exploratorios iniciales. El piloto y HPO nuevos solo dimensionan D7.
- **Equivalencia y utilidad:** no se afirman. TOST/ROPE requieren otro planteamiento si se quisiera sostener equivalencia; un margen académico no demuestra utilidad financiera.

## Proyección a E2

E2 deberá fijar explícitamente una **versión inmutable del baseline**, incluyendo receta, semilla, selección y protocolo: GRU/D7 de 32 unidades de fase I y GRU/D7-c03 de 16 unidades de fase II no son el mismo modelo entrenado. La propuesta avanzada requiere su propio piloto o dispersión pareada, planificación y datos que no hayan sido utilizados para ajustarla o escogerla. Los R=465 y el nuevo periodo ahora evaluado no se transfieren automáticamente a ese estudio.

No se entrenó aquí un modelo avanzado ni se inventan requisitos de documentos completos de E2/E3 que no se han aportado.

La DE definitiva de GRU–TCN supera aproximadamente 1,64 veces su cota de planificación obtenida con el piloto. Junto con el rechazo de normalidad, esto limita la transferencia de los supuestos: el R se calculó antes de la ejecución, pero no se afirma potencia alcanzada o demostrada.
