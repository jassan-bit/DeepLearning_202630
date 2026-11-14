# 17. Contrato sincronizado de E1, E2 y E3

**Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

**Actualización editorial del 08/11/2026.** El candidato activo de continuidad es **CNN-1D + LSTM + atención temporal**. Sustituye HAR + red residual, cuyo código y recibos permanecen históricos e inactivos. Esta página no es un preregistro, no registra entrenamiento de mercado ni modifica resultados; la implementación provisional y el smoke se documentan en [§18](18_integracion_sintetica.md).

## 17.1 Qué aporta cada entregable

| Entregable | Referencia vigente | Estado y límite |
|---|---|---|
| E1 | GRU32, LSTM32 y MLP16, estudio `e1_608_remediation_20261107` | Ejecutado sobre 608 fechas conocidas; comparación histórica exploratoria |
| E2 | Diez modelos de §15 y candidato 11 CNN-1D + LSTM + atención temporal | Selección editorial; integración, preregistro y benchmark vigentes pendientes |
| E3 | Continuidad temática elegida con el mismo híbrido | Diseño conceptual; guía académica específica pendiente, sin implementación ni resultados propios |

La fuente principal es [E1, benchmark de 608 fechas](https://jassan-bit.github.io/DeepLearning_202630/sections/11_reejecucion_608.html). El complemento anterior de 60 fechas y la ejecución original de diez semillas mantienen identidad propia. Los archivos `e1_reference.json` y `current_protocol.json` enlazados por capítulos históricos de E2 no son el contrato vigente: no se renombran ni se sobrescriben.

## 17.2 Objetivo e información disponible

La serie es **BTCUSDT de Binance Spot, días UTC**. Al comienzo del día t se predice:

$$
r_t=\log(C_t/C_{t-1}),\quad \bar r_t=\frac{1}{7}\sum_{j=0}^{6}r_{t-j},\quad
\sigma_7(t)=\sqrt{\frac{1}{7}\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

Es desviación **poblacional (`ddof=0`), sin anualización**, no varianza, precio ni volatilidad realizada intradía. Requiere ocho cierres consecutivos. La etiqueta incluye el retorno de t y sólo existe al terminar ese día; el horizonte es un día.

Las entradas son las siete barras anteriores, **[t−7,t)**, ordenadas de antigua a reciente, con columnas `(open, high, low, close, volume)`. No se añaden rezagos explícitos de σ₇, canales calendario, sentimiento, transacciones o historia mensual. Una barra pasada observada durante evaluación puede entrar en la ventana siguiente; eso no autoriza ajustar pesos con etiquetas de evaluación.

## 17.3 Preprocesamiento exacto de E1

D7×5 describe cinco canales **derivados**, no OHLCV crudo:

1. Para cada ventana, transformar cada precio P mediante `log(P) − log(C[t−1])`, usando el último cierre conocido como ancla común.
2. Transformar volumen mediante `log1p(V)`.
3. Estandarizar cada canal con media y DE poblacional del entrenamiento permitido. Los estadísticos usan todas las N×7 posiciones de las ventanas, sin deduplicar días; una DE cero se sustituye por 1.
4. Escalar el objetivo dividiéndolo por la media positiva de σ₇ del entrenamiento, **sin centrar ni aplicar MinMax**. Invertir esa escala antes de calcular métricas.
5. Usar escaladores independientes para selección interna y reajuste, ajustados sólo con sus respectivos entrenamientos. Los del ajuste definitivo quedan fijos durante evaluación.

Se conserva el calendario, sin comprimir huecos ni imputar el día incompleto 24/03/2023. Se excluyen ventanas dependientes de días inválidos con una máscara común verificable. Fuentes de implementación: [dl_data.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_data.py) y [dl_data_v2.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_data_v2.py), modalidad `relative_window`.

## 17.4 Fechas: referencia ejecutada y uso futuro

| Etapa de E1 actual | Condición ejecutada, no nueva ejecución E2/E3 |
|---|---|
| Desarrollo | 2023–2024 |
| CV | Diez bloques mensuales enero–octubre de 2024; 305 fechas concatenadas por trayectoria |
| Regla por origen o | Etiquetas de entrenamiento hasta o−8 días; embargo de siete días |
| Validación interna | Desde o−90 hasta o−8, 83 etiquetas en los cortes completos; entrenamiento interno hasta o−98 |
| Piloto E1 | 01/11–30/12/2024, 60 fechas, pesos fijos; entrenamiento hasta 24/10/2024 |
| Ajuste final E1 | Origen 01/01/2025; 709 etiquetas hasta 24/12/2024; 25–31/12 excluidos de etiquetas de entrenamiento |
| Selección interna final | 619 etiquetas hasta 25/09/2024 y validación 03/10–24/12/2024, 83 etiquetas; reajuste desde cero |
| Evaluación principal | **01/01/2025–31/08/2026: 608 fechas**, pesos y escaladores fijos, sin refit con etiquetas 2025–2026 |

Los índices forward chaining proceden de `timeseries-cv`; no se sustituye por partición aleatoria. En CV, el RMSE se calcula sobre las 305 fechas concatenadas de cada trayectoria antes de promediar semillas; no es una media no ponderada de RMSE mensuales.

El híbrido y las diez redes deberán comprobar exactamente sus máscaras, fechas comunes y disponibilidad causal antes de compararse. Esta sincronización toma la cohorte de 608 fechas como **referencia de comparabilidad**, no declara una evaluación nueva ni congelada. Las 608 fechas ya se conocen; reutilizarlas seguirá siendo exploratorio. La reserva complementaria 01/09–30/10/2026 también fue evaluada y no recupera independencia. Un estudio confirmatorio necesitaría un periodo no utilizado para selección y una decisión registrada previa, aún pendiente.

## 17.5 Referencias, métricas y límites de inferencia

**GRU32 es el baseline neuronal elegido por CV**, no el ganador frente a persistencia en evaluación. E1 usa GRU32 c03 (LR 0,003, dropout 0,1; 3.777 parámetros), LSTM32 c02 (0,003, 0; 5.025) y MLP16 c03 (0,003, 0,1; 593). Permanecen referencias separadas, no nuevas posiciones del catálogo E2.

La métrica primaria es **media entre semillas de las RMSE individuales**, en unidades originales sobre fechas comunes, no RMSE de un ensemble ni raíz de MSE agregado. Secundarias: MAE, MSE, R², MAPE y MASE. MAPE usa y>0 y declara cobertura; MASE se escala con cambios absolutos entre etiquetas consecutivas elegibles de entrenamiento. Persistencia σ₇(t−1) es una referencia determinista, no múltiples réplicas independientes.

Los [resultados cerrados de E1](https://jassan-bit.github.io/DeepLearning_202630/sections/05_benchmark.html) permanecen intactos. Persistencia tuvo menor RMSE que las tres redes en esa ejecución; el R² histórico ≈0,854 no pertenece a los nuevos pesos. No se traslada ninguna métrica al híbrido.

El margen documentado es **δ=0,00025772842664981106**. Su procedencia histórica no se recalcula usando la nueva CV. **R=309 y las semillas 18101–18409 pertenecen exclusivamente a la fase definitiva de las tres redes de E1, dimensionada con su piloto y sus hipótesis. El piloto usó diez semillas por red, 17101–17110.** No acreditan potencia para once modelos, atención, nuevos pares o inferencia temporal. E2/E3 necesitan hipótesis, multiplicidad, HPO, piloto y dimensionamiento propios antes de entrenar; no se heredan p-valores ni intervalos.

## 17.6 Diseño del candidato e implementación provisional

D7×5 relativo → CNN-1D causal → LSTM unidireccional que conserva siete estados → atención temporal sobre esos estados observados → contexto → cabeza escalar para σ₇.

La causalidad de la CNN se define sobre el tensor ya transformado. `relative_window` ancla todos los pasos al cierre C[t−1], por lo que no se afirma invariancia causal de cada prefijo sobre OHLCV crudo; sí disponibilidad completa de la ventana en el origen de pronóstico.

La atención puede combinar los siete estados porque todos pertenecen al pasado disponible en el origen. No puede atender al día objetivo ni a futuros orígenes, mezclar muestras o estimar escaladores con evaluación. Sus pesos no son explicaciones causales por sí solos.

Las recetas provisionales de §18 ya fijan estructura y pasos sintéticos. Su elección definitiva de kernels, anchos, capas, regularización, atención, enlace, optimizador, presupuesto y selección para mercado sigue abierta. No se presenta esta composición como novedosa, receta literal del curso o superioridad demostrada. Las ablaciones propuestas —CNN sola, LSTM sola y CNN–LSTM sin atención— son controles del candidato, no modelos añadidos tácitamente al catálogo. Deben definirse antes de evaluar.

## 17.7 Estado verificable

Esta sección conserva el contrato de referencia; la implementación nueva se documenta aparte en §18. Los cuatro prototipos compactos de E2 y el corrector HAR anterior conservan sus pruebas y limitaciones; **esas pruebas no validan el nuevo híbrido**. La implementación provisional y sus pruebas específicas ya están disponibles en [§18](18_integracion_sintetica.md). El preregistro, entrenamiento de mercado, evaluación y resultados del benchmark siguen pendientes. La guía de E3 también sigue pendiente.

El [contrato estructurado y las huellas de fuentes](../_static/books_alignment_20261108.json) tienen copias idénticas en los tres libros. Las huellas permiten comprobar preservación de fuentes y resultados sin publicar datos originales adicionales.
