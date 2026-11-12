# 26. Registro por receta: fuentes, NAS, HPO e integración

**Vigencia documental (12/11/2026):** el [catálogo obligatorio (§43)](43_catalogo_obligatorio.md) contiene dieciséis sistemas; híbrido y Ridge son opcionales. [§44](44_arquitecturas_publicadas.md) conserva o restaura nueve cuerpos definidos del Top 10 y mantiene NAS para TiDE. La selección HPO/NAS ejecutada corresponde al plan anterior de catorce redes; no acredita las nuevas recetas SAE, TimesNet, TimeMixer y PatchTST. El benchmark final, k, R y el cierre inferencial siguen pendientes. [Enmienda registrada (§45)](45_enmienda_protocolo.md) · [Corte histórico de resultados (§41)](41_avance_optimizacion_11_origenes.md).

Las seis familias NAS de [§27](27_nas_hpo_presupuesto.md) identifican el plan anterior. Las tablas que siguen documentan el inventario del 09/11/2026: sus IDs, parámetros, estados de integración y decisiones abiertas se interpretan a esa fecha. Las nuevas identidades `SAE-D7-PUBLISHED-v1`, `TN-D7-PUBLISHED-v1`, `TM-D7-PUBLISHED-v1` y `PT-D7-PUBLISHED-v1` y sus adaptaciones se consultan en §44; los recibos anteriores conservan su identidad.

**Secuencia histórica:** [§27 fijó seis variantes NAS y cinco cuerpos publicados](27_nas_hpo_presupuesto.md); sus ejecuciones posteriores se publican por separado. La decisión vigente de §44 y el calendario de §45 rigen el trabajo nuevo afectado y no modifican los artefactos de aquellos cortes.

**Corte documental: 09/11/2026.** Esta tabla entrega el inventario por modelo previsto en [§25](25_matriz_cierre.md), R01 y R03–R06. Se cotejaron fichas, fábrica y recibos del [snapshot eca22e8](https://github.com/jassan-bit/DeepLearning_202630/blob/eca22e87085edca79e53f60196c8f64458659b5d/book_e2/sections/23_implementacion_recetas.md). El registro distingue una receta implementada de una receta experimental congelada: **no se ha adjudicado NAS, cerrado HPO ni ejecutado un benchmark nuevo**.

[Registro estructurado JSON](../_static/recipe_register_20261109/registry.json) · {download}`Descargar registro por receta <../_static/recipe_register_20261109/registry.json>`.

## 26.1 Qué versiones documenta este inventario histórico

Las [fichas de §22](22_fichas_recetas.md) se implementaron en [§23](23_implementacion_recetas.md) mediante `scripts/e2_recipe_catalogue.py`. Se conservan sus IDs, incluidos los sufijos `proposed`, para enlazar especificación, código y recibo sin cambiar retrospectivamente identidades.

La [matriz NAS de §21](../_static/controls_nas_20261108/nas_matrix.json) es un corte anterior: sus descripciones de FCN, ResNet e Inception compactas no describen los cuerpos restaurados de §23. Este registro aporta la síntesis posterior; el JSON histórico permanece intacto. El SAE también tiene ahora dos fases de reconstrucción implementadas.

Contrato compartido: D7×5 OHLCV con `relative_window`, h=1, σ₇ poblacional sin anualizar, escaladores ajustados sólo al entrenamiento y salida escalar Softplus multiplicada por la media positiva del objetivo de entrenamiento. **Los seis adaptadores agregan diez momentos de la misma ventana; los cuatro cuerpos restaurados y el híbrido no.** Las entradas informativas son comunes, pero esta asimetría del readout debe controlarse; no se presenta como comparación de backbones puros.

## 26.2 Fuentes y adaptación por modelo: corte del 09/11/2026

Los enlaces bibliográficos identifican originales arquitectónicos; no acreditan elegibilidad final del Top 10 ni resultados en Bitcoin. Los snapshots de código y las versiones exactas de los seis textos están en el JSON descargable y las fichas.

| Modelo / ID de receta | Fuente | Cuerpo y adaptación de la receta de aquel corte |
|---|---|---|
| **FCN-1D** · `fcn1d_wang_d7_v1_proposed` | [Wang et al. · FCN](https://arxiv.org/abs/1611.06455) | Conv1D 128/256/128; kernels 8/5/3; BN, ReLU y media temporal. Port Conv2D singleton a Conv1D; D7×5 y cabeza escalar Softplus. |
| **ResNet-1D** · `resnet1d_wang_d7_v1_proposed` | [Wang et al. · ResNet](https://arxiv.org/abs/1611.06455) | Tres bloques 64/128/128, nueve convoluciones principales y tres proyecciones. Port 1D; BN inicial sólo en rama principal y proyección del tercer atajo; cabeza de regresión. |
| **Inception-1D** · `inception1d_single_d7_v1_proposed` | [InceptionTime](https://arxiv.org/abs/1909.04939) | Seis módulos, cuatro ramas de 32 canales y dos conexiones residuales; kernels 40/20/10. Red individual en D7 y cabeza positiva; padding largo sobre siete observaciones. |
| **LSTM-autoencoder con predictor** · `lstm_sae_25_26_d7_v1_proposed` | [Sagheer y Kotb · SAE](https://doi.org/10.1038/s41598-019-55320-6) | Encoders LSTM 25/26; dos preentrenamientos separados y transferencia al predictor. Ambas fases reconstruyen X/s; segundo encoder recibe vector repetido; escala auxiliar propia; Softplus. |
| **TimesNet** · `TN-D7-r1` | [TimesNet v3](https://arxiv.org/abs/2210.02186v3) | Dos bloques, d_model32 y d_ff32; proyección temporal 7→8 latente. Cuatro frecuencias admisibles frente a top-k5 fuente, cabeza sobre último token y diez momentos. |
| **TSMixer** · `TSM-D7-r1` | [TSMixer v5](https://arxiv.org/abs/2303.06053v5) | Seis bloques de mezcla y MLP de variables 512; semántica BN del port. Aplanado 35 + diez momentos, Softplus y dropout inicial 0,1. |
| **TiDE** · `TIDE-D7-r1` | [TiDE v5](https://arxiv.org/abs/2304.08424v5) | Preset encoder/decoder 256, latente8 y decoder temporal128. Sentinel futuro cero, decoder temporal emite8 frente a1; 40 latentes + diez momentos. |
| **TimeMixer** · `TM-D7-r1` | [TimeMixer v1](https://arxiv.org/abs/2405.14616v1) | PDM/FMM, dos bloques, d_model16/d_ff32; escalas7/3/1. Dos descensos en vez de tres para evitar escala vacía; 80 latentes + diez momentos. |
| **PatchTST** · `PT-D7-r1` | [PatchTST v2](https://arxiv.org/abs/2211.14730v2) | Tres capas, cuatro cabezas, d_model16/d_ff128; patch16, stride8. RevIN antes del padding; dos parches de datos replicados de D7; 160 latentes + diez momentos. |
| **iTransformer** · `IT-D7-r1` | [iTransformer v4](https://arxiv.org/abs/2310.06625v4) | Dos capas, ocho cabezas, d_model256/d_ff256; cinco tokens de variables. Embedding de siete valores, retirar projector/desnormalización OHLCV; 1280 latentes + diez momentos. |
| **CNN-1D + LSTM + atención temporal** · `HIB-D7-r1` | Composición propia | CNN causal32/k3 → LSTM32 → atención aditiva16; cabeza escalar. Composición propia, estado reiniciado por ventana y sin momentos adicionales. |

**Conciliación bibliográfica de este corte.** El expediente [de textos completos](../_static/literature_fulltext/models.json) registra las versiones leídas de los seis adaptadores: TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. Esto no cierra el PRISMA global ni la calidad editorial por año de [§3](03_literatura.md). Para el SAE, la receta exacta está sustentada en el código del autor y sigue pendiente el texto íntegro verificable. Para el híbrido no se verificó un artículo que defina la composición completa.

Hay diferencias que deben conservarse al justificar la selección: FCN/ResNet proceden de clasificación; Inception es un solo miembro y no el ensemble de cinco; TSMixer usa dropout inicial distinto de la fuente; TiDE cambia la salida del decoder temporal; TimeMixer reduce escalas; PatchTST replica datos de siete días para formar parches. Los límites individuales se incluyen en el JSON. Ninguna de estas fuentes demuestra superioridad de la receta D7.

## 26.3 Decisión NAS por receta

La columna siguiente identifica la vía de decisión y su fundamento **en el registro histórico** (`applies: null`, sin exenciones concedidas a esa fecha). La adjudicación posterior de §44 sustituye esas decisiones para el trabajo nuevo: nueve cuerpos definidos y TiDE NAS. La implementación sintética, por sí sola, no decide la admisibilidad académica de una adaptación.

| Modelo | Clasificación documental actual | Decisión que falta resolver |
|---|---|---|
| FCN-1D | Cuerpo definido y puente de tarea | Documentar traducción y cabeza para resolver la vía de cuerpo publicado; no heredar decisión del compacto. |
| ResNet-1D | Cuerpo definido y puente de tarea | Resolver cuerpo publicado con convenciones del snapshot y adaptación de tarea. |
| Inception-1D | Cuerpo definido y puente de tarea | Resolver aceptación de un miembro individual y geometría D7 frente al ensemble publicado. |
| LSTM-autoencoder con predictor | Variante de código; texto pendiente | Recuperar texto completo verificable y adjudicar variante de código, escala y transferencia. |
| TimesNet | Cambio estructural | Adjudicar cambios estructurales de frecuencias y puente D7, no sólo la cabeza. |
| TSMixer | Cuerpo definido y puente de tarea | Resolver cuerpo publicado y cabeza conjunta; conciliar discrepancias de dropout y rango de ancho. |
| TiDE | Cambio estructural | Adjudicar decoder y covariables sustitutas como adaptación estructural o diseño propio. |
| TimeMixer | Cambio estructural | Adjudicar reducción de escalas y readout; necesidad geométrica no concede exención. |
| PatchTST | Geometría de parches adaptada | Resolver geometría corta y cabeza conjunta; exención del modelo publicado no se extiende automáticamente. |
| iTransformer | Cuerpo definido y puente de tarea | Resolver embedding7 y cabeza de tarea con momentos como puente admisible. |
| CNN-1D + LSTM + atención temporal | Composición propia | Si se evalúa como diseño propio E2, aplicar propuesta §14: w1 CNN, w2 LSTM, w3 atención; falta resolver e implementar tres dropouts. |

Cuando corresponda §14 de la guía: tres bloques, cinco anchos {16,32,64,128,256} por bloque y ocho patrones binarios de dropout p=0,2 producen **1.000 arquitecturas únicas**. La referencia es (64,64,64) con patrón (0,1,0). Primero se ajustan los otros hiperparámetros sobre la referencia; después fase 1 con los mil diseños y una semilla, y fase 2 con k≥10 finalistas, más referencia si no está incluida, y al menos cinco semillas distintas de fase 1. La traducción de bloques a cada familia y las restricciones requieren especificación previa; enumerar nombres o repetir diseños no satisface la cardinalidad.

Este inventario no fijaba cuántas recetas entraban a NAS. La adjudicación vigente fija TiDE como único NAS obligatorio; el híbrido queda opcional. La presencia de un nombre publicado nunca sustituye la verificación del cuerpo y de su adaptación.

## 26.4 HPO: factores que deben convertirse en espacios ejecutables

El mínimo de §11.2 de la guía incluye **learning rate en escala logarítmica, weight decay, dropout, batch, profundidad y ancho**, con ejes específicos por familia. El cotejo de [§20](20_contraste_guia.md) explica por qué la cuadrícula reducida anterior no basta.

La tabla concreta qué debe revisarse por receta. Los complementos estructurales se proponen para especificación, no son dominios aprobados ni una búsqueda ya implementada. Los rangos numéricos, distribuciones, optimizador, pérdida de entrenamiento, épocas, parada, número de oportunidades y presupuesto común permanecen sin congelar. El Adam/MSE de un paso sintético no se convierte por omisión en la configuración final.

| Modelo | Ejes adicionales o restricciones por resolver, además del mínimo común |
|---|---|
| FCN-1D | Kernel y normalización; conciliar profundidad/ancho con cuerpo publicado. |
| ResNet-1D | Kernel, normalización y estructura residual; conciliar profundidad/ancho. |
| Inception-1D | Kernel, ramas, bottleneck y módulos; mantener identidad de miembro individual. |
| LSTM-autoencoder con predictor | Celdas, anchos y transferencia; dos fases AE con presupuestos propios. |
| TimesNet | Frecuencias admisibles, kernels, embedding y bloques; compatibilidad D7. |
| TSMixer | Bloques, ancho del MLP y normalización; conciliar dropout con fuente. |
| TiDE | Anchos encoder/decoder/latente y estructura residual; condicionados a la decisión NAS. |
| TimeMixer | Escalas, pooling, descomposición y bloques; evitar longitudes vacías. |
| PatchTST | Cabezas, embedding, FF, capas y geometría de parches; D7 fijo. |
| iTransformer | Cabezas, embedding, FF y capas; cinco tokens y D7 fijo. |
| CNN-1D + LSTM + atención temporal | Canales/kernel CNN, unidades LSTM, atención y ubicaciones de dropout. |

Para Transformers deben revisarse cabezas, contexto y warmup, y learning rate por capa cuando haya fine-tuning. D7 es heredado: no se amplía la historia para aparentar búsqueda de contexto. Tampoco se inventa fine-tuning de pesos externos cuando estas recetas no los utilizan. Deben justificarse los factores fijos o no aplicables. En recurrentes se revisan celdas y bidireccionalidad; teacher forcing no corresponde al predictor escalar h=1 actual. En CNN se revisan kernel, resolución y normalización sin añadir observaciones futuras.

No se variarán profundidad y ancho silenciosamente dentro de un cuerpo que se quiera declarar predefinido: el alcance permitido de HPO y su relación con NAS debe resolverse por receta. La fábrica vigente construye las recetas del smoke; eso no acredita implementación de todos estos ejes.

## 26.5 Coste e integración: evidencia disponible

Los parámetros se transcriben del [recibo de §23](../_static/recipe_implementation_20261109/synthetic_receipt.json). Son todos los parámetros del predictor contabilizados por el código, **no necesariamente todos entrenables**, ni FLOPs, memoria pico o horas de mercado. No se reejecutaron los modelos en este corte documental.

| Modelo | Parámetros del predictor | Fases por etapa de entrenamiento | Integración comprobada |
|---|---:|---|---|
| FCN-1D | 268.929 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| ResNet-1D | 522.955 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| Inception-1D | 490.561 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| LSTM-autoencoder con predictor | 8.739 | AE1 → AE2 → supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| TimesNet | 1.172.875 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| TSMixer | 35.044 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| TiDE | 168.537 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| TimeMixer | 2.615 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| PatchTST | 16.664 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| iTransformer | 795.403 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |
| CNN-1D + LSTM + atención temporal | 9.537 | supervised | Receta completa en smoke sintético; integración de mercado pendiente |

El SAE tiene además autoencoders auxiliares de **3.840 y 6.172 parámetros**; sus decoders no forman parte del predictor de 8.739. Deben presupuestarse AE1, extracción de representaciones, AE2 y supervisado, incluidas repeticiones y refits. No sumar sin más los tres tamaños como memoria pico ni omitir las fases auxiliares.

§23 acredita once pasos supervisados y dos auxiliares sobre datos artificiales. [§24](24_ejecutor_comun.md) acredita un recorrido sintético de **18 sistemas** y, en una capa posterior, un núcleo temporal real probado. Sigue pendiente conectar y auditar todas las fábricas y adaptadores con ese núcleo. No se extrapolan los tiempos de un paso artificial a entrenamientos completos.

El presupuesto completo debe separar HPO, NAS cuando aplique, selección interna, refits, piloto de potencia, calibración, controles de fuga, robustez y ablaciones. Requiere hardware, mediciones comparables, límites y contabilidad de fallos. Parámetros pequeños no garantizan menor latencia; tiempos o memoria aún no medidos se registran como desconocidos.

## 26.6 Las referencias: seis obligatorias y Ridge histórico opcional

MLP16, LSTM32, GRU32, Persistence, SeasonalNaive y ARIMA son las seis referencias/comparadores que completan el catálogo obligatorio de dieciséis. La tabla conserva las configuraciones sintéticas del 09/11/2026; no acredita sus resultados finales. Ridge permanece para trazabilidad y no requiere nuevas corridas para cerrar el catálogo obligatorio.

| Sistema de referencia | Configuración que sí se probó sintéticamente | Selección o cierre pendiente |
|---|---|---|
| MLP16 de E1 | Arquitectura E1, inicializada de nuevo para E2 | Receta de entrenamiento y oportunidades comparables; no importar pesos o métricas históricas |
| LSTM32 de E1 | Arquitectura E1, inicializada de nuevo para E2 | Igual requisito; declarar costes y soportes |
| GRU32 de E1 | Arquitectura E1, inicializada de nuevo para E2 | Igual requisito; declarar costes y soportes |
| Persistencia | σ₇(t−1) | Auditoría causal y soporte común; no necesita búsqueda de periodo |
| Ridge, opcional histórico | α=1 en smoke | Fuera del alcance obligatorio vigente |
| Naive estacional | m=7 en smoke | Seleccionar entre {7,14,28} según desarrollo; no afirmar estacionalidad demostrada |
| ARIMA | (1,0,0) en smoke | Espacio propuesto p,q∈{0,1,2}, d∈{0,1}: 18 órdenes; cerrar selección A/B, fallos y cómputo |

El naive estacional es un sistema con tres periodos candidatos; ARIMA es un sistema con dieciocho órdenes candidatos. No se cuentan como modelos adicionales del Top 10. Sus espacios propuestos no están ejecutados por haber probado una receta. ARIMA exige conservar calendario, filtrar el puente sin refit y registrar no convergencia, sin sustituirla por un pronóstico cero.

## 26.7 Qué queda listo y qué sigue

Este inventario aporta una identidad verificable por receta de su corte: fuente, adaptación, duda NAS, factores HPO, parámetros, fases y alcance de integración. El cierre bibliográfico global sigue separado de la adjudicación arquitectónica posterior y de la ejecución experimental.

La adjudicación y los dominios actuales están documentados en §44–§45. El siguiente cierre operativo es conciliar presupuesto e intentos anteriores, completar la integración y ejecutar los pilotos y ajustes pendientes de las recetas vigentes, conservando D7, las referencias E1, la exposición conocida y todos los artefactos. El trabajo de [§46](46_artefactos_y_pilotos.md) prepara esa ejecución; no aporta por sí solo nuevos resultados de mercado.

La disponibilidad común de las 608 fechas finales sigue sin certificarse y el periodo conocido conserva alcance exploratorio. Este registro no abre esa reserva ni publica resultados de pronóstico.
