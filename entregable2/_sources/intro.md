# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

**Avance verificado al 12/11/2026, 22:22 de Bogotá.** La campaña reducida registra **97 corridas HPO exitosas: 60 entrenamientos nuevos y 37 resultados anteriores reutilizados**. Hay **dos fallos conocidos y un intento con desenlace desconocido**, conservados sin repetirlos. El corte acredita dos ajustes nuevos completados en **NVIDIA A100-SXM4-80GB**. [Instantánea y procedencia](_static/e2_progress_20261113/progress_snapshot.json) · [CSV de entrenamiento y validación](_static/e2_progress_20261113/top10_training_validation.csv).

**Protocolo vigente:** diez candidatos, tres referencias E1 y tres comparadores, **16 sistemas**. Se mantienen seis configuraciones HPO por red, una semilla y tres orígenes internos; el presupuesto autorizado es de **72 horas acumuladas**, con un máximo prospectivo de **30 minutos por fase**. La transición de T4 a A100 conserva los resultados históricos y sus cargos; GitHub es el destino persistente. [Reglas, fuentes y actualizaciones (§48)](sections/48_protocolo_reducido.md).

**El testing final sigue pendiente.** Los recibos actuales acreditan entrenamiento y validación interna; todavía no aportan la tabla final A2 ni un ganador del catálogo. Los cortes consolidados anteriores conservan su fecha y alcance.

**Versión de entrega consolidada con la evidencia disponible al 12/11/2026.** Se reúnen problema, EDA, literatura, implementaciones, experimentos ejecutados y conclusiones. Este cierre utiliza resultados guardados y no inicia nuevas corridas.

**[Leer el informe consolidado, resultados y conclusiones (§47)](sections/47_entrega_consolidada.md)** · **[Descargar el paquete de entrega](_static/delivery_20261112/Entregable2_evidencia_disponible.zip)**

## Resumen del trabajo

Se estudia el pronóstico diario de volatilidad de Bitcoin: **siete días de OHLCV → σ₇ poblacional de retornos de Close**, horizonte de un día y sin anualización. Se heredan de E1 la RMSE primaria, el margen δ=0,00025772842664981106 y el tratamiento temporal de los datos.

El libro aporta un EDA orientado a decisiones, revisión bibliográfica, modelos y adaptaciones D7, una réplica parcial de TimeMixer en ETTh1 y selección experimental trazable. Los historiales conservan **1.705 resultados de la campaña optimizada y 1.802 registros anteriores: 3.507 registros acumulados**. Son resultados de selección y antecedentes identificados; no equivalen a una evaluación final común de los dieciséis sistemas.

La réplica ETTh1, con cinco semillas, satisface su criterio previo de compatibilidad numérica. El antecedente E1 muestra persistencia competitiva frente a sus tres redes. Estos experimentos tienen tareas, calendarios y recetas propios; no se transfieren a modelos E2 con otra identidad.

La entrega declara sus límites: **todavía no hay un benchmark final completo de dieciséis sistemas ni un ganador demostrado del catálogo vigente**. Los requisitos sin evidencia quedan identificados para una revisión posterior.

## Guía de lectura

| Contenido | Dónde revisar |
|---|---|
| Resultados disponibles, discusión, conclusiones y archivos | **[Informe consolidado (§47)](sections/47_entrega_consolidada.md)** |
| Problema, hipótesis, datos y contrato heredado | [§1](sections/01_punto_partida.md) · [§17](sections/17_contrato_sincronizado.md) |
| EDA y auditoría temporal | [§2](sections/02_eda.md) · [§10](sections/10_auditoria_datos.md) |
| Literatura, catálogo y arquitecturas | [§3](sections/03_literatura.md) · [§43](sections/43_catalogo_obligatorio.md) · [§44](sections/44_arquitecturas_publicadas.md) |
| Protocolo y criterios registrados | [§48, actual](sections/48_protocolo_reducido.md) · [§45, anterior](sections/45_enmienda_protocolo.md) |
| Réplica TimeMixer ETTh1: ejecución y métricas | [§5](sections/05_paper_guia.md) |
| Benchmark: evidencia existente y alcance | [§6](sections/06_benchmark_final.md) · [Cortes históricos (§41)](sections/41_avance_optimizacion_11_origenes.md) |
| Reproducibilidad y exigencias de la guía | [§8](sections/08_reproducibilidad.md) · [§38](sections/38_cumplimiento_guia.md) |

## Avance de entrenamiento y validación

**Corte de consulta: 12/11/2026, 22:22 de Bogotá; última admisión registrada a las 22:20:06.** Cada candidato tiene 18 oportunidades HPO previstas: **C01–C06 × enero, julio y octubre de 2024**. Una corrida exitosa incluye entrenamiento y evaluación en las 83 fechas comunes de validación de su origen. Las cifras de la tabla son conteos, no valores de RMSE de training o testing.

| Modelo | Nuevos completados | Reutilizados | Validaciones exitosas | Fallos conocidos | Desenlace desconocido | Por cerrar |
|---|---:|---:|---:|---:|---:|---:|
| FCN-1D | 6 | 12 | **18/18** | 0 | 0 | 0 |
| ResNet-1D | 6 | 12 | **18/18** | 0 | 0 | 0 |
| Inception-1D | 11 | 7 | **18/18** | 0 | 0 | 0 |
| LSTM-autoencoder con predictor | 18 | 0 | **18/18** | 0 | 0 | 0 |
| TimesNet | 16 | 0 | **16/18** | 2 | 0 | 0 |
| TSMixer | 3 | 6 | **9/18** | 0 | 1 | 8 |
| TiDE | 0 | 0 | **0/18** | 0 | 0 | 18 |
| TimeMixer | 0 | 0 | **0/18** | 0 | 0 | 18 |
| PatchTST | 0 | 0 | **0/18** | 0 | 0 | 18 |
| iTransformer | 0 | 0 | **0/18** | 0 | 0 | 18 |
| **Total Top 10** | **60** | **37** | **97/180** | **2** | **1** | **80** |

Se cerraron **100 de las 180 oportunidades del Top 10**, contando los fallos; las **80 sin cierre incluyen una admitida y 79 aún no admitidas**. La última admisión del corte corresponde a **TSMixer C05, julio de 2024, en A100**; una reserva no acredita por sí sola progreso del optimizador en tiempo real. Los dos ajustes A100 completados son TSMixer C03 y C04 de julio. La caché reutiliza resultados de selección ya auditados, sin proporcionar pesos finales.

Las referencias **MLP16, LSTM32 y GRU32** conservan otras **54 oportunidades HPO**, todavía sin cierre en este corte. Por tanto, las trece redes suman **234 oportunidades**, de las que quedan 134 sin cerrar. Persistence, SeasonalNaive y ARIMA completan el catálogo de 16 sistemas y su comparación final sigue pendiente. Los ceros describen esta campaña reducida, sin borrar los antecedentes históricos.

[Tabla de las trece redes (CSV)](_static/e2_progress_20261113/all13_hpo_progress.csv) · [Instantánea JSON](_static/e2_progress_20261113/progress_snapshot.json) · [Fuentes y hashes](_static/e2_progress_20261113/source_evidence.json) · [Manifiesto de estos archivos](_static/e2_progress_20261113/manifest.json) · [Control remoto más reciente en GitHub](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/control/github-primary-v1/head.json).

## Trabajo pendiente para cerrar E2

1. Cerrar el HPO de las trece redes y seleccionar cada configuración exclusivamente con validación.
2. Completar los pilotos de semillas/potencia, justificar las repeticiones y ejecutar los reajustes finales con sus curvas, pesos, configuraciones y recursos.
3. Congelar predictores y calibradores, evaluar el soporte común y rellenar la tabla A2 con métricas, incertidumbre, costes y contrastes.
4. Verificar los análisis de robustez, interpretabilidad, ablaciones y reproducibilidad que exige la guía. La [matriz de requisitos (§38)](sections/38_cumplimiento_guia.md) distingue evidencia y pendientes.

## Catálogo y protocolo vigentes

Los **16 sistemas obligatorios** son **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**; las referencias **MLP16, LSTM32 y GRU32**; y **persistencia, naive estacional y ARIMA**. El híbrido CNN-LSTM-atención y Ridge conservan alcance opcional/histórico.

La ejecución actual incorpora los cuerpos definidos del catálogo, incluida la nueva receta **TIDE-D7-PUBLISHED-R0-v2**, vinculada a la fuente oficial y a su [evidencia de paridad](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_tide_published_r0_20261112/source_parity.json). Las restauraciones de SAE, TimesNet, TimeMixer, PatchTST y TiDE tienen identidades propias; los ajustes anteriores no se renombran como resultados de esos cuerpos. [Fuentes y adaptaciones (§44)](sections/44_arquitecturas_publicadas.md) conserva el corte previo; la actualización está en [§48](sections/48_protocolo_reducido.md).

El protocolo reducido usa **tres folds HPO** —enero, julio y octubre de 2024— para seleccionar una receta global por red. El ajuste final común termina el **31/10/2024**; la calibración queda en noviembre–diciembre y la evaluación hereda las 608 fechas conocidas de E1. Conserva embargo y transformaciones ajustadas sólo con entrenamiento. Los diez reajustes mensuales del [protocolo anterior (§45)](sections/45_enmienda_protocolo.md) permanecen como antecedente; para las nuevas ejecuciones rige [§48](sections/48_protocolo_reducido.md).

## Evidencia histórica de cierre

El corte de selección se fija al commit [`795844308c07`](https://github.com/jassan-bit/DeepLearning_202630/commit/795844308c07cbc160d3dd1feb6284b5f443bb20), con cierre registrado el **12/11/2026 a las 12:53:37 UTC**. Contiene cincuenta selecciones modelo/origen del plan anterior; sus contadores no prueban ejecución de las recetas restauradas o de la enmienda nueva.

La [instantánea](_static/delivery_20261112/selection_snapshot.json) y el [CSV de ensayos](_static/delivery_20261112/trials.csv) conservan ese detalle histórico, cuyo límite era de **64 horas**. La campaña vigente utiliza las **72 horas acumuladas autorizadas**, sin reiniciar consumos, y la transición A100 documentada en §48. Esta actualización verifica recibos remotos; no inspecciona ni modifica directamente el kernel privado de Colab.

La revisión global de requisitos queda en [§38](sections/38_cumplimiento_guia.md). Los capítulos anteriores conservan su fecha y ámbito. La [preparación técnica de pilotos (§46)](sections/46_artefactos_y_pilotos.md) conserva el alcance del registro anterior; los pilotos y reajustes del plan reducido se describen en [§48](sections/48_protocolo_reducido.md).

