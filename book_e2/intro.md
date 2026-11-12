# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

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
| Protocolo y criterios registrados | [§45](sections/45_enmienda_protocolo.md) |
| Réplica TimeMixer ETTh1: ejecución y métricas | [§5](sections/05_paper_guia.md) |
| Benchmark: evidencia existente y alcance | [§6](sections/06_benchmark_final.md) · [Cortes históricos (§41)](sections/41_avance_optimizacion_11_origenes.md) |
| Reproducibilidad y exigencias de la guía | [§8](sections/08_reproducibilidad.md) · [§38](sections/38_cumplimiento_guia.md) |

## Catálogo y protocolo vigentes

Los **16 sistemas obligatorios** son **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**; las referencias **MLP16, LSTM32 y GRU32**; y **persistencia, naive estacional y ARIMA**. El híbrido CNN-LSTM-atención y Ridge conservan alcance opcional/histórico.

La política mantiene nueve cuerpos definidos del Top 10 y NAS para TiDE. Las restauraciones de SAE, TimesNet, TimeMixer y PatchTST tienen recetas nuevas; los ajustes anteriores no se renombran como resultados de esos cuerpos. [Fuentes y adaptaciones (§44)](sections/44_arquitecturas_publicadas.md).

La enmienda registrada distingue **tres anclas HPO** —enero, julio y octubre de 2024— de **diez reajustes mensuales de evaluación**. Conserva embargo, equidad y selección causal. Los periodos históricos evaluados mantienen carácter conocido y exploratorio. [Protocolo (§45)](sections/45_enmienda_protocolo.md).

## Evidencia de cierre

El corte de selección se fija al commit [`795844308c07`](https://github.com/jassan-bit/DeepLearning_202630/commit/795844308c07cbc160d3dd1feb6284b5f443bb20), con cierre registrado el **12/11/2026 a las 12:53:37 UTC**. Contiene cincuenta selecciones modelo/origen del plan anterior; sus contadores no prueban ejecución de las recetas restauradas o de la enmienda nueva.

La [instantánea](_static/delivery_20261112/selection_snapshot.json) y el [CSV de ensayos](_static/delivery_20261112/trials.csv) conservan el detalle. La conciliación mantiene el techo de **64 horas acumuladas**, sin tiempo nuevo. Esta publicación no comprueba ni modifica el kernel privado de Colab.

La revisión global de requisitos queda en [§38](sections/38_cumplimiento_guia.md). Los capítulos anteriores conservan su fecha y ámbito. La [preparación técnica de pilotos (§46)](sections/46_artefactos_y_pilotos.md) es un apéndice opcional para una continuación posterior.
