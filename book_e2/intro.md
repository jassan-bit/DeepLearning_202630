# Proyecto Final de Deep Learning: Entregable 2

**Guía original cotejada:** [correcciones y pendientes por página](sections/20_contraste_guia.md). La cuadrícula reducida no cubre el espacio mínimo; controles F2a, NAS condicional y batería estadística requieren completar el plan antes de entrenar.


**Actualización de planificación:** [§19: auditoría real de desarrollo y plan propuesto](./sections/19_plan_auditoria.md). Sin entrenamiento de mercado, HPO ni preregistro nuevo.

**Deep Learning · 2026-30 · Jassan Arteta**

## Catálogo vigente

Esta presentación considera únicamente los siguientes diez modelos:

1. **FCN-1D**.
2. **ResNet-1D**.
3. **Inception-1D**.
4. **LSTM-autoencoder con predictor**.
5. **TimesNet**.
6. **TSMixer**.
7. **TiDE**.
8. **TimeMixer**.
9. **PatchTST**.
10. **iTransformer**.

**Candidato 11: CNN-1D + LSTM + atención temporal.** Su superioridad no está demostrada. Es una hipótesis que debe compararse con controles y puede arrojar resultados negativos.

El catálogo sustituido se retira de la presentación. Estos nombres provienen de la literatura y de adaptaciones a las familias del curso; no son una transcripción literal de sus modelos. MLP, LSTM y GRU de E1 son referencias separadas. [Especificación y límites del catálogo](sections/15_revision_modelos.md).

## Qué está implementado y qué está pendiente

- **Implementación provisional disponible:** los once candidatos comparten fábrica y flujo sintético; 11/11 ejecutados sin omisiones, dos pasos supervisados cada uno y un paso de reconstrucción para el autoencoder. El híbrido ya tiene código y pruebas propias. **No hay entrenamiento de mercado ni benchmark completo.** [Recetas y evidencia](sections/18_integracion_sintetica.md).
- **Benchmark vigente:** sin entrenamiento integral ni resultados comparativos publicados. No hay ganador. Deben fijarse recetas, entradas, controles, HPO, dimensionamiento e inferencia en una nueva enmienda antes de ajustar datos de mercado. [Plan experimental](sections/06_benchmark_final.md).
- **Evidencia histórica real:** el piloto previo completó 220 trayectorias y 440 ajustes. El recibo del 06/11/2026 acredita HPO previo iniciado e incompleto, no ejecución actual. La réplica TimeMixer/ETTh1 es externa a Bitcoin. Estos recibos no se atribuyen al catálogo vigente. [Trazabilidad](sections/09_protocolo_registrado.md).

El contrato vigente se sincroniza con E1: D7×5 derivado de OHLCV mediante `relative_window`, σ₇ poblacional sin anualización, horizonte de un día y referencia histórica principal de 608 fechas. [Contrato completo y diferencias con los recibos anteriores](sections/17_contrato_sincronizado.md). No se reentrena ni se trasladan resultados al híbrido.

## Evidencia bibliográfica

**Ampliación bibliográfica del 08/11/2026:** la [sección 3](sections/03_literatura.md) incorpora una ronda nueva de WoS, Scopus e IEEE (**1.634 ocurrencias → 1.078 grupos**) y cuatro consultas arXiv (**1.199 apariciones → 1.004 IDs**). Los universos se mantienen separados hasta conciliarlos. Hay triaje provisional, tres lecturas focales y fuentes para las nuevas propuestas; no se declara elegibilidad global cerrada ni se modifican resultados experimentales.

**Avance de revisión bibliográfica:** los 59 prioritarios ya tienen lectura individual asistida por IA: 40 candidatos a texto completo, 17 inciertos y dos secundarios. La lectura de exclusiones propuestas alcanza 337/337: las 295 posteriores aportan 184 fuera del núcleo, 98 secundarios, 12 inciertos y un candidato directo. De los 40 candidatos iniciales, se examinaron 22: ocho textos completos (seis editoriales y dos preprints con equivalencia editorial pendiente) y catorce accesos parciales/no recuperados; 18 siguen sin búsqueda individual. No son inclusiones finales ni doble revisión humana. [Conteos y límites](sections/03_literatura.md).

## Ruta de lectura

- [Problema, datos e hipótesis](sections/01_punto_partida.md), [EDA](sections/02_eda.md) y [literatura](sections/03_literatura.md).
- [Implementación](sections/04_implementacion.md), [réplica externa](sections/05_paper_guia.md) y [benchmark pendiente](sections/06_benchmark_final.md).
- [Ruta temporal](sections/07_ruta_temporal.md), [reproducibilidad](sections/08_reproducibilidad.md) y [registro histórico](sections/09_protocolo_registrado.md).
- [Auditoría de datos](sections/10_auditoria_datos.md), [HPO](sections/11_hpo_equidad.md), [incertidumbre](sections/12_incertidumbre_errores.md), [inferencia](sections/13_agregacion_estadistica.md) y [arquitecturas](sections/14_arquitecturas.md).
- [Diez modelos y candidato 11](sections/15_revision_modelos.md), [fichas de las cuatro nuevas arquitecturas](sections/16_fichas_nuevas.md).

Los protocolos, muestras, scripts y recibos anteriores se preservan como evidencia inmutable. No se reetiquetan como entrenamientos del diseño actual ni se declara cerrada la revisión bibliográfica.
