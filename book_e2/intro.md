# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30**

**Autor:** Jassan Arteta

**Estado:** secciones 1–5 desarrolladas. Las capturas de las secciones 1–4 se recibieron el 5 de noviembre de 2026; la sección 5, el 6 de noviembre (America/Bogota), con página 16 y continuación sin folio visible. Se ejecutaron el EDA, la réplica externa de TimeMixer en ETTh1 y un piloto externo sobre TimeMixer, iTransformer y PatchTST. Los adaptadores y el benchmark **Bitcoin D7** siguen pendientes; esas corridas externas no son resultados de Bitcoin. Faltan las secciones siguientes y el cierre del protocolo del benchmark.

Este es el Jupyter Book del **Entregable 2**. Tiene su propia configuración, índice y capítulos dentro del repositorio común [DeepLearning_202630](https://github.com/jassan-bit/DeepLearning_202630). Su dirección es distinta de las de E1 y E3.

## Objetivo y estándar de calidad

El entregable requiere una revisión sistemática del estado del arte, un EDA orientado a decisiones y un **Top 10 de modelos** y su benchmark con referencias adicionales bajo un protocolo común, reproducible y con prevención de fuga de datos. Debe seguir la ruta metodológica del tipo de problema y respaldar sus conclusiones con inferencia estadística, análisis de robustez, calibración, incertidumbre e interpretabilidad. Culminará en una brecha de investigación y una hipótesis contrastable.

Se mantiene el **pronóstico de volatilidad de Bitcoin: regresión temporal**, con la definición de datos y objetivo heredada de E1. Cada decisión deberá justificarse, cada afirmación apoyarse en evidencia y cada resultado poder reproducirse desde el repositorio. El propósito es un benchmark académico; los márgenes no acreditan utilidad financiera.

## Cómo leer el libro

La parte recibida de la guía describe esta organización: secciones 1–6, productos; sección 7, rutas por tipo de problema y pruebas; sección 8, reproducibilidad, ética y comunicación; secciones 9–14, procedimientos de mayor riesgo metodológico; apéndices, plantillas y listas de verificación. **Ese resumen no sustituye el contenido completo todavía no recibido.**

El índice actual distingue las secciones desarrolladas de las páginas de trabajo:

1. [Definición del problema, preguntas de investigación y dataset](sections/01_punto_partida.md): formulación, tres RQ, contrato heredado y Data Card.
2. [EDA orientado a decisiones](sections/02_eda.md): distribución, transformaciones, calidad, duplicados, sesgos, partición, desplazamiento y diagnóstico temporal.
3. [Revisión de literatura y diez modelos propuestos](sections/03_literatura.md): fuentes WoS/Scopus, deduplicación, cuatro familias, código de autores y adaptación a Bitcoin.
4. [Implementación de modelos benchmark](sections/04_implementacion.md): diseño temporal, HPO y costes propuestos, requisitos de implementación, piloto, auditoría de fuga, evaluación, robustez, explicabilidad y ablación. La auditoría distingue documentación válida de preparación real para entrenar.
5. [Paper guía y replicación](sections/05_paper_guia.md): TimeMixer como referencia, réplica parcial ETTh1 con cinco semillas, criterio previo de compatibilidad y piloto externo de tres modelos para cuantificar limitaciones. Presenta resultados observados, discrepancias y límites de transferencia a Bitcoin.

Además, se conservan las páginas de trabajo de [catálogo y propuesta](sections/02_propuesta.md), [diseño experimental](sections/03_diseno_experimental.md), [benchmark pendiente](sections/04_experimentos_resultados.md), [reproducibilidad](sections/05_reproducibilidad_entrega.md) y [referencias de contexto](sections/referencias.md).

Las páginas de trabajo son provisionales y no se presentan como el desarrollo de otras secciones oficiales cuyo texto aún no se ha recibido. La sección 3.2 recibida aclara que los diez se añaden a las tres bases de E1, una referencia no profunda y una trivial: **mínimo quince configuraciones**. La propuesta es TSMixer, TiDE, TimeMixer, TimesNet, PatchTST, Informer, Autoformer, iTransformer, Crossformer y S-Mamba. Se procesaron exportaciones de WoS y Scopus y se leyeron sus diez originales completos, con métricas, páginas y limitaciones trazables. Falta validar las adaptaciones y cerrar el protocolo operativo; el cribado global y la búsqueda transversal siguen abiertos.

## Relación con E1 y E3

Se heredan **los tres modelos base, la métrica primaria, el margen práctico δ, la unidad de independencia y la partición** de la versión de E1 identificada en la [sección 1](sections/01_punto_partida.md). El tamaño muestral de E1 no determina automáticamente el de E2. E3 reutilizará y congelará los resultados que efectivamente produzca E2.

Los periodos de prueba de E1 ya son conocidos. Por ello, las preguntas formuladas sobre ellos se clasifican como **exploratorias**; escribir hipótesis formales ahora no los convierte en un test nuevo y sellado. Cuando una disposición posterior de la guía precise otra anterior, prevalece la más restrictiva. Su aplicación completa requiere las páginas restantes.

El diagnóstico residual del EDA registra una **alerta para revisar la ruta de series de tiempo**. No cambia silenciosamente las entradas heredadas ni demuestra que añadir rezagos del objetivo mejore el pronóstico.

Descargar el [registro de la guía recibida](./_static/guia_e2_recibida.json), el [contrato de referencia de E1](./_static/e1_reference.json) y la [tabla hallazgo–decisión](./_static/eda_traceability/hallazgo_decision.csv). También está disponible el [paquete de la réplica y el piloto ETTh1](./_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip), con código propio, diseño registrado y agregados verificables.

[Todos los entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) · [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html).
