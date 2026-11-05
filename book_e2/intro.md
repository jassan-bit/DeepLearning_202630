# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30**

**Autor:** Jassan Arteta

**Estado:** introducción y secciones 1–2 desarrolladas con las capturas de la guía recibidas el 5 de noviembre de 2026 (America/Bogota). El EDA se amplió con diagnósticos ejecutados y trazables. El resto de la guía y el protocolo completo siguen pendientes; el benchmark de diez modelos de E2 no se ha ejecutado.

Este es el Jupyter Book del **Entregable 2**. Tiene su propia configuración, índice y capítulos dentro del repositorio común [DeepLearning_202630](https://github.com/jassan-bit/DeepLearning_202630). Su dirección es distinta de las de E1 y E3.

## Objetivo y estándar de calidad

El entregable requiere una revisión sistemática del estado del arte, un EDA orientado a decisiones y un **benchmark de 10 modelos** bajo un protocolo común, reproducible y con prevención de fuga de datos. Debe seguir la ruta metodológica del tipo de problema y respaldar sus conclusiones con inferencia estadística, análisis de robustez, calibración, incertidumbre e interpretabilidad. Culminará en una brecha de investigación y una hipótesis contrastable.

Se mantiene el **pronóstico de volatilidad de Bitcoin: regresión temporal**, con la definición de datos y objetivo heredada de E1. Cada decisión deberá justificarse, cada afirmación apoyarse en evidencia y cada resultado poder reproducirse desde el repositorio. El propósito es un benchmark académico; los márgenes no acreditan utilidad financiera.

## Cómo leer el libro

La parte recibida de la guía describe esta organización: secciones 1–6, productos; sección 7, rutas por tipo de problema y pruebas; sección 8, reproducibilidad, ética y comunicación; secciones 9–14, procedimientos de mayor riesgo metodológico; apéndices, plantillas y listas de verificación. **Ese resumen no sustituye el contenido completo todavía no recibido.**

El índice actual distingue las secciones desarrolladas de las páginas de trabajo:

1. [Definición del problema, preguntas de investigación y dataset](sections/01_punto_partida.md): formulación, tres RQ, contrato heredado y Data Card.
2. [EDA orientado a decisiones](sections/02_eda.md): distribución, transformaciones, calidad, duplicados, sesgos, partición, desplazamiento y diagnóstico temporal.

Además, se conservan las páginas de trabajo de [catálogo y propuesta](sections/02_propuesta.md), [diseño experimental](sections/03_diseno_experimental.md), [benchmark pendiente](sections/04_experimentos_resultados.md), [reproducibilidad](sections/05_reproducibilidad_entrega.md) y [referencias de contexto](sections/referencias.md).

Las páginas de trabajo son provisionales y no se presentan como el desarrollo de otras secciones oficiales cuyo texto aún no se ha recibido. El catálogo de diez modelos y su relación exacta con los tres modelos heredados se cerrarán con la guía completa y la revisión sistemática.

## Relación con E1 y E3

Se heredan **los tres modelos base, la métrica primaria, el margen práctico δ, la unidad de independencia y la partición** de la versión de E1 identificada en la [sección 1](sections/01_punto_partida.md). El tamaño muestral de E1 no determina automáticamente el de E2. E3 reutilizará y congelará los resultados que efectivamente produzca E2.

Los periodos de prueba de E1 ya son conocidos. Por ello, las preguntas formuladas sobre ellos se clasifican como **exploratorias**; escribir hipótesis formales ahora no los convierte en un test nuevo y sellado. Cuando una disposición posterior de la guía precise otra anterior, prevalece la más restrictiva. Su aplicación completa requiere las páginas restantes.

El diagnóstico residual del EDA registra una **alerta para revisar la ruta de series de tiempo**. No cambia silenciosamente las entradas heredadas ni demuestra que añadir rezagos del objetivo mejore el pronóstico.

Descargar el [registro de la guía recibida](./_static/guia_e2_recibida.json), el [contrato de referencia de E1](./_static/e1_reference.json) y la [tabla hallazgo–decisión](./_static/eda_traceability/hallazgo_decision.csv).

[Todos los entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) · [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html).
