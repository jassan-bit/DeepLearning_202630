# Entregable 3 · Deep Learning

**Autor:** Jassan Arteta

**Curso:** Deep Learning · 2026-30

**Estado:** continuidad propuesta con CNN-1D + LSTM + atención temporal; guía específica y alcance académico pendientes. Sin implementación ni resultados propios.

Este es el Jupyter Book del **Entregable 3**, con portada, índice y configuración propios. Comparte el repositorio del curso con los otros entregables y dispone de una dirección web separada.

La guía de E3 todavía no se ha incorporado. Los capítulos siguientes constituyen una **organización inicial orientativa**, no una transcripción de requisitos oficiales. El objetivo específico, el desarrollo y los productos se definirán al revisar esa guía. No se han registrado ejecuciones ni resultados de E3.

## Continuidad elegida con E1 y E2

Por decisión del autor, el candidato activo compartido es **CNN-1D + LSTM + atención temporal**. Sustituye la propuesta HAR residual, que queda como antecedente inactivo. La propuesta combina extracción convolucional causal, memoria recurrente unidireccional y ponderación de estados temporales ya observados; no hay superioridad demostrada ni receta congelada.

El [contrato sincronizado de E1/E2/E3](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) fija la referencia: BTCUSDT, siete días y cinco canales derivados de OHLCV mediante `relative_window`, objetivo σ₇ poblacional y horizonte de un día. La comparación principal ya ejecutada en E1 usa 608 fechas de 01/01/2025 a 31/08/2026. Su calendario es conocido: reutilizarlo sería evaluación histórica exploratoria, no un test nuevo independiente. Las métricas, pesos y R=309 de E1 no se atribuyen a E3.

Esta continuidad temática **no sustituye la guía oficial de E3**. El alcance evaluable, productos exigidos, receta del híbrido, HPO, semillas, dimensionamiento y evaluación nueva siguen pendientes. Compilar este libro sólo publica documentación.

## Recorrido del libro

| Capítulo | Contenido inicial | Estado |
|---|---|---|
| [1. Alcance y requisitos](sections/01_alcance_requisitos.md) | Enunciado, criterios y productos que deberá precisar la guía | Pendiente de guía |
| [2. Planificación](sections/02_planificacion.md) | Decisiones, versiones y secuencia de trabajo | Propuesta temática elegida; diseño pendiente |
| [3. Desarrollo y evidencias](sections/03_desarrollo_evidencias.md) | Material que se producirá para E3 | Diseño propuesto; sin implementación |
| [4. Resultados y discusión](sections/04_resultados_discusion.md) | Hallazgos propios y sus límites | Sin resultados E3 |
| [5. Reproducibilidad y entrega](sections/05_reproducibilidad_entrega.md) | Construcción del libro y futuras evidencias | Libro inicial |

## Libros del curso

- [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html).
- [Entregable 2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html).
- [Índice de los tres entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html).

Los enlaces a E1 y E2 son referencias externas. La compilación de este libro utiliza sus propias fuentes y no incorpora sus páginas o resultados como si pertenecieran a E3.

## Avance reutilizable de E2

El [prototipo integrado y las pruebas sintéticas de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/18_integracion_sintetica.html) ya incluyen CNN-1D + LSTM + atención temporal. Son evidencia de software de E2, no una ejecución, evaluación o resultado propio de E3; su guía sigue pendiente.
