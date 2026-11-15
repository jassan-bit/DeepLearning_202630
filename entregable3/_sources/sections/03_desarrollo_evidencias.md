# 3 Desarrollo y evidencias

## Componentes que pueden reutilizarse

| Componente | Procedencia | Uso en E3 |
|---|---|---|
| Contrato BTCUSDT D7×5 y transformación `relative_window` | E2, contrato vigente §17 | Mantener objetivo, disponibilidad, máscaras y escalas; auditar su ejecución real. |
| Catálogo, recetas y resultados | E2, A2 y congelación de 81 predictores | Referencia histórica y punto de partida del inventario; no representan una nueva evaluación E3. |
| Prototipo CNN-LSTM-atención y pruebas sintéticas | [Integración de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/18_integracion_sintetica.html) | Reutilización de software atribuida; no demuestra novedad, propiedad teórica ni desempeño del mecanismo nuevo. |
| Robustez, calibración, explicaciones y reproducción | E2, §§54–63 | Motivación y formatos reutilizables con su alcance posterior e histórico. |

El [avance de los bloques A y B](08_bloques_a_b.md) presenta el inventario y la evidencia de ingeniería obtenida de estos componentes. La comprobación de un archivo, su exportación o su paridad no acredita por sí sola ausencia de fuga ni superioridad predictiva.

## Desarrollo del Bloque C

El [candidato y sus pruebas](09_modelo_original.md) ya tienen código desde la formulación y una demostración de estabilidad de la cabeza latente. Las 35 pruebas usan datos artificiales; no completan la búsqueda de novedad ni la comparación experimental.

## Productos nuevos necesarios

El mecanismo original deberá programarse desde su formulación, con docstrings, pseudocódigo, complejidad y trazabilidad de procedencia. El [diseño preliminar](07_diseno_preliminar.md) delimita una hipótesis que aún debe superar la búsqueda de novedad y los experimentos de desarrollo.

Los tests exigidos incluyen sobreajuste de un lote diminuto, determinismo, formas y entradas inválidas, gradientes, propiedad formal y coincidencia al guardar/cargar/exportar. Las pruebas previas del híbrido convencional no validan automáticamente sus modificaciones. El diseño debe permitir al menos tres ablaciones aisladas y justificadas.

## Dashboard previsto

Dash Plotly tendrá Contexto, EDA y Modelos. Para esta tarea temporal se prevén cronología y particiones, distribución y dependencia de volatilidad, leaderboard completo, predicción con validación del esquema D7×5, incertidumbre, explicación local, aviso fuera de dominio y diagnóstico de errores. Los límites de memoria se resuelven con carga selectiva y resultados precalculados justificados para modelos pesados.

La aplicación sólo cargará artefactos y hará inferencia CPU. Los agregados del dashboard y del libro deberán generarse desde la misma tabla maestra y comprobarse automáticamente.
