# Entregable 3 de Deep Learning

**Autor del proyecto:** Jassan Arteta · **Curso:** Deep Learning 2026-30

**Estado al 15 de noviembre de 2026:** diagnóstico y plan de trabajo preparados, con avance verificable en los bloques A y B a partir de los artefactos de E2. No se ha entrenado ni evaluado el modelo original de E3; las mediciones actuales verifican artefactos heredados de E2.

## Problema y punto de partida

El proyecto pronostica la volatilidad diaria de Bitcoin, BTCUSDT de Binance Spot: siete barras diarias OHLCV anteriores al día objetivo, cinco canales transformados mediante `relative_window`, horizonte de un día y desviación poblacional σ₇ de siete retornos logarítmicos de Close, sin anualización. El [contrato vigente de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) define disponibilidad, escalas y máscaras.

E2 ya ejecutó 16 sistemas sobre 608 fechas, con seis semillas por cada una de las trece redes y una trayectoria por comparador determinista. El LSTM-autoencoder obtuvo la menor RMSE media observada, 0,00380343; las veinte comparaciones primarias no acreditaron la mejora práctica registrada tras Holm. Estos son antecedentes de E2, no resultados de E3.

## Objetivo y productos de E3

La guía exige cerrar las correcciones de E2, exportar las mejores configuraciones, construir un modelo neuronal original con una propiedad formal demostrada y verificada, evaluarlo rigurosamente y entregar un dashboard en Dash Plotly desplegado en Render junto con el Jupyter Book reproducible.

El candidato temático continúa siendo CNN causal + LSTM unidireccional + atención temporal. La combinación de bloques no acredita por sí sola originalidad. El [diseño preliminar](sections/07_diseno_preliminar.md) propone una línea de investigación y conserva abierta su admisibilidad.

## Primer avance

| Documento | Para qué sirve |
|---|---|
| [Bloques A y B](sections/08_bloques_a_b.md) | Auditoría de artefactos, selección de configuraciones y exportación con evidencia. |
| [Alcance y matriz de requisitos](sections/01_alcance_requisitos.md) | Relaciona cada exigencia de la guía con evidencia y criterio de cierre. |
| [Plan de trabajo](sections/02_planificacion.md) | Ordena tareas y dependencias antes de entrenar. |
| [Trazabilidad de correcciones](sections/06_trazabilidad_correcciones.md) | Registra pendientes de la auditoría interna; no se han recibido observaciones docentes. |
| [Diseño preliminar del modelo](sections/07_diseno_preliminar.md) | Presenta hipótesis, propiedad candidata y pruebas necesarias. |
| [Desarrollo y evidencias](sections/03_desarrollo_evidencias.md) | Distingue código reutilizable de productos nuevos. |
| [Resultados y discusión](sections/04_resultados_discusion.md) | Conserva el estado sin resultados propios. |
| [Reproducibilidad y entrega](sections/05_reproducibilidad_entrega.md) | Fija procedencia, verificaciones y estructura final prevista. |

El inventario de artefactos y la revisión de configuraciones ya tienen evidencia ejecutable en los bloques A y B. El cierre de los controles de datos pendientes sigue condicionando el congelamiento definitivo de E3. Los periodos ya evaluados en E1/E2 siguen siendo conocidos; repetirlos no crea una confirmación independiente.

## Libros del curso

- [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html).
- [Entregable 2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html).
- [Selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html).
