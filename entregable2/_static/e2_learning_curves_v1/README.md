# Curvas guardadas de todos los modelos neuronales del benchmark

Esta galería presenta registros por época que ya existían, pero estaban disponibles principalmente en el ZIP de auditoría. Acompaña el punto 4 y se enlaza desde el diagnóstico F2a y el punto 5. La curva TimeMixer de ETTh1 pertenece a la réplica externa del paper guía; estas figuras corresponden al benchmark Bitcoin.

| Figura PNG/SVG | Contenido |
|---|---|
| `final_supervised` | Trece redes, seis semillas individuales por modelo: 78 curvas finales de entrenamiento. Se conservan las épocas fijadas previamente y los 655 ejemplos de train. |
| `hpo_2024-01-01` | Pares entrenamiento–validación de las recetas ganadoras en el primer origen: cinco curvas disponibles y ocho ausencias de caché identificadas. |
| `hpo_2024-07-01` | Segundo origen: once curvas disponibles y dos ausencias de caché. |
| `hpo_2024-10-01` | Tercer origen: las trece curvas de entrenamiento y validación disponibles. |
| `sae_auxiliary` | AE1 y AE2 separadas por origen HPO y ajuste final: ocho paneles y dieciocho historias auxiliares. |

Los tres orígenes son fechas de mercado del diseño, no fechas de ejecución del software. HPO utiliza una semilla, 271201, y se muestra por origen sin promediarlos. La galería contiene las trece recetas seleccionadas, no las 234 oportunidades completas de búsqueda. Los finales usan 271801–271806, con épocas fijas y sin otra curva de validación o test por época.

## Unidades e interpretación

Las curvas supervisadas muestran RMSE en unidades originales de σ₇. El riesgo se guardó al terminar cada época, con evaluación sin gradientes sobre todo train; no es la pérdida online de un minibatch. Los paneles usan rangos verticales propios y muestran las épocas completas. Sus alturas entre paneles no son un ranking de calidad.

El riesgo por época utiliza el batch configurado y la A2 usa pronósticos guardados con batch 1. Esas vías no son necesariamente idénticas, especialmente para TimesNet. La separación visual train–validación es descriptiva; no sustituye las pruebas formales de ajuste o sobreajuste.

AE1 y AE2 reconstruyen las mismas entradas escaladas con una cantidad calculada sólo con train; AE2 recibe la representación del primer encoder. Sus curvas son RMSE de reconstrucción en escala auxiliar, distintas del objetivo σ₇. Cada fase y cada origen permanecen separados. Las fases finales de 172 y 177 épocas no tienen validación por época.

Las diez pérdidas HPO no disponibles en caché no se interpolan ni se reconstruyen a partir de una métrica terminal. Los paneles conservan la mejor época nominal como metadato, sin inventar una línea. Persistence, SeasonalNaive y ARIMA son comparadores sin historias de épocas neuronales equivalentes; aparecen como no aplicables en la cobertura.

## Fuente, cobertura y reproducción

La fuente es `book_e2/_static/e2_f2a_audit_20261113/training_evidence.zip`, SHA-256 `daaf2cfcf050a37dbc992260011cfbb919a5d577c14f9003aaf6cf1c8be1900d`. Contiene 6.440 filas supervisadas, 3.250 auxiliares y metadatos de 117 ajustes. Se verifican hashes internos y vínculos con recetas y checkpoints congelados.

`coverage.csv` tiene 120 filas: 117 ajustes neuronales y tres comparadores no aplicables. `learning_curves.json` conserva las 18 historias auxiliares aparte, versiones, fuentes, controles, límites y hashes de los productos. `manifest.json` identifica este expediente y el script.

Desde un checkout completo, con Python 3.12, NumPy 2.3.5 y Matplotlib 3.10.8:

```bash
python scripts/plot_e2_benchmark_learning_curves.py --output /tmp/e2-curvas-guardadas
```

Elegir una ruta nueva; los archivos existentes sólo se aceptan si son idénticos. Los doce productos generados se reprodujeron byte por byte en el entorno registrado. La generación lee metadatos guardados: no carga pesos, entrena, ejecuta inferencia, calcula nuevas puntuaciones o pruebas, ni abre archivos brutos de mercado. Los resultados originales, las curvas fuente y la A2 permanecen intactos.
