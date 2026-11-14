(e2-reproducibilidad)=
# 8. Reproducibilidad de la entrega vigente

**Estado documental del 14/11/2026:** la tabla principal compara **16 sistemas sobre 608 fechas**. La reproducción registrada de sus **81 predictores** se completó en un checkout remoto y un entorno Python nuevos, con discrepancia máxima absoluta de predicción **0** en calibración y prueba. También se completaron **18 ablaciones**. Los recibos y sus límites se presentan en [§55](55_reproduccion_ablaciones.md). El Apéndice B vigente está en [§61](61_reproduccion_consolidada.md), con estados completos, parciales, no aplicables e incumplimientos identificados.

## 8.1 Punto de entrada de la entrega

Desde la raíz del repositorio, con Python 3.12 o posterior:

```bash
python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

Este comando funciona **sin red, GPU ni paquetes externos**. Verifica los manifiestos y archivos publicados, reconstruye las 81 RMSE a partir de los pronósticos congelados y las 16 medias de A2, valida calendario/semillas, comprueba los informes de 81 reproducciones y 18 ablaciones y el refinamiento de los veinte contrastes a 20.000 remuestreos. El informe registra el entorno de esta ejecución y sus discrepancias.

El comando ejecuta una auditoría aritmética y de integridad. **No vuelve a entrenar ni a generar pronósticos, no repite los sorteos bootstrap y no deserializa pesos.** La repetición de los 81 ajustes y las comparaciones tensoriales ya fueron ejecutadas por los workers A100 y están acreditadas separadamente en §55. La auditoría no declara aprobados los requisitos metodológicos pendientes.

## 8.2 Entorno y fuentes científicas

| Etapa | Entorno o fuente identificada |
|---|---|
| Ajuste final y reproducción neuronal | Python **3.12.12**, Torch **2.8.0+cu126**, CUDA **12.6**, NVIDIA **A100-SXM4-80GB** |
| Política registrada | FP32, algoritmos deterministas; AMP y TF32 desactivados; semillas explícitas |
| Motor científico | [`1e8745497b484518f766b6da5c19be8362d9add7`](https://github.com/jassan-bit/DeepLearning_202630/tree/1e8745497b484518f766b6da5c19be8362d9add7) |
| Resultados complementarios cerrados | [`d604ede6bf0978e14a1b1d7bafaae6250fb8ed07`](https://github.com/jassan-bit/DeepLearning_202630/tree/d604ede6bf0978e14a1b1d7bafaae6250fb8ed07) |
| Reproducción registrada | Nuevo checkout y venv sin paquetes del sistema; [recibo del entorno](../_static/e2_reproduction_ablations_20261113/historical_clean_runtime.json) |
| Persistencia | **GitHub**, con inventarios y recibos por intento; sin Google Drive |

No se promete igualdad numérica entre hardware o versiones distintos. Los recibos acreditan la reproducción efectivamente observada, condicionada a esas versiones y a la misma cronología. El [manifiesto de resultados](../_static/e2_reproduction_ablations_20261113/manifest.json) y la [procedencia por ajuste](../_static/e2_reproduction_ablations_20261113/source_evidence.json) enlazan las tablas con los resultados productores.

## 8.3 Reconstrucción y análisis de pronósticos guardados

Los [insumos congelados públicos](../_static/e2_final_audit_inputs_20261114/manifest.json) permiten repetir cálculos sin descargar precios nuevos. [§56](56_inferencia_estabilidad_final.md) incluye el comando completo para recalcular los veinte contrastes con 20.000 réplicas y la fluctuación Giacomini–Rossi. [§57](57_errores_techo.md) incluye el comando de la taxonomía y las referencias de dificultad. Sus resultados son posteriores y no reemplazan los originales ni modifican recetas o calibradores.

La [tabla A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) conserva su SHA256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. El auditor de entrega comprueba ese digest antes de reconstruir sus métricas. Las métricas son medias de RMSE individuales, no la RMSE de promediar pronósticos.

## 8.4 Reproducción del entrenamiento y presupuesto

La ejecución original y los 81 reajustes de reproducción están cerrados. El límite autorizado fue **84 horas acumuladas**, conservando todos los cargos previos; el diario final contiene **351 eventos**, **80,2561 horas cargadas** y ninguna reserva abierta. Es contabilidad de tiempo de campaña, no GPU activa, energía o dinero. El [balance cerrado](../_static/e2_reproduction_ablations_20261113/budget_final.json) conserva sus hashes.

Las instrucciones antiguas P0/OOF y sus comandos `--stage all` se archivan al final de esta página; corresponden a otros modelos y otro periodo. No se presentan como un comando para reproducir el benchmark actual ni se propone relanzar el controlador cerrado. Un replay futuro del entrenamiento requeriría una salida independiente y autorización presupuestaria propia, conservando los resultados anteriores.

## 8.5 Límites de la reproducción

La guía acepta reproducción por otra persona **o en un entorno limpio**. Los 81 reajustes acreditan la segunda vía; no crean un test nuevo. El periodo 2025–2026 ya se conocía desde E1. El margen δg previo, la potencia objetivo y otras condiciones del Apéndice B no quedan aprobados por una discrepancia numérica cero. La calibración y los análisis estadísticos requieren sus propios supuestos y evidencias.

El programa de auditoría se comprueba con pruebas de manipulación de hashes, calendarios, semillas, estimando y número de remuestreos:

```bash
python -m unittest tests.test_audit_e2_delivery -v
```


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/08_reproducibilidad.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
