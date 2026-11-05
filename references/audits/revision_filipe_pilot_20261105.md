# Auditoría independiente del piloto Filipe

**PASS.** Se verificaron registro, código, cronología, cohortes, escaladores, curvas y hashes de checkpoints sin cargar modelos. Las métricas se reconstruyeron desde las predicciones de validación retenidas y el dimensionamiento se contrastó con statsmodels.

- 60 trayectorias; 720 segmentos; 1440 ajustes contabilizados.
- 21960 predicciones de validación; cinco semillas nuevas por combinación.
- R común con DE puntual: 302; recomendación con cota superior marginal de DE: 1680.

El piloto es prospectivo respecto de estas nuevas ejecuciones, sobre el calendario ya conocido de 2024. No repara retrospectivamente el estudio anterior ni acredita independencia temporal, potencia lograda o potencia conjunta. La publicación remota previa queda atestada por el recibo; se verificó localmente el commit que contiene el protocolo sellado. Los cinco registros históricos permanecen sin cambios. El auditor no ajustó redes, no cargó checkpoints y no abrió datos individuales del test anterior.
