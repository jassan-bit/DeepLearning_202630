# Escenario retrospectivo P9

`preferred_intrapaper_scenario.csv` contiene nueve P numéricos, cada uno normalizado sólo en su conjunto de comparación. Inception-1D queda ausente por falta de una comparación numérica verificable para la red individual. El integrador puede expandirlo a diez modelos conservando ese valor nulo.

La regla se escribió a las 11:11:24 UTC en `preferred_intrapaper_scenario_rule.json` antes de generar la selección. `preferred_intrapaper_rule.json` es una copia exacta para compatibilidad. Los valores fuente ya eran visibles: es una regla retrospectiva explícita, sin prerregistro ni cegamiento. El recibo registra ambos tiempos y los hashes.

La selección forecast se verificó por código: ETTh1/H96/L96 y MSE; más comparadores primero; DOI fuente ascendente para desempates. No usa el valor de rendimiento para escoger tabla. El enlace documental de Table 23 al setting L96 unificado se detalla en la regla, con su limitación de verificación.

SAE usa RMSE conforme a la regla y obtiene 0 dentro de sus cuatro comparadores. MAE y SMAPE se guardan por separado como sensibilidad: ambos dan 1. No se promedian. El CSV mantiene vacíos P original y P contextual global; no se afirma un ranking global controlado y no se usó A2.
