# 4 Resultados y discusión

Todavía no hay resultados de desempeño del modelo original de E3. Los [bloques A y B](08_bloques_a_b.md) añaden verificaciones de artefactos, selección y paridad de exportación del benchmark heredado; esas mediciones no evalúan la propuesta neuronal nueva.

El [Bloque C](09_modelo_original.md) añade resultados de verificación de software y propiedad matemática sobre estados artificiales. Se distinguen de los resultados predictivos o confirmatorios del candidato, que aún no existen.

## Antecedentes de E2

La [tabla A2 de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/e2_benchmark_final_20261113/benchmark_A2.csv) reúne 16 sistemas y 608 fechas. El LSTM-autoencoder obtiene la menor RMSE media observada (0,00380343). Las veinte comparaciones primarias no acreditan una mejora mayor que δ tras Holm. El piloto de E2 tiene potencia condicional insuficiente para su efecto planificado; su cálculo sobre dos modelos no representa potencia de toda la familia de E3.

## Evidencia que deberá incorporarse

Los resultados nuevos identificarán protocolo, versión, dataset, partición y semilla. Se reportarán métricas individuales, incertidumbre temporal y entre semillas, tamaños de efecto, p crudos/ajustados y veredicto frente a δ. La conclusión de R3 debe considerar todas las pruebas exigidas, adecuación del modelo y consistencia adicional; mejorar una RMSE puntual o sólo la robustez no basta.

Las comparaciones sobre periodos ya conocidos se conservarán como históricas o exploratorias. No rechazo no prueba equivalencia; los resultados adversos y desviaciones se documentarán con su alcance real.
