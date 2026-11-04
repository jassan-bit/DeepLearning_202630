# Alineación de los modelos con la guía 6.4 — 4 de noviembre de 2026

La propuesta del proyecto es **MLP, LSTM y CNN1D poco profunda**, con GRU como alternativa para el puesto recurrente. Procede de la guía de regresión temporal aportada por el usuario. Es una propuesta de familias; no congela arquitecturas ni acredita extracción aplicada ya completada.

Los perfiles previos de LSTM, TCN y encoder Transformer son antecedentes recuperados de la literatura. Se conservan sus 27 campos, ecuaciones, diagramas, valores publicados, hiperparámetros fuente y limitaciones. No se renombran TCN o encoder como si fueran los modelos nuevos. La TCN ganadora es ancha y de seis niveles, por lo que no demuestra una CNN1D simple.

Se mantienen 35 referencias y cuatro columnas bibliográficas, cinco textos completos, 81 cifras de autores y todas las frecuencias. MLP independiente y CNN1D simple siguen en 0/5 dentro del corpus accesible: faltan referencias aplicadas y sus fichas. LSTM unidireccional tiene tres apoyos de cinco y una variante pequeña identificada; esto justifica proponer la opción recurrente sin probar superioridad predictiva frente a GRU.

El libro, los protocolos documentales, el registro vigente y las copias publicadas distinguen propuesta, perfiles de fuente y cierre experimental. El campo legacy `selected_families` de las fichas se aclara como listado de perfiles; la selección experimental autoritativa sigue vacía en `selection_evidence.json`. El hash de la dependencia vigente del contrato de benchmark se actualiza; los hashes de auditorías anteriores conservan su papel histórico.

No se realizaron nuevas búsquedas, entrenamientos, cálculos de rendimiento ni lecturas de datos de mercado/test. Se mantienen OHLCV, n=7, h=1, D=7/14/21/28, exclusión estricta, RMSE, el 5 % solo frente a persistencia y la ejecución reservada para E2. La alineación curricular no certifica el cumplimiento completo de literatura, implementación o benchmark.
