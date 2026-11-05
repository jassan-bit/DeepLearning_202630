# Síntesis del nuevo trío

Baseline neural fijado en CV: **GRU/D7**, 3777 parámetros.

RMSE CV: **0.005154568533 ± 0.0001615570771**. RMSE del test histórico ya observado: **0.003775111281 ± 0.000194100703**.

Estas cifras son medias y desviaciones estándar entre diez semillas. Los IC del paquete son condicionales al calendario, no incertidumbre de generalización a futuros mercados.

Los tres contrastes D7 por fase usan t pareada y Holm3; los doce contrastes de todas las entradas usan Holm12 como sensibilidad. Ambos análisis se formularon después del test. Se conserva cada comparación, incluso si no es significativa. Ausencia de significancia no significa equivalencia.

BDS y Ljung–Box se calculan por semilla e intervalo. Sus frecuencias de rechazo y medias de p-valores son descriptivas: no son pruebas combinadas. Un rechazo BDS no identifica por sí solo no linealidad explotable. Los bloques mensuales cortos limitan su calibración.

Diebold–Mariano compara pérdidas cuadradas diarias promediadas sobre semillas, con HAC Bartlett de seis rezagos por el solapamiento σ7 y corrección HLN para h=1. Su estimando es diferente del promedio de RMSE anual por semilla. Ninguno elimina la observación previa del test.

Margen académico futuro: **δ = 0.00025772842665**, el 5 % del RMSE CV de GRU/D7. Se declara después del test para nuevos estudios; no es un margen previo del experimento actual ni una afirmación de utilidad financiera. Se conserva el margen original únicamente como lectura histórica.

Los futuros modelos deben contrastarse tanto con GRU como con persistencia y declarar sus propios datos, comparaciones y piloto antes de la evaluación. Las ventajas pueden diferir entre RMSE, MAE y MAPE; las tablas publican las seis métricas.

El entrenamiento inicial conserva 2023; la selección usa forward chaining en 2024; el test comprende enero de 2025–agosto de 2026. Septiembre–octubre de 2026 permanece sin asignar.
