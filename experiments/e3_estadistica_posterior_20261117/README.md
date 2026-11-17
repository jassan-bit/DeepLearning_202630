# Análisis estadístico posterior de E3 — 17 de noviembre de 2026

Este registro se fija después de conocer las predicciones y las métricas medias de E2 y E3, antes de calcular los p-valores de esta etapa. Es un plan de análisis retrospectivo y exploratorio. No es un preregistro confirmatorio y no restaura el cegamiento del calendario de prueba.

## Insumos y estimando

Se conservan los dieciséis rivales: los diez modelos principales de E2, MLP16, LSTM32, GRU32 y las referencias Persistence, SeasonalNaive y ARIMA. Cada red tiene diez semillas, 271801–271810; cada referencia determinista tiene una sola trayectoria. La semilla histórica de E3, 20261116, queda fuera de esta comparación.

Las 608 fechas comunes abarcan del 1 de enero de 2025 al 31 de agosto de 2026. Las fechas, los objetivos, las máscaras y las métricas guardadas deben coincidir antes del análisis. Se verifica la conservación de los 81 predictores originales de E2 dentro de su ampliación a 133. Los once insumos, sus commits y sus SHA256 están fijados en `plan.json`.

La métrica primaria es la media de las RMSE individuales por semilla, en unidades originales de sigma7 poblacional, sin anualización. No se transforma en RMSE de un ensemble ni en raíz del MSE agregado. La ganancia es la métrica del rival menos la del candidato; positivo favorece a E3. Se conserva el margen heredado delta=0.00025772842664981106 del borrador de Bloque D, sin ajustarlo al resultado.

## Procedimientos fijados antes de estos p-valores

1. Bootstrap temporal circular pareado de bloques de catorce días, con 9999 réplicas y semilla 2026111701. Cada réplica usa las mismas fechas remuestreadas para todos los modelos, conservando enteras las trayectorias e inicializaciones ajustadas. Se recalcula la media de RMSE individuales. Para la nula ganancia<=delta, el p unilateral compara el ruido bootstrap centrado con la distancia observada al margen y utiliza la corrección Monte Carlo de uno en numerador y denominador. Los intervalos básicos del 95% son marginales y reflejan incertidumbre temporal condicionada a los modelos ajustados.
2. Aproximación de la media de RMSE mediante su función de influencia y varianza Bartlett-HAC, con catorce rezagos. Se deriva cada raíz antes de promediar entre semillas. El estadístico contrasta la ganancia de RMSE menos delta, en las mismas unidades. No se denomina DM–HLN ni se inserta delta de RMSE directamente en una pérdida cuadrática.
3. t pareada sobre las diez ganancias por identificador de semilla, con nueve grados de libertad, condicionada al calendario fijo. Para un clásico se conserva su única RMSE y se compara con las diez inicializaciones de E3; no se inventan diez ajustes clásicos. Este cálculo asume inicializaciones independientes y ganancias aproximadamente normales. Shapiro se reporta como diagnóstico, sin elegir otro procedimiento o retirar rivales según su resultado.

Se aplica Holm una vez a los 48 contrastes principales (16 rivales por 3 procedimientos), con alfa=0.05. Este control es más amplio que las tres familias de dieciséis propuestas en el borrador y no depende de independencia entre procedimientos. Los p no estimables se conservan en la familia usando uno para el ajuste y se identifican como no estimables, no como ausencia de diferencia.

Los tres cálculos tienen ámbitos de incertidumbre diferentes. No constituyen por su sola presencia tres confirmaciones temporales homogéneas ni una batería acreditada de la Tabla 4 de la guía. La t entre semillas no mide incertidumbre entre mercados y el bootstrap/HAC de una serie no incorpora la selección previa del diseño.

## Sensibilidad, tamaños de efecto y análisis auxiliares

Los bloques y rezagos siete y veintiocho se fijan como sensibilidades. Cada configuración conserva su familia completa y se reporta junto a la principal; no se reemplaza la conclusión por la configuración de menor p. Se informan la ganancia absoluta y relativa, el intervalo temporal y Cohen dz entre semillas con intervalo bootstrap nominal. La dispersión entre semillas no se presenta como intervalo predictivo.

MAE es la métrica secundaria. Su familia separada incluye dieciséis contrastes bootstrap temporales frente a cero con Holm16; no se inventa un margen práctico de MAE. Las primeras cinco y las últimas cinco semillas se examinan descriptivamente como subconjuntos prefijados.

El MCS auxiliar usa Tmax, alfa=0.10, 4999 réplicas, bloques catorce y semilla 2026111702. Su pérdida es el MSE diario medio entre semillas: es otro estimando, distinto de la media de RMSE. Incluye los diecisiete sistemas desde el inicio. La pertenencia al conjunto no demuestra equivalencia ni una ganancia mayor que delta. No se aplica Friedman usando semillas como si fueran conjuntos de datos independientes.

## Interpretación y reproducción

Los métodos temporales requieren una aproximación razonable de dependencia débil/estacionariedad dentro de este calendario conocido. E2 se entrenó con 655 ejemplos y E3 con 532, y sus reglas de selección no son iguales. El análisis compara las recetas ejecutadas; no aísla un efecto causal de arquitectura ni acredita igualdad de presupuesto.

No se estima potencia prospectiva, potencia conjunta ni diferencia mínima detectable a partir del propio test y del efecto observado. Esos productos requieren un diseño y un piloto adecuados. Tampoco se ejecutan conjuntos adicionales, nuevos entrenamientos o HPO. R1, la defensa humana de R2 y R3 conservan sus requisitos pendientes.

El ejecutor verificará fuentes, registrará versiones, semillas, plan y código, y producirá tablas y figuras reproducibles. Se ejecuta en CPU con NumPy2.3.5 y SciPy1.17.0. Esta etapa conserva separados los consumos cerrados de E2 y E3; no reinicia esos contadores.
