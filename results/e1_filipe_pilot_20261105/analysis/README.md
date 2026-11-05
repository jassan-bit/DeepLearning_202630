# Piloto registrado y dimensionamiento de nuevas ejecuciones

Se ejecutaron cinco semillas nuevas para MLP, LSTM y CNN1D en D=7/14/21/28: 60 trayectorias anuales sobre las 366 fechas de validación 2024. Cada trayectoria agrupa los doce folds mensuales. Las fechas, las arquitecturas y el calendario ya eran conocidos; el registro es anterior a estos nuevos ajustes, no a la investigación histórica.

`pilot_metrics.csv` informa media, DE muestral, CV porcentual e IC t marginal del 95 % entre cinco inicializaciones. `paired_dimensioning.csv` contiene los 12 pares neuronales definidos antes del piloto: tres por historia. La aleatoriedad es de entrenamiento condicionada a un calendario; las fechas y los meses no son réplicas independientes.

El margen absoluto fijado para este nuevo piloto es δ=0.00023944847859645605, el 5 % del RMSE histórico de persistencia. Se detecta una diferencia de tamaño δ frente a cero mediante una referencia t bilateral de diferencias normales. Esto no prueba una mejora superior al 5 % ni es el criterio futuro de un modelo avanzado frente a dos referencias.

La planificación usa α global=0,05 y α por contraste=0,05/12, conservadora para una familia futura de Holm-12. La potencia objetivo es 0,80 por contraste, no la probabilidad conjunta de detectar todos los pares. El efecto medio observado solo se describe; no se utiliza para calcular potencia observada ni elegir el tamaño de efecto de planificación.

Se informa la aproximación normal de la guía y la inversión de t no central. El máximo común con DE puntual es R=302; con el límite superior unilateral marginal del 95 % de la DE bajo normalidad, R=1680. El segundo es la recomendación condicional común, si no está ausente. Los límites de DE son marginales, no un intervalo simultáneo para las 12 varianzas. Un resultado que exceda 10000 se declara sin recortarlo como tamaño suficiente. La MDE se informa para las cinco semillas y para los tamaños comunes calculados.

Cinco diferencias no acreditan normalidad. Shapiro–Wilk y los gráficos Q–Q son diagnósticos de baja resolución, no una garantía ni un mecanismo para escoger una prueba favorable. La incertidumbre de la dispersión piloto limita estas recomendaciones. Para Wilcoxon bilateral exacto, 9 pares sin ceros ni empates solo permiten aritméticamente alcanzar el primer umbral de Holm-12; no garantizan potencia.

El R propuesto corresponde a semillas definitivas nuevas, excluyendo las cinco del piloto. Este análisis no inicia esas ejecuciones ni consulta el test 2025–agosto de 2026. Tampoco reasigna septiembre–octubre de 2026, crea una evaluación independiente de mercado o subsana retroactivamente el piloto ausente del benchmark anterior. La evaluación temporal independiente futura permanece por definir.
