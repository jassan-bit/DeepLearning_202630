# Plan posterior de pantalla univariada

Se fija antes de calcular las 35 regresiones con datos reales. Cada OLS con intercepto usa sólo train655; los mismos coeficientes puntúan train/cal53/test608. R²≥0,95 en una cohorte activa revisión, sin eliminar atributos ni ajustar el benchmark. Se conservan R² negativos, coordenadas constantes y R² indefinidos por objetivo constante. Las coordenadas usan barras t−7…t−1; OHLC log-relativo requiere Close(t−1), y volumen log1p usa su propia barra pasada.

La ejecución requiere QA ligada al SHA exacto de plan y código, y se cobra mediante el wrapper de presupuesto existente con un límite de 30 segundos. Preparar este archivo y el plan no ejecutó cálculos de regresión. El diseño es posterior al benchmark y al conocimiento del calendario heredado; no restaura preregistro histórico ni ceguera.
