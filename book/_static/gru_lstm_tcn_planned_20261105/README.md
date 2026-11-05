# Fase principal D7 planificada: GRU–LSTM–TCN

Esta fase se registró antes de su HPO, piloto, dimensionamiento y evaluación sobre una reserva histórica adicional. El primer benchmark conserva sus resultados y alcance exploratorio. La reserva nueva no constituye otro mercado ni predicción en vivo.

| Elemento | Evidencia ejecutada |
|---|---|
| HPO | Ocho candidatos y dos semillas por familia; diez meses; 960 ajustes nuevos |
| Piloto | Diez semillas nuevas por receta; 60 días de desarrollo; 60 ajustes nuevos |
| Dimensionamiento | R=465 por familia; potencia marginal objetivo0,80 bajo diferencias normales y DE conservadora |
| Ajuste definitivo | 1395 modelos nuevos; 2790 ajustes inner/refit; pesos congelados antes del acceso de precios |
| Reserva | 01/09–30/10/2026; 60 fechas; 83700 predicciones |
| Baseline fijado con HPO | GRU/D7; no se reelige con los resultados de reserva |

`chronology.json` verifica el orden real de registros. `dimensioning.json` conserva DE, incertidumbre, potencia de planificación y sensibilidades. Esa potencia corresponde a aleatoriedad de entrenamiento condicionada al calendario; no acredita potencia temporal80 %.

Las tres comparaciones primarias usan RMSE pareada por semilla y Holm3. DM/HAC6/HLN y el bootstrap temporal de bloques7, con sensibilidades3/14, son análisis secundarios con seis pares fijos. Significación frente a cero y una mejora práctica superior al5 % son afirmaciones distintas.

Los archivos públicos contienen sólo agregados, metadatos y figuras. `predictions_fixed_seed.png` usa la semilla 8101 fijada antes del test; no se entrega su CSV individual. No se publican brutos, pesos, escaladores ni vectores diarios. La adquisición había inspeccionado timestamps/esquema de los archivos; no se afirma que los ZIP nunca se hubieran abierto.

El adaptador de exportación es posterior a la ejecución; las firmas del código científico registrado permanecen separadas.
