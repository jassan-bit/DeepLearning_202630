# E1N057 — revisión íntegra del original suministrado

Poudel et al. (2023), *Cryptocurrency price and volatility predictions with machine learning*, Journal of Marketing Analytics 11, 642–660. DOI: [10.1057/s41270-023-00239-1](https://doi.org/10.1057/s41270-023-00239-1).

Se leyeron las 19 páginas del original suministrado mediante la extracción completa de 1075 líneas y se verificaron visualmente las tablas y figuras clave en el PDF. Acceso: `user_supplied_complete_original`. Esto acredita lectura completa, sin atribuir al archivo acceso abierto: el original informa derechos exclusivos y no se verificó una licencia abierta. SHA256 del PDF: `dabfe50b1a7f5684783902aa879e7e336a3f32e2cfbd2cfe54587867fcc4eee6`. Solo se guardan análisis derivados, metadatos y hash; no se incorpora el original ni su texto o resumen.

## Decisión para selección de modelos

El artículo compara **LSTM, Bi-LSTM y GRU como modelos autónomos**, con filas independientes en Table 2 (p. 652) y figuras propias. No utiliza TCN. Las menciones generales a NN/RNN no prueban un modelo autónomo adicional. GARCH, TARCH y VAR son referencias estadísticas, sin naturaleza de Deep Learning.

**Conservar como evidencia contextual y original completo revisado; excluir del denominador estricto de estudios elegibles para extraer baselines reproducibles.** La guía §3.1 excluye fuentes sin suficiente arquitectura o protocolo. Solo GRU tiene dos hiperparámetros concretos —800 épocas y lote 50, pp. 652–654—; no informa capas, unidades, ventana, regularización, pérdida ni optimizador. El artículo tampoco acredita una evaluación cronológica libre de fuga. El registro sí pertenece al denominador de originales completos evaluados, aunque no al de incluidos elegibles.

La presencia contextual por familia normalizada es LSTM=1, GRU=1, TCN=0; LSTM y Bi-LSTM no crean dos votos de la misma familia en un artículo. En el denominador estricto, este artículo aporta cero. Una arquitectura concreta basada en estas familias requiere una fuente suficiente adicional o la búsqueda de arquitectura de Entregable 2 §14 que prescribe la guía §3.2. No debe presentarse una implementación propia como reproducción del autor.

## ¿Pronostica volatilidad continua o precio?

Hay salidas continuas y análisis de volatilidad, pero **no se verifica que los resultados DL de Table 2 correspondan a volatilidad realizada o a σ7**:

- P. 646 declara como objetivo la columna `Change %` del día anterior; Table 1 p. 647 la describe como variación porcentual del precio.
- Pp. 645 y 649–651 denominan volatilidad a un índice Google Trends 0–100, comparado manualmente con precios y usado para ajustar regresiones.
- Figs. 13–17 pp. 654–655 muestran precio de cierre para GRU, LSTM y Bi-LSTM. Table 2 denomina su tarea «Bitcoin price volatility prediction», sin aclarar cómo se reconcilian estos objetivos y escalas.
- Fig. 18 p. 656 **sí presenta volatilidad realizada**, con ventanas 7/30/60/180/365 días según la leyenda. Es una figura descriptiva: no define la fórmula ni vincula esas series con el entrenamiento y las métricas DL. La ventana de 7 días no demuestra un horizonte de pronóstico de 7 días ni equivalencia con σ7.
- Fig. 20 p. 657 presenta pronóstico TARCH de volatilidad sobre validación; su título informa TARCH(1,2,0), media constante, distribución t de Student asimétrica y bootstrap. No define la construcción del objetivo ni su horizonte.

El horizonte concreto de LSTM/GRU no está especificado. La explicación genérica de autoregresión al siguiente paso, p. 650, no establece el protocolo de esas redes.

## Datos y protocolo

Se reportan datos diarios Bitcoin/DogeCoin y Google Trends. P. 645 declara historia Bitcoin agosto 2013–marzo 2023, nueve años de predicción marzo 2014–marzo 2023 y foco visual septiembre 2021–marzo 2023. Sin embargo, los ejes DogeCoin en Figs. 14, 16 y 19 alcanzan abril de 2023. No se da proveedor de mercado preciso, descarga reproducible, número de observaciones, versión ni cortes de partición.

P. 645 describe **diez folds aleatorios**, probando cada parte contra las otras nueve y mencionando un fold mediano para modelar. No constituye walk-forward. Figs. 11–12 hablan de train/test, pero no dan límites o proporciones que resuelvan la relación con esa CV. La aleatorización permite entrenar con información posterior a otras observaciones evaluadas. La normalización anual y ajustes manuales tampoco tienen ámbito de ajuste especificado. La bidireccionalidad por sí sola no demuestra fuga; faltan la ventana y el corte de información para evaluarla.

No se informan semillas, repeticiones independientes de entrenamiento, dispersión por fold, intervalos, pruebas estadísticas, búsqueda de hiperparámetros reproducible ni repositorio de código/datos. No se reprodujeron los experimentos del autor.

## Métricas exactas

Table 2, p. 652 (página 11 del PDF), cifras del autor sin recálculo:

| Modelo | MSE | RMSE | MAE | RMSPE |
|---|---:|---:|---:|---:|
| Linear regression | 14,914,797 | 3,861 | 2,963 | 0.1033 |
| LSTM | 2,845,040 | 1,686 | 1,191 | 0.0535 |
| Bi-LSTM | 3,425,114 | 1,850 | 1,403 | 0.0506 |
| GRU | 2,003,410 | 1,415 | 1,030 | 0.0418 |
| TARCH, impreso «TRACH» | 384,400 | 620 | 603 | 0.2009 |
| VAR | 613,715 | 783.4 | 253.4 | 3.2254 |

GRU tiene menor RMSPE; TARCH tiene menor MSE/RMSE y VAR menor MAE. No se demuestra superioridad general de GRU ni significancia estadística. La p. 642 usa RMSPE GRU 0.0468, incompatible con 0.0418 de Table 2 y p. 650. Se registra la discrepancia sin corregir ni promediar las cifras. P. 655 afirma 94.67% de «accuracy» para GARCH/DogeCoin, sin definición suficiente para usarlo como error de regresión o exactitud de clasificación.

La falta de definición operacional de RMSPE, escala coherente del objetivo y protocolo común impide trasladar estos valores a un benchmark de σ7. Otros problemas verificables son el rango de fechas de Table 1 y una fila donde Low supera High (p. 647), así como el pie de Fig. 18 que limita ventanas a 7–60 días aunque su leyenda incluya 180 y 365. El detalle estructurado y sus localizadores se conservan en [review.json](review.json); la referencia bibliográfica está en [reference.bib](reference.bib).
