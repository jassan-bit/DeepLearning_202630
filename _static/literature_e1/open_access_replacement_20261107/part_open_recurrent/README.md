# Lectura recurrente y comparadores simples, 7 de noviembre de 2026

Cinco fichas de lectura primaria pública. Tang ofrece la receta LSTM pequeña más concreta encontrada; Dudek2024 aporta un contraste directo donde MLP/HAR superan LSTM según la pérdida. La recuperación de Brauneis/Sahiner completa E1X001. Dudek2025 añade contexto probabilístico con los mismos pronósticos base de 2024. Sun confirma GRU autónomo pero tiene problemas que impiden usar sus efectos o su receta.

| ID | Citekey | Acceso y versión | Uso permitido por la evidencia |
|---|---|---|---|
| E1OA-R01 | Tang2024RoughVolatility | Autor, arXiv:2311.04727v2; editorial final de suscripción sin leer | LSTM autónomo,1 capa,2/4 unidades,SiLU,Adam0.001,ventanas7/30. Faltan epochs/batch y CV detallada. Diez semillas forman un ensemble. |
| E1OA-R02 | Dudek2024VolatilityComparative | PDF público maquetado en página del coautor | Comparación RV cripto,LSTM/MLP/HAR y otros. Anchos finales y entrenamiento LSTM ausentes. MCS no implica equivalencia. |
| E1OA-R03 | Sun2025VMDGRU | Editorial,CC BY-NC 4.0 | Sólo contexto de GRU/LSTM autónomos. Sin split/receta; inconsistencias MSE–RMSE,DM y texto ajeno al mercado cripto. |
| E1OA-R04 | Dudek2025ProbabilisticVolatility | Autor,arXiv:2508.15922v1,CC BY-NC-ND 4.0 | Contexto de cuantiles; no receta recurrente nueva ni réplica independiente de R02. Versión IEEE no leída. |
| E1X001 | Brauneis2026VolatilityHorserace | Editorial,CC BY 4.0,HTML/PDF | Actualización de fuente complementaria previa: cuerpo y Tablas 4–8 completos. BTC HAR tiene menor MAE/RMSE; faltan hiperparámetros neuronales seleccionados. |

`access_full` registra que el cuerpo completo se pudo leer. `oa_license_verified` registra una licencia abierta explícita de la versión leída. Son hechos distintos: el manuscrito Tang tiene autorización de distribución arXiv, sin licencia CC verificada, y el PDF del coautor Dudek2024 declara reserva de derechos. Su disponibilidad pública permite auditarlo; no autoriza redistribuirlo. No se guardaron PDFs, textos completos, abstracts ni contactos.

`full_text_read` se refiere al cuerpo de la versión identificada y las tablas/localizadores declarados; no convierte el DOI editorial de un preprint en versión final leída. Las celdas se extrajeron del texto primario parseado. El servicio de screenshots no devolvió imágenes inspeccionables; no se afirma cotejo visual. Campos `null` indican información no localizada, no valores supuestos.

Se cotejaron DOI/título contra los 182 registros E1 y los siete estudios históricos N7 (S1/S2/S5/S6/S7/S8/S9). Ninguna ficha pertenece a esos grupos. R02 sí estaba en E2 como WOS:001137349000001 con lectura de abstract y texto completo pendiente: es una recuperación nueva, no una referencia global nueva. E1X001 conserva su identidad previa. R01/R03/R04 no tuvieron coincidencia exacta previa en `references`/`book`. Las huellas de los archivos consultados quedan en `search_log.json`.

R01, R02 y X001 pueden alimentar el estrato complementario de comparación puntual cripto (`include_in_new_stratum=true`), con sus límites. R04 queda como contexto probabilístico y R03 como contexto con calidad insuficiente. Ninguno aporta receta exacta reproducible de E1: difieren targets, escalas, horizontes o faltan ajustes. No se modifica N7, no se recalculan sus frecuencias y los 14 originales inaccesibles conservan su estado. E1X003 Zhou/EMGNN ya tenía lectura primaria completa en `internet_closure_20261106/part_d`; no se vuelve a contar. Su LSTM es autónomo; su módulo temporal no es una TCN autónoma.

Archivos publicables de esta entrega: `reviews.json`, `search_log.json`, `README.md`, `references.bib`. Son metadatos, paráfrasis breves, hechos y valores seleccionados con localizadores; no constituyen anexos de texto fuente. Las búsquedas fueron dirigidas en web pública, no una nueva ejecución nativa Scopus/WoS ni una revisión sistemática exhaustiva. No se entrenaron modelos ni se reprodujeron resultados.
