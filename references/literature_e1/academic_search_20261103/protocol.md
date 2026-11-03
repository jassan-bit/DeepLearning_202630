# Protocolo de búsqueda académica — 3 de noviembre de 2026

Estado: cadenas preparadas; búsqueda bibliográfica no ejecutada de forma verificable. Los cuatro intentos actuales fueron bloqueados por el proxy del entorno antes de alcanzar las fuentes. Total de resultados y registros recuperados por cada búsqueda: desconocidos (`null`), no cero.

## Pregunta y alcance

¿Qué arquitecturas de Deep Learning relativamente sencillas se utilizan como comparadores independientes para pronosticar volatilidad o varianza de Bitcoin? El objetivo propio es diario y usa OHLCV, pero las diferencias de mercado, frecuencia, etiqueta e información de los artículos se extraen y critican, sin equipararlos al dataset del proyecto.

Periodo preferente preparado: 01/01/2019–03/11/2026, ocho años calendario. Los fundamentos anteriores se conservan como contexto justificado. En arXiv se filtra la fecha de envío, que no equivale a año editorial; registrar también las fechas de publicación o disponibilidad que declare cada fuente.

B1 es la consulta general con términos DL. B2 es una sensibilidad que no exige DL en los metadatos para recuperar estudios cuyos comparadores se describen solo en el texto. Las cuatro consultas F por familia son complementarias y opcionales: no se utilizan para estimar prevalencia global. Q1 se reserva para el cuartil de revista, que requiere verificación independiente por índice, revista, año y categoría.

No se excluyen automáticamente palabras como price, hybrid o GARCH: el cribado debe verificar si el trabajo realmente evalúa volatilidad y contiene una referencia DL independiente. No se decide de antemano que las tres familias candidatas serán las tres definitivas.

## Cadenas nativas completas

Son sintaxis candidatas: la aceptación de la cadena, la colección y los filtros aplicados solo se acreditarán con una respuesta de búsqueda de la plataforma. Preparar una cadena o recuperar una página de entrada no acredita ejecución.

### Web of Science

**B1 — general_DL**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=("deep learning" OR "neural network" OR "neural networks")
AND PY=(2019-2026)
```

**B2 — sensitivity_without_DL_requirement**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND PY=(2019-2026)
```

**F_feedforward — targeted_family_optional**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=(MLP OR "multilayer perceptron" OR feedforward OR "feed-forward")
AND PY=(2019-2026)
```

**F_recurrent — targeted_family_optional**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=(RNN OR LSTM OR GRU OR BiLSTM)
AND PY=(2019-2026)
```

**F_convolutional — targeted_family_optional**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=(CNN OR TCN OR "temporal convolutional" OR "convolutional neural")
AND PY=(2019-2026)
```

**F_attention — targeted_family_optional**

Campos: TS: Topic (database-defined title, abstract and keyword fields). Filtro solicitado: PY=(2019-2026).

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=(Transformer OR attention OR Synthesizer)
AND PY=(2019-2026)
```

### Scopus

**B1 — general_DL**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY("deep learning" OR "neural network" OR "neural networks")
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**B2 — sensitivity_without_DL_requirement**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**F_feedforward — targeted_family_optional**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY(MLP OR "multilayer perceptron" OR feedforward OR "feed-forward")
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**F_recurrent — targeted_family_optional**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY(RNN OR LSTM OR GRU OR BiLSTM)
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**F_convolutional — targeted_family_optional**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY(CNN OR TCN OR "temporal convolutional" OR "convolutional neural")
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**F_attention — targeted_family_optional**

Campos: TITLE-ABS-KEY. Filtro solicitado: PUBYEAR > 2018 AND PUBYEAR < 2027.

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY(Transformer OR attention OR Synthesizer)
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

### IEEE Xplore

**B1 — general_DL**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":"deep learning" OR "All Metadata":"neural network" OR "All Metadata":"neural networks")
```

**B2 — sensitivity_without_DL_requirement**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
```

**F_feedforward — targeted_family_optional**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":MLP OR "All Metadata":"multilayer perceptron" OR "All Metadata":feedforward OR "All Metadata":"feed-forward")
```

**F_recurrent — targeted_family_optional**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":RNN OR "All Metadata":LSTM OR "All Metadata":GRU OR "All Metadata":BiLSTM)
```

**F_convolutional — targeted_family_optional**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":CNN OR "All Metadata":TCN OR "All Metadata":"temporal convolutional" OR "All Metadata":"convolutional neural")
```

**F_attention — targeted_family_optional**

Campos: All Metadata, Command Search. Filtro solicitado: Publication Year: 2019–2026, applied in source filters and recorded separately.

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":Transformer OR "All Metadata":attention OR "All Metadata":Synthesizer)
```

### arXiv

**B1 — general_DL**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:"deep learning" OR abs:"deep learning" OR ti:"neural network" OR abs:"neural network" OR ti:"neural networks" OR abs:"neural networks")
AND submittedDate:[201901010000 TO 202611032359]
```

**B2 — sensitivity_without_DL_requirement**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND submittedDate:[201901010000 TO 202611032359]
```

**F_feedforward — targeted_family_optional**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:MLP OR abs:MLP OR ti:"multilayer perceptron" OR abs:"multilayer perceptron" OR ti:feedforward OR abs:feedforward OR ti:"feed-forward" OR abs:"feed-forward")
AND submittedDate:[201901010000 TO 202611032359]
```

**F_recurrent — targeted_family_optional**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:RNN OR abs:RNN OR ti:LSTM OR abs:LSTM OR ti:GRU OR abs:GRU OR ti:BiLSTM OR abs:BiLSTM)
AND submittedDate:[201901010000 TO 202611032359]
```

**F_convolutional — targeted_family_optional**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:CNN OR abs:CNN OR ti:TCN OR abs:TCN OR ti:"temporal convolutional" OR abs:"temporal convolutional" OR ti:"convolutional neural" OR abs:"convolutional neural")
AND submittedDate:[201901010000 TO 202611032359]
```

**F_attention — targeted_family_optional**

Campos: ti and abs, API/advanced search. Filtro solicitado: submittedDate:[201901010000 TO 202611032359].

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:Transformer OR abs:Transformer OR ti:attention OR abs:attention OR ti:Synthesizer OR abs:Synthesizer)
AND submittedDate:[201901010000 TO 202611032359]
```

## Registro y exportación

Guardar la cadena exacta, colección/campos, idioma y tipo documental, filtros solicitados frente a confirmados, fecha y zona horaria, orden, tamaño de página y regla de parada. Conservar exportaciones originales RIS/BibTeX/CSV/Atom con SHA-256 antes de deduplicar. Si hay límite de exportación, informar total declarado, registros realmente recuperados y cobertura parcial; no sumar totales solapados entre cadenas o fuentes.

La bitácora actual distingue GET intentado, cadena incluida o no en la URL, llegada a la plataforma y ejecución bibliográfica verificada. En el intento WoS solo se probó el portal de entrada: la cadena no llegó a enviarse al buscador. En las otras tres rutas la URL llevaba la consulta candidata, pero el túnel fue rechazado antes de que la plataforma la recibiera.

## Cribado, deduplicación y calidad

Incluir documentos identificables con Bitcoin, objetivo de volatilidad/varianza, uso comparativo de Deep Learning y arquitectura, protocolo y resultados verificables. Una propuesta híbrida puede contener comparadores independientes pertinentes. Un documento no recuperado queda pendiente; no se excluye por suponer que carece de un método o por un título ambiguo.

Deduplicar por DOI normalizado; identificador arXiv sin versión y vínculo comprobado preprint–artículo; en ausencia de identificadores, título/autores/año con revisión de coincidencias ambiguas. Un estudio puede originar varios registros y textos, pero cuenta como máximo una vez por familia comparadora. No duplicar artículo, preprint y código.

Evaluar claridad de arquitectura y protocolo, dataset/objetivo/horizonte, partición y disponibilidad, posible fuga, tuning, repeticiones, dispersión y contrastes. Distinguir lo declarado de lo verificado y de lo no reportado. La indexación no acredita automáticamente Q1 y arXiv no acredita revisión por pares. Mantener literatura gris identificada y sensibilidades sin ella cuando corresponda.

La selección corresponde al autor responsable, Jassan Arteta, con asistencia de Codex. No se acredita un segundo revisor humano ni consenso; registrar dudas y decisiones. El nuevo flujo académico quedará pendiente hasta disponer de exportaciones, deduplicación y cribado verificables.

## Estado y reanudación

La política activa solo incluye destinos de Binance y los dominios predefinidos del entorno; los cinco dominios académicos examinados en la política no están activadas. Se guardó una configuración de red pendiente, conservando los destinos previos y añadiendo export.arxiv.org para la API pública. Guardar el borrador no aplica la política a esta instancia.

Para habilitar las solicitudes, revisar y guardar los cambios en la configuración del entorno y publicarlo. Después se verificará la política activa y el acceso. La posible necesidad posterior de autenticación institucional, verificación de navegador o claves API no ha sido diagnosticada: no se ha alcanzado ninguna plataforma. No incluir claves, cookies o contraseñas en exportaciones públicas o en el chat.

El intento no modifica 35 referencias, cinco textos incluidos, 81 cifras de autores ni las frecuencias previas; el trío final sigue abierto. No se entrenaron modelos ni se abrieron datos de mercado o test.
