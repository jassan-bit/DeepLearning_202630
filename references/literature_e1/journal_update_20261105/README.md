# Actualización de respaldo metodológico y procedencia editorial

Se mantienen **seis textos completos del corpus específico de Bitcoin** y sus frecuencias: MLP independiente 1/6, CNN genérica independiente 1/6 y CNN1D simple explícita 0/6. El catálogo bibliográfico pasa de 36 a 39 referencias; esto no convierte el corpus BTC en nueve estudios.

## Tres niveles de evidencia diferentes

| Nivel | Referencia | Material realmente consultado | Qué se puede sostener |
|---|---|---|---|
| T1 | Bucci (2020), Journal of Financial Econometrics, DOI10.1093/jjfinec/nbaa008 | Artículo editorial completo | FNN/FNNX independientes, una capa oculta, para volatilidad financiera S&P 500 mensual; valores exactos de Tablas 4/5. |
| T2 | Reisenhofer, Bayer y Hautsch (2022), HARNet, arXiv:2205.07719 | Código primario de autores y README | Conv1D causal, inicialización HAR y configuración SPX/rv5_ss verificadas en código. Original, revista y métricas publicadas pendientes. |
| T3 | Moreno-Pino y Zohren (2024), DeepVol, Quantitative Finance, DOI10.1080/14697688.2024.2387222 | Entrada de publicación y enlace DOI en CV primario del autor | Identidad editorial y DOI; sin arquitectura, comparadores o resultados atribuidos al original no recuperado. |

Los tres aportes son de **respaldo metodológico**, con distinto alcance. Se añade un texto editorial completo nuevo, una fuente de código de autores y un comprobante bibliográfico primario en CV. No se describen como tres artículos íntegros leídos ni como aplicaciones nuevas de nuestras tres recetas a Bitcoin.

## Archivos

- `methodological_sources.json` y `methodological_comparison.csv`: tres filas T1/T2/T3 con la comparación de fuentes y sus límites.
- `bucci2020_primary_extraction.json`: identidad, datos, ecuación, arquitectura, evaluación y límites del original.
- `bucci2020_metrics.csv`: **24 filas largas y 48 valores de pérdidas MSE/QLIKE** de las Tablas 4/5; copia exacta de la transcripción contrastada visualmente.
- `harnet2022_primary_code.json`: configuración verificada, URLs y seis hashes de archivos primarios, sin copiar el código completo.
- `deepvol2024_publication.json`: comprobante en CV, DOI y campos originales pendientes.
- `journal_index_matches.json/.csv`: siete revistas, separando cinco comprobaciones históricas y dos adicionales. Todo es evidencia secundaria de revista, no de artículo ni cuartil.
- `editorial_candidates.json`: Huang, D’Amato y Wang pendientes; PreBit excluido por predecir movimientos extremos de precio. No se añaden citas nuevas para estos candidatos.
- `search_and_access_log.json`: procedencia y hashes de las evidencias integradas, sin fingir búsquedas nativas en plataformas no consultadas.
- `references_access_39.json/.csv/.bib`: los 36 registros históricos más los tres nuevos, con niveles explícitos. Los catálogos35/36 no se modifican.
- `manifest.json`: inventario SHA-256 de esta actualización pública.

## Límites de interpretación

Bucci usa logaritmo de volatilidad mensual y datos S&P 500, no sigma7 diaria. Sus pérdidas son promedios sobre 245/22 observaciones fuera de muestra; **no son media ± DE de 300 semillas del benchmark**. Las 300 inicializaciones corresponden al ajuste/selección de la fuente. El original conserva ambigüedades sobre retardos exógenos, activaciones empíricas y detalles de evaluación: no se armonizan retrospectivamente.

HARNet carga `rv5_ss` directamente; el cargador no aplica raíz cuadrada. Para retardos[1,5,20], sus fórmulas dan kernels5/4 y dilataciones1/5 con padding causal, pero son hechos del código. `label_length=5` es un bloque de etiquetas, no un horizonte de cinco pasos acreditado. No se atribuyen resultados numéricos publicados ni equivalencia con nuestra CNN de 16 canales.

DeepVol se registra como publicación corroborada en CV. No se reconstruyen sus hiperparámetros desde el título, un notebook educativo o una reconstrucción generada automáticamente.

Las siete coincidencias de revistas proceden de un mirror independiente de listas WoS/Scopus. Se conservan títulos, ISSN, filas, fuentes y hashes; **no se verifican registros individuales, Q1 ni cobertura histórica WoS por año editorial**. Elsevier es una editorial; WoS y Scopus son índices.

Esta carpeta contiene hechos bibliográficos, paráfrasis, cifras publicadas y comprobantes. **No redistribuye PDF, texto completo de artículos, CV completo, notebooks, código íntegro de terceros, datos de mercado ni resultados privados del proyecto.**
