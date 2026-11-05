# Recuperación aplicada y comparabilidad — 4 de noviembre de 2026

## Resultado

Se ejecutaron **13 cadenas literales**, **18 páginas de búsqueda pública** y se registraron **113 ocurrencias/92 repositorios únicos** en este recorrido. Se inspeccionaron nueve raíces/README y siete archivos de código o notebook, sin ejecutar su código. No se recuperó un nuevo manuscrito científico completo: **el corpus sigue en cinco estudios y 35 referencias**. Los repositorios no se cuentan como artículos.

Se ampliaron páginas 2–4 de dos cadenas enfocadas en CNN; el total indexado por GitHub no equivale al número de artículos ni garantiza que todas las coincidencias sean recuperables. No se realizaron consultas nativas WoS, Scopus, Xplore o arXiv ni se acreditó Q1.

## Pistas verificadas, no nuevas inclusiones

| Candidato | Evidencia comprobada | Límite y siguiente paso |
|---|---|---|
| C01 · Ala’anzy/Kaliyev | README de autores identifica trabajo bajo revisión; código CNN-only: Conv1D(32, k=3), BN, GAP y Dense64→1 para RV | Falta manuscrito completo y tabla de ese comparador; under review no significa publicado/indexado. |
| C02 · crypto_vol_prediction | Código de referencia TS-only MLP64→32→16→1Softplus; después añade noticias | Falta identidad de artículo y texto completo. Objetivo intradiario de retornos cuadrados distinto de σ7. |
| C04 · Realized_Volatility_Forecasting | Notebook con CNN1D independiente sencilla | Falta firma de estudio/fuente de datos/manuscrito; no se incluye como paper. |

C03 documenta software que declara Binance Spot / BTCUSDT a un minuto, pero pronostica dispersión futura de diez minutos. Esto aporta contexto de procedencia, no uso científico verificado de la versión o etiqueta del proyecto. C08 usa CNN2D para clasificación de dirección y declara figuras sintéticas/tablasTBD: no respalda una CNN1D de regresión.

## Comparabilidad con fuentes ya revisadas

Referencia propia: Binance Spot BTCUSDT a un minuto agregado a diario UTC; σ7 centrada poblacional, h1día; desarrollo2023–2024 y test2025–agosto2026.

| Fuente | Datos/mercado | Objetivo/horizonte | Comparación directa con el proyecto |
|---|---|---|
| S1 · Akbiyik etal. | Bitcoin15min yTwitter | RV agregada de96retornos intradiarios, próximo día | No: otra definición, escala, información y periodo. |
| S2 · Zeri | Binance Futures perpetuo, CryptoLake | LogRV futura a 30 minutos, microestructura/onchain | No: futuros noSpot; otro objetivo/frecuencia. |
| S5 · Gao etal. | Yahoo BTC-USD diario | Ventana de 30 días sincentramiento, desplazada siete días | No: proveedor/par, denominador, ventana y horizonte distintos. |
| S6 · Wang etal. | Bitcoin OHLCV diario | RV sin fórmula suficiente | No se puede acreditar igualdadσ7 ni trasladar métricas. |
| S7 · Alakhouch | Bitcoin diario | Volatilidad futura de 30 días anualizada en código | No: futuro30d/anualización frente móvil de siete días y h=1. |

La matrizJSON/CSV incluye también las fuentes contextuales S3/S4 con estado de recuperación. **No se acredita todavía que una publicación haya usado la misma versión histórica BinanceSpot/σ7**. La mejora consiste en declarar de forma verificable dónde coinciden y dónde difieren los datos, no en atribuirles igualdad.

## Archivos y criterio de cierre

`search_log.json` conserva cadenas, páginas, URLs, fechas, conteos devueltos y SHA256 de las respuestas; `candidate_assessment.*` registra nueve decisiones. `source_code_manifest.json` conserva commit y hash de siete archivos inspeccionados. Los documentos completos/HTML quedan fuera del libro.

Para cerrar A01 se requiere recuperar un documento científico identificable de los candidatos pertinentes —o de otras fuentes— y extraer el papel independiente, arquitectura, dataset, protocolo, resultados y ubicación. Para A07 sigue pendiente ampliar cobertura académica y corroborar la procedencia individual. A08 ahora dispone de comparación explícita, pero no de uso previo de la misma versión acreditado. No se cambian las frecuencias0/5 de MLP y CNN simple por contar README o componentes de modelos híbridos.
