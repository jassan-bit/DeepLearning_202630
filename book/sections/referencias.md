# Referencias y materiales

## Guía y materiales recibidos

Los originales se conservan en `references/course/` del repositorio fuente, con sus hashes en `SHA256SUMS.txt`.

- **Entregable1DeepLearningProject.pdf:** guía de 34 páginas. La Sección 1 aparece en páginas 2–3. Es la fuente de requisitos, no una fuente de datos de mercado.
- **entregable1_texto.txt:** texto pegado de la misma guía; no es el Entregable 2.
- **volatilidad_fragmento.png:** definición histórica y frecuencia sugerida en el material recibido. La convención adoptada —adquisición a un minuto, modelado diario, siete retornos para la volatilidad principal y horizonte de un día— se explicita en §§1.1.1 y 1.3.2; no es el Entregable 3.
- **seccion1_pagina1.png**, **seccion1_pagina2.png**, **seccion1_pagina3.png:** capturas de alcance recibidas.
- **periodos_anteriores.png** y **calendario_aclarado.png:** referencia previa y calendario aclarado por el usuario; la última fija test hasta el 31/08/2026.
- **modelos_clase.png:** familias de modelos mostradas en clase.

Estos materiales se conservan como referencia académica suministrada por el usuario. No se les atribuye licencia abierta ni autorización de distribución pública. El usuario decidió hacer público el repositorio fuente para habilitar GitHub Pages; esa visibilidad no modifica los derechos de los materiales originales.

## Bibliografía académica: catálogo inicial y fuentes de las fichas

El catálogo inicial de treinta referencias respalda fundamentos y motivación. Los cortes de 33, 35 y 36 entradas añadieron fuentes aplicadas con sus fechas y niveles de lectura; la ampliación financiera T1/T2/T3 llevó después a 39. El catálogo actual reúne **40 referencias**, tras incorporar S9, *Cryptocurrency Volatility Forecasting with Applications in Trading* {cite:p}`djanga2025commonality`, manuscrito completo sobre veinte pares cripto incluido BTC. La {ref}`tabla del capítulo 3 <tabla-referencias-e1>` distingue citas de los **siete estudios cripto incluidos**: todos incluyen BTC y uno también otras monedas, con solapamiento. T1/T2/T3 permanecen separados por lectura original, código o identidad. Los antecedentes fundacionales no se convierten en frecuencias aplicadas; DOI/revista no prueban indexación individual o Q1.

La consulta inicial de fuentes en internet se realizó el **2 de noviembre de 2026** y se amplió el **3 de noviembre de 2026**, como documenta §3.1. El [registro de verificación](../_static/bibliografia/verification_30.csv) identifica qué fuente se consultó y si se accedió a un original, a documentación de autores o a metadatos en otra bibliografía. Los [detalles por referencia](../_static/bibliografia/references_30.json) conservan enlaces, ubicaciones de evidencia y limitaciones. No se afirma lectura de treinta textos originales completos, consulta efectiva de bases bloqueadas ni clasificación Q1 sin verificar.

Se incluye DOI cuando se dispone de un identificador corroborado o consignado en la ficha bibliográfica; su resolución directa no se presenta como comprobada si la red la bloqueó. Cuando no se corroboró el DOI, se enlaza la fuente donde se verificó la identidad bibliográfica, que puede ser distinta del texto original. El antecedente de Huang y colaboradores conserva su verificación previa en el proyecto. No se inventan DOI ni paginación para completar campos.

[Catálogo inicial de 30 referencias en PDF](../_static/bibliografia/bibliografia_30.pdf) · [Bibliografía actual de 40 referencias en BibTeX](../_static/literature_e1/crypto_extension_20261105/references_access_40.bib) · [Catálogo actual CSV](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv) · [JSON con procedencia](../_static/literature_e1/crypto_extension_20261105/references_access_40.json). Los catálogos de [39](../_static/literature_e1/journal_update_20261105/references_access_39.csv), [36](../_static/literature_e1/recovery_20261105/references_access_36.csv) y [35](../_static/literature_e1/references_access_35.csv) permanecen como cortes históricos.

```{bibliography}
:all:
:style: plain
```

## Herramientas

[Jupyter Book 1: construcción HTML](https://jupyter-book.readthedocs.io/v1/start/build.html). Se fija la versión 1.0.4.post1 por compatibilidad con la configuración `_config.yml`/`_toc.yml`. Es la línea v1 en mantenimiento, no la versión más reciente del producto.

Consulta bibliográfica inicial: 2 de noviembre de 2026; ampliación y verificación pública: 3 de noviembre de 2026. La recuperación complementaria posterior de Miura y su trazabilidad se documentan en §3.1; no modifican retrospectivamente la búsqueda ni el benchmark anteriores.

## Fuente directa de datos

[Binance Spot: archivos públicos de BTCUSDT de un minuto](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/) y [documentación oficial Binance Public Data](https://github.com/binance/binance-public-data). Fuente directa utilizada: descarga y checksums de 75 ZIP verificados para el 01/01/2023–30/10/2026 inclusive UTC. Auditoría de adquisición en §2.3.2; EDA de calidad, precios, retornos y ventanas candidatas de volatilidad de desarrollo ejecutado; las decisiones finales de modelado permanecen identificadas en sus secciones.

[Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), versión 1.0, actualizados el 26 de agosto de 2026; consultados el 2 de noviembre de 2026. Declaran CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Véase §2.3.4 del [EDA](02_eda.md) para el registro de condiciones y atribución.


## Validación temporal con timeseries-cv

[Timeseries-cv en PyPI](https://pypi.org/project/timeseries-cv/), versión 0.1.5. Proyecto [DidierRLopes/timeseries-cv](https://github.com/DidierRLopes/timeseries-cv), desarrollado con coautoría de Filipe Roberto Ramos. Código y documentación bajo MIT; copyright 2021 DidierRLopes. La figura de forward chaining se conserva desde el notebook oficial, commit `8e9137c2774e92a6d7a80446e4022c27ca1b7991`, con su aviso de licencia.

Ramos, F. R. (2021). *Data Science na Modelação e Previsão de Séries Económico-financeiras: das Metodologias Clássicas ao Deep Learning*. Tesis doctoral, Instituto Universitário de Lisboa — ISCTE Business School. Referencia de la tesis indicada por la documentación de la biblioteca.
