# Referencias y materiales

## Guía y materiales recibidos

Los originales se conservan en `references/course/` del repositorio fuente, con sus hashes en `SHA256SUMS.txt`.

- **Entregable1DeepLearningProject.pdf:** guía de 34 páginas. La Sección 1 aparece en páginas 2–3. Es la fuente de requisitos, no una fuente de datos de mercado.
- **entregable1_texto.txt:** texto pegado de la misma guía; no es el Entregable 2.
- **volatilidad_fragmento.png:** definición histórica y frecuencia sugerida, con inconsistencia minuto/día que debe resolverse; no es el Entregable 3.
- **seccion1_pagina1.png**, **seccion1_pagina2.png**, **seccion1_pagina3.png:** capturas de alcance recibidas.
- **periodos_anteriores.png** y **calendario_aclarado.png:** referencia previa y calendario aclarado por el usuario; la última fija test hasta el 31/08/2026.
- **modelos_clase.png:** familias de modelos mostradas en clase.

Estos materiales se conservan como referencia académica suministrada por el usuario. No se les atribuye licencia abierta ni autorización de distribución pública. El usuario decidió hacer público el repositorio fuente para habilitar GitHub Pages; esa visibilidad no modifica los derechos de los materiales originales.

## Bibliografía académica: 30 referencias

Las treinta referencias siguientes respaldan la síntesis del capítulo 3 y la motivación del problema. Comprenden **siete aplicaciones a Bitcoin**, **siete fundamentos de redes neuronales** y **dieciséis fundamentos de volatilidad, diagnóstico y evaluación**. Los trabajos fundacionales anteriores al periodo reciente se incluyen expresamente como antecedentes metodológicos; no se contabilizan como evidencia de frecuencia de modelos base en Bitcoin.

La consulta de fuentes en internet se realizó el **2 de noviembre de 2026**. El [registro de verificación](../_static/bibliografia/verification_30.csv) identifica qué fuente se consultó y si se accedió a un original, a documentación de autores o a metadatos en otra bibliografía. Los [detalles por referencia](../_static/bibliografia/references_30.json) conservan enlaces, ubicaciones de evidencia y limitaciones. No se afirma lectura de treinta textos originales completos, consulta efectiva de bases bloqueadas ni clasificación Q1 sin verificar.

Se incluye DOI cuando se dispone de un identificador corroborado o consignado en la ficha bibliográfica; su resolución directa no se presenta como comprobada si la red la bloqueó. Cuando no se corroboró el DOI, se enlaza la fuente donde se verificó la identidad bibliográfica, que puede ser distinta del texto original. El antecedente de Huang y colaboradores conserva su verificación previa en el proyecto. No se inventan DOI ni paginación para completar campos.

[Descargar la bibliografía en PDF](../_static/bibliografia/bibliografia_30.pdf) · [Descargar las 30 referencias en BibTeX](../_static/bibliografia/references_30.bib).

```{bibliography}
:all:
:style: plain
```

## Herramientas

[Jupyter Book 1: construcción HTML](https://jupyter-book.readthedocs.io/v1/start/build.html). Se fija la versión 1.0.4.post1 por compatibilidad con la configuración `_config.yml`/`_toc.yml`. Es la línea v1 en mantenimiento, no la versión más reciente del producto.

Consulta de las fuentes externas: 2 de noviembre de 2026.

## Fuente directa de datos

[Binance Spot: archivos públicos de BTCUSDT de un minuto](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/) y [documentación oficial Binance Public Data](https://github.com/binance/binance-public-data). Fuente directa utilizada: descarga y checksums de 75 ZIP verificados para el 01/01/2023–30/10/2026 inclusive UTC. Auditoría de adquisición en §2.3.2; EDA de calidad, precios, retornos y ventanas candidatas de volatilidad de desarrollo ejecutado; las decisiones finales de modelado permanecen identificadas en sus secciones.

[Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), versión 1.0, actualizados el 26 de agosto de 2026; consultados el 2 de noviembre de 2026. Declaran CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Véase §2.3.4 del [EDA](02_eda.md) para el registro de condiciones y atribución.


## Validación temporal con timeseries-cv

[Timeseries-cv en PyPI](https://pypi.org/project/timeseries-cv/), versión 0.1.5. Proyecto [DidierRLopes/timeseries-cv](https://github.com/DidierRLopes/timeseries-cv), desarrollado con coautoría de Filipe Roberto Ramos. Código y documentación bajo MIT; copyright 2021 DidierRLopes. La figura de forward chaining se conserva desde el notebook oficial, commit `8e9137c2774e92a6d7a80446e4022c27ca1b7991`, con su aviso de licencia.

Ramos, F. R. (2021). *Data Science na Modelação e Previsão de Séries Económico-financeiras: das Metodologias Clássicas ao Deep Learning*. Tesis doctoral, Instituto Universitário de Lisboa — ISCTE Business School. Referencia de la tesis indicada por la documentación de la biblioteca.
