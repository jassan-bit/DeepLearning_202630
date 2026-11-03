# Búsqueda pública y verificación posterior — 3 de noviembre de 2026

Se ejecutaron doce búsquedas web públicas en GitHub: cuatro de descubrimiento, cuatro de recuperación dirigida de documentos y cuatro de recursos de indexación. Se recuperó la primera página por cadena, con relevancia predeterminada. Es una búsqueda pública acotada en repositorios de autores y curadores; no una consulta de Google ni una búsqueda interna en WoS, Scopus, IEEE Xplore o arXiv.

## Consultas realmente ejecutadas

| ID | Función | Cadena literal | Total mostrado por el buscador | Registros recuperados |
|---|---|---|---:|---:|
| W1 | document_discovery | `bitcoin volatility forecasting in:readme` | 4577 | 10 |
| W2 | document_discovery | `bitcoin realized volatility neural in:readme` | 245 | 10 |
| W3 | document_discovery | `bitcoin realised volatility neural in:readme` | 45 | 10 |
| W4 | document_discovery | `"bitcoin volatility" arxiv in:readme` | 10 | 10 |
| IDX1 | index_resource_lookup | `"Journal of International Financial Markets" "Scopus" in:readme` | 0 | 0 |
| IDX2 | index_resource_lookup | `"Web of Science" "master journal" in:readme` | 28 | 10 |
| IDX3 | index_resource_lookup | `"Scopus" "source list" in:readme` | 97 | 10 |
| IDX4 | index_resource_lookup | `"Forecasting Bitcoin volatility" "Scopus" in:readme` | 0 | 0 |
| P1 | targeted_document_recovery | `"Bitcoin volatility forecasting with a glimpse" in:readme` | 1 | 1 |
| P2 | targeted_document_recovery | `"Forecasting Bitcoin Volatility Spikes" in:readme` | 6 | 6 |
| P3 | targeted_document_recovery | `"2211.08281" in:readme` | 0 | 0 |
| P4 | targeted_document_recovery | `bitcoin volatility "buy and sell orders" arxiv in:readme` | 0 | 0 |

Los registros son repositorios, no artículos. Las cuatro consultas de descubrimiento recuperaron 40 ocurrencias y 36 repositorios distintos; las otras ocho son verificaciones dirigidas. No se suman las estimaciones del índice ni se infiere exhaustividad o prevalencia de arquitecturas. Las consultas con cero resultados representan respuestas válidas de GitHub, no cero resultados en WoS/Scopus.

## Procedencia documental corroborada

| Trabajo | Hallazgo público | Fuente comprobada | Alcance |
|---|---|---|---|
| Akbiyik et al. (2023) | arXiv:2110.14317v2 y DOI10.1145/3539597.3570387; ACM WSDM 2023 | PDF público de autores, página1 | Identificador corroborado; registro arXiv en vivo no consultado. |
| Herremans y Low (2025) | IEEE Access 13, 117788–117807; enlace Xplore 11058926 | README público de autores | Enlace declarado por autores; registro Xplore en vivo no consultado. |
| Guo et al. (2018) | IEEE ICDM 2018, pp. 989–994 | README público de autores | Venue declarado; DOI y registro Xplore concreto no corroborados en el material nuevo. |

URLs fijadas, ubicaciones, commits y SHA-256 están en `publication_provenance.json`. La portada de Akbiyik se leyó para identidad/procedencia; no se extrajeron métricas adicionales ni se amplió el corpus. Los trabajos de congreso no se evalúan con una lista limitada a revistas.

## Coincidencias secundarias de revista en listas WoS/Scopus

Fuente: repositorio independiente wosaide/wosaide-journal-lists, commit4d0b1a32915b04180c37e6f697a7da04a6f05de6. Declara copias WoS de agosto de 2026 y Scopus de julio de 2026 obtenidas de listados oficiales; no está avalado por sus proveedores. Se cotejaron título e ISSN.

| Referencia | Revista declarada e ISSN | Colección WoS en la copia de 2026 | SourceID y cobertura Scopus declarada |
|---|---|---|---|
| Huang et al. (2024) | Journal of International Financial Markets, Institutions and Money · ISSN 1042-4431 | SSCI | 17531 · 1997-2026 |
| Barjašić y Antulov-Fantulin (2021) | Frontiers in Physics · ISSN 2296-424X | SCIE | 21100831025 · 2013-2026 |
| Katsiampa (2017) | Economics Letters · ISSN 0165-1765 | SSCI | 28490 · 1978-2026 |
| Dyhrberg (2016) | Finance Research Letters · ISSN 1544-6123 | SSCI | 15519 · 2004-2026 |
| Herremans y Low (2025) | IEEE Access · ISSN 2169-3536 | SCIE | 21100374601 · 2013-2026 |

Estas cinco coincidencias son evidencia secundaria de revista, no confirmación oficial de cada artículo. La cobertura Scopus declarada incluye el año de las referencias, pero no demuestra por sí sola su registro individual; el snapshot WoS2026 tampoco acredita la cobertura histórica del año editorial. No se asigna cuartil Q1, que requiere índice, revista, año y categoría propios.

## Asociación errónea descartada

Un mirror atribuía a Herremans el IDarXiv:2211.00759 y un PDF con nombre concordante. La portada pertenece a otro trabajo, Online Control of Adaptive Large Neighborhood Search Using Deep Reinforcement Learning, de Reijnen y colaboradores. Se rechazó esa asociación. La pista secundaria 2211.08281 para Herremans se mantiene como candidata, sin confirmar su registro oficial ni su fecha de envío. Identidad por título/autores/DOI/portada prevalece sobre el nombre del archivo o el resultado de un buscador.

## Cribado y alcance de esta fase

Los cinco README adicionales inspeccionados son pistas de recuperación; no se incorporan automáticamente como estudios. La tesis/proyecto de b-zahra, el informe híbrido GARCH–LSTM de Zeri y otros repositorios de código requieren recuperar documentos, determinar identidad y resolver duplicados antes de extracción. Los artículos citados por un README no heredan su pertinencia a Bitcoin ni su indexación.

El corpus de extracción permanece en cinco textos y 81 cifras de autores; el catálogo conserva 35 referencias y la selección definitiva de tres modelos sigue abierta. Los recuentos históricos118/106/20/7/5 no se recalculan con esta búsqueda web. No se solicitaron accesos, no se consultaron directamente las cuatro plataformas, no se entrenaron modelos y no se abrieron datos de mercado/test.
