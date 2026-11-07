# Cierre bibliográfico por internet · E1

La cola congelada contiene **22 grupos nativos**. Se revisó evidencia para los 22: **4 originales leídos**, **1 transcripción completa de espejo**, **5 decisiones estrictas cerradas** y **17 pendientes**. Las cuatro fuentes E1X001–004 se mantienen separadas; solo E1X003 tiene lectura de original completa.

| Dimensión nativa | Cantidad | Identificadores |
| --- | ---: | --- |
| Original de autor/editorial leído | 4 | E1N053, E1N075, E1N118, E1N163 |
| Transcripción completa, no original editorial | 1 | E1N091 |
| Decisión estricta cerrada: excluido/contexto | 5 | E1N053, E1N075, E1N091, E1N163, E1N170 |
| Pendiente de original | 16 | Véase decisions.csv |
| Original leído, suplemento/objetivo pendiente | 1 | E1N118 |

Estas dimensiones se superponen; no se suman como un flujo de artículos. E1N075 se leyó como manuscrito aceptado de autor, pero se excluye del núcleo estricto actual por proxy/receta no conciliadas e inconsistencias; se conserva como contexto y puede reabrirse ante evidencia nueva. E1N091 es una transcripción secundaria completa, sin PDF editorial recuperado. E1N170 cierra pertinencia desde método primario parcial que identifica precio; no suma lectura íntegra. E1N118 necesita confirmar en suplemento la salida de volatilidad evaluada.

Pendientes estrictos: E1N007, E1N057, E1N074, E1N080, E1N118, E1N120, E1N121, E1N124, E1N129, E1N133, E1N138, E1N142, E1N152, E1N169, E1N173, E1N175, E1N177.

**N7 permanece en 7; cero nuevas inclusiones nativas y cero incrementos de recurrencia estricta GRU/LSTM/TCN.** model_evidence.csv distingue presencia autónoma, objetivo, receta simple y elegibilidad. Una fila de TCN en clasificación, una salida multivariable sin volatilidad aislada o un módulo convolucional no resuelven conjuntamente esos criterios. E1X002 confirma un builder TCN autónomo en código de autor, con versión/tablas pendientes; E1X003 respalda LSTM en RV diaria, sin receta final simple equivalente a E1. No se combinan proxies ni horizontes.

reviews.json conserva íntegra cada ficha derivada bajo source_review y añade la adjudicación vigente, sin reescribir las fuentes. decisions.csv es el índice de estados; summary.json ofrece conteos e identificadores. Los 20 archivos de las subpartes son entradas públicas explícitamente revisadas, con huellas en manifest.json. No se incluyen PDFs, textos originales completos, resúmenes nativos ni contactos.

Reproducción: python scripts/build_e1_internet_closure.py. Verificación sin escritura: python scripts/build_e1_internet_closure.py --verify. El compilador valida cobertura exacta, identidad DOI, estados, hashes, privacidad de campos y espejo público. Empaqueta evidencia guardada: no ejecuta búsquedas, descarga originales ni entrena modelos. Las consultas públicas no acreditan historial nativo WoS/Scopus/IEEE; los faltantes de cadenas literales, fechas y totales siguen en search_recovery_20261106.

Esta consolidación es posterior a los experimentos y preserva su cronología; no constituye un prerregistro retrospectivo. El espejo de publicación está en book/_static/literature_e1/internet_closure_20261106.
