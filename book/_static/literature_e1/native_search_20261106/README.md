# Búsquedas nativas recibidas el 6 de noviembre de 2026

Esta capa documenta cuatro exportaciones recibidas: 244 ocurrencias y 182 grupos candidatos provisionales. No sustituye el corpus histórico de E1 ni acredita exhaustividad de las búsquedas. El denominador histórico de textos completos cripto sigue siendo N=7; esta importación no modifica sus frecuencias.

## Archivos

- `source_manifest.json`: nombre, tamaño, SHA-256 y cobertura observable de cada exportación. Consulta exacta, filtros, total mostrado y rango exportado quedan sin establecer.
- `occurrences.csv`: las ocurrencias originales por archivo, sin borrar identidades de origen.
- `records.csv`: grupos candidatos por DOI, identificador de base o título normalizado; conserva años e identificadores discrepantes.
- `deduplication.csv`: cada enlace o conflicto entre ocurrencias. Un enlace por título es siempre provisional.
- `screening.csv` y `review_queue.csv`: primera pasada automática por título, resumen y palabras clave, con reglas en el script, razones y prioridades. Todas las filas requieren revisión humana.
- `summary.json`: recuentos, incertidumbres y enlaces por DOI a la evidencia anterior.
- `flow.svg`: flujo de descubrimiento provisional; no es un PRISMA de selección final.
- `manifest.json`: integridad de los archivos públicos derivados.

## Identidades y decisiones pendientes

Los 62 enlaces internos no equivalen a duplicados definitivamente descartados. Hay 4 grupos con enlaces por título; 1 contiene identificadores diferentes de la misma base. En particular, el título *Enhancing Cryptocurrency Price Prediction Using Hybrid Regression and RNN Models* aparece con dos EID Scopus y años 2024/2025: ambos registros y el registro WoS se conservan como ocurrencias distintas dentro de un grupo candidato pendiente de validación. El total definitivo de documentos únicos sigue sin resolverse.

Las prioridades 1, 2 y 3 ordenan la lectura; ninguna es inclusión o exclusión final. La prioridad 1 señala volatilidad en el título y contexto cripto, pero aún puede corresponder a objetivos mixtos, índices, clasificación de regímenes o robustez de precios. La prioridad 2 revisa precio/retorno/riesgo/trading y señales de resumen. La prioridad 3 conserva revisiones, contexto y alcance incierto. Las menciones de GRU, LSTM, TCN u otros modelos no acreditan comparadores independientes. No se revisaron textos completos mediante este importador. Los años nominales posteriores a 2026 requieren comprobar publicación anticipada o metadatos; no se excluyen automáticamente.

## Reproducir y verificar

Requiere Python 3.10 o posterior, sin paquetes externos. Desde la raíz del repositorio:

```sh
python scripts/import_e1_literature.py \
  --scopus '/ruta/export_4a31d458-6ce4-4435-9507-df9ee7cc2881_2026-11-06T220747.614885599.csv' \
  --scopus '/ruta/export_694623e0-7e5c-4597-aca5-c8fc7cb8b431_2026-11-06T220948.394330608.csv' \
  --wos '/ruta/savedrecs (4).txt' \
  --wos '/ruta/savedrecs (3).txt'
python scripts/import_e1_literature.py --verify-public references/literature_e1/native_search_20261106
python -m unittest discover -s tests -p 'test_e1_literature_import.py'
```

El orden de los cuatro argumentos fija los identificadores S01/S02/W01/W02; el orden de salida y las reglas son deterministas. El script lee la matriz histórica `references/literature_e1/crypto_extension_20261105/evidence_matrix.json` únicamente para enlazar DOI y registrar su huella. `--output-dir` y `--mirror-dir` permiten otros destinos; `--no-mirror` evita la copia estática. `--verify-public` verifica hashes, recuentos, cobertura de ocurrencias y campos permitidos sin necesitar las exportaciones originales.

Los resúmenes se usan en memoria y solo se publica su disponibilidad y huella del texto normalizado por el parser. No se redistribuyen exportaciones completas, resúmenes, correos, afiliaciones ni referencias citadas brutas. Las fuentes originales son necesarias para repetir el cribado; sus SHA-256 permiten comprobar que son los mismos archivos.
