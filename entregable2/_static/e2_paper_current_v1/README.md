# Evidencias vigentes del paper guía y su réplica

Este expediente acompaña el punto 5 de E2. Actualiza la lectura de evidencias ya ejecutadas; conserva los resultados, las figuras, los registros y los paquetes originales.

## Tres ámbitos distintos

| Ámbito | Qué acredita |
|---|---|
| Réplica externa TimeMixer | Cinco semillas en ETTh1 horario, siete variables, historia y horizonte 96. MSE media 0,3778376162 frente a 0,375; IC95 % de discrepancia [−0,0007261706; 0,0064014030]. Cumple el criterio previo de contener el intervalo completo en ±0,01875. |
| Piloto externo | Quince ajustes: TimeMixer, iTransformer y PatchTST, con cinco semillas cada uno. Cuantifica dependencia del batch con calendario y estudia alta variabilidad observable, con estimandos y presupuestos propios. |
| Benchmark y reproducción Bitcoin | D7×5 OHLCV → σ₇, h=1, dieciséis sistemas y 608 fechas. Sus 81 predictores y las 81 reproducciones/18 ablaciones complementarias no son más réplicas del paper ETTh1. |

El criterio de compatibilidad del paper se registró antes de los ajustes; su tolerancia no es el margen δ de Bitcoin. La MSE publicada se trata como referencia fija redondeada, sin inventar una varianza original. El IC entre las cinco semillas usa t de Student y el mismo calendario retenido; no mide incertidumbre entre mercados. La incertidumbre temporal se informa por separado.

La operacionalización del preset de autores conserva `drop_last=True`: se evaluaron 2.688 de 2.785 orígenes test, omitiendo 97. No se conoce la cobertura exacta de la fila histórica del artículo. CPU, librerías y semillas también difieren del entorno histórico no íntegramente identificado. Por eso se acredita una réplica parcial compatible, no identidad de todas las condiciones del paper.

## Limitación y conexión con E3

L1 identifica dependencia de otras muestras del batch en la implementación fijada con marcadores de calendario. Los controles y la intervención sin calendario delimitan su alcance: no demuestra una deficiencia de toda la familia multiescala, ni un deterioro significativo de precisión, ni el mismo defecto en el adaptador D7 que omite esos marcadores. Corregir un índice no basta como arquitectura original de E3.

L2 resulta inconclusa con el bloque principal de 192 horas; se conserva el rechazo de iTransformer en la sensibilidad de 96 horas, sin sustituir el análisis principal. Los análisis usan 4.999 remuestreos circulares pareados, con sensibilidades 96/384 horas y Holm dentro de cada familia de tres contrastes. El mínimo transversal de 10.000 para contrastes pareados queda señalado en la lista de requisitos; los 20.000 posteriores de Bitcoin no modifican estos resultados ETTh1.

La revisión actual de mecanismos y los diagnósticos Bitcoin ya están disponibles para motivar E3. Son evidencia con su propia cronología: no se presentan como hipótesis previas de este piloto ni como demostración retroactiva de L2. La elección de un modelo original permanece abierta.

## Archivos

- `guide_checklist.json` y `.csv`: 34 condiciones de la guía, con evidencia, estado y límites.
- `decision_traceability.csv`: decisiones y verificaciones vinculadas a esas condiciones.
- `current_evidence_index.json`: índice de 22 fuentes con tamaños y SHA-256, cifras originales y separación de ámbitos.
- `05_paper_guia_before_update.md`: copia exacta del capítulo previo a esta consolidación.
- `manifest.json`: tamaños y hashes del expediente.

Las cifras bibliométricas conservan sus cortes: 609 citas Scopus al seleccionar TimeMixer y 619 en el snapshot uniforme posterior del mismo EID. No se suman ni se reescribe la decisión de selección.

## Verificación y reproducción

Los hashes del índice permiten verificar los archivos originales del checkout. El paquete ya publicado `paper_guide_download/Paper_guia_E2_replica_piloto.zip`, su `MANIFIESTO.json` y el capítulo 5 conservan diseño, código, condiciones, métricas, curvas e instrucciones de las fases `validate`, `register`, `fit`, `score` y `pilot`. Los datos y fuentes de autores se adquieren por separado y están fijados por commit/hash.

Esta actualización documental comprobó la cronología, los cinco resultados, sus medias/DE/IC, las disposiciones del piloto, los enlaces y los hashes. No volvió a entrenar, inferir, remuestrear ni cambiar el código científico. El entorno CPU registrado de ETTh1 y el entorno A100 de Bitcoin tienen identidades distintas; construir el libro no ejecuta esos experimentos.
