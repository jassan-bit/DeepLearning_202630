# Cribado editorial de las cuatro exportaciones nativas

El 6 de noviembre de 2026 se examinaron individualmente **título y resumen exportado de los 182 grupos provisionales**. La pasada fue realizada por **Codex, asistente de IA**, con lectura paralela de lotes dentro del mismo sistema. No se declara doble revisión humana ni resolución de desacuerdos entre revisores independientes. Los resúmenes se consultaron desde los cuatro archivos originales recibidos; no se redistribuyen aquí.

Esta etapa añade decisiones editoriales a la primera clasificación automática conservada en `native_search_20261106`. No modifica ni presenta como final aquella clasificación anterior. El objetivo es resolver pertinencia observable y ordenar la obtención de originales conforme a §3.1 de la guía de E1, pp. 12–13.

## Criterios y alcance de las decisiones

El **núcleo de esta etapa** es el pronóstico de una medida numérica de volatilidad de criptomonedas con modelos profundos. Se conservan para revisión original los estudios de volatilidad realizada, dispersión histórica o CVI implícita, diferenciando sus constructos; no se equiparan entre sí ni con σ₇. También se conservan los casos ambiguos de riesgo, rango o regímenes cuando el resumen no permite decidir si hay una salida continua de volatilidad.

- `exclude_core`: el resumen declara precio, retorno, dirección, sentimiento, política de trading, liquidez u otro activo como objetivo, sin una tarea de volatilidad cripto diferenciada. La mera mención de mercados volátiles, indicadores de volatilidad o GARCH no acredita el objetivo. Se excluye del núcleo, no se elimina el registro ni se prohíbe su uso contextual.
- `contextual_only`: revisión, volumen de actas o aporte metodológico sin experimento primario pertinente identificable. Puede servir para búsqueda secundaria, sin contar sus estudios citados como propios.
- `seek_full_text`: pertinencia posible o constructo ambiguo. Requiere original para comprobar definición de respuesta, horizonte, roles de las redes, arquitectura y calidad experimental; **no es inclusión final**.
- `full_text_documented_elsewhere`: enlace a una ficha completa ya documentada. La evidencia leída en esta pasada sigue siendo título/resumen y la ficha se referencia expresamente; **no se afirma una nueva lectura del original**.

Los códigos `E_*`, `C_*`, `F_*` y `D_*` denotan, respectivamente, exclusión por objetivo/dominio observado, contexto, necesidad de original y documentación completa existente. Cada fila de `decisions.csv` contiene su motivo propio, constructo, identificadores de ocurrencia y huellas de los resúmenes usados. No se infiere la ausencia de un modelo o protocolo del texto completo sólo porque no aparezca en el resumen.

El resultado es **145 excluidos del núcleo, 11 contextuales, 22 que requieren original y 4 enlazados a fichas completas existentes**. Los cuatro enlaces corresponden a **dos estudios históricos** (E1N037=S8; E1N038=S1) y **dos originales adicionales ya documentados** (E1N073=CVI; E1N111=GARCH–DL). En esta pasada se leyeron **cero originales nuevos** y se añadieron **cero inclusiones definitivas por texto completo**. Las dos fichas nuevas no son dos comparaciones autónomas equivalentes de σ₇: CVI es otro constructo y las redes de GARCH–DL son híbridas.

## Identidad, fecha y conciliación

`identity_review.csv` contrasta los cuatro enlaces inicialmente basados en título. En E1N011, E1N094 y E1N150, autor inicial, revista, año, volumen, número y páginas coinciden entre exportaciones: se acepta editorialmente el enlace entre bases, con alcance limitado a esos metadatos. No se atribuye un DOI al registro WoS que carecía de él ni se afirma verificación con el editor. E1N097 conserva dos EID Scopus y años/números discrepantes: su identidad sigue pendiente de confirmación editorial. La exclusión temática de ese grupo por precio no resuelve cuántos documentos representa.

Los **cinco grupos con año nominal 2027** mantienen pendiente la fecha efectiva. Ninguno se excluye por ese año: cuando se excluyen por tarea, la razón es una salida de precio/retorno/clasificación observada independientemente. No se inventa publicación anticipada.

El denominador histórico **N=7 no se reevalúa ni cambia** con esta etapa. Su alcance y criterios no se declaran idénticos a los de este núcleo específico. Los enlaces por DOI evitan volver a sumar S1/S8 y las fichas nuevas. La conciliación con todas las búsquedas anteriores sigue incompleta: el flujo mostrado cubre solamente **244 ocurrencias de cuatro archivos → 182 grupos de trabajo** y no es un PRISMA acumulado de estudios incluidos. Los grupos tampoco son un total definitivo de documentos únicos mientras subsista el conflicto de identidad.

## Evidencia que todavía falta

Los 22 originales pendientes necesitan una decisión final por texto completo y evaluación de calidad: comparadores autónomos frente a componentes híbridos, métricas verificables, arquitectura, separación temporal, repeticiones y pruebas. El cribado no completa las cadenas exactas, fechas de ejecución, totales del universo, rangos exportados ni consulta de IEEE Xplore. La correspondencia de las búsquedas y el filtro 2019–2026 siguen siendo información confirmada por el usuario, no un dato nuevo verificado en las exportaciones.

La pasada es posterior a los experimentos y no modifica su cronología ni convierte en previa la elección de GRU–LSTM–TCN. Tampoco busca incluir sólo estudios que confirmen esa terna: los candidatos se retienen por tarea y evidencia, con independencia del modelo favorecido.

## Archivos y reproducción

- `decisions.csv`: decisiones editoriales congeladas, con cobertura de los 182 grupos.
- `identity_review.csv`: adjudicación de los cuatro enlaces por título.
- `summary.json`: recuentos, solapamientos y pendientes, con huellas de insumos.
- `flow.svg`: flujo de esta etapa y límites de sus denominadores.
- `manifest.json`: SHA-256 y tamaño de los cinco archivos públicos.

```sh
python scripts/build_e1_editorial_screening.py
python scripts/build_e1_editorial_screening.py --verify
```

El script **valida y resume juicios editoriales ya registrados**; no toma esas decisiones mediante reglas automáticas ni reproduce una nueva lectura independiente. Las huellas de los originales y de los resúmenes normalizados permiten comprobar los insumos cuando se dispone de las exportaciones originales. La copia en `book/_static/literature_e1/editorial_screening_20261106` es idéntica a esta carpeta.
