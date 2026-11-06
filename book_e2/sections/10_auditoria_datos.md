(e2-auditoria-datos)=
# 10. Contrato de datos y auditoría formal de fuga

**Estado:** la auditoría ejecutada de soporte P0/OOF de desarrollo está aprobada. Esto no certifica todas las verificaciones del pipeline final de veinte configuraciones exigidas en la tabla10.1 de la guía (PDF pp44–45). La regla de p46 exige corregir cualquier fallo y repetir el experimento afectado; no permite publicar un benchmark inválido con una nota al pie. La ausencia de un control todavía no ejecutado se informa como pendiente, sin declararlo aprobado.

## 10.1 Evidencia ejecutada, fecha y frontera de acceso

El [recibo `data_support_audit.json`](../_static/remediation_preflight/data_support_audit.json) se generó el **06/11/2026 a las 14:52:31 UTC** con `scripts/e2_remediation_data_audit.py` de la fuente registrada [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0). Lee exclusivamente los dos derivados diarios de desarrollo 2023–2024, comprueba su integridad y produce agregados e índices, sin precios individuales en el recibo.

| Comprobación | Resultado y alcance |
|---|---|
| Calendario | 731 fechas diarias UTC de 2023–2024, únicas y sin compresión |
| Día OHLCV incompleto | 24/03/2023; la máscara conserva su posición |
| Reconstrucción σ₇ | Diferencia absoluta máxima0,0 en la reconstrucción comparada por este auditor; consistencia aritmética, no ausencia de ruido de proxy |
| Orígenes de desarrollo | Once: diez mensuales para OOF y uno para noviembre–diciembre |
| OOF enero–octubre de 2024 | 305 fechas de resultado únicas |
| P0 noviembre–30 de diciembre de 2024 | Sesenta fechas de resultado únicas |
| Identidades de resultados | Cero intersecciones train/evaluación e inner-train/inner-validación dentro de cada separación auditada |
| Duplicados exactos D7 cruzados | Cero colisiones de contenido entre train/evaluación en los once orígenes |
| Corte temporal | Última etiqueta y ajuste de escalado no posteriores a origen−8; embargo de siete días |
| Preflight de datos | Cero ajustes y cero predicciones realizados por **este auditor**, sin acceso a precios de test de 2025/2026 |

El último contador describe el proceso de auditoría, no el estado global posterior del proyecto: **P0 está en ejecución**, con un plan de 220 trayectorias/440 ajustes de desarrollo. No son resultados del benchmark final, ni nuevos mercados independientes.

Los hashes de fuentes son:

```text
daily_market_development.csv
bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171
daily_volatility_development.csv
20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a
```

Un hash identifica bytes y detecta cambios respecto al registro; **no prueba por sí solo ausencia de fuga**. El calendario, la disponibilidad, los soportes y el código de transformación requieren controles adicionales. Tampoco la ausencia de duplicados exactos excluye dependencia, casi duplicados o decisiones hechas con test.

## 10.2 Contrato operativo de las observaciones

Se hereda la [Data Card](../_static/dataset_card.json) y el [contrato científico E1](../_static/e1_reference.json), sin cambiar D7×5, objetivo, h=1, δ, unidad ni partición. La [ruta F2a](07_ruta_temporal.md) añade sus referencias explícitas; no añade Y rezagada a las redes.

| Campo | Regla al cargar y construir ejemplos |
|---|---|
| Identificador temporal | Fecha diaria UTC; su posición no se sustituye por número de fila retenida |
| Covariables | Cinco magnitudes OHLCV; valores finitos, OHLC positivos y coherentes, volumen no negativo en barras completas |
| Disponibilidad | Para resultado t, solo barras completas t−7…t−1; no OHLCV del propio día t |
| Etiqueta | Desviación poblacional de siete retornos logarítmicos consecutivos, ocho cierres, sin anualización |
| Ausencia | No imputar ni unir días separados por un hueco; excluir ejemplos dependientes de información inválida |
| Transformación | OHLC relativos al último cierre observable de la ventana y log1p de volumen; escalas y media de Y aprendidas en entrenamiento admisible |
| Selección y parada | Validación interna anterior al origen, con su embargo; datos de evaluación no eligen época ni parámetros |
| Referencias endógenas | Persistencia alcanza t−8; naive semanal alcanza t−14; ARIMA usa historia declarada, con parámetros y actualizaciones separados |

El 24/03/2023 reúne 80 minutos ausentes y una vela presente con cierre temporal irregular según E1. No se presentan como 81 filas ausentes. Se conserva el día incompleto y se propaga su máscara a retornos, σ₇ y ventanas. Los recuentos detallados de calidad y elegibilidad, incluidas las 350/366 etiquetas válidas de desarrollo, permanecen en §2; 731 días no equivalen a 731 muestras supervisadas ni a 731 unidades independientes.

El driver P0 descodifica para cada ajuste únicamente el prefijo admisible hasta su corte; el scoring de desarrollo es una fase posterior a los cierres programados. La auditoría previa sí inspecciona desarrollo completo para comprobar su esquema y calendarios: no se confunde ese control con una evaluación desconocida ni con permiso para ajustar escalas con validación.

## 10.3 Once separaciones auditadas

| Origen | Último resultado train/escalado | Train elegible | Inner-train | Inner-validación | Evaluación |
|---|---|---:|---:|---:|---:|
| 01/01/2024 | 24/12/2023 | 343 | 253 | 83 | 31 |
| 01/02/2024 | 24/01/2024 | 374 | 284 | 83 | 29 |
| 01/03/2024 | 22/02/2024 | 403 | 313 | 83 | 31 |
| 01/04/2024 | 24/03/2024 | 434 | 344 | 83 | 30 |
| 01/05/2024 | 23/04/2024 | 464 | 374 | 83 | 31 |
| 01/06/2024 | 24/05/2024 | 495 | 405 | 83 | 30 |
| 01/07/2024 | 23/06/2024 | 525 | 435 | 83 | 31 |
| 01/08/2024 | 24/07/2024 | 556 | 466 | 83 | 31 |
| 01/09/2024 | 24/08/2024 | 587 | 497 | 83 | 30 |
| 01/10/2024 | 23/09/2024 | 617 | 527 | 83 | 31 |
| 01/11/2024 | 24/10/2024 | 648 | 558 | 83 | 60 |

Los diez primeros orígenes reúnen 305 fechas OOF; la última fila representa todo el bloque P0 hasta el 30/12, no un undécimo mes OOF. La validación interna nominal de noventa días deja 83 resultados elegibles por su separación temporal; inner-train más inner-validación no suman todo train porque también existe embargo interno.

El activo BTCUSDT aparece legítimamente en todas las fases: se evalúa una sola serie, no generalización a activos excluidos de entrenamiento. Para esta ruta se comprueban fechas de resultados y soportes cronológicos; no se exige inventar identificadores de mercados disjuntos. Una fecha validada puede incorporarse a un entrenamiento posterior cuando ya es pasada y respeta embargo. Ese uso expansivo no vuelve independientes los folds, las semillas o las etiquetas solapadas.

## 10.4 Correspondencia con las verificaciones obligatorias

| Tabla10.1, PDF pp44–45 | Evidencia disponible | Estado y límite |
|---|---|---|
| Intersección de identificadores | Índices de resultados train/evaluación e inner-train/validación en once orígenes | Aprobado para P0/OOF; una serie común no se presenta como particiones iid por mercado |
| Duplicados exactos | Hash de los tensores D7 numéricos entre train/evaluación | Cero colisiones en desarrollo auditado; controles finales/test por completar |
| Casi duplicados | Vecinos en representación relativa, escalado de entrenamiento y umbral0,01 documentados en §2 | Diagnóstico de desarrollo ejecutado; sensibilidad de desempeño con/sin casos no acreditada por este recibo |
| Fuga temporal | Train y escalado hasta origen−8; controles internos y calendario no comprimido | Aprobado en once separaciones D7; cada referencia con otro soporte requiere su comprobación específica |
| Fuga espacial | Un mercado sin coordenadas/unidades espaciales | No aplica a la tarea F2a; no sustituirlo por un test ficticio |
| Transformaciones | Escalado de train comprobado y pruebas sintéticas de adaptadores; EDA con parámetros train-only | Evidencia P0 parcial; auditoría completa de cada pipeline final pendiente |
| Atributos posteriores al evento | Diccionario causal y prefijos de lectura anteriores al origen; análisis de rezagos en §2 | Disponibilidad documentada; revisión final y controles univariados aplicables pendientes de cierre |
| Validación adversarial | EDA de shift con KS/Wasserstein y remuestreo por bloques | Ese análisis no equivale a un clasificador adversarial; la comprobación correspondiente sigue pendiente |
| Modelos preentrenados | P0 construye sus cuatro redes desde inicialización, usando código identificado de autores | Sin pesos preentrenados en este piloto; confirmar procedencia de todas las implementaciones finales |
| Sanidad con etiquetas permutadas | No ejecutada por el preflight de soporte | Pendiente a nivel de modelos; verificar contra una referencia nula apropiada para regresión, sin confundirla con azar de clasificación |

Los controles del preflight de adaptadores verifican forma y finitud de la salida y que una predicción no cambie al acompañarla de otros orígenes en el batch. Son controles relevantes para disponibilidad causal, pero **no equivalen** a entrenar con etiquetas permutadas, demostrar ausencia de atajos o aprobar automáticamente los veinte pipelines finales. Si una comprobación futura falla, se conservará el fallo y se repetirá el experimento afectado tras corregir su causa; no se seleccionarán fechas favorables ni un test que produzca el resultado deseado.

## 10.5 Reconstrucción y cierre pendiente

Desde el checkout registrado, en el entorno de §8 y con una ruta de salida separada, este comando regenera el recibo agregado sin ajustar modelos:

```bash
python scripts/e2_remediation_data_audit.py --output ../e2-data-audit-reproduction.json
```

El [registro de entrada](../_static/remediation_preflight/registration.json) preserva sus hashes y el [recibo publicado](../_static/remediation_preflight/data_support_audit.json) preserva los resultados observados. La [descarga unificada](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) incluye el procedimiento y las evidencias de esta etapa. Una reproducción puede cambiar la fecha de generación del recibo; comparar esa fecha o el hash del JSON completo no sustituye contrastar los mismos campos científicos e identidades de entrada.

Antes del scoring final faltan los recibos de la auditoría completa sobre las implementaciones, las cohortes congeladas de referencias con soportes distintos, los checkpoints cerrados y los análisis registrados. Los periodos 2025–2026 conocidos de E1 siguen fuera de este preflight; su reutilización futura conserva alcance exploratorio. Un hash nuevo no crea un test nuevo y una auditoría de soporte aprobada no demuestra por sí sola generalización independiente ni superioridad predictiva.
