# Revisión de limitaciones inferenciales y entregables — 3 de noviembre de 2026

Se contrastaron las páginas 32–34 de la guía con los capítulos 4–7 y sus artefactos documentales. Esta revisión no ejecutó modelos, pruebas empíricas o simulaciones ni abrió datos de mercado o test. Los mínimos exactos de Wilcoxon son resultados teóricos, no métricas del proyecto.

## Limitaciones y correspondencia con los productos

| ID | Requisito | Estado | Evidencia o corrección | Límite o pendiente |
|---|---|---|---|---|
| L01 | Potencia de Friedman con tres modelos y cinco bloques | Advertencia precisada | Baja potencia posible; no se afirma imposibilidad de rechazo ni una potencia numérica del proyecto. | La potencia requiere alternativa, dispersión, dependencia, empates y calibración. |
| L02 | Bloques completos y post-hoc | Reglas sincronizadas con §5.4.3 | Cada bloque reúne los tres modelos; semillas, fechas y configuraciones no son réplicas intercambiables. | Referencia exacta/aproximada, puerta global y familia por registrar; Nemenyi no recibe Holm automáticamente. |
| L03 | Límite exacto bilateral de Wilcoxon | Condiciones y unidad explícitas | p mínimo 2/2^R; R cuenta pares útiles, bajo signos equiprobables, sin ceros o empates absolutos. | Ceros, empates, redondeo, simetría, independencia pertinente y método de software requieren documentación. |
| L04 | Holm para tres comparaciones | Imposibilidad frente a mera posibilidad corregida | Con R=5 o 6 ningún p bilateral exacto alcanza el primer paso 0,05/3; R=7 lo permite en el extremo. | No se eligen siete semillas; posibilidad discreta no garantiza potencia. Holm continúa con α/2 y α solo tras rechazos previos. |
| L05 | Mínimos de repeticiones y elección de prueba | Mínimo separado de suficiencia | Se conservan cinco ejecuciones y seis para rangos de la guía, sujetos al dimensionamiento mayor. | No cambiar a t, unilateral o bootstrap para obtener significación; R definitivo sigue abierto. |
| L06 | Validación repetida y Nadeau–Bengio | Adaptación temporal justificada | No se adopta K-fold aleatorio ni 5×5 automático; origen móvil, cortes internos y embargo vigentes. | Correcciones no crean independencia ni reparan fuga o selección con evaluación; bloques temporales por justificar. |
| L07 | MDE, potencia y margen práctico | Preguntas separadas | MDE frente a cero, ganancia superior al 5 % frente a persistencia y potencia temporal son distintas. | Sin R/MDE numéricos; dispersión piloto incierta; no usar potencia retrospectiva del efecto observado como suficiencia. |
| L08 | Efecto e intervalos pareados | Alcance de incertidumbre precisado | Diferencia en unidades de la métrica; emparejamiento y dependencia conservados. | Remuestrear predicciones guardadas condiciona a ajustes existentes; no incorpora automáticamente volver a entrenar. |
| L09 | Probabilidad de superioridad | Estimando descriptivo identificado | Fracción de victorias pareadas más medio peso a empates; unidad, dirección y precisión declaradas. | No es p-value, posterior, probabilidad universal o prueba de superar 5 %; puede acompañar efectos minúsculos. |
| L10 | Equivalencia TOST o ROPE | Interpretación corregida | TOST α=0,05 usual: IC de decisión 90 % dentro de banda, distinto de IC95 % de reporte; ROPE requiere modelo posterior. | Diferencia frente a cero y equivalencia práctica pueden coexistir; banda y reglas siguen abiertas, sin heredar el 5 %. |
| L11 | Honestidad y selección del baseline | Regla de desarrollo conservada | Una referencia operativa puede elegirse aun con comparación inconclusa, comunicando incertidumbre. | Selección con desarrollo antes de test; ranking posterior descriptivo; no hay baseline DL medido. |
| L12 | Siete productos obligatorios | Correspondencia literal explicitada | §7.4 identifica siete productos de la página 33, sin sumar evidencias auxiliares como nuevos requisitos. | La matriz no certifica cumplimiento completo ni asigna calificación. |
| L13 | Código completo de extremo a extremo | Parcial | Scripts versionados de adquisición, EDA, referencias Ridge e índices CV existentes. | Pipeline implementado EDA + tres redes + inferencia pendiente; build HTML no lo ejecuta. |
| L14 | Dataset, instrucciones y reserva | Trazabilidad existente conservada | Instrucciones, scripts, hashes y términos enlazados; reserva de 20 ZIP desde metadatos. | Muestras supervisadas y protocolo pendientes; sin abrir mercado/test, imputar o cambiar datos/folds en esta revisión. |
| L15 | Síntesis, extracción y flujo bibliográfico | Estado acumulado sincronizado | 20 repositorios inspeccionados, siete textos evaluados, cinco incluidos, 81 cifras publicadas; 35 referencias. | Búsqueda indexada parcial, tres documentos grises y trío final abierto; se corrigieron dos recuentos históricos en pendientes. |
| L16 | Tabla comparativa, baseline e inferencia | Pendiente de ejecución | Contratos de diseño/benchmark existentes y dos plantillas solo de cabeceras. | Cero resultados DL; literatura y diagnósticos Ridge no sustituyen métricas, costos, IC, efectos o contrastes entre redes. |
| L17 | Dependencias, semillas y configuración por corrida | Producto parcial identificado | Locks de libro/EDA, requisitos CV y configuración Ridge; diseño actual separado del borrador histórico. | Entorno DL, semillas finales, manifiestos y logs pendientes; propuestas históricas no son corridas ni plan congelado. |
| L18 | Declaración a priori y cronología | Alcance precisado | n7/h1, cuatro D, MSE/RMSE, 5 % solo frente a persistencia, una serie y partición temporal registrados. | Previo a DL pendiente, posterior al EDA y al mercado; no preregistro completo ni pronóstico prospectivo de 2025–2026. |
| L19 | Pesos de evaluación | Rúbrica conservada | EDA 30 %, literatura 30 %, implementación 20 %, benchmark 20 %, total 100 %. | Pesos no son avance o nota; tampoco se afirma que el 60 % restante esté aprobado. |
| L20 | Reproducibilidad y buenas prácticas | Cierre verificable especificado | Repeticiones, media/DE/IC95 %, efectos, supuestos, esfuerzo comparable, versiones y decisiones previas. | Compilar y validar enlaces comprueba publicación; no superioridad estadística ni reproducción experimental completa. |
| L21 | Margen para propuesta avanzada | Conclusión futura delimitada | Separación del margen aceptado de persistencia; referencia y evaluación por configuración/periodo. | Margen avanzado sin fijar; un test usado para diseñar después una propuesta no la confirma independientemente. |
| L22 | Estado global y alcance autorizado | Sin nuevos experimentos | Se preservan cuatro Ridge de EDA, 1.464 predicciones existentes y contrato tipo 4/ruta 6.4. | Trío, piloto, entrenamiento DL, benchmark y test pendientes; reservarlos a E2 no modifica la guía original de E1. |

## Resolución exacta y multiplicidad

La comparación bilateral exacta usual supone diferencias útiles sin ceros ni empates absolutos y signos equiprobables bajo la nula. El mínimo es 2/2^R. Para tres pares de modelos, el primer paso de Holm usa 0,05/3; los siguientes usan 0,05/2 y 0,05 solo si los anteriores rechazan.

| Pares útiles R | Mínimo bilateral exacto | Rechazo sin ajuste aritméticamente posible | Primer paso Holm aritméticamente posible |
|---|---:|---|---|
| 5 | 0,0625 | No | No |
| 6 | 0,03125 | Sí | No |
| 7 | 0,015625 | Sí | Sí |
| 8 | 0,0078125 | Sí | Sí |

Siete pares solo permiten rechazo en un extremo para esa familia. No fijan siete semillas ni acreditan potencia 0,80. No se eligieron nuevos contrastes, márgenes o configuraciones experimentales.

## Entregables y estado

El capítulo identifica los siete productos literales de la guía: informe; código completo; dataset o instrucciones/scripts según licencia; síntesis y flujo bibliográfico; comparativa/baseline/inferencia; dependencias/semillas/configuración por corrida; declaración a priori. Los diagnósticos Ridge y los manifiestos son evidencia complementaria, no sustitutos de las tres redes.

Los pesos 30/30/20/20 totalizan 100 % y no representan una nota ni avance. El registro precede a futuras corridas DL, pero sucede después del EDA y del periodo de mercado: no es un preregistro completo congelado o una predicción prospectiva de 2025–2026.

La página de pendientes se sincronizó con los metadatos acumulados: 20 repositorios inspeccionados, siete textos evaluados, cinco incluidos y 81 cifras de autores. Se mantienen 35 referencias y el trío definitivo abierto. Las dos plantillas comparativas siguen sin filas de resultados.

No se modificaron los contratos experimentales padre, los datos, los índices o las cuatro Ridge existentes. No se declara E1 completo; la reserva de ejecución para E2 es una instrucción de alcance, no una excepción docente de la guía.
