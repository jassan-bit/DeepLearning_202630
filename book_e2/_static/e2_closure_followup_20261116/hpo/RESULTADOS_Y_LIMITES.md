# HPO: evidencia descriptiva y piloto de coste

La selección original conserva 78 configuraciones, 234 oportunidades y 231 resultados válidos. Las seis tuplas por red sólo aportan **un par de 195** que difiere en un único parámetro. Las 237 filas marginales y los puntos por configuración describen asociaciones; no identifican importancia independiente ni toda la variación de selección. iTransformer tiene rango de diseño seis con seis tuplas, sin grados de libertad residuales en ese diseño lineal. No se ajustó aquí un modelo de respuesta.

El nuevo dominio de 16 tuplas por red y tres semillas HPO provisionales permanece **sin ejecutar**. Mantiene los cuerpos publicados y D7; NAS del capítulo 14 no aplica. Profundidad/ancho y las dimensiones familiares no exploradas requieren alternativas publicadas adjudicadas y una enmienda antes de ampliar el dominio; el diseño de factores de entrenamiento por sí solo no cierra §11.2.

## Piloto operacional ejecutado

Se eligió C03 por la misma posición registrada, con semilla 271211 y el primer fold: 253 ejemplos train, 83 de validación. El parser sólo decodificó valores hasta el 24/12/2023; no leyó valores de la calibración final ni de prueba. Se mantuvieron máximo 200 épocas, paciencia 20 y horizonte de scheduler 200. No se sustituyó una selección completa por pocos batches.

| Modelo | Estado | Épocas completadas / mejor | Pasos | Cargo de trabajador |
|---|---|---:|---:|---:|
| MLP16 | Completo | 200 / 180 | 800 | 5,753766 s |
| iTransformer | Completo | 24 / 4 | 192 | 12,817986 s |
| TimesNet | Censurado por el límite de tiempo | 39 / no cerrado | 624 | 266,563939 s |

TimesNet acumuló 262,968779 s dentro de la fase antes del límite sincronizado. Sus 39 épocas y 624 pasos no equivalen a un resultado HPO completo; no se le asigna una RMSE final ni se usa su tiempo como coste completo. Los dos trabajos que terminaron permiten observar el coste de sus recetas/fold/semilla concretos. Tres representantes no identifican el coste de los otros diez cuerpos, las demás configuraciones o todos los folds. La carga concurrente del equipo y los tamaños de batch también limitan extrapolaciones.

El primer intento de TimesNet fue bloqueado antes de construir/ajustar el modelo porque cambió un ejecutor de portabilidad ajeno a los imports científicos. Se conservó y cobró ese fallo. La nueva admisión fijó la clausura de dependencias científicas y sus metadatos; sólo se reanudó TimesNet. MLP16 e iTransformer no se repitieron. Se preservan `pilot_v2` y sus fuentes/recibos; `pilot_v3` registra la recuperación y su auditoría de imports.

Los cuatro cargos de trabajadores físicos, incluido el bloqueo previo, suman **285,250329 s**. Con las dos admisiones, el grupo consume **291,208123 s**, dentro del máximo 450 s autorizado. Los tiempos de fase están contenidos en los cargos de trabajador y no se vuelven a sumar. El ledger global continúa el límite acumulado de 84 h. Estos costes nuevos no reconstruyen ni sustituyen los costes históricos perdidos.

## Qué permite cerrar y qué continúa parcial

El inventario histórico concilia 233 cargos observados: **25.833,898918 s** con relojes/hardware heterogéneos. La asignación desconocida de 1.860 s sigue separada. Hay 180 desgloses de fase y 51 ausentes; sus cargos de pared disponibles no desaparecen por esa ausencia. No se afirma coste histórico completo, GPU activa, dinero o energía.

El piloto identifica costes observados de dos ajustes completos y una censura del representante caro; evita una ampliación de horas fundada en un coste imaginado. Una nueva campaña debe medir/calibrar las restantes cargas, adjudicar las dimensiones familiares y fijar una suma de límites de trabajo que quepa en el presupuesto autorizado. No se presenta una proyección de los trece cuerpos como si estuviese identificada, ni se solicita una cantidad arbitraria de horas. La cobertura HPO, la importancia independiente, la variación de selección y la comparabilidad completa con clásicos permanecen parciales. A2, las recetas originales y el periodo de prueba no se modificaron.

## Reanálisis independiente

`reanalisis_hpo_descriptivo_v1.zip` se ejecuta con Python estándar fuera del repositorio y sin red, Torch o datos de mercado. Reconstruye **702 cifras marginales**, con diferencia máxima **0**, y verifica los cargos y la separación de la asignación desconocida. Incluye el formulario bibliográfico original y la cola de 174 informes; no redistribuye resúmenes completos o PDF. El ZIP verifica los descriptivos existentes: no ejecuta este piloto ni una nueva búsqueda HPO.
