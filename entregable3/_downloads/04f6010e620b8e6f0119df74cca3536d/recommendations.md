# Revisión metodológica del protocolo de evaluación ampliada

Preparada con asistencia de IA, 18 de noviembre de 2026 (America/Bogota), antes de observar resultados de los nuevos conjuntos. Es una revisión de diseño: las recomendaciones siguientes no acreditan R1/R3, no constituyen un protocolo congelado y no declaran ejecutados modelos o admisiones todavía pendientes.

## Separar la auditoría ejecutable de la acreditación confirmatoria

Se recomienda una **auditoría adicional preespecificada**, con fuentes admitidas, particiones, modelos, presupuesto, métricas y perturbaciones fijados antes de evaluar los bloques reservados. Puede producir resultados reales aun cuando no exista potencia suficiente para una afirmación confirmatoria. El informe debe conservar esa limitación; la ausencia de un dimensionamiento de 0,90 bloquea la acreditación de R3, no toda generación honesta de evidencia.

La confirmación de R3 exige además cerrar sus requisitos propios: pertinencia de las pruebas, margen práctico, potencia prospectiva de la batería, comparación justa y consistencia entre conjuntos. Un tag o unos tests de software no sustituyen esos requisitos.

## Modelos e información comparable

- Conservar los **17 sistemas históricos**: la aditiva E3 de 1210 parámetros y sus dieciséis rivales. V3 queda fuera de esta familia mientras no se adopte explícitamente como candidato; sus propiedades o métricas no se transfieren a la aditiva.
- Registrar por separado los controles analíticos `floor_only` y `floor_plus_mean_excess`. Antes de ejecutar debe decidirse si forman parte de la familia primaria o si son diagnósticos adicionales; no deben añadirse o retirarse después por su desempeño.
- Predefinir una comparación con **proyección física común** `max(predicción, sqrt(A))` para todos los sistemas, conservando también sus salidas crudas. La auditoría anterior mostró que el piso común modifica sustancialmente la ventaja descriptiva. Una proyección exclusiva del candidato no aísla el mecanismo frente al conocimiento compartido del soporte.
- Declarar que Persistence, SeasonalNaive y ARIMA usan historia rezagada del objetivo, mientras las redes reciben siete barras OHLCV. Esta diferencia de información debe ser visible y debe existir un control analítico del piso. Ningún control puede incorporar el objetivo que está pronosticando.

## Particiones, HPO y entrenamiento

Una propuesta operativa es reservar 55 % para ajuste, 15 % para validación, 10 % para calibración y el resto para prueba, con siete objetivos purgados en cada frontera. Los soportes de ocho cierres de las etiquetas deben comprobarse, además de la disyunción de identificadores. Los conteos dependen del constructor definitivo, y deben fijarse por hash antes de evaluar.

| Serie | Ventanas potenciales | Ajuste | Validación | Calibración | Prueba | Purga |
|---|---:|---:|---:|---:|---:|---:|
| D01, 3270 barras | 3263 | 1794 | 489 | 326 | 633 | 21 |
| D02, 1461 barras | 1454 | 799 | 218 | 145 | 271 | 21 |
| D03, 1583 barras | 1576 | 866 | 236 | 157 | 296 | 21 |

Los escaladores de características y objetivo se ajustan exclusivamente en el bloque de ajuste. La calibración queda después de la selección y antes de prueba. AAPL mantiene su reloj de sesiones; no se fabrican fines de semana o días UTC para imitar las criptomonedas.

La selección automática de hiperparámetros en la validación de cada conjunto es admisible como **ajuste rutinario** si la arquitectura, la rejilla, el criterio, el número de intentos, el presupuesto y las reglas de fallo se fijaron previamente. Mirar esos resultados y rediseñar después el mecanismo convierte ese conjunto en desarrollo. El perfilado de costes y el diseño de potencia deben usar BTC de desarrollo, sin consultar resultados reservados de D01–D03 para escoger una configuración favorable.

Para la primera auditoría se puede mantener el ajuste final en FIT y la parada en VAL, de manera común. Si se decide fusionar FIT y VAL para un refit, el número de épocas de cada fase debe fijarse desde la selección previa, y no debe usarse el entrenamiento fusionado como una falsa validación para elegir mejores pesos.

## Presupuesto viable y medible

La igualdad de épocas por fase no implica igualdad de cómputo: SAE tiene AE1, AE2 y fase supervisada. Debe compartir **un límite total por configuración o réplica**, con asignación de fases declarada, pasos y tiempo de cada fase. Los intentos fallidos y los reinicios consumen presupuesto; un timeout o un fallo no autoriza a omitir un rival o sustituirlo silenciosamente.

Como techo inicial por dimensionar únicamente en BTC, tres tasas de aprendizaje y una semilla HPO prefijada, con 120–180 segundos por configuración y 120–180 segundos por réplica final, permiten una campaña de días:

- Catorce sistemas neuronales, diez semillas y cuatro series producen 560 ajustes finales, más doce trayectorias clásicas.
- Tres configuraciones HPO por red y serie producen 168 intentos de selección.
- Los techos nominales anteriores suman unas 25–37 horas, antes de I/O, exportación y análisis. Son límites calculados, no consumos medidos ni una promesa de terminación.

La misma GPU, precisión, paralelismo y perfil de medición deben emplearse al comparar tiempos. Reportar tiempo real, pasos, parámetros, FLOPs cuando estén medidos, memoria pico y energía sólo con método explícito. No mezclar una latencia caliente local de ONNX con tiempo de entrenamiento GPU como si pertenecieran a un mismo perfil de Pareto.

## Robustez de los modelos congelados

Las tres severidades de ruido deben quedar fijadas antes de prueba y usar escala estimada sólo en ajuste. Un log-desplazamiento común de O/H/L/C por fecha conserva coherencia de precios; un cambio multiplicativo del volumen mantiene su unidad original y no negatividad. La ruta perturbada debe ser común entre modelos y coherente entre ventanas solapadas, con semillas y tiempos fijos.

Si se estudia ruido de medición en entradas, la **verdad futura permanece limpia e idéntica** y el calibrador permanece congelado. No recalcular el objetivo de prueba a partir de las entradas perturbadas ni recalibrar cada severidad. Para los clásicos debe definirse cómo se perturba su información histórica causalmente; si el estudio sólo afecta redes OHLCV, se declara ese ámbito sin inventar una comparación de toda la familia.

Las curvas de tamaño de muestra requieren ajustes reales sobre los prefijos especificados. Para un primer estudio viable se pueden prefijar candidato, TSMixer y un control analítico, tamaños 25/50/100 % y las primeras tres semillas de ingeniería, manteniendo cal/test fijos. Es un estudio de ese subconjunto, no una curva de todos los rivales ni evidencia de potencia confirmatoria.

Los subgrupos temporales pueden ser trimestres prefijados; los umbrales de régimen, si se usan, deben provenir de información disponible al pronosticar y estimada en ajuste. Un grupo definido después por el objetivo realizado es un diagnóstico descriptivo, no una regla de decisión predictiva.

## Estadística y potencia

La métrica primaria es la **media de las RMSE individuales**, con la raíz calculada por trayectoria antes de promediar. No sustituirla por RMSE de un ensamble. El remuestreo debe sincronizar el calendario entre modelos y conservar su dependencia temporal; no tratar días × semillas como observaciones IID.

La potencia actual de `campaign/power.py` pertenece a otro alcance: dieciocho sistemas, 51 pruebas de localización entre semillas y un modelo gaussiano condicional. No dimensiona automáticamente la familia histórica de diecisiete sistemas, los nuevos calendarios ni una batería temporal. Una planificación futura debe incorporar por separado número de semillas y número de fechas, alternativas y márgenes declarados sin usar el efecto favorable del test, y sensibilidad a bloques/dependencias. Siempre seguirá siendo condicional a los supuestos del plan, no una garantía sobre mercados nuevos.

Para la ruta temporal, una batería puede incluir bootstrap estacionario, DM–HLN y una prueba condicional de Giacomini–White cuando su diseño sea pertinente. Deben conservarse sus **distintas nulas, pérdidas y supuestos**: DM sobre MSE no prueba un margen de RMSE, y el rechazo omnibus de GW no demuestra por sí solo dirección de superioridad marginal. El HAC de influencia de RMSE no se renombra DM–HLN. Las pruebas entre semillas añaden otra fuente de incertidumbre, pero no se convierten en tres pruebas temporales.

Las familias estadísticas deben enumerarse antes de ejecutar. Si se publican afirmaciones por rival y por conjunto, el control familiar debe abarcar esas afirmaciones o justificar un procedimiento jerárquico previamente fijado. No fraccionar la familia después de ver resultados para obtener rechazos. Las ausencias no estimables permanecen visibles y no se convierten en p = 0.

La calibración residual se ajusta sólo en CAL, con igual nivel nominal para toda la familia, y se informa cobertura, anchura e interval score globales y por subgrupo. Un intervalo central del 90 % no define por sí solo una distribución suficiente para calcular CRPS o PIT.

## Alcance de los tres conjuntos

D01 aporta un cambio claro de acciones frente a BTC y de tamaño. D02 y D03 modifican instrumento, producto o periodo, pero comparten factores cripto; D03 comparte subyacente BTC y fechas 2023–2024 ya conocidas. No constituyen tres mercados independientes o tres calendarios globalmente ciegos.

La guía pide diferencias en al menos dos ejes pertinentes. Un periodo distinto no acredita automáticamente otro eje de dominio, ruido, tamaño o dimensión. Deben documentarse las magnitudes del cambio y las características observables de cada fuente antes de presentar la diversidad como requisito plenamente satisfecho. Las escalas distintas de RMSE se reportan por serie; no se promedian como si fueran comparables en unidades originales.

Esta auditoría puede mejorar de forma material la evidencia de evaluación ampliada y robustez aun cuando el candidato no gane. El informe debe conservar los resultados desfavorables y distinguir falta de superioridad, potencia limitada, diferencia práctica pequeña y error de implementación.
