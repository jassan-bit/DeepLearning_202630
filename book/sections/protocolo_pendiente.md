# Decisiones, revisión y pendientes de E1

El procedimiento vigente utiliza **CV forward chaining de Filipe con D=7/14/21/28**, objetivo fijo n=7, horizonte h=1 y MLP/LSTM/CNN1D. Los índices de `timeseries-cv` alimentan los doce ajustes mensuales de 2024; los escaladores y purgas se aplican al pasado permitido. Después de seleccionar con desarrollo se cerraron configuraciones, épocas y política de calidad; se ajustaron y firmaron 120 modelos finales antes de abrir el test del 01/01/2025 al 31/08/2026.

## Decisiones incorporadas

| Elemento | Criterio y evidencia vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT, velas oficiales de un minuto |
| Modelado | Diario UTC, cinco canales OHLCV; tipo 4, ruta 6.4, sin sigma rezagada explícita |
| Entradas y objetivo | **D=7/14/21/28** días; **n=7** retornos diarios, poblacional ddof0, sin anualizar; **h=1** |
| Sensibilidad del objetivo | n=14/21/28 explorados en EDA; no entrenados en el benchmark |
| Preprocesamiento | `relative_window`: logOHLC respecto al último cierre observable, log1pV y escalas aprendidas solo con train |
| Control histórico | `absolute_log` se ejecutó únicamente en el ciclo inicial D7, no en las otras tres entradas |
| Pérdida y métricas | MSE; RMSE primaria; MAE/MSE/MAPE/R²/MASE complementarias, en escala original y con cobertura |
| Margen operativo | 5 % de reducción relativa de RMSE frente a persistencia, no margen entre redes, de equivalencia o avanzado |
| MAPE y MASE | MAPE sobre etiquetas positivas sin epsilon; MASE con diferencias de un día de calendario conocidas en train |
| Calidad y embargo | Calendario íntegro, exclusión estricta del día incompleto y ventanas afectadas, sin imputación; u≤origen−D−1 |
| CV | Historia inicial 2023, doce bloques mensuales de 2024 con entrenamiento creciente, pesos fijos dentro del mes y diez semillas |
| Ejecución primaria CV | 120 trayectorias, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones; treinta trayectorias D7 reutilizadas y noventa nuevas D14/21/28 |
| Cierre antes de test | Selección de historia por familia y baseline por RMSE media CV; épocas tomadas de diciembre de 2024; doce combinaciones a reportar ya declaradas |
| Baseline neuronal cerrado | **MLP con D=7**, sin reabrirlo por el ranking de test |
| Ajuste final | 120 redes con desarrollo permitido de 2023–2024, embargo D y escaladores de entrenamiento; pesos firmados antes del primer ZIP de test |
| Test | Enero de 2025–agosto de 2026: 608 fechas comunes evaluables, 72.960 predicciones; veinte meses de reporte, cero reajustes o selección en test |
| Fuera del test | Septiembre–octubre de 2026 descargados pero sin asignación automática |
| Métricas y BDS por bloque | Todas las historias y diez semillas; BDS h=1, dimensiones 2/3, radio de 1,5 DE muestral, familias Holm separadas por fase; media/DE de p descriptivas |
| Inferencia inicial D7 | Holm-12 por semillas, Friedman, DM/HAC/HLN, LB/BDS-15 y bootstrap conservan su alcance de desarrollo; no se trasladan a las otras D o test |
| Parámetros | MLP: 80D+33, es decir 593/1.153/1.713/2.273; LSTM: 5.025; CNN1D: 1.057 |
| Referencias contextuales | Persistencia determinista; cuatro Ridge de EDA como diagnóstico separado, no cuarto modelo DL de test |
| Independencia | Una serie y un mercado; fechas, semillas, historias y meses no crean mercados independientes |

Los porcentajes 27,26/27,33/45,41 describen los días de calendario elegidos: 365/366/608 sobre 1.339. No son proporciones impuestas por Filipe ni una partición aleatoria; entrenamiento crece en los folds y los recuentos supervisados dependen de historia, purga y calidad.

El [método vigente](../_static/filipe_multihistory_20261104/method.json), la [congelación](../_static/filipe_multihistory_20261104/freeze.json), la [auditoría de test](../_static/filipe_multihistory_20261104/test/raw_test_audit.json) y el [calendario de cohorte](../_static/filipe_multihistory_20261104/test/cohort_manifest.json) permiten comprobar fechas, firmas, reglas y elegibilidad. El libro publica resúmenes agregados, código y manifiestos; los vectores individuales, derivados diarios, pesos y paquetes completos permanecen locales. El exportador no entrena ni vuelve a abrir brutos.

## Revisión integral contra la guía

Se conservan **390 incisos numerados** de los capítulos 1–7. El [inventario](../_static/e1/revision_items_libro_20261104.csv) identifica la revisión; “revisado” no equivale a cumplimiento científico total. La comparación de cuatro entradas, el BDS por fold y las tablas finales de test ya tienen evidencia empírica. Las sensibilidades del objetivo n=14/21/28 siguen fuera del alcance entrenado.

La revisión bibliográfica permanece en **35 referencias, cinco textos completos y 81 valores de autores**. LSTM unidireccional 3/5, familia LSTM 4/5, GRU 2/5, TCN 1/5 y encoder 2/5 describen el corpus acotado. MLP independiente y CNN1D simple continúan 0/5: más corridas propias no completan apoyo aplicado independiente ni acreditan consulta nativa WoS/Scopus o cuartil Q1.

## Estado de los siete cambios solicitados

| Prioridad | Producto | Estado verificable |
|---|---|---|
| 1 | Búsqueda trazable | Consultas web públicas y comprobantes; límites de cobertura e indexación explícitos |
| 2 | Cribado y flujo | 118 ocurrencias, 12 duplicadas, 106 repositorios, 20 inspeccionados, 7 textos evaluados y 5 incluidos |
| 3 | Matriz de literatura | Cinco textos, 81 cifras publicadas y evaluación con ubicaciones y riesgos |
| 4 | Tres modelos básicos | MLP/LSTM/CNN1D ejecutados para cuatro entradas; respaldo aplicado de MLP/CNN incompleto |
| 5 | Fichas técnicas | 27 campos numerados del trío, con fórmulas dependientes de D y resultados propios por historia |
| 6 | Comparación crítica | Literatura y resultados propios separados; CV, cierre y test con periodos y límites |
| 7 | Coherencia | n=7, h=1, D=7/14/21/28, RMSE, 5 % frente a persistencia, MAPE/MASE, ruta 6.4 y procedencia sincronizados |

## Límites que permanecen abiertos

- **Cobertura académica:** búsqueda pública acotada; faltan apoyos aplicados independientes de MLP/CNN1D simple, corroboración individual de indexación y extracción definitiva por los criterios de la guía.
- **Calidad documental:** tres fuentes grises y una sin año editorial corroborado; los perfiles suplementarios no sustituyen la evidencia del trío.
- **Potencia y piloto:** diez semillas y MDE/R retrospectivo no acreditan piloto prospectivo ni potencia 0,80; no se inventa un registro anterior a observar desarrollo.
- **Equivalencia y modelo avanzado:** los márgenes específicos siguen sin fijarse; 5 % frente a persistencia no los reemplaza y no se fabrica TOST/ROPE.
- **Generalización:** un activo y proveedor, desarrollo 2024 expuesto y un backtest histórico final; sin validación anidada de toda la selección, otros mercados o estrés controlado.
- **Diagnósticos:** BDS mensual asintótico con muestras cortas, dependencia de sigma7 y posible deriva; promediar p no es combinar evidencia ni prueba exclusiva de no linealidad aprovechable.
- **Calidad externa:** mecanismo MCAR/MAR/MNAR y ruido externo de precio/etiqueta no identificados; la reconstrucción exacta de sigma7 no mide esos errores.

## Proyección a E2

La evaluación final cierra el trabajo experimental solicitado para estas cuatro entradas, sin declarar completo todo el entregable. Si el test actual se usa para diseñar después un modelo avanzado, deberá separarse una nueva evaluación independiente. Los documentos completos de E2/E3 no se han aportado y no se inventan requisitos o arquitecturas. La rúbrica conserva sus pesos originales; no se asigna una calificación ni un porcentaje de cumplimiento.
