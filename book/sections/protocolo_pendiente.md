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
| Margen del estudio ejecutado | 5 % de reducción relativa de RMSE frente a persistencia; conserva su fecha y alcance original |
| Criterio académico futuro | Mejora superior al 5 % frente a MLP/D7 y persistencia sobre la misma evaluación nueva, con incertidumbre y dos comparaciones; definido después del test actual |
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

La revisión bibliográfica vigente reúne **36 referencias y seis textos completos**. La recuperación de Miura, Pichl y Kaizoji (2019) incorpora comparadores MLP, LSTM y CNN independientes en volatilidad de Bitcoin; el [conteo actualizado](../_static/literature_e1/recovery_20261105/family_frequencies.csv) acredita MLP en 1/6 estudios y CNN en 1/6 como familias. El artículo no declara dimensionalidad CNN ni configura las recetas exactas del proyecto. El corte anterior de **35 referencias, cinco textos y 81 cifras** conserva sus frecuencias originales como antecedente histórico, no como estado actual. La revisión es posterior al benchmark; no acredita selección bibliográfica a priori, recurrencia amplia, consulta nativa WoS/Scopus o cuartil Q1.

## Matriz de cierre derivada de la auditoría

La auditoría del **PDF de 34 páginas** distingue requisitos explícitos, observaciones de coherencia y mejoras. Los identificadores A01–A23 permiten seguir el trabajo sin convertir el recuento de hallazgos en una nota o porcentaje de cumplimiento. Los archivos de aquella auditoría se conservan como corte previo; la [actualización documental de S8](../_static/literature_e1/recovery_20261105/selection_update.json) sustituye la ausencia anterior de fuente aplicada MLP/CNN por evidencia de familia con límites. No se declara por ello cerrado todo A01. Los cinco cierres prioritarios quedan:

| Prioridad | Requisito y origen | Estado actual | Evidencia o criterio de cierre |
|---|---|---|---|
| **A01 · Alta** | Extraer los tres baselines de literatura aplicada; PDF §§3–3.3, pp. 12–15 | **Respaldo aplicado de familias actualizado**: S8 acredita MLP y CNN independientes; subsisten recurrencia limitada, dimensionalidad CNN no reportada y adaptación exacta | [Fuente primaria y Tabla 1](../_static/literature_e1/recovery_20261105/primary_source.json), [síntesis de seis estudios](../_static/literature_e1/recovery_20261105/synthesis_six_columns.csv) y selección con fecha posterior al benchmark. Las recetas ejecutadas no se presentan como réplicas ni como selección a priori derivada de S8. |
| **A02 · Alta** | Margen y planificación frente al baseline; PDF §5.3, p. 23 | **Criterio académico definido para modelos futuros; sensibilidad y planificación calculadas** | Mejora >5 % frente a MLP/D7 y persistencia; 36 sensibilidades y 27 escenarios conjuntos de frontera práctica. Faltan piloto de dispersión real, potencia real y una evaluación nueva; no se atribuye este criterio al test ya realizado. |
| **A03 · Alta** | Comparación formal del benchmark final; PDF §§5.3–5.4 y 6.4 | **Ejecutado como suplemento post-hoc**; [resultados](../_static/e1_audit_improvements_20261104/inference/summary.json) | Tres pares entre las configuraciones D7 seleccionadas, diferencias/IC/efectos, multiplicidad y diagnóstico temporal. Se declarará posterior a observar test; no requiere otros entrenamientos o pronósticos. |
| **A04 · Alta** | Piloto y dimensionamiento previos del procedimiento vigente; PDF §§4.1–4.3, pp. 17–19 | **Pendiente para una nueva evaluación**; borrador preparado | [Plantilla prospectiva DRAFT](../_static/e1_audit_improvements_20261104/next_study_pilot_DRAFT.json) con al menos cinco semillas por base, media/DE/CV, δ/σd y R/MDE antes de ejecutar. No es un piloto realizado ni un registro completo; el piloto histórico pertenece a otro procedimiento. |
| **A05 · Alta** | Entrega reproducible de extremo a extremo; PDF §§4.2–4.3 y entregables p. 33 | **Ruta privada de restauración y auditoría validada**; [recibo](../_static/e1_audit_improvements_20261104/reproducibility/summary.json) | Un evaluador restaura o replica en una carpeta limpia siguiendo instrucciones exactas y verifica hashes sin borrar el marcador del estudio original. Los paquetes completos pueden entregarse privadamente. |


La ruta A05 valida una **réplica de evidencia**, no una repetición nueva de los 120 entrenamientos: conserva 113 entradas, seis TAR y 16.890 miembros verificados en una copia aislada, con auditoría de métricas/firmas y cero ajustes, nuevos pronósticos o lecturas de ZIP de mercado. [BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) distingue el sitio de resúmenes del paquete privado para un evaluador. El registro original y su marcador de test se mantienen intactos.

El suplemento A03 se calculó después de observar test; ese momento queda declarado. Los seis pares RMSE, seis DM principales, nueve Ljung–Box y bootstrap temporal tienen unidades/familias distintas. Las tablas completan evidencia del benchmark, sin volver retrospectivamente prospectivo el experimento ni reemplazar MLP/D7 por otro ganador de test.

**Aclaración sobre la partición:** el embargo D está exigido por la ruta 6.4 de la guía, página 29, porque D es el máximo rezago de covariables. No es una condición nativa de la API `timeseries-cv`. Los porcentajes 27,26/27,33/45,41 son duraciones del calendario elegido, no proporciones impuestas por Filipe.

## Mejoras concretas y evidencia de cierre

| Identificadores | Acción | Estado verificable |
|---|---|---|
| A06 · HPO y procedencia | Conectar búsqueda histórica, receta elegida y transferencia a la CV relativa | **Documentado en §3.2.5**: 96 corridas históricas igualadas, anchos 16/32 y LR 0,001/0,003; las diez semillas actuales son otra etapa. No se afirma que nunca hubo HPO ni que la transferencia garantiza optimalidad. |
| A07–A08 · Literatura/dataset | Ampliar recuperación y contrastar proveedor, par, frecuencia, periodo y objetivo de cada estudio | **Etapas separadas**: el corte anterior de 13 cadenas/18 páginas y 92 repositorios no incluyó textos nuevos; la recuperación posterior incorpora S8. Su RV de tres horas, Bitstamp e historia de RV difieren de la σ7 diaria y covariables OHLCV propias; no se afirma igualdad de dataset ni comparabilidad numérica directa. |
| A09 · Faltantes | Mantener exclusión causal y aportar evidencia del mecanismo | **Parcial**: conteos, calendario y ventanas afectadas verificados; MCAR/MAR/MNAR no identificado. No se asigna una categoría sin evidencia. |
| A10–A14 · Resultados y diagnóstico | Añadir resultados anuales, variabilidad cuantitativa, residuos/segmentos finales y diferencias de MAE | **Ejecutado**: [tablas anuales](../_static/e1_audit_improvements_20261104/descriptive/annual_metrics_summary.csv), [descomposición](../_static/e1_audit_improvements_20261104/descriptive/variance_components.csv), [distribuciones](../_static/e1_audit_improvements_20261104/descriptive/distribution_by_period.csv), residuos por bins y diferencias MAE/RMSE. Usa predicciones guardadas, sin volver a puntuar modelos. |
| A15–A16 · Coherencia | Separar el control absoluto D7 de las cuatro historias y distinguir snapshots históricos | **Corregido en §3.2.2 y §3.2.5**; los registros históricos conservan su contenido y su identificador. |
| A17 · Diagramas | Mostrar formas de tensor, capas y parámetros reales | **Ejecutado en §3.3 y §4.2.1**: SVG de MLP16, LSTM32 y CNN16, entradas D×5, cabeza positiva y parámetros verificados contra la implementación. |
| A18–A21 · Robustez | Interpretar BDS corto, justificar épocas finales y añadir diagnóstico agregado por periodo | **Mejoras con alcance**: BDS mensual es exploratorio; las épocas vienen del último corte de desarrollo; no se reoptimizan mirando test. Ridge bajo otro procedimiento sigue siendo diagnóstico contextual. |
| A22 · Calendario real | Separar el ejemplo sintético de la ejecución y los porcentajes de calendario de las muestras | **Ejecutado en §2.1.4**: figura reconstruida desde manifiestos con doce bloques mensuales por D, embargo y test de pesos finales fijos; no dibuja 366 reajustes diarios. |
| A23 · Equivalencia | Separar requisito obligatorio de análisis condicional | **Aclarado**: TOST/ROPE y su margen solo se requieren si se quiere afirmar equivalencia. Ausencia de significancia no prueba equivalencia. |

## Límites que permanecen abiertos

- **Alcance del respaldo aplicado:** la ausencia de fuente primaria para las familias MLP y CNN quedó resuelta con S8. Ampliar recurrencia y documentar dimensionalidad/configuraciones desconocidas sigue siendo pertinente; la Conv1D propia es una adaptación y los datasets no son idénticos. No se acredita un registro individual WoS/Scopus o Q1 a partir de un DOI o del nombre de una editorial. La preferencia por Q1 no es una obligación universal ni se inventa un mínimo de papers.
- **Diseño prospectivo:** hubo piloto histórico de cinco semillas por familia; diez semillas y sensibilidad retrospectiva del procedimiento actual no lo convierten en un piloto previo de esta CV. Se necesita un registro nuevo antes de la evaluación futura.
- **Criterio académico avanzado:** queda definido para el futuro como mejora >5 % frente a MLP/D7 y persistencia, con incertidumbre y dos comparaciones. Los escenarios conjuntos son ilustrativos; siguen pendientes el piloto con dispersión real, la potencia real y una evaluación nueva. No se afirma utilidad financiera y la equivalencia sigue siendo condicional.
- **Generalización:** un activo/proveedor y desarrollo 2024 expuesto; las semillas, historias y meses no son mercados independientes. Usar este test para diseñar otra propuesta exige una nueva evaluación independiente.
- **Calibración:** BDS mensual con 28–31 residuos y etiquetas solapadas tiene referencia asintótica limitada; promediar p-valores no combina evidencia ni identifica exclusivamente no linealidad aprovechable.
- **Calidad externa:** mecanismo MCAR/MAR/MNAR y ruido externo de precio/etiqueta no identificados. La reconstrucción exacta de σ7 acredita implementación de la etiqueta, no ausencia de error en su proxy.

## Proyección a E2

La evaluación final cierra el trabajo experimental solicitado para estas cuatro entradas, sin declarar completo todo el entregable. Si el test actual se usa para diseñar después un modelo avanzado, deberá separarse una nueva evaluación independiente. Los documentos completos de E2/E3 no se han aportado y no se inventan requisitos o arquitecturas. La rúbrica conserva sus pesos originales; no se asigna una calificación ni un porcentaje de cumplimiento.
