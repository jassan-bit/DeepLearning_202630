(seleccion-hpo-ejecutada-y-equidad)=
# 11. Protocolo de Entrenamiento, Ajuste y Costo Computacional

**Lectura vigente:** la selección reducida de las **trece redes obligatorias** está cerrada. La comparación final utiliza los diez candidatos, las tres referencias E1 y los comparadores Persistence, SeasonalNaive y ARIMA. No hay un candidato 11 adicional activo dentro de la obligación de E2. Las recetas y recibos se consultan en [§48](48_protocolo_reducido.md). La [implementación vigente](04_implementacion.md) y la [lista del punto 4](../_static/e2_implementation_current_v1/guide_checklist.csv) relacionan diseño, entrenamiento, recursos y evidencia de evaluación con la guía.

## 11.1 Dominio y oportunidades conservadas

Cada red recibió **seis configuraciones y los mismos tres orígenes internos**: enero, julio y octubre de 2024, con semilla HPO 271201. Son **234 oportunidades**: 231 resultados válidos, dos fallos conocidos y un desenlace desconocido. Cada ganador dispone de tres folds completos. Se mantuvieron las incidencias sin sustituirlas para favorecer un modelo. Las 231 oportunidades válidas incluyen reutilizaciones admitidas; no equivalen a 231 entrenamientos nuevos. El plan identificó 61 candidatos de caché sujetos a sus controles de identidad; este cotejo verifica el total de resultados y disposiciones, sin reconstruir una atribución exhaustiva caché/nuevo del diario. El dominio original no se sustituye por un supuesto de cobertura exhaustiva.

La selección usa exclusivamente las métricas de validación de los folds admitidos. Los dominios de seis plazas proceden de una reducción explícita de la búsqueda histórica, no de una afirmación de exploración exhaustiva. Cada reutilización requiere identidad de receta, configuración, firma de datos y trabajador compatible; no se convierte en un nuevo ajuste. La cobertura de curvas HPO seleccionadas permanece incompleta: se recuperaron 29 de 39 registros, mientras los 78 ajustes finales sí conservan sus curvas en [§49](49_diagnosticos_f2a.md).

La [auditoría del calendario ejecutado](../_static/e2_eda_current_v1/partition_audit.json) identifica **253/435/527 ejemplos de entrenamiento y 83 de validación en cada fold**. Los 249 resultados de selección preceden a la evaluación externa. Después se usan **655 ejemplos de entrenamiento final, 53 fechas de calibración y 608 de evaluación**; los entrenamientos expansivos se solapan y los folds no son réplicas independientes.

## 11.2 Presupuesto común y diferencias estructurales

El número de plazas y folds es común para las redes. SAE conserva sus fases de preentrenamiento y supervisión propias; igualdad de oportunidades no implica igual número de parámetros, pasos o segundos. Los límites de admisión y la transición T4–A100 se registraron por separado. La política de recursos para fases nuevas fijó **1.800 segundos, máximo 200 épocas y paciencia 20**; las reservas fueron 1.860 segundos por ajuste ordinario y 5.460 para las tres fases SAE. Los resultados reutilizados conservan la política con la que se obtuvieron. Estos límites y un número común de plazas no demuestran igual coste acumulado de optimización. Los [costes medidos](51_eficiencia_costes.md), el [perfil A100](54_robustez_interpretabilidad.md) y los [FLOPs core](58_flops_predictores.md) describen magnitudes diferentes.

ARIMA evaluó **18 órdenes × tres folds**, con 51 resultados válidos, dos no convergencias y un fallo de implementación que consumió su oportunidad. Quince órdenes tuvieron tres folds válidos y ganó (2,0,1). SeasonalNaive eligió periodo 7 entre tres periodos. Los comparadores deterministas aportan una trayectoria cada uno, sin duplicarlos como seis entrenamientos independientes. El ajuste clásico y el neuronal tienen dominios diferentes; no se certifica igualdad de coste global únicamente por compartir la cohorte final.

## 11.3 Arquitectura definida y NAS condicional

Los cuerpos actuales están adjudicados en [§44](44_arquitecturas_publicadas.md) y la actualización TiDE r=0 de [§48](48_protocolo_reducido.md). Sus identidades y adaptaciones D7 se preservan. La exigencia de **1.000 arquitecturas y dos fases NAS** corresponde a cuerpos no definidos por sus fuentes. Las seis plazas HPO no se presentan como cumplimiento de NAS. Introducir una arquitectura indefinida nueva exigiría su protocolo propio; no se hace en este cierre.

## 11.4 Ajustes finales y alcance de las comparaciones

Cada red se reinicializó con **seis semillas, 271801–271806**, después de elegir su configuración por validación. La A2 promedia las seis RMSE individuales sobre las mismas 608 fechas; la fase de selección no se mezcla con las ejecuciones finales. El piloto de cinco pares iTransformer/GRU32 sobre 53 fechas estimó potencia condicional **0,122953** con R=6 y α=0,0025 para una ganancia de 2δ; el objetivo de 0,80 no se alcanzó. Su MDE fue **0,000878674**, con los supuestos y límites del [recibo del piloto](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json). La ausencia de rechazo se interpreta como resultado no concluyente y no como equivalencia; no se aumenta R después de observar la prueba.

El periodo evaluado ya se conocía en E1. Se declara su alcance retrospectivo y no se elige una receta distinta usando el ranking final. La [matriz de cumplimiento](38_cumplimiento_guia.md) mantiene las condiciones metodológicas parciales, incluida la auditoría integral de fuga y la cobertura del dominio reducido.


## 11.5 Registro, entorno y cierre de recursos

Los 78 ajustes finales y los diez pilotos se ejecutaron en **NVIDIA A100-SXM4-80GB, Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6**, con FP32 determinista, AMP y TF32 desactivados y dos hilos CPU. El historial T4 pertenece a selección/caché; no se atribuye igualdad numérica entre GPU distintas. Los máximos de memoria de entrenamiento y los perfiles de inferencia posteriores tienen ámbitos distintos, declarados en §§51/54.

El seguimiento usa **configuraciones versionadas, eventos, manifiestos, hashes, checkpoints y diarios aditivos respaldados en GitHub**, como herramienta de tracking del proyecto. Se registran semillas, intentos/fallos, curvas, costes y estados terminales; no se exige atribuir el seguimiento a MLflow o W&B. La reproducción posterior cerró **81 predictores y 18 ablaciones**, conservando recetas y cargos. El límite final fue **84 horas totales**, con **80,2561 horas contabilizadas**, **3 h 44 min 38 s de saldo** y cero reservas abiertas. Son horas contables de campaña, no GPU-horas activas ni dinero. [Cierre verificado](55_reproduccion_ablaciones.md).

## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/11_hpo_equidad.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
