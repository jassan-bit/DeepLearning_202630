# 11. Selección HPO ejecutada y equidad

**Estado vigente del 14/11/2026:** la selección reducida de las **trece redes obligatorias** está cerrada. La comparación final utiliza los diez candidatos, las tres referencias E1 y los comparadores Persistence, SeasonalNaive y ARIMA. No hay un candidato 11 adicional activo dentro de la obligación de E2. Las recetas y recibos se consultan en [§48](48_protocolo_reducido.md).

## 11.1 Dominio y oportunidades conservadas

Cada red recibió **seis configuraciones y los mismos tres orígenes internos**: enero, julio y octubre de 2024, con semilla HPO 271201. Son **234 oportunidades**: 231 resultados válidos, dos fallos conocidos y un desenlace desconocido. Cada ganador dispone de tres folds completos. Se mantuvieron las incidencias sin sustituirlas para favorecer un modelo.

La selección usa exclusivamente las métricas de validación de los folds admitidos. Los dominios de seis plazas proceden de una reducción explícita de la búsqueda histórica, no de una afirmación de exploración exhaustiva. Los 61 resultados reutilizados requieren identidad de receta, configuración, firma de datos y trabajador compatible; no se convierten en nuevos ajustes. La cobertura de curvas HPO seleccionadas permanece incompleta: se recuperaron 29 de 39 registros, mientras los 78 ajustes finales sí conservan sus curvas en [§49](49_diagnosticos_f2a.md).

## 11.2 Presupuesto común y diferencias estructurales

El número de plazas y folds es común para las redes. SAE conserva sus fases de preentrenamiento y supervisión propias; igualdad de oportunidades no implica igual número de parámetros, pasos o segundos. Los límites de admisión y la transición T4–A100 se registraron por separado. Los [costes medidos](51_eficiencia_costes.md), el [perfil A100](54_robustez_interpretabilidad.md) y los [FLOPs core](58_flops_predictores.md) describen magnitudes diferentes.

ARIMA evaluó **18 órdenes × tres folds**, con 51 resultados válidos, dos no convergencias y un fallo de implementación que consumió su oportunidad. Quince órdenes tuvieron tres folds válidos y ganó (2,0,1). SeasonalNaive eligió periodo 7 entre tres periodos. Los comparadores deterministas aportan una trayectoria cada uno, sin duplicarlos como seis entrenamientos independientes. El ajuste clásico y el neuronal tienen dominios diferentes; no se certifica igualdad de coste global únicamente por compartir la cohorte final.

## 11.3 Arquitectura definida y NAS condicional

Los cuerpos actuales están adjudicados en [§44](44_arquitecturas_publicadas.md) y la actualización TiDE r=0 de [§48](48_protocolo_reducido.md). Sus identidades y adaptaciones D7 se preservan. La exigencia de **1.000 arquitecturas y dos fases NAS** corresponde a cuerpos no definidos por sus fuentes. Las seis plazas HPO no se presentan como cumplimiento de NAS. Introducir una arquitectura indefinida nueva exigiría su protocolo propio; no se hace en este cierre.

## 11.4 Ajustes finales y alcance de las comparaciones

Cada red se reinicializó con **seis semillas, 271801–271806**, después de elegir su configuración por validación. La A2 promedia las seis RMSE individuales sobre las mismas 608 fechas; la fase de selección no se mezcla con las ejecuciones finales. La potencia estimada del piloto fue insuficiente para el efecto planificado. La ausencia de rechazo se interpreta como resultado no concluyente y no como equivalencia.

El periodo evaluado ya se conocía en E1. Se declara su alcance retrospectivo y no se elige una receta distinta usando el ranking final. La [matriz de cumplimiento](38_cumplimiento_guia.md) mantiene las condiciones metodológicas parciales, incluida la auditoría integral de fuga y la cobertura del dominio reducido.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/11_hpo_equidad.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
