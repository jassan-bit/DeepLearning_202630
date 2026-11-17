# 4 Bloque A Correcciones del Entregable 2

No se han aportado observaciones del docente sobre E2 en el registro recibido. Las páginas de la guía fijan requisitos; no son retroalimentación individual ni aprobación del trabajo. Las filas siguientes identifican **hallazgos internos y evidencia ejecutada**, con sus alcances y pendientes.

El corte actual conserva los 16 sistemas de E2 y amplía las trece redes de seis a diez semillas: **133 predictores, 130 neuronales y tres deterministas**. Esa ampliación es posterior al conocimiento del test y está identificada como tal. No reescribe la A2 histórica, no es una corrección de sus etiquetas y no cierra automáticamente el Bloque A.

## 4.1 Trazabilidad de correcciones

| Observación del docente | Hallazgo interno | Acción realizada y estado | Ubicación en el Jupyter Book | Evidencia | Impacto en resultados |
|---|---|---|---|---|---|
| No aportada | A01 Organización y estado del informe. | Actualizados alcance, estándar, panorama y trazabilidad, conservando capítulos 1–16 y anexos. | [Sección 1](01_introduccion.md), [sección 3](03_panorama_general.md) y [concordancia](concordancia_secciones.md). | Índice, fuentes y matriz de esta sección. | Corrección documental; no altera predicciones ni resultados históricos. |
| No aportada | A02 Auditoría de fuga incompleta y descripción desactualizada de los datos disponibles. | Incorporado el cierre de datos con train655, cal53 y prueba608; 12 controles PASS, 3 PARTIAL y 1 NA. | Sección 4.2. | [Informe de datos fijado](https://github.com/jassan-bit/DeepLearning_202630/blob/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit/audit_result.json). | Reconstrucción de los 608 objetivos con diferencia máxima cero; quedan controles de ports, etiquetas permutadas y clasificación adversarial. |
| No aportada | A03 Alcance del diagnóstico histórico de brecha δg. | Conservado el suplemento CPU: dos de trece redes establecieron tolerancia condicional; la puerta CPU–CUDA estricta no se dio por aprobada. | Sección 4.3. | [Trece contrastes CPU](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/CPU_delta_g_Holm13.csv). | No se convierte ese resultado de seis pares por red en adecuación de diez semillas o de la CNN-LSTM E3. |
| No aportada | A04 Seis semillas históricas y dimensionamiento insuficiente. | Conservadas las seis semillas y añadidas cuatro por cada red, con recetas fijas y todos los ajustes registrados. | Sección 4.4 y [sección 8](08_verificacion_estadistica.md). | [Extensión E2 final](https://github.com/jassan-bit/DeepLearning_202630/tree/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1). | 81→133 predictores y cambios de medias/rangos publicados abajo. Es una ampliación posterior; no un dimensionamiento prospectivo ni prueba de potencia. |
| No aportada | A05 Calendario conocido. | Mantenida la exposición previa de las 608 fechas y el carácter retrospectivo de la comparación. | [Sección 7.2](07_diseno_riguroso.md) y [sección 8.1](08_verificacion_estadistica.md). | Planes e insumos fijados de E2/E3. | Repetir el calendario o ampliar semillas no crea un test ciego ni mercados independientes. |
| No aportada | A06 Novedad del híbrido y separación entre mecanismo y desempeño. | La referencia CNN-LSTM con atención y salida aditiva positiva ya tiene diez semillas y evidencia predictiva favorable; R1 y la defensa académica de R2 permanecen sin acreditar. | [Secciones 6](06_modelo_original.md), [7](07_diseno_riguroso.md) y [8](08_verificacion_estadistica.md). | Revisión de antecedentes, métricas y contrastes posteriores. | RMSE medio de E3 0,0029394324; ventaja observada de 24,10 % frente al mejor E2 ampliado. No demuestra que la atención cause la mejora ni acredita novedad. |
| No aportada | A07 Equidad de particiones y presupuesto. | Documentadas las recetas conservadas y los consumos separados; no se declara equivalencia del tratamiento. | [Sección 8.2](08_verificacion_estadistica.md). | Plan de extensión y registro de ejecución. | E2 usa 655 ejemplos de ajuste y 53 de calibración; E3, 532 de ajuste y 116 de validación. Diez semillas por red no eliminan esas diferencias. |
| No aportada | A08 Familia y procedimientos de E3. | Analizados los 16 rivales y 48 contrastes frente al margen heredado, con Holm global y sensibilidades prefijadas. | [Sección 8](08_verificacion_estadistica.md). | Plan posterior, p crudos/ajustados, intervalos y MCS. | 48/48 rechazos bajo los supuestos de cada procedimiento; MCS conserva diez sistemas. No altera los p históricos de E2 ni completa R3, el preregistro o la potencia exigida. |
| No aportada | A09 CPU y exportación. | Conservadas tres referencias TorchScript y sus reportes de ida y vuelta. | [Sección 5](05_configuraciones_exportacion.md). | Metadatos y paridad CPU–TorchScript. | Paridad puntual sobre calibración histórica; no se equipara a la puerta CUDA, a prueba608 o a exportación de todos los sistemas. |
| No aportada | A10 Reproducibilidad integral. | Fuentes, manifiestos y pruebas de CI disponibles; el análisis posterior reutiliza predicciones fijadas. | [Secciones 8.9](08_verificacion_estadistica.md) y [11](11_informe_tecnico.md). | Registro de 42 pruebas de insumos/métodos y construcción del libro. | Verificación de software y archivos; no sustituye contenedor final y reproducción humana independiente de la campaña. |
| No aportada | A11 Cobertura bibliográfica. | Conservadas las fuentes heredadas y revisiones focales; búsqueda completa y verificación externa R1 pendientes. | [Sección 6.3](06_modelo_original.md). | Matriz, manifiestos y bitácoras de consultas. | No se infiere ausencia de antecedentes de una consulta bloqueada ni se presenta un agente de IA como la persona externa exigida. |

Una observación docente futura debe incorporarse con su fuente, acción, ubicación, evidencia e impacto; ninguna queda corregida por omitirla de esta tabla. Las casillas abiertas permanecen abiertas.

## 4.2 Auditoría de artefactos y datos reales

### Inventario histórico y ampliación actual

El inventario del primer corte de E3 concilió **16 sistemas y 81 predictores**, con 78 trayectorias neuronales y tres deterministas. Sus **167 metadatos** —incluidos 81 escaladores y 81 calibradores— y los **245 hashes** del freeze histórico conservan ese alcance. No se extrapolan esas cantidades a los 133 predictores posteriores. Listar 79 assets de pesos en Releases tampoco equivale a verificar todos sus contenidos.

La ampliación conserva íntegros los 81 predictores y los campos globales del insumo histórico, y agrega **52 ajustes nuevos**, cuatro por red. La auditoría de entrada del análisis posterior verifica los 133 predictores E2 y diez trayectorias E3, las fechas y objetivos exactos y las métricas guardadas. Su [informe](../_static/e3_estadistica_posterior_20261117/input_validation.json) no se presenta como una nueva deserialización independiente de todos los pesos.

### Cierre de datos disponible

El [cierre de datos del 16 de noviembre](https://github.com/jassan-bit/DeepLearning_202630/blob/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit/audit_result.json) tiene estado global **PARTIAL** y clasifica **12 controles PASS, 3 PARTIAL y 1 NA**. La descripción anterior de ocho verificados/siete parciales y de ventanas de prueba no disponibles corresponde a un corte previo y queda reemplazada por esta evidencia.

El archivo fijado `current_cohort_arrays.npz`, SHA-256 `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`, contiene las **1316 ventanas D7×5**: 655 de entrenamiento, 53 de calibración y 608 de prueba, con representaciones crudas, relativas y escaladas, fechas y objetivos. Los controles publicados verifican reconstrucción de sigma7 poblacional, ausencia de covariables del día objetivo/futuro en las ventanas, transformaciones ajustadas en entrenamiento y escaladores físicos. Para los 608 objetivos de prueba, la diferencia máxima respecto al registro congelado fue **cero**.

```{csv-table} Controles del cierre de datos: estado y alcance
:file: ../_static/e3_revision_guia_20261117/data_audit_controls.csv
:header-rows: 1
```

Los soportes crudos de entrenamiento están separados de calibración y prueba. Los identificadores objetivo de calibración/prueba son distintos, pero sus historias móviles comparten **seis cierres, cinco barras de entrada y cinco retornos**. Es pasado disponible para el pronóstico; no se declara independencia entre esas cohortes o entre días solapados.

Siguen parciales **D13**, comportamiento específico de los ports; **D14**, entrenamiento con etiquetas permutadas; y **D15**, clasificador adversarial train/test. Este último diagnosticaría diferencias de distribución, no demostraría fuga por sí solo. El estado NA de D16 responde a que no hay anotadores humanos de una etiqueta continua determinista; no identifica ruido de medida externo ni un límite de Bayes. Los conteos de controles no son un porcentaje de cumplimiento de la guía.

Esta revisión lee informes y archivos existentes; no ejecuta nuevos entrenamientos, inferencias o pruebas de hipótesis. El {download}`readout de Bloque A <../_static/e3_revision_guia_20261117/block_a_readout.json>` conserva fuentes, hashes y límites.

## 4.3 Brecha, determinismo y límites numéricos

El suplemento CPU histórico completó 78 trayectorias y contrastó la brecha **calibración–train** frente a δg, con seis pares por red y Holm13. **Dos de trece redes establecieron tolerancia condicional; once no la establecieron.** Una brecha negativa puede reflejar una calibración más sencilla. Este resultado no se transforma en adecuación de E3 ni en análisis actualizado de los diez ajustes por red.

La puerta anterior de coincidencia CPU frente a CUDA, con tolerancia 10⁻⁷, conserva su fallo y sus ejecuciones parciales. El suplemento CPU no la convirtió retrospectivamente en aprobada. La paridad de exportación CPU–TorchScript de la sección 5 compara otras salidas y tiene su propio alcance.

La extensión E2 documenta **Python 3.12.12, PyTorch 2.8.0+cu126, NumPy 2.3.5, FP32, dos hilos CPU y un hilo inter-op**, con TF32 y AMP desactivados. El ejecutor exige `CUBLAS_WORKSPACE_CONFIG=:4096:8` y algoritmos deterministas de Torch; las semillas 271801–271810 identifican las inicializaciones, no nuevos mercados. El código original fija `random.seed`, `torch.manual_seed` y las semillas CUDA y usa generadores Torch registrados para el muestreo.

No se atribuye al entrenamiento una llamada global `numpy.random.seed` que no existe. Los remuestreos estadísticos sí usan generadores NumPy con semillas registradas en su plan. Documentar esos mecanismos no demuestra igualdad bit a bit entre CPU, CUDA, dispositivos o versiones; el desacuerdo numérico histórico no se oculta. Una auditoría integral debe comprobar cada fuente de aleatoriedad efectivamente usada, no completar esa evidencia con una llamada añadida después de obtener el resultado.

## 4.4 Resultados actualizados y congelamiento

### Antes/después: de seis a diez semillas

La A2 original conserva SHA-256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. El siguiente cuadro **no reemplaza la A2 ni corrige sus etiquetas**: compara sus trayectorias históricas con la ampliación posterior a diez semillas, sobre las mismas 608 fechas y con las mismas recetas. Para cada clásico R sigue siendo uno. La métrica es la media de RMSE individuales, en unidades originales de sigma7.

```{csv-table} Benchmark E2 histórico y ampliado: medias y rangos entre sus dieciséis sistemas
:file: ../_static/e3_revision_guia_20261117/benchmark_seed_extension_before_after.csv
:header-rows: 1
```

El LSTM-autoencoder permanece primero entre los rivales E2, aunque su media cambia de **0,0038034296 a 0,0038728895**. ResNet-1D pasa de rango 8 a 7 y MLP16 de 7 a 8; TimesNet pasa de 11 a 10 y LSTM32 de 10 a 11. También se conservan los cambios desfavorables de las medias de FCN-1D y MLP16. Ningún modelo o semilla se retira según su resultado.

Las diferencias del cuadro proceden de **agregar cuatro inicializaciones**, no de corregir una fuga demostrada, alterar el conjunto de prueba o repetir HPO. La ampliación y los contrastes posteriores de E3 están declarados como posteriores al conocimiento del test: no se presentan como un aumento de semillas dimensionado antes de una confirmación. Los análisis primarios y la conclusión de la A2 histórica conservan su identidad. La [sección 8](08_verificacion_estadistica.md) contiene la comparación de los diecisiete sistemas, incluido E3, sin mezclar esas inferencias con las históricas.

### Versiones fijadas y reglas para cambios futuros

| Evidencia | Versión fijada | SHA-256 o ancla |
|---|---|---|
| A2 histórica | `f65b14d38e38f8d5fd80c717d71f2abd9ea708e2` | `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1` |
| Insumo histórico de 81 predictores | `2121b048d3b379c2c28dfe366cba35a1c08a3617` | `65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187` |
| Insumo ampliado de 133 predictores | `5e9023ea821bfa9fb0b92340d518413f6ab89161` | `df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655` |
| Predicciones E3 de diez semillas | `510c3bddfdc9926322e3210ac61ee1f5eeffad2e` | Fuentes y hashes del plan en [sección 8](08_verificacion_estadistica.md) |

Si se demuestra un error que modifica datos, escalado, predicciones o selección, deben recalcularse **todos los productos dependientes** —métricas, ranking, inferencias, tablas, figuras e informe— en una nueva versión, con causa e impacto antes/después. No basta actualizar una cifra dejando otras inconsistentes; tampoco se sobrescriben las descargas históricas protegidas.

Las recetas y los artefactos fijados de esta evaluación no se retunean para favorecer al candidato. Cualquier corrección documentada conservará el estado anterior, sus hashes y la justificación; los 52 ajustes adicionales no autorizan continuar buscando semillas favorables. El cierre definitivo de Bloque A requiere resolver o delimitar formalmente los controles pendientes y atender la retroalimentación docente que se aporte. **Este capítulo no lo declara cerrado.**

### Evidencias de referencia

- {download}`Inventario E3 de 81 predictores <../_static/e3_bloques_ab_20261115/audit_predictors.csv>`.
- {download}`Resumen del corte de auditoría de artefactos <../_static/e3_bloques_ab_20261115/audit_summary.json>`.
- {download}`Antes/después de la ampliación de semillas <../_static/e3_revision_guia_20261117/benchmark_seed_extension_before_after.csv>`.
- {download}`Controles del cierre de datos <../_static/e3_revision_guia_20261117/data_audit_controls.csv>`.
- {download}`Readout y fuentes fijadas de Bloque A <../_static/e3_revision_guia_20261117/block_a_readout.json>`.
- {download}`Manifiesto del readout <../_static/e3_revision_guia_20261117/block_a_readout_manifest.json>`.
- [Auditorías y diagnósticos posteriores de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html).
- [Tabla anterior de trazabilidad](06_trazabilidad_correcciones.md), conservada como corte de trabajo.
