# Cierre provisional, reproducibilidad, autoría y bibliografía

Este anexo cierra la navegación del informe con los requisitos de reproducción y autoría de las secciones 11–12 y 15. Es un **registro provisional**, no un acta de entrega, firma o reproducción independiente. Las [conclusiones de la sección 16](16_conclusiones.md) mantienen la diferencia entre documentación publicada, software comprobado y resultados científicos pendientes.

## Conclusiones provisionales y límites

E2 aporta un benchmark de 16 sistemas sobre 608 fechas, con diez semillas por cada una de sus trece redes y una trayectoria por clásico: **133 predictores**. E3 dispone de diez réplicas de la **CNN causal → LSTM unidireccional → atención temporal por contenido, con salida aditiva positiva y 1210 parámetros**. El prototipo local de Dash representa los 17 sistemas y ejecuta cuatro ONNX CPU —MLP16, LSTM32, GRU32 y E3— para la semilla representante 271801. Las tres exportaciones TorchScript históricas se conservan; el servicio público en Render sigue pendiente.

El primer candidato tiene formulación, implementación asistida por IA, demostración de una cota de su cabeza latente y 35 pruebas históricas con datos artificiales. Los pilotos v1/v2 aportan cinco recetas entrenadas en desarrollo y la referencia aditiva v2. El [piloto posterior por momentos](https://github.com/jassan-bit/DeepLearning_202630/tree/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1) **completó sus tres recetas**: `moment_joint` obtuvo RMSE de validación 0,0032589201 frente a 0,0031446148 de la referencia aditiva y no superó la heurística de continuación del 2 %. Las diez réplicas aditivas repiten una receta; no son diez diseños nuevos. El Nd global continúa pendiente de conciliación.

La aditiva evaluada tiene **RMSE media 0,0029394324**, DE muestral **0,0000471254** y MAE media **0,0018928273** sobre las mismas 608 fechas del 1 de enero de 2025 al 31 de agosto de 2026. El [análisis estadístico posterior](08_verificacion_estadistica.md) ya completó los 48 contrastes primarios y todos rechazan tras Holm global, bajo los supuestos y ámbitos de incertidumbre declarados. El MCS90 auxiliar conserva diez sistemas, incluido E3; no determina un ganador único. E2 usó 655 ejemplos de ajuste y 53 de calibración; E3, 532 de ajuste y 116 de validación, sin abrir el bloque de calibración. No se afirma igualdad de tratamiento ni se convierte el calendario conocido en una confirmación ciega.

La revisión focal y la bibliografía no establecen ausencia de antecedentes. **R1, la defensa académica de R2 y R3 permanecen pendientes.** La guía requiere tres pruebas distintas apropiadas; sus rechazos no sustituyen el diseño justo, la potencia previa, los conjuntos adicionales ni los restantes requisitos. Los tests de software conservan su alcance y las propiedades de los mecanismos anteriores no se transfieren automáticamente al modelo de 1210 parámetros.

La paridad ONNX se comprobó en 608 fechas frente a memoria CPU y pronósticos congelados, con tolerancia absoluta 10⁻⁶ en σ₇. La etapa de ingeniería ejecutó inferencias CPU, sin ajustar pesos o calibradores ni usar GPU; el análisis estadístico anterior mantiene su alcance sin inferencia nueva. El historial de entrenamiento de una semilla se recuperó del checkpoint y no es una ejecución nueva.

Las etapas siguientes son cerrar auditorías y fuentes admisibles, revisión/hipótesis/diseño, separación de confirmación, datasets, presupuesto y potencia antes de un protocolo definitivo; completar evaluación ampliada, explicaciones y calibración faltantes; verificar el paquete final, desplegar el servicio y reproducir la campaña de forma independiente. La app local, las fichas y los reportes de ingeniería son productos con alcance concreto; no convierten los borradores históricos en preregistros ni el prototipo en una entrega completa.

La revisión de los bloques F/G conecta el procedimiento de despliegue y humo de [10](10_despliegue_render.md), la estructura científica y el acta del tercero de [11](11_informe_tecnico.md), la matriz de los trece entregables de [12](12_lista_entregables.md) y el guion de exposición de [13](13_sustentacion.md). El resumen con aportes técnicos y las ecuaciones numeradas mejoran la comunicación y trazabilidad del informe; no acreditan por sí solos novedad, confirmación o una reproducción ejecutada. Render todavía no tiene un servicio del proyecto cuya URL pueda verificarse aquí.

## Lista de reproducibilidad al cierre

| Elemento | Evidencia disponible | Pendiente para cierre completo |
|---|---|---|
| Versión de fuentes | Git, manifiestos de los cortes, CITATION.cff y fuentes del libro. | Vincular la entrega exacta con archivo verificable, tag y DOI/identificador persistente; CITATION sin ese identificador no lo sustituye. |
| Entorno documental | `requirements-book.lock.txt`, configuración y workflow de construcción de los tres libros; ecuaciones MyST numeradas y referencias cruzadas del informe. | Reconstruir/verificar la versión final con sus enlaces y recursos preservados; esto no ejecuta la campaña científica. |
| Entornos científicos y de servicio | Dependencias fijadas e imagen Docker local de ONNX Runtime sin Torch, comprobada en ocho rutas con usuario sin privilegios y sólo lectura; entorno científico separado. | Desplegar y verificar el servicio público; contenedor científico completo y reproducción humana siguen pendientes. |
| Hardware y precisión | Perfiles CUDA de las corridas E3 y CPU del análisis estadístico y exportaciones, con versiones y alcance. | Consolidar hardware/backend/hilos/lotes de la entrega, sin mezclar eager CPU, TorchScript y GPU. |
| Semillas y determinismo | Redes de E2 y E3 con semillas 271801–271810; semilla histórica E3 20261116 separada y tres clásicos con una trayectoria cada uno. | Dimensionar por potencia las ejecuciones confirmatorias, documentar políticas de generadores y límites entre plataformas; diez semillas no prueban potencia suficiente. |
| Hiperparámetros y esfuerzo | Recibos E2, cinco recetas de v1/v2, tres por momentos y réplicas aditivas congeladas. | Espacios/presupuestos y ajuste final comparables, Nd global conciliado, fallos/decisiones y protocolo previo a la confirmación. |
| Métricas y tabla maestra | Tabla ampliada de 17 sistemas y bundle del dashboard derivado de ella con igualdad probada; curvas de semilla representante, efectos e IC con sus unidades declaradas. | Tres conjuntos adicionales, calibración de E3 y todos los campos comparables de recursos/exportación del catálogo final. |
| Datos y derechos | DATA_STATEMENT.md, hashes y 1316 observaciones de ventanas auditadas; fechas t−1/t diferenciadas y auditoría 12 PASS, 3 PARTIAL y 1 NA. | Resolver controles abiertos, ratificar derechos por fuente y aprobar la licencia propia del proyecto. La declaración y los hashes no cierran por sí solos el Bloque A o esas licencias; las MIT de recursos externos conservan su alcance. |
| Artefactos y paridad | Cuatro ONNX y metadatos verificados en 608 fechas para la semilla 271801 con tolerancia 10⁻⁶; tres TorchScript históricos conservados. | Trece exportaciones restantes, política común de ajuste y verificaciones completas de incertidumbre y explicación. El intervalo E3 no está calibrado. |
| Checkpoints y estados | Snapshots de las fases nuevas de E2 y diez checkpoints E3 con pesos, estados de optimización/RNG y metadatos; los modelos seleccionados de E2 y los artefactos de servicio tienen otro alcance. | Completar la política común de ajuste final y la verificación integral de carga/reanudación/exportación. La inspección de código e inventario no equivale a deserializar todos los binarios. |
| Cómputo y energía | Consumos cerrados: ampliación E2 5130,9707 s; E3 acumulado 948,4396 s, incluidos consumos previos. Análisis estadístico CPU sin ajustes ni inferencias nuevos. | Costes comparables completos y energía medida o estimada con método explícito; no inventar kWh ni confundir reloj, CPU y horas GPU. |
| Verificación de software | 42 pruebas estadísticas históricas y 50 A/B con sus alcances. El [CI del dashboard del corte de base](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213072051) aprobó 25 casos y seis recorridos de navegador; científico: 60 aprobados y una referencia SciPy opcional omitida, incluida en los 61 aprobados localmente. | Conservar los recibos de la versión final y la prueba pública sin sumar coberturas como ejecuciones de una campaña científica; estos tests y los checks de bytes no demuestran reproducción humana. |
| Reproducción independiente | [Acta prevista en 11](11_informe_tecnico.md), con filas completas emparejadas, tolerancias previas y discrepancia máxima separada por campo/unidad. | Una persona ajena al código, contenedor científico limpio, tabla principal regenerada, discrepancias observadas y concepto firmado. |
| Servicio público | Prototipo local de tres páginas y cuatro modelos CPU descrito en [9](09_dashboard_predictivo.md); guion y procedimiento de humo en [10](10_despliegue_render.md). | Crear el servicio Render y registrar URL HTTPS, humo público por página/modelo, logs, tiempos y contingencia de esa versión desplegada. Un guion o recibo local no se registra como éxito público. |
| Autoría y defensa | Procedencia con asistencia IA y [guion de 20 minutos](13_sustentacion.md), con 18 de contenido y dos de margen, nueve preguntas y comandos de práctica comprobados. | Contribuciones individuales confirmadas, firmas humanas, ensayo y dominio del algoritmo comprobado durante la defensa. |

Las verificaciones `python scripts/e3_build_handoff_book.py --check` y `python scripts/e3_verify_original.py --check` tienen ámbitos de evidencia y software definidos. La construcción `jupyter-book build book_e3 --all --warningiserror --keep-going` y el comprobador de enlaces verifican documentación. Ninguno de esos comandos, por sí solo, entrena la campaña E3, reproduce un servicio Render o convierte a la IA en una persona independiente del código.

## Registro de reproducción independiente por completar

El informe deberá identificar revisor humano y relación con el proyecto; fecha; commit/tag y archivo/DOI; digest del contenedor científico; hardware; versiones; hashes de datos/artefactos; comandos y logs; semillas; productos regenerados y productos sólo comprobados. La [sección 11.4](11_informe_tecnico.md) define la máxima discrepancia por campo y unidad sobre todas las filas emparejadas. Se fijarán antes las tolerancias absoluta y relativa, se conservarán ausencias y se reportarán diferencias, incumplimientos, fallos y causa investigada, sin mezclar segundos, memoria y métricas en un único máximo.

**Estado: no ejecutado ni firmado.** No hay una discrepancia cero que informar de esa reproducción. Las comprobaciones de las tres TorchScript históricas y cuatro ONNX recientes conservan sus tolerancias y alcances propios, igual que CI.

## Autoría, contribuciones y uso de IA

La preparación del [piloto exploratorio de momentos](07_diseno_riguroso.md) empleó ChatGPT/Codex para implementación del modelo, ejecutor, guardado y reanudación en GitHub, pruebas sobre datos artificiales, revisión de código y documentación del notebook. La ejecución posterior del piloto está respaldada por sus resultados guardados y se distingue del corte inicial de preparación. La asistencia de IA también intervino en los ejecutores de réplicas, el análisis estadístico posterior y la actualización del libro. Sus commits, versiones, métricas y consumos conservan el alcance de cada etapa; esto no equivale a una verificación humana externa y la persona autora mantiene la responsabilidad de revisar y defender el código.

El [dictamen posterior sobre la cabeza de momentos](22_dictamen_momentos.md) también usa asistencia de ChatGPT/Codex para lectura focal, búsqueda, crítica matemática, contraejemplos escalares, código de auditoría y documentación. Su [procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/provenance.md) declara ese alcance. El paquete de revisión humana está sin diligenciar y no se presenta como aval externo ni firma.

La [propuesta posterior al cierre v2](21_propuesta_momentos.md) también fue preparada con asistencia de ChatGPT/Codex para revisión focal, formulación, demostración, crítica matemática, verificación escalar, pruebas y documentación. Su [declaración de procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_mecanismo_momentos_20261116/provenance.md) atribuye los componentes conocidos y mantiene pendiente la revisión humana, la novedad y la defensa del estudiante. No se entrenó una red nueva en ese corte.

La autoría declarada en la portada se mantiene como dato del proyecto. No se atribuyen automáticamente contribuciones individuales ni se afirma una firma o defensa que no se haya verificado.

La implementación del dashboard, sus componentes de datos e inferencia, exportaciones, validaciones, pruebas, fichas y documentación también empleó asistencia de IA. Los recibos de navegador y de software describen comprobaciones técnicas, no la revisión humana independiente ni una defensa académica. La persona autora debe revisar y poder modificar este código y sus supuestos.

La revisión de las secciones 10–13 también emplea asistencia de ChatGPT/Codex para el guion de humo, la estructura y numeración del informe, el registro de productos y la preparación técnica de la exposición. Los comandos de práctica sintética y de inferencia son comprobaciones de ingeniería; no constituyen un ensayo personal realizado, acta del tercero, firma o defensa del estudiante.

La [procedencia existente](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/procedencia.md) declara que el módulo candidato fue escrito con asistencia de IA desde la formulación. La asistencia también se ha usado para organizar requisitos, redactar documentación, preparar herramientas de auditoría/exportación y comprobar software. Los capítulos deben conservar ese alcance y la revisión humana pendiente; no se presentan como implementación o demostración comprendidas y defendidas por la persona autora sólo porque existan archivos.

Antes de la entrega, cada integrante deberá completar y confirmar este registro y firmar su propia declaración:

| Campo | Contenido a registrar | Estado |
|---|---|---|
| Identidad y participación | Nombre de cada integrante y relación con la versión entregada. | Por confirmar. |
| Contribuciones individuales | Actividades concretas, archivos/decisiones y evidencia de su revisión. | Por declarar; no asignadas por la IA. |
| Herramientas | Nombre/versión o identificación disponible, usos, alcance y archivos afectados, incluidas herramientas IA. | Asistencia descrita; inventario final por validar. |
| Comprensión y responsabilidad | Qué código, derivaciones, resultados y limitaciones revisó y puede explicar/modificar cada persona. | No certificado por los tests. |
| Confirmación | Declaración personal, fecha real y firma de cada integrante. | Sin firmas. |

Esta plantilla no es una declaración suscrita. La [sustentación](13_sustentacion.md) deberá permitir explicar y modificar el mecanismo y sus pruebas sin ocultar la asistencia recibida.

## Bibliografía de partida y alcance de lectura

La bibliografía configurada tiene dos fuentes separadas. `book_e3/references.bib` conserva intactos los siete registros del corte del candidato. Los originales de Kim y colaboradores {cite:p}`kim2021lipschitz` y Dasoulas y colaboradores {cite:p}`dasoulas2021lipschitz` tuvieron lectura focal documentada; la matriz de la [sección 6](06_modelo_original.md) distingue los otros antecedentes, metadatos heredados y límites de consulta.

`referencias_metodologicas_v1.bib` incorpora las 37 sugerencias de la guía, indexadas con su [número original y aplicabilidad](referencias_sugeridas.md). No son 37 búsquedas o lecturas nuevas, ni se agregan automáticamente al corpus de novedad o al PRISMA. Su inclusión no afirma que todos sus procedimientos sean aplicables a F2a.

Se conservan la descarga original de {download}`references.bib, siete antecedentes del candidato <../references.bib>` y la nueva descarga de {download}`referencias_metodologicas_v1.bib, 37 sugerencias de la guía <../referencias_metodologicas_v1.bib>`. La lista renderizada reúne los 44 registros. Cerrar R1 requiere completar/repetir la revisión, clasificar diferencias y obtener la verificación externa exigida; una lista BibTeX no demuestra novedad.

```{bibliography}
:all:
:style: unsrt
```
