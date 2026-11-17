# Cierre provisional, reproducibilidad, autoría y bibliografía

Este anexo cierra la navegación del informe con los requisitos de reproducción y autoría de las secciones 11–12 y 15. Es un **registro provisional**, no un acta de entrega, firma o reproducción independiente. Las [conclusiones de la sección 16](16_conclusiones.md) mantienen la diferencia entre documentación publicada, software comprobado y resultados científicos pendientes.

## Conclusiones provisionales y límites

E2 aporta un benchmark de 16 sistemas sobre 608 fechas, con diez semillas por cada una de sus trece redes y una trayectoria por clásico: **133 predictores**. Su ampliación conserva las seis semillas históricas y las recetas seleccionadas. E3 dispone de diez réplicas de la **CNN causal → LSTM unidireccional → atención temporal por contenido, con salida aditiva positiva y 1210 parámetros**. Las tres exportaciones TorchScript verificadas en calibración histórica pertenecen a referencias de E2; el servicio en Render sigue pendiente.

El primer candidato tiene formulación, implementación asistida por IA, demostración de una cota de su cabeza latente y 35 pruebas históricas con datos artificiales. Los pilotos v1/v2 aportan cinco recetas entrenadas en desarrollo y la referencia aditiva v2. El [piloto posterior por momentos](https://github.com/jassan-bit/DeepLearning_202630/tree/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1) **completó sus tres recetas**: `moment_joint` obtuvo RMSE de validación 0,0032589201 frente a 0,0031446148 de la referencia aditiva y no superó la heurística de continuación del 2 %. Las diez réplicas aditivas repiten una receta; no son diez diseños nuevos. El Nd global continúa pendiente de conciliación.

La aditiva evaluada tiene **RMSE media 0,0029394324**, DE muestral **0,0000471254** y MAE media **0,0018928273** sobre las mismas 608 fechas del 1 de enero de 2025 al 31 de agosto de 2026. El [análisis estadístico posterior](08_verificacion_estadistica.md) ya completó los 48 contrastes primarios y todos rechazan tras Holm global, bajo los supuestos y ámbitos de incertidumbre declarados. El MCS90 auxiliar conserva diez sistemas, incluido E3; no determina un ganador único. E2 usó 655 ejemplos de ajuste y 53 de calibración; E3, 532 de ajuste y 116 de validación, sin abrir el bloque de calibración. No se afirma igualdad de tratamiento ni se convierte el calendario conocido en una confirmación ciega.

La revisión focal y la bibliografía no establecen ausencia de antecedentes. **R1, la defensa académica de R2 y R3 permanecen pendientes.** La guía requiere tres pruebas distintas apropiadas; sus rechazos no sustituyen el diseño justo, la potencia previa, los conjuntos adicionales ni los restantes requisitos. Los tests de software conservan su alcance y las propiedades de los mecanismos anteriores no se transfieren automáticamente al modelo de 1210 parámetros.

Las etapas siguientes son cerrar auditorías y fuentes admisibles, revisión/hipótesis/diseño, separación de confirmación, datasets, presupuesto y potencia antes de un protocolo definitivo; completar la evaluación confirmatoria y ampliada; construir y verificar la aplicación y su reproducción. La comparación posterior ya realizada se conserva. Los borradores y las páginas de diseño mantienen sus fechas y no se convierten retrospectivamente en preregistros o productos funcionales.

## Lista de reproducibilidad al cierre

| Elemento | Evidencia disponible | Pendiente para cierre completo |
|---|---|---|
| Versión de fuentes | Git, manifiestos de los cortes y fuentes del libro. | Vincular la entrega exacta con archivo verificable, tag y DOI/identificador persistente. |
| Entorno documental | `requirements-book.lock.txt`, configuración y workflow de construcción de los tres libros. | Mantener instrucciones verificadas para reconstruir esta versión; esto no ejecuta la campaña científica. |
| Entornos científicos y de servicio | Locks CPU del software existente y reportes con versiones. | Contenedores probados separados: científico con framework; servicio ONNX Runtime sin framework de entrenamiento, conforme a sección 10. |
| Hardware y precisión | Perfiles CUDA de las corridas E3 y CPU del análisis estadístico y exportaciones, con versiones y alcance. | Consolidar hardware/backend/hilos/lotes de la entrega, sin mezclar eager CPU, TorchScript y GPU. |
| Semillas y determinismo | Redes de E2 y E3 con semillas 271801–271810; semilla histórica E3 20261116 separada y tres clásicos con una trayectoria cada uno. | Dimensionar por potencia las ejecuciones confirmatorias, documentar políticas de generadores y límites entre plataformas; diez semillas no prueban potencia suficiente. |
| Hiperparámetros y esfuerzo | Recibos E2, cinco recetas de v1/v2, tres por momentos y réplicas aditivas congeladas. | Espacios/presupuestos y ajuste final comparables, Nd global conciliado, fallos/decisiones y protocolo previo a la confirmación. |
| Métricas y tabla maestra | Métricas de 17 sistemas, diez semillas por red, contrastes, efectos e IC posteriores; calendario y ámbitos de incertidumbre declarados. | Tres conjuntos adicionales, calibración del candidato, campos de recursos/exportación completos y fuente común de informe/dashboard. |
| Datos y derechos | Data Card/contrato, hashes y 1316 ventanas verificadas, incluidas las 608 de prueba; auditoría 12 PASS, 3 PARTIAL y 1 NA. | Resolver controles pendientes, consolidar instrucciones exactas y derechos por fuente y DATA_STATEMENT final; la auditoría no cierra por sí sola el Bloque A. |
| Artefactos y paridad | Tres TorchScript y paridad puntual en 53 fechas de calibración. | Exportaciones de producción ONNX y demás artefactos aplicables, metadata y pruebas completas de preprocesamiento/salida/intervalo con tolerancias previas. |
| Checkpoints y estados | Snapshots de las fases nuevas de E2 y diez checkpoints E3 con pesos, estados de optimización/RNG y metadatos; los modelos seleccionados de E2 y los artefactos de servicio tienen otro alcance. | Completar la política común de ajuste final y la verificación integral de carga/reanudación/exportación. La inspección de código e inventario no equivale a deserializar todos los binarios. |
| Cómputo y energía | Consumos cerrados: ampliación E2 5130,9707 s; E3 acumulado 948,4396 s, incluidos consumos previos. Análisis estadístico CPU sin ajustes ni inferencias nuevos. | Costes comparables completos y energía medida o estimada con método explícito; no inventar kWh ni confundir reloj, CPU y horas GPU. |
| Verificación de software | Workflows de handoff/modelo y **42 pruebas de métodos e insumos estadísticos en CI**, más comprobaciones del libro. | Tests de la aplicación futura y su coherencia con el informe. Un check de bytes no demuestra reproducción integral de resultados. |
| Reproducción independiente | Requisitos y procedimiento previsto en [11](11_informe_tecnico.md). | Una persona ajena al código, entorno limpio en contenedor, tabla principal regenerada y discrepancia máxima reportada. |
| Servicio público | Plan [10](10_despliegue_render.md) y diseño [9](09_dashboard_predictivo.md). | URL Render, prueba de humo por página/modelo, logs, tiempos y contingencia basados en una app funcional. |
| Autoría y defensa | Procedencia del candidato declara asistencia IA. | Contribuciones individuales confirmadas, firmas humanas y dominio del algoritmo comprobado durante preparación/defensa. |

Las verificaciones `python scripts/e3_build_handoff_book.py --check` y `python scripts/e3_verify_original.py --check` tienen ámbitos de evidencia y software definidos. La construcción `jupyter-book build book_e3 --all --warningiserror --keep-going` y el comprobador de enlaces verifican documentación. Ninguno de esos comandos, por sí solo, entrena la campaña E3, reproduce un servicio Render o convierte a la IA en una persona independiente del código.

## Registro de reproducción independiente por completar

El informe deberá identificar revisor humano y relación con el proyecto; fecha; commit/tag y archivo/DOI; digest del contenedor; hardware; versiones; hashes de datos/artefactos; comandos y logs; semillas; productos regenerados y productos sólo comprobados. La tabla principal se comparará con la versión entregada usando tolerancias fijadas previamente, y se reportarán discrepancia máxima, localización de diferencias, fallos y causa investigada.

**Estado: no ejecutado ni firmado.** No hay una discrepancia cero que informar de esa reproducción. Los errores cero de las tres exportaciones y las comprobaciones CI mantienen su alcance propio.

## Autoría, contribuciones y uso de IA

La preparación del [piloto exploratorio de momentos](07_diseno_riguroso.md) empleó ChatGPT/Codex para implementación del modelo, ejecutor, guardado y reanudación en GitHub, pruebas sobre datos artificiales, revisión de código y documentación del notebook. La ejecución posterior del piloto está respaldada por sus resultados guardados y se distingue del corte inicial de preparación. La asistencia de IA también intervino en los ejecutores de réplicas, el análisis estadístico posterior y la actualización del libro. Sus commits, versiones, métricas y consumos conservan el alcance de cada etapa; esto no equivale a una verificación humana externa y la persona autora mantiene la responsabilidad de revisar y defender el código.

El [dictamen posterior sobre la cabeza de momentos](22_dictamen_momentos.md) también usa asistencia de ChatGPT/Codex para lectura focal, búsqueda, crítica matemática, contraejemplos escalares, código de auditoría y documentación. Su [procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/provenance.md) declara ese alcance. El paquete de revisión humana está sin diligenciar y no se presenta como aval externo ni firma.

La [propuesta posterior al cierre v2](21_propuesta_momentos.md) también fue preparada con asistencia de ChatGPT/Codex para revisión focal, formulación, demostración, crítica matemática, verificación escalar, pruebas y documentación. Su [declaración de procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_mecanismo_momentos_20261116/provenance.md) atribuye los componentes conocidos y mantiene pendiente la revisión humana, la novedad y la defensa del estudiante. No se entrenó una red nueva en ese corte.

La autoría declarada en la portada se mantiene como dato del proyecto. No se atribuyen automáticamente contribuciones individuales ni se afirma una firma o defensa que no se haya verificado.

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
