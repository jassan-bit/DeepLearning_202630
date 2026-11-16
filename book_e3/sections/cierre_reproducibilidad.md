# Cierre provisional, reproducibilidad, autoría y bibliografía

Este anexo cierra la navegación del informe con los requisitos de reproducción y autoría de las secciones 11–12. Es un **registro provisional**, no un acta de entrega, firma o reproducción independiente. Se conserva la diferencia entre documentación publicada, software comprobado y resultados científicos pendientes.

## Conclusiones provisionales y límites

E2 aporta un benchmark histórico de 16 sistemas y 608 fechas, con recetas/artefactos, auditorías parciales y análisis posteriores de alcance declarado. E3 dispone de inventario, selección documentada y tres exportaciones TorchScript de referencias, verificadas numéricamente en calibración histórica. Estas comprobaciones no aportan confirmación independiente del modelo original ni certifican el servicio en Render.

El candidato tiene formulación, implementación asistida por IA, demostración de una cota de su cabeza latente y 35 pruebas con datos artificiales. La revisión focal y la bibliografía no establecen una ausencia de antecedentes. **R1, la defensa académica de R2 y R3 permanecen pendientes.** No se infieren ganancias de RMSE, adecuación, calibración o generalización de los tests artificiales.

Las etapas siguientes son cerrar auditorías y fuentes admisibles, revisión/hipótesis/diseño, separación de confirmación, datasets, presupuesto y potencia antes del protocolo; ejecutar después la comparación completa; construir y verificar la aplicación y su reproducción. Los borradores y las páginas de diseño conservan sus fechas y no se convierten retrospectivamente en preregistros o productos funcionales.

## Lista de reproducibilidad al cierre

| Elemento | Evidencia disponible | Pendiente para cierre completo |
|---|---|---|
| Versión de fuentes | Git, manifiestos de los cortes y fuentes del libro. | Vincular la entrega exacta con archivo verificable, tag y DOI/identificador persistente. |
| Entorno documental | `requirements-book.lock.txt`, configuración y workflow de construcción de los tres libros. | Mantener instrucciones verificadas para reconstruir esta versión; esto no ejecuta la campaña científica. |
| Entornos científicos y de servicio | Locks CPU del software existente y reportes con versiones. | Contenedores probados separados: científico con framework; servicio ONNX Runtime sin framework de entrenamiento, conforme a sección 10. |
| Hardware y precisión | Perfiles identificados en los reportes de E2/exportación. | Registrar hardware/backend/hilos/lotes de cada ejecución final, sin mezclar eager CPU, TorchScript y GPU. |
| Semillas y determinismo | E2 registra seis semillas por red; el software del candidato conserva su configuración y pruebas. | Lista y número final de semillas E3 dimensionados por potencia, políticas de todos los generadores y límites de determinismo documentados. No multiplicar trayectorias clásicas deterministas. |
| Hiperparámetros y esfuerzo | Recibos de selección de E2 y bitácora artificial del candidato. | Espacios/presupuestos comparables, Nd total, fallos, decisiones y protocolo previo al cálculo definitivo. |
| Métricas y tabla maestra | Inventario de los 16 sistemas heredados y reportes por cohorte. | Métricas por ejecución del candidato y tres conjuntos; fuente común de informe/dashboard, IC y dispersiones con unidades de incertidumbre. |
| Datos y derechos | Data Card/contrato de E2, hashes y auditorías de desarrollo con alcance. | Datos admisibles faltantes, instrucciones exactas y licencias/condiciones verificadas para cada fuente; DATA_STATEMENT final. |
| Artefactos y paridad | Tres TorchScript y paridad puntual en 53 fechas de calibración. | Exportaciones de producción ONNX y demás artefactos aplicables, metadata y pruebas completas de preprocesamiento/salida/intervalo con tolerancias previas. |
| Cómputo y energía | Registros existentes con sus ámbitos históricos. | Tiempos/costes y energía medida o estimada con método explícito de la campaña E3; no inventar kWh ni confundir reloj, CPU y horas GPU. |
| Verificación de software | Workflows `verify-e3-handoff.yml` y `verify-e3-original.yml`, más comprobaciones del libro. | Tests de la aplicación futura y su coherencia con el informe. Un check de bytes no demuestra reproducción integral de resultados. |
| Reproducción independiente | Requisitos y procedimiento previsto en [11](11_informe_tecnico.md). | Una persona ajena al código, entorno limpio en contenedor, tabla principal regenerada y discrepancia máxima reportada. |
| Servicio público | Plan [10](10_despliegue_render.md) y diseño [9](09_dashboard_predictivo.md). | URL Render, prueba de humo por página/modelo, logs, tiempos y contingencia basados en una app funcional. |
| Autoría y defensa | Procedencia del candidato declara asistencia IA. | Contribuciones individuales confirmadas, firmas humanas y dominio del algoritmo comprobado durante preparación/defensa. |

Las verificaciones `python scripts/e3_build_handoff_book.py --check` y `python scripts/e3_verify_original.py --check` tienen ámbitos de evidencia y software definidos. La construcción `jupyter-book build book_e3 --all --warningiserror --keep-going` y el comprobador de enlaces verifican documentación. Ninguno de esos comandos, por sí solo, entrena la campaña E3, reproduce un servicio Render o convierte a la IA en una persona independiente del código.

## Registro de reproducción independiente por completar

El informe deberá identificar revisor humano y relación con el proyecto; fecha; commit/tag y archivo/DOI; digest del contenedor; hardware; versiones; hashes de datos/artefactos; comandos y logs; semillas; productos regenerados y productos sólo comprobados. La tabla principal se comparará con la versión entregada usando tolerancias fijadas previamente, y se reportarán discrepancia máxima, localización de diferencias, fallos y causa investigada.

**Estado: no ejecutado ni firmado.** No hay una discrepancia cero que informar de esa reproducción. Los errores cero de las tres exportaciones y las comprobaciones CI mantienen su alcance propio.

## Autoría, contribuciones y uso de IA

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

La bibliografía configurada en `book_e3/references.bib` conserva los siete registros ya presentes en el corte del candidato; no es una nueva búsqueda ni acredita lectura completa de todos los trabajos. Los originales de Kim y colaboradores {cite:p}`kim2021lipschitz` y Dasoulas y colaboradores {cite:p}`dasoulas2021lipschitz` tuvieron lectura focal documentada. La matriz de la [sección 6](06_modelo_original.md) distingue los otros antecedentes, metadatos heredados y límites de consulta.

El archivo {download}`references.bib <../references.bib>` y la bibliografía renderizada permiten seguir las citas. Cerrar R1 requiere además completar/repetir la revisión, clasificar diferencias y obtener la verificación externa exigida; una lista BibTeX no demuestra novedad.

```{bibliography}
:all:
:style: unsrt
```
