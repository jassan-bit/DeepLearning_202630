# 38. Exigencias de la guía, evidencia y pendientes

**Versión de entrega con lo disponible:** resultados, discusión, conclusiones y descargas en [§47](47_entrega_consolidada.md). Esta matriz conserva los requisitos sin evidencia para revisión posterior; el cierre documental no los convierte en experimentos completados ni ordena nuevas corridas.

**Actualización de alcance: 12/11/2026; matriz de evidencia del corte 11/11/2026.** Fuente: *Entregable2DeepLearningProject.pdf*, 65 páginas, aportado nuevamente por el usuario. SHA256 del PDF:

```text
93499dcfda6a36d9621524a81821470017ba114ca6b08241a35de5a54f74a654
```

Las referencias usan páginas impresas. Esta actualización organiza el informe y no ejecuta mercado, no inicia Colab y no certifica cumplimiento integral.

## Qué exige el PDF y qué decidió el proyecto

| Elemento | Exigencia de la guía | Decisión propia vigente |
|---|---|---|
| Catálogo | Top 10 con ≥4 familias, tres referencias E1 y comparadores F2a | **16 obligatorios:** diez modelos, MLP16/LSTM32/GRU32 y naive/estacional/ARIMA; híbrido adicional y Ridge opcionales (§43) |
| Validación temporal | Origen móvil, embargo, ajuste sin futuro y esquema común | **§45:** tres anclas HPO (enero/julio/octubre de 2024), diez orígenes mensuales de evaluación con receta causal y un refit histórico en enero de 2025 sin nueva búsqueda |
| Aplicación de NAS | Sólo diseños no definidos por su fuente; cuerpo publicado definido exento | **Política prospectiva de §44:** nueve cuerpos definidos del Top 10, con cuatro restauraciones de receta nueva; TiDE conserva la variante NAS. El plan sellado anterior permanece sin migrar |
| NAS fase 1 | Todas las 1.000 arquitecturas, una semilla, presupuesto reducido | Máximo 50 épocas por fase y parada temprana |
| NAS fase 2 | k≥10 según ruido, referencia y ≥5 semillas diferentes de fase 1 | k de 20/40 según piloto registrado; cota conservadora k40, referencia una sola vez, cinco semillas y máximo 200 épocas por fase |
| HPO | Espacios/ejes aplicables justificados y mismo esfuerzo | 18 oportunidades y dos semillas por red en cada una de las tres anclas; LR/WD/batch y ejes aplicables. Las restauraciones fijan knobs estructurales/dropout al cuerpo fuente (§44); §45 declara calendario, selección y contabilidad |
| Repeticiones finales | ≥5 ejecuciones; ≥6 cuando se usan pruebas de rangos y dimensionamiento por potencia | R todavía sin cerrar; los escenarios de coste no fijan R |
| Coste | Estimar antes de buscar; declarar y justificar desviaciones | Cota prospectiva de selección ≤2.609 ajustes neuronales, más 54 ARIMA y trabajo adicional declarado. Techo acumulado de 64 horas; saldo actual pendiente de conciliación. 85.074 conserva alcance histórico (§40) |

**La [enmienda de catálogo de §43](43_catalogo_obligatorio.md) retira el híbrido y Ridge del alcance obligatorio.** La [enmienda de arquitectura de §44](44_arquitecturas_publicadas.md) conserva cinco cuerpos del Top 10, restaura cuatro con identidades nuevas y mantiene NAS para TiDE. La posterior [enmienda del protocolo de §45](45_enmienda_protocolo.md) registra tres anclas HPO, una NAS inicial TiDE y evaluación mensual causal. Conserva semillas, límites, gasto e historial; no reescribe los resultados anteriores. La campaña de Colab sellada con 14 redes aún debe aplicar una transición explícita.

**La política prospectiva vigente está en [§45](45_enmienda_protocolo.md): tres anclas de búsqueda y diez orígenes de evaluación con reajuste.** Se justifica por cobertura temporal y coste, con resultados desagregados y límites por comprobar. Es una enmienda distinta de la propuesta de §37; [§40 conserva el plan sellado anterior](40_calendario_original.md). El registro no acredita ejecución completa, potencia alcanzada ni aceptación institucional.

## Diversidad del Top 10

La siguiente taxonomía funcional es una decisión explícita del proyecto, con los cuerpos vigentes adjudicados en [§44](44_arquitecturas_publicadas.md) y sus antecedentes en [§26](26_registro_recetas.md). Una familia no se multiplica por cambiar el nombre de un modelo; las adaptaciones a D7 se declaran por separado.

| Familia funcional | Modelos del Top 10 | Mecanismo que justifica la asignación |
|---|---|---|
| Convolucional | FCN-1D, ResNet-1D, Inception-1D, TimesNet | Convoluciones temporales; en TimesNet, transformaciones por periodicidad y convoluciones 2D en TimesBlock |
| Recurrente con representación autoencoder | LSTM-autoencoder con predictor | Codificadores/decodificadores LSTM y transferencia a predictor |
| Mezcla mediante MLP / encoder-decoder denso | TSMixer, TiDE, TimeMixer | Mezcla temporal/de variables, proyecciones densas y, en TimeMixer, descomposición multiescala |
| Atención Transformer | PatchTST, iTransformer | Atención sobre parches o tokens de variables |

Así se documentan **cuatro familias dentro de los diez modelos**, sin contar el híbrido candidato ni los controles E1. Esta clasificación no cierra la tabla de puntajes, la selección bibliográfica, las fichas/diagramas ni la evaluación experimental exigidas por la guía; G05 permanece parcial.

## Matriz de evidencia

**Parcial** significa que existe evidencia reutilizable, pero quedan condiciones por verificar. **Diseño documentado** no significa ejecución completada. **Pendiente** no se rellena con cifras de E1, ETTh1, artículos o datos sintéticos. Los estados se apoyan en lo documentado en los capítulos enlazados; no equivalen a una nueva auditoría de todos los artefactos.

| ID | Guía | Exigencia | Estado del catálogo vigente | Evidencia y trabajo para cerrar |
|---|---|---|---|---|
| G01 | §1, pp. 1–2 | Problema, RQ, hipótesis, primaria, δ y criterio de decisión | **Parcial** | [Evidencia documentada](01_punto_partida.md). Cerrar correspondencia RQ–contraste–unidad–criterio en el preregistro vigente; declarar exposición al test. |
| G02 | §2, pp. 3–9 | EDA que determina decisiones y su trazabilidad | **Parcial** | [Evidencia documentada](02_eda.md). Vincular cada transformación y clasificación F2a con evidencia del EDA; no trasladar diagnósticos de otro modelo. |
| G03 | §3.1, pp. 9–10 | Búsqueda reproducible, PRISMA, extracción y calidad | **Parcial** | [Evidencia documentada](03_literatura.md). Cerrar elegibilidad, duplicados y razones; conservar cadenas, fechas y conteos nativos. arXiv es complementario. |
| G04 | §3.1, p. 10 | Al menos tres cadenas transversales y tabla de mecanismos transferibles | **Pendiente de cierre** | [Evidencia documentada](03_literatura.md). Acreditar consultas ejecutadas y fuentes primarias; una cadena propuesta no cuenta como búsqueda. |
| G05 | §3.2–3.3, pp. 11–12 | Top 10, ≥4 familias, puntajes completos y ficha/diagrama por modelo | **Parcial** | [Evidencia documentada](15_revision_modelos.md). Taxonomía de cuatro familias documentada en §38; completar puntajes reproducibles, excluidos y diagramas. No contar el candidato 11 para reparar la diversidad del Top 10. |
| G06 | §§3.2, 7.4 F2a, pp. 11, 30–31 | Tres bases E1 y baselines naive, estacional y ARIMA o ETS | **Implementación parcial** | [Evidencia documentada](07_ruta_temporal.md). Ejecutar comparadores con soporte común y ajuste justo. Los resultados E1 conocidos no sustituyen reajustes de E2. |
| G07 | §§4.1, 7.4 F2a, pp. 12–13, 30–31 | Cronología, origen móvil, embargo y disponibilidad de covariables | **Diseño documentado** | [Calendario común de §45](45_enmienda_protocolo.md). Aplicar las tres anclas HPO y los diez reajustes mensuales con la receta más reciente disponible, embargo y máscaras comunes; reportar por origen. Migración y ejecución pendientes; once búsquedas no son mandato literal del PDF. |
| G08 | §10, pp. 44–46 | Data Card, contrato y auditoría formal de fuga | **Parcial** | [Evidencia documentada](10_auditoria_datos.md). Completar evidencias aplicables de duplicados, atributos sospechosos, controles negativos y auditoría adversarial; no declarar auditoría completa por verificar hashes. |
| G09 | §§4.3, 9, pp. 14, 42–43 | Preregistro real, repeticiones y potencia ≥0,80 o limitación/MDE | **Pendiente de cierre** | [Evidencia documentada](13_agregacion_estadistica.md). Completar implementación y dimensionamiento de la familia primaria de veinte contrastes de §45 (Top 10 frente a GRU32 y Persistence, Holm FWER 0,05), bloques de 14 días, δ y R antes de nuevos contrastes. Piloto de cinco semillas pareadas en TimeMixer/PatchTST/GRU32, 08/11–30/12/2024, separado de HPO. R sigue pendiente en 6–30: alternativa planificada 2δ y límite inferior Wilson de potencia ≥0,80, con sensibilidad temporal declarada; no heredar R ni afirmar potencia por alcanzar un techo. |
| G10 | §11, pp. 46–48 | Mismo esfuerzo de HPO y registro de tiempo, fallos y recetas | **Implementación y selección parciales** | [Evidencia documentada](11_hpo_equidad.md). Completar búsqueda y selección de todos los modelos; justificar rangos/ejes aplicables y mantener recibos de fallos sin sustituciones silenciosas. |
| G11 | §14, pp. 52–58 | NAS condicional: 1.000 diseños, dos fases, finalistas y selección sólo con validación | **Adjudicación documentada; selección vigente pendiente** | [Política y fuentes de §44](44_arquitecturas_publicadas.md): nueve cuerpos definidos y TiDE NAS. La campaña histórica conserva otras recetas; migrar antes de nuevos ajustes afectados. Para TiDE, §45 registra una sola búsqueda inicial con 1.000 arquitecturas, dos fases, referencia y confirmación; k≥10 y ≥5 semillas. El piloto fija k20/40 antes de NAS-1. Completar conjunto indistinguible y sesgo del ganador; solapar IC no demuestra equivalencia. |
| G12 | §5, pp. 16–17 | Réplica parcial, limitación cuantificada y piloto en ≥3 del Top 10 | **Evidencia externa parcial** | [Evidencia documentada](05_paper_guia.md). Conservar réplica TimeMixer/ETTh1; documentar métrica, magnitud e IC de la limitación y sus límites de transferencia a Bitcoin. No convertir el híbrido en solución validada. |
| G13 | §6 y Apéndice A, pp. 17–20, 61–62 | Tabla de benchmark con medias, DE, IC y costes; ranking y Pareto | **Pendiente de resultados finales** | [Evidencia documentada](06_benchmark_final.md). Evaluar todos los sistemas en soporte común con R fijado; añadir parámetros, FLOPs, latencia, recursos, calibración y robustez. No rellenar con métricas de otros estudios. |
| G14 | §§6.3, 7.3, 7.4 F2a, pp. 18–19, 21–31 | Bondad de ajuste, sobreajuste y diferencias con pruebas admisibles | **Código/plan parciales** | [Evidencia documentada](13_agregacion_estadistica.md). Preespecificar batería pertinente, al menos tres pruebas admisibles por contraste confirmatorio, supuestos y multiplicidad. Incluir diagnósticos temporales, calibración y DM-HLN donde corresponda; no tratar semillas/orígenes solapados como datasets iid. |
| G15 | §§7.3, 13, pp. 22–27, 50–51 | IC, efectos y remuestreo que respete dependencia | **Pendiente de análisis final** | [Evidencia documentada](13_agregacion_estadistica.md). Justificar bloques temporales, ≥2.000 remuestreos para IC y ≥10.000 para contraste pareado aplicable; reportar p exactos y ajustados, efecto e IC, unidad y familia. |
| G16 | §4.6, p. 16 | Robustez con ≥3 severidades por perturbación y cambio de distribución | **Pendiente para benchmark vigente** | [Evidencia documentada](12_incertidumbre_errores.md). Registrar perturbaciones causales sin cambiar el objetivo, severidades y curvas de degradación; evaluar recetas seleccionadas. |
| G17 | §4.6, p. 16 | Explicabilidad del Top 3, fidelidad y estabilidad entre semillas | **Pendiente de identificar Top 3** | [Evidencia documentada](12_incertidumbre_errores.md). Aplicar técnica pertinente con borrado/inserción y estabilidad; mapas de atención aislados no acreditan fidelidad. |
| G18 | §4.6, p. 16 | Ablación del mejor modelo con ≥3 componentes e inferencia | **Pendiente de identificar ganador** | [Evidencia documentada](06_benchmark_final.md). Aislar tres aportes con las mismas semillas y controles; no asumir que el híbrido será el mejor. |
| G19 | §12, pp. 48–50 | Referencia de techo/margen γ, incertidumbre y errores por segmentos | **Parcial** | [Evidencia documentada](12_incertidumbre_errores.md). Justificar referencia y γ sin llamar a persistencia techo de Bayes; calibración disjunta, cobertura y ≥100 casos estratificados por modelo principal, según aplicabilidad. |
| G20 | §8, pp. 40–41; Entregables, pp. 58–59 | Entorno fijado, código, datos/instrucciones, reproducción y entregables | **Parcial** | [Evidencia documentada](08_reproducibilidad.md). Ejecutar reproducción integral en entorno limpio, completar Apéndice B y empaquetar artefactos. Contenedor opcional en E2; obligatoriedad de E3 no se presenta como requisito E2. |
| G21 | §6.4, p. 20; estructura, p. 62 | Resumen ≤250 palabras, discusión por RQ, conclusiones y amenazas | **Provisional** | [Evidencia documentada](06_benchmark_final.md). Redactar resumen final con cifras y responder hipótesis sólo tras análisis; conservar explícitos resultados negativos, incertidumbre y limitaciones. |

## Orden para completar la entrega

1. **Cerrar el método:** consolidar preregistro/enmiendas con fecha real, taxonomía y puntajes Top 10, auditoría de fuga, unidades/familias de contraste y presupuesto completo. Registrar cualquier cambio antes de nuevos trabajos afectados; no afirmar preregistro retroactivo.
2. **Completar selección y dimensionamiento:** migrar el controlador conservando saldo y caché compatible, ejecutar HPO/NAS del calendario de §45 y fijar R con el piloto de potencia. Conservar fallos y costes desconocidos. El contador de intentos no es un porcentaje de terminación del proyecto.
3. **Completar evaluación común:** reajustar modelos/referencias, cerrar soporte y calendario de evaluación y producir tablas con IC, costes y diagnósticos. El test conocido mantiene interpretación exploratoria.
4. **Completar análisis exigidos:** inferencia temporal con multiplicidad, robustez, incertidumbre/calibración, explicabilidad del Top 3, ablaciones del mejor y análisis de errores.
5. **Cerrar informe y reproducción:** responder cada RQ, distinguir significancia de relevancia práctica, completar discusión/amenazas, ejecutar reproducción limpia y generar las descargas verificadas.

El orden anterior es el trabajo que resta; no declara que el método esté cerrado porque ya existan ajustes parciales.

## Paquete de entrega y estructura del informe

| Entregable del PDF | Evidencia que debe acompañarlo |
|---|---|
| Jupyter Book reproducible | Resumen final ≤250 palabras; introducción/RQ; datos/EDA/auditoría; revisión/PRISMA/paper guía; métodos/preregistro; resultados; discusión; conclusiones; referencias y apéndices |
| Código completo | EDA, modelos, baselines, auditoría, entrenamiento, pruebas de ajuste/sobreajuste, inferencia, robustez, incertidumbre e interpretabilidad |
| Datos o adquisición verificable | Data Card, contrato, procedencia, fechas, permisos/instrucciones y comprobación de hashes |
| Preregistro y registro experimental | Versiones fechadas, configuraciones, semillas, intentos/fallos, métricas y recursos por ejecución; desviaciones explícitas |
| Ficha de ruta F2a | Partición, disponibilidad, métricas y evidencia de pruebas de bondad de ajuste, sobreajuste y diferencias (Tablas A, B y C) |
| Benchmark y eficiencia | Tablas de Apéndice A, efectos/IC, p ajustados, comparación frente a baselines, Pareto y limitaciones |
| Informe NAS aplicable | Las 1.000 arquitecturas, ambas fases, finalistas/referencia, selección, sesgo del ganador, efectos e inventario reproducible |
| Paper guía e insumo para E3 | Réplica parcial verificable, limitación cuantificada con IC y mecanismos transferibles con fuentes originales |
| Incertidumbre y techo | Referencia justificada, margen γ, calibración y cobertura, errores estratificados y límites |
| Lista de reproducción y defensa | Apéndice B completado con evidencia de entorno limpio y presentación oral de las decisiones |

[Lista estructurada de requisitos, corte anterior a la reducción del catálogo](../_static/guide_compliance_20261111/requirements_11_origins_v2.json). El catálogo vigente de 16 sistemas se encuentra en [§43](43_catalogo_obligatorio.md). Cada ID mantiene el vínculo entre requisito, estado y evidencia de cierre. La guía pondera problema/EDA/preregistro 15 %, revisión/paper/gap 15 %, implementación/NAS/rigor 25 %, benchmark/inferencia 20 %, ruta/robustez/incertidumbre/interpretabilidad 15 % y reproducción/redacción/defensa 10 % (p. 59). No se calcula una calificación ni un porcentaje de cumplimiento a partir del número de filas o de entrenamientos.

El [corte documental anterior a la decisión de conservar once orígenes](../_static/guide_compliance_20261111/requirements.json) se conserva como descarga histórica, con sus bytes originales. Ambos cortes estructurados se leen junto con las enmiendas de catálogo (§43), arquitectura (§44) y protocolo (§45).
