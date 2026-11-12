# Proyecto Final de Deep Learning: Entregable 2

**Catálogo obligatorio vigente (12/11/2026): 16 sistemas: diez modelos del Top 10, tres referencias E1 y tres comparadores temporales.** El híbrido adicional y Ridge quedan fuera del alcance obligatorio. [Catálogo y enmienda de alcance (§43)](sections/43_catalogo_obligatorio.md). La [enmienda del protocolo de §45](sections/45_enmienda_protocolo.md) fija tres anclas HPO y conserva diez orígenes mensuales de evaluación con reajuste común. La campaña sellada de Colab todavía corresponde al plan anterior de catorce redes: esta publicación no modifica la sesión privada.

**Arquitecturas y NAS (12/11/2026):** la ruta prospectiva conserva o restaura nueve cuerpos publicados del Top 10 y mantiene NAS para la variante TiDE. Las cuatro restauraciones tienen recetas nuevas y adaptaciones D7 explícitas; sus resultados anteriores no se renombran. [Adjudicación, fuentes y pruebas (§44)](sections/44_arquitecturas_publicadas.md).

**Protocolo temporal (12/11/2026):** HPO común en enero, julio y octubre de 2024; una sola NAS inicial para TiDE; reajustes mensuales sin repetir búsquedas. El techo sigue siendo 64 horas acumuladas, con saldo actual pendiente de conciliación. Los periodos históricos ya evaluados se separan de una prueba independiente prospectiva. [Calendario, criterios y pendientes (§45)](sections/45_enmienda_protocolo.md).

**Deep Learning · 2026-30 · Jassan Arteta**

**Avance experimental verificado: 1.484 ajustes HPO nuevos completados, sin fallos, al 12/11/2026 a las 02:29:43 de Bogotá.** Se suman a 1.802 registros anteriores: **3.286 registros de resultados** entre ambos historiales. **Estado al cierre del corte: pausada por límite de sesión, con cierre limpio.** El saldo de unas 4 h 9 min pertenece exclusivamente a ese corte histórico; el saldo actual debe conciliarse antes de ejecutar la enmienda, dentro del techo acumulado de 64 horas. Son corridas de selección; el benchmark final sigue pendiente. [Detalle y fuentes (§41)](sections/41_avance_optimizacion_11_origenes.md).

**Auditoría contra la guía:** [§42: qué podemos completar mientras entrena Colab](sections/42_auditoria_y_trabajo_paralelo.md). Incluye prioridades, 21 ítems del Apéndice B y plantillas de reporte; distingue preparación de cumplimiento.

## Estado de la entrega y guía de lectura

**Actualización documental al 12/11/2026.** Se conserva el contraste con la guía de 65 páginas aportada por el usuario. El libro contiene código, revisión bibliográfica, preparación experimental y resultados parciales de selección. **El benchmark final no está terminado y no se declara cumplimiento integral de la guía.**

La [matriz vigente de exigencias, evidencias y pendientes](sections/38_cumplimiento_guia.md) organiza la entrega según la guía y distingue requisitos del PDF, decisiones propias y resultados observados. Cada requisito se cierra con evidencia verificable; describirlo o disponer de código no acredita su ejecución.

| Componente | Estado documental verificable | Dónde revisar |
|---|---|---|
| Problema, datos, EDA y ruta temporal | Contrato definido; auditoría completa del benchmark pendiente | [Problema](sections/01_punto_partida.md), [EDA](sections/02_eda.md), [F2a](sections/07_ruta_temporal.md), [contrato](sections/17_contrato_sincronizado.md) |
| Revisión sistemática y Top 10 | Búsquedas y lecturas parciales; cierre de elegibilidad y puntajes por completar | [Literatura](sections/03_literatura.md), [catálogo](sections/15_revision_modelos.md) |
| Implementación y HPO/NAS | Política prospectiva: nueve cuerpos definidos del Top 10 y NAS para TiDE; selección y migración del controlador pendientes | [Arquitecturas y verificación](sections/44_arquitecturas_publicadas.md), [avance del plan sellado](sections/41_avance_optimizacion_11_origenes.md), [recetas históricas](sections/26_registro_recetas.md) |
| Réplica del paper guía | Evidencia externa de TimeMixer/ETTh1; no sustituye Bitcoin D7 | [Réplica y limitaciones](sections/05_paper_guia.md) |
| Benchmark e inferencia | Pendientes de recetas seleccionadas, R, reajustes y evaluación común | [Benchmark](sections/06_benchmark_final.md), [inferencia](sections/13_agregacion_estadistica.md) |
| Robustez, incertidumbre e interpretabilidad | Preparación parcial; resultados del catálogo completo pendientes | [Incertidumbre y errores](sections/12_incertidumbre_errores.md), [matriz de cierre](sections/38_cumplimiento_guia.md) |
| Reproducibilidad y entrega | Fuentes y procedimientos disponibles; reproducción integral pendiente | [Reproducibilidad](sections/08_reproducibilidad.md), [lista de entrega](sections/38_cumplimiento_guia.md) |

## Problema, contrato y referencias

Se estudia **OHLCV diario D7×5 → σ₇**, desviación estándar poblacional de siete retornos logarítmicos diarios de Close, **sin anualización y con horizonte de un día**. Se conserva RMSE como métrica primaria heredada de E1 y **δ=0,00025772842664981106**; las recomendaciones de métricas temporales no cambian por silencio ese contrato. Las métricas auxiliares y sus denominadores deben declararse antes de evaluar.

La ruta es **F2a, series de tiempo**: partición cronológica, embargo y transformaciones ajustadas sólo con el pasado. Las **608 fechas históricas conocidas** mantienen carácter exploratorio; una actualización editorial o un nuevo programador no recuperan independencia del test. [Contrato y límites](sections/17_contrato_sincronizado.md).

El catálogo comprende diez modelos base:

1. FCN-1D.
2. ResNet-1D.
3. Inception-1D.
4. LSTM-autoencoder con predictor.
5. TimesNet.
6. TSMixer.
7. TiDE.
8. TimeMixer.
9. PatchTST.
10. iTransformer.

Los controles **MLP16, LSTM32 y GRU32 de E1** son las tres referencias adicionales, con pesos reinicializados. **Persistencia (naive), naive estacional y ARIMA** completan los tres comparadores: **13 redes + 3 comparadores = 16 sistemas**. El Top 10 comprende cuatro familias: convolucionales, recurrentes/autoencoder, mezcladores MLP y atención; [§38 explicita la asignación](sections/38_cumplimiento_guia.md). **CNN-1D + LSTM + atención temporal y Ridge son opcionales/históricos y no forman parte de la entrega obligatoria.** Sus implementaciones y recibos se conservan. [Catálogo vigente y fuentes ejecutables (§43)](sections/43_catalogo_obligatorio.md).

## Diseño vigente, guía y cómputo

La [enmienda vigente de §45](sections/45_enmienda_protocolo.md) distingue **tres búsquedas HPO comunes** —01/01, 01/07 y 01/10/2024— de **diez reajustes mensuales de evaluación** —enero a octubre de 2024—. Cada origen usa únicamente la receta que ya estaba disponible: enero–junio utiliza enero, julio–septiembre utiliza julio y octubre utiliza octubre. TiDE tiene una sola NAS inicial; su arquitectura queda congelada antes de la primera evaluación externa. El refit histórico de enero de 2025 usa la receta de octubre, sin otra búsqueda.

La decisión reduce repeticiones de búsqueda y conserva la evaluación de origen móvil, el embargo de siete días, las 18 oportunidades HPO por red/ancla y las demás exigencias. La guía no obliga a once búsquedas completas; tres anclas tampoco garantizan suficiencia por sí solas. La cobertura y sus límites se reportarán por origen. Esta enmienda sustituye el calendario prospectivo anterior; no adopta sin cambios la propuesta histórica de §37.

La cota nueva es **hasta 2.609 ajustes neuronales de selección**, antes de reutilización compatible y sin pilotos, reajustes o análisis complementarios. Incluye 1.404 HPO, 1.000 NAS-1 TiDE y hasta 205 NAS-2 con k=40; los ajustes SAE incluyen etapas auxiliares y ARIMA añade 54 ajustes de selección. El máximo de **85.074** de [§40](sections/40_calendario_original.md) conserva el significado del plan sellado anterior de catorce redes; ninguna de las dos cifras es un mínimo literal del PDF.

El techo autorizado permanece en **64 horas acumuladas**, conservando el gasto de ambas campañas y los intentos inciertos. No se asignan otras 64 horas al migrar. R, el piloto de ruido, el saldo actual, la integración GPU y la prueba independiente siguen pendientes de sus evidencias de cierre. El notebook histórico de once orígenes no aplica automáticamente la nueva política. [Registro y pendientes de §45](sections/45_enmienda_protocolo.md) · [Corte histórico de recibos](sections/41_avance_optimizacion_11_origenes.md).

## Resultados parciales y límites

El [corte vigente de §41](sections/41_avance_optimizacion_11_origenes.md) se fija al commit `dfd7d3c035c015aac9cad454b1aec30f190a631d`. La primera ola de ocho redes/referencias con cuerpo fijo tiene cobertura HPO cerrada de enero a mayo de 2024. En junio, FCN-1D, ResNet-1D, Inception-1D y TSMixer registran 36/36 cada uno; iTransformer lleva 8/36. Los 1.484 resultados nuevos tienen estado `completed`; el total combinado de 3.286 incluye 1.802 registros históricos cuyos estados no se reauditán aquí. El controlador pausó limpiamente porque la siguiente reserva completa no cabía en el tiempo de esa sesión. El informe de campaña está actualizado; este libro publica un corte fijo, sin comprobar en vivo el kernel de Colab. El CSV mantiene configuraciones, costes y la identificación del ajuste recuperado; no contiene un ranking final.


El [corte documentado en §36](sections/36_avance_campana_cuda.md), del **11/11/2026 a las 06:55:42 de Bogotá**, contiene **1.748 archivos de resultados**. Registra selección del primer origen para FCN-1D, ResNet-1D, Inception-1D y LSTM-autoencoder y avance NAS-1 de TimesNet. Es un corte histórico, no el contador actual ni evidencia de activación del calendario reducido. Los resultados de E1, los perfiles de recursos y la réplica ETTh1 conservan su propio ámbito.

**No hay ranking final, superioridad demostrada del híbrido ni R definitivo.** La [matriz de cumplimiento](sections/38_cumplimiento_guia.md) indica las evidencias necesarias antes de publicar conclusiones.

## Recorrido recomendado

- **Problema y literatura:** [§1](sections/01_punto_partida.md), [§2](sections/02_eda.md), [§3](sections/03_literatura.md) y [§5: paper guía](sections/05_paper_guia.md).
- **Catálogo obligatorio y arquitectura:** [§43: los 16 sistemas](sections/43_catalogo_obligatorio.md) y [§44: cuerpos publicados, adaptaciones D7 y NAS aplicable](sections/44_arquitecturas_publicadas.md).
- **Protocolo y separación de datos:** [§45: calendario común, presupuesto, métricas y prueba prospectiva](sections/45_enmienda_protocolo.md).
- **Método y modelos:** [§7: ruta F2a](sections/07_ruta_temporal.md), [§10: auditoría](sections/10_auditoria_datos.md), [§11: equidad](sections/11_hpo_equidad.md), [§14: arquitecturas](sections/14_arquitecturas.md) y [§26: recetas](sections/26_registro_recetas.md).
- **Ejecución y evidencia:** [§29: entrenamiento](sections/29_entrenamiento_temporal.md), [§35: CUDA](sections/35_busqueda_cuda_reanudable.md), [§36: corte histórico](sections/36_avance_campana_cuda.md), [§40: calendario histórico](sections/40_calendario_original.md) y [§41: ejecución y avance](sections/41_avance_optimizacion_11_origenes.md).
- **Entrega y cierre:** [§6: benchmark](sections/06_benchmark_final.md), [§12: incertidumbre](sections/12_incertidumbre_errores.md), [§13: inferencia](sections/13_agregacion_estadistica.md), [§8: reproducción](sections/08_reproducibilidad.md) y [§38: exigencias de la guía](sections/38_cumplimiento_guia.md).

Los capítulos con cortes anteriores conservan sus fechas y recibos. Las notas de vigencia distinguen la preparación histórica del diseño actual; no se trasladan resultados entre protocolos ni se rellenan tablas con cifras de otros experimentos.
