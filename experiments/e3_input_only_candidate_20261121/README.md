# Input-only overlap MSE: expediente y decisión

**Decisión: CONTINUAR para preparar confirmación de una candidata con identidad propia, `input-only-overlap-mse-1`.** La revisión de R1 es técnicamente favorable y acotada para el algoritmo completo. No se basa en inventar normalización ni retirar CRPS: se vuelve a justificar la lectura propia heredada de V3, aplicando la transformación del componente desconocido de la volatilidad solapada antes de la esperanza. El dictamen y su crítica IA separada no son aceptación docente ni sustentación observada.

Se ha cerrado la decisión de candidata. **No se ha cerrado ni ejecutado un protocolo confirmatorio.** La identidad del algoritmo está fijada; el protocolo distingue los datos, métodos y tamaños que todavía carecen de evidencia. R3 no se declara acreditada.

## Evidencia verificada y resultado adverso conservado

El ZIP y los recibos del commit `05c1a637f13356dd1df6b8c5aa8f4676dda61c12` confirman 30 ajustes V5 completos, 1.760 predicciones guardadas y cinco semillas por receta neuronal sobre 55 fechas de desarrollo ya expuestas. La auditoría reproduce 1.035 comprobaciones, sin discrepancias numéricas materiales. Se conservan todas las recetas y los siete hashes internos de resultados.

| Receta | Media de RMSE por semilla | MAE medio | Alcance |
|---|---:|---:|---|
| `input_only_mse` | 0,0020515608242855487 | 0,0015484962302801841 | Señal seleccionada después de ver desarrollo. |
| `global_mse` | 0,0021097806965662083 | 0,0016536738452131044 | Control directo; cuatro de cinco pares favorecen input-only. |
| V5 completo `seoi_mse` | 0,002160960420031140 | 0,001702061588126237 | `ABANDON_CURRENT_SEOI_RECIPE`, sin cambiar su regla. |

La ganancia de input-only frente a global es **2,759522 %**. La receta completa V5 fue **5,332506 % peor** que input-only; no se cambia el denominador ni se llama éxito al piloto V5. La señal tiene límites: input-only pasó de último en selección a primero en evaluación expuesta, presenta 7,27 veces la SD entre semillas de global y empeora 0,591 % en el cuartil retrospectivo de menor volatilidad realizada. Ninguna partición posterior recupera independencia confirmatoria.

## Archivos que resuelven la solicitud

- [Dictamen R1](docs/R1_VERDICT.json), [expediente matemático y bibliográfico](docs/R1_THEORY_LITERATURE.md), [matriz de 21 requisitos](docs/R1_REQUIREMENTS_MATRIX.csv) y [crítica IA independiente](docs/R1_INDEPENDENT_AI_REVIEW.json).
- [Auditoría empírica](analysis/EMPIRICAL_AUDIT.md), [métricas recalculadas](analysis/metrics_recomputed.csv), [comparaciones completas](analysis/all_pairwise_summary.csv) y [sensibilidad temporal retrospectiva](analysis/subset_input_only_comparisons.csv).
- [Identidad del algoritmo](CANDIDATE_IDENTITY.json), [implementación fijada](runtime/candidate.py), [procedencia del código preservado](runtime/vendor/ORIGIN.json) e [inventario del evento de diseño](DESIGN_EVENT.json).
- [Protocolo de confirmación propuesto](confirmation/CONFIRMATION_PROTOCOL_DRAFT.json), [plan de potencia](confirmation/POWER_PLAN.json), [admisión de datos](confirmation/DATA_ADMISSION_TEMPLATE.json) y [control ejecutable de preparación](runtime/preflight.py).
- [Intervenciones concretas y evidencia que entregar](AUTHOR_HANDOFF.md) y [procedencia, derechos e IA](docs/RIGHTS_AND_AI.md).

Las pruebas generadas comprueban identidad de estados, salida y pérdida respecto de la receta ejecutada, formas, soporte, causalidad al emitir y un contraejemplo de la equivariancia de salida que esta variante **no** conserva. MSE sola tampoco identifica o calibra una verdadera distribución de U. No se ejecutaron ajustes o predicciones de mercado nuevos para este expediente.

## Reproducir ahora la preparación

El cuaderno `preparation_colab.ipynb` y `start_colab.py` de la entrega fijan un commit de fuentes y los hashes del lanzador/manifiesto. Ejecutan la auditoría guardada y el informe de admisión; una celda adicional instala un entorno CPU aislado para pruebas con fixtures generados. No requieren GPU, Drive, token o clave del checkpoint. No contienen una celda de entrenamiento.

Con las fuentes locales:

```bash
python analysis/audit_saved_results.py --zip inputs/public_results.zip --output audit/recalculated
python -m runtime.preflight --root . --output audit/preflight.json
```

El informe debe distinguir integridad correcta de admisión confirmatoria incompleta. Que el comando termine correctamente significa que pudo auditar y reportar el estado; no que autorice una campaña. El cuaderno es una **preparación ejecutable**, no un entrenador confirmatorio ya integrado para la familia completa.

## Qué queda fijado y qué falta resolver

El algoritmo es la misma celda funcional input-only del factorial V5; asignarle identidad propia no añade otro diseño al Nd. Se conservan los 796 trabajos aditivos, 30 ajustes V3, 20 V4 y 30 V5; sus versiones, datos, resultados, reglas y maestros no se modifican.

La familia prospectiva propuesta incluye los 16 rivales históricos obligatorios y cuatro extensiones declaradas: aditiva, los dos controles del piso y `global_mse`: **20 rivales, 21 sistemas incluida la candidata**. No se excluye su control directo favorable por conveniencia. Las oportunidades de selección, el soporte común y la equivalencia de presupuestos aún deben fijarse para los nuevos paneles.

La ruta de planificación elegida propone inferencia por semillas condicionada a calendarios fijos, preservando media de RMSE individuales. Su t, bootstrap-t y Wilcoxon no son tres procedimientos temporales F2 ya acreditados: se exponen sus estimandos, supuestos y correspondencia pendiente. Tampoco se hereda la aprobación de las extensiones cruzadas históricas no validadas. Ninguna prueba o potencia confirmatoria se ejecutó aquí.

Los cuatro paneles históricos están expuestos; la plantilla registra candidatos de adquisición sin certificar reserva, derechos o dimensiones finales inexistentes. Los números definitivos R/T y la potencia por contraste/conjunta permanecen sin inventar. Un umbral prospectivo propuesto del 2 % es una decisión nueva por justificar, no una regla universal R3 ni una reescritura de la heurística V5.

## Coste

El piloto conservado registra unos 76,02 s de ajuste y 10,00 s de I/O agregados; no incluyen instalación, transporte GitHub ni todo el tiempo de la VM. No se extrapolan esos tiempos de una pequeña CNN a todo el Top10.

Como **escenario nominal de planificación, no presupuesto final admitido**, con cuatro paneles y diez semillas habría 640 ajustes neuronales, cuatro ajustes ARIMA y dieciséis trayectorias analíticas. A 120 s por ajuste neuronal, la oportunidad nominal suma **21 h 20 min**, más ARIMA, selección, calibración, pruebas, respaldos y demás costes. Más semillas, configuraciones o presupuestos mayores aumentan ese total. Diez semillas es el mínimo de la guía, no evidencia de potencia suficiente.

Las reglas de respaldo del piloto anterior sí se verificaron en su propia ejecución. Su controlador está fijado en el commit de código original; no se afirma haber validado su integración en un nuevo entrenador de 21 sistemas. Ese trabajo debe seguir a un protocolo admitido y no se disfraza de entrenamiento ejecutado.
