# Resolución final de las cuatro observaciones

**Corte final del 16/11/2026, por instrucción del autor.** Se cierra la revisión de auditoría, HPO, reproducción y alcance histórico con la información disponible. Cada observación tiene una decisión terminal, evidencia y afirmaciones permitidas. **No hay tareas correctivas activas para estos cuatro puntos dentro de esta versión congelada de E2.**

La decisión de terminar el seguimiento **no acredita los requisitos que la evidencia no permite aprobar**. Los cinco estados parciales del Apéndice B conservan su significado científico. Un evaluador puede mantener descuentos por esas limitaciones; este cierre no modifica la rúbrica ni garantiza una nota. La tarea sigue siendo **F2a, serie de tiempo**, y **NAS del capítulo 14 no aplica**.

## Dictamen por observación

| Observación | Resolución con la evidencia disponible | Aceptación científica | Estado del seguimiento |
| --- | --- | --- | --- |
| **C01 · Auditoría integral** | Procedencia, cohortes, transformaciones, ports, causalidad clásica y controles negativos ejecutados; se conserva el resultado adverso. | **Aceptación integral no establecida.** La cercanía a la referencia constante no está acreditada y la causa de la señal retenida no fue identificada. Tampoco se ha demostrado una vía de fuga mediante estos controles. | **Cerrado con dictamen; B03 parcial.** |
| **C02 · Cobertura HPO** | Selección reducida de seis tuplas por red terminada; 234 oportunidades, 231 resultados completos, trece ganadores y cargos observados conciliados. | Igualdad de plazas acreditada. **Amplitud mínima integral e importancia independiente no acreditadas.** Las asociaciones disponibles son descriptivas. | **Cerrado con limitación del diseño; B06/B09 parciales.** |
| **C03 · Reproducción** | Reproducciones y portabilidad con recibos propios; los módulos de reanálisis disponibles se integran en un comando independiente. | **Reanálisis integrado verificable.** No se acredita que ese comando repita toda la campaña histórica, HPO, entrenamientos y figuras. | **Cerrado en el alcance entregado; B21 parcial.** |
| **C04 · Anterioridad y alcance** | Cronología auditada por etapa, comparación OOF–prueba ejecutada y cohorte Scopus original fijada. | Registro anterior a 170 HPO nuevos y 78 ajustes finales. **No acredita anterioridad histórica integral, cegamiento restaurado, validación externa independiente o exhaustividad bibliográfica.** | **Cerrado con alcance retrospectivo; B01 parcial.** |

[Decisiones estructuradas](../_static/e2_resolucion_final_20261116/decisions.json) · [Tabla descargable de las cuatro resoluciones](../_static/e2_resolucion_final_20261116/observaciones_finales.csv) · [Apéndice B final, CSV](../_static/e2_resolucion_final_20261116/appendix_B_final.csv) · [Apéndice B final, JSON](../_static/e2_resolucion_final_20261116/appendix_B_final.json).

## Auditoría: resultado adverso conservado y aceptación no establecida

Los trece controles iniciales y los seis adicionales están ejecutados. En el suplemento de TSMixer e iTransformer, TOST/Holm2 no establece equivalencia con la constante dentro del margen registrado. La revisión estática no identifica una ruta explícita de etiquetas futuras; las funciones antes del entrenamiento tampoco explican íntegramente la señal posterior. **No se atribuye una causa que los datos no han demostrado.** [Resultados, planes y límites de §66](66_seguimiento_pendientes.md).

El control permuta las etiquetas del ajuste final, mientras que arquitectura, receta y épocas proceden de selección previa con etiquetas reales. No reproduce toda la selección bajo el nulo. Además, σ₇(t) contiene seis retornos conocidos legítimamente al emitir el pronóstico. Si A y B son la suma y la suma de cuadrados de esos seis retornos, respectivamente:

```{math}
\sigma_7(t)^2 = \frac{B+r_t^2}{7}-\frac{(A+r_t)^2}{49}.
```

Por ello, superar una constante después de permutar no identifica por sí solo una fuga: pueden existir funciones legítimas no constantes de X. **Esta observación tampoco demuestra que explique las ganancias concretas del control.** Se rechazan tanto la acusación automática de fuga como la certificación de ausencia de fuga.

La regla de la guía, §10/p.46, exige corregir la causa y repetir el experimento afectado para aprobar una verificación fallida; **este cierre no acredita que eso haya ocurrido**. A2 y los análisis quedan disponibles como resultados retrospectivos reproducibles, sin presentarse como un benchmark íntegramente validado conforme a la Tabla 10.1. El dictamen final de B03 es **parcial, con aceptación integral no establecida**, y no una aprobación acompañada de una nota al pie.

## HPO: dominio ejecutado y límites de identificabilidad

El dominio ejecutado tiene seis configuraciones por red y tres folds. La alternativa de igualdad del número de configuraciones de §11.1 está acreditada; no se exige además igualdad de segundos. Se conservan los fallos, el desenlace desconocido y los ámbitos distintos de medición de recursos. Los **233 cargos observados suman 25.833,898918 s**; la asignación desconocida de **1.860 s** permanece separada.

Sólo **uno de 195 pares** cambia un único parámetro: PatchTST C02–C06 modifica LR y conserva WD, batch, dropout y warmup. Sus dos puntuaciones permiten describir una sensibilidad observada de esa receta, no una importancia global ni una incertidumbre poblacional. Las 237 asociaciones descriptivas ya publicadas no se renombran como estimaciones independientes. No se ajustan retrospectivamente modelos de importancia para obtener un cierre favorable.

La propuesta posterior de dieciséis tuplas queda **archivada como diseño no ejecutado**, sin trabajos pendientes en esta versión. El piloto temporal está terminado: dos ajustes completos y TimesNet censurado. No se extrapola su coste a las trece redes. El benchmark se interpreta condicionado a las recetas realmente exploradas; **B06/B09 conservan su cumplimiento parcial**. [Dominio y costes](11_hpo_equidad.md).

## Un comando para los reanálisis disponibles

El [paquete integrado independiente](../_static/e2_resolucion_final_20261116/reanalisis_integrado_v1.zip) incluye sus entradas, módulos, hashes, ejecutor e instrucciones. Después de extraerlo e instalar las dependencias indicadas en su `LEEME.md`, se ejecuta el comando de su carpeta:

```bash
python run.py
```

El ejecutor encadena los módulos incluidos y genera un informe de ejecución con su alcance. La [verificación fuera del repositorio](../_static/e2_resolucion_final_20261116/reanalisis_integrado/summary.json) identifica los módulos completados, comparaciones y cargos. No necesita clonar el proyecto ni descargar otro entregable. No entrena ni hace inferencia sobre modelos en este modo.

Este cierre resuelve la ejecución conjunta de esos reanálisis. **No afirma reconstruir toda la campaña histórica desde los veinte ZIP de minutos, ni repetir HPO, todos los ajustes o todas las figuras.** Las reproducciones originales A100 y la repetición posterior CPU de recetas fijas conservan sus propios recibos. B21 sigue parcial para el literal integral de la guía; no se aprueba por disponer de un nuevo ejecutor.

## Alcance definitivo de las conclusiones

Se conservan como resultados de E2:

- SAE tiene la menor RMSE media observada de A2 en las 608 fechas comunes.
- Ninguno de los veinte contrastes primarios demuestra mejora mayor que δ tras Holm20, bajo los supuestos declarados; eso no demuestra igualdad.
- RQ2 y RQ3 CPU están ejecutadas como suplementos posteriores, con sus familias y límites.
- OOF249–prueba608 es una comparación ejecutada entre ajustes y periodos distintos. La diferencia de SAE no identifica su causa ni valida externamente el procedimiento de selección.
- La descomposición con IC describe las pérdidas del catálogo y las inicializaciones observadas; no identifica variación entre mercados o particiones nuevas.
- La cronología acredita las etapas verificadas; no certifica que todo E2 se registró antes de cualquier entrenamiento o conocimiento del test.

Scopus queda cerrado con **7.500 registros originales**. Se preservan **9.559 grupos multibase, 42 estudios y 174 informes no recuperados**, junto con los límites de consultas, cobertura y acceso ya documentados. No se solicitan nuevas exportaciones para esta entrega. [Decisión bibliográfica](62_cierre_literatura_focal.md).

## Verificación de la resolución y versiones anteriores

El [ZIP pequeño de resoluciones y evidencia](../_static/e2_resolucion_final_20261116/resolucion_cuatro_observaciones_v1.zip) permite comprobar hashes, decisiones y estados con **Python estándar**, sin red, modelos o remuestreos:

```bash
python verify.py
```

El verificador comprueba que **cerrar el seguimiento no cambie a aprobado un requisito parcial**. Este comando documental es distinto del `run.py` del reanálisis numérico integrado. Ambos paquetes son independientes y declaran sus entradas y operaciones.

El [recibo del último reanálisis](../_static/e2_resolucion_final_20261116/reanalisis_integrado/accounting_receipt.json) y el [diario aditivo final](../_static/e2_resolucion_final_20261116/compute_ledger.jsonl) conservan el límite acumulado de 84 horas, los cargos anteriores y cero reservas abiertas. El diario anterior mantiene sus bytes; el nuevo prolonga su cadena con la comprobación integrada.

Los bytes originales de A2, pronósticos, p-valores, IC, márgenes y descargas se conservan. El Apéndice B final mantiene **14 completos, 5 parciales y 2 no aplicables** y añade un campo separado de resolución. Las listas previas y §66 siguen disponibles como cortes fechados. Las propuestas de ampliación se archivan; no son compromisos activos ni se ejecutan nuevas corridas para buscar aprobación. Las limitaciones permanecen visibles y evaluables.
