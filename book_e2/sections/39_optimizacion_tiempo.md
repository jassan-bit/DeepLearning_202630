# 39. Optimizar tiempo sin recortar las exigencias de E2

**Enmienda de implementación del 11/11/2026.** Se preservan las catorce redes, cuatro comparadores, contrato D7×5, tres orígenes, seis familias NAS y sus espacios registrados. La actualización no activa un Colab privado ni declara terminado el benchmark. [Exigencias y pendientes de E2](38_cumplimiento_guia.md).

## Decisión sobre las arquitecturas publicadas

El escenario de 8.742 intentos dependía de justificar cuatro nuevas exenciones de NAS. **No se aplica ese escenario:** la inspección del código actual no demuestra cuatro cuerpos publicados intactos.

| Modelo | Diferencia que impide una exención automática de la variante actual |
|---|---|
| TimesNet | Selección espectral por ejemplo, correspondencia de bloques variables y decisiones propias para la geometría D7 |
| TiDE | Decoder temporal de ocho dimensiones en lugar de la salida escalar de referencia y covariables sustitutas |
| TimeMixer | Profundidad multiescala limitada por D7 y cuerpo de tres bloques NAS con anchos variables |
| PatchTST | Variante NAS de anchos/cabezas variables, cambios en atención residual y geometría de parches/readout adaptada |

Usar el mismo nombre o congelar una variante propia no acredita la excepción de p. 53. Restaurar otra arquitectura publicada sería una receta nueva que exigiría comparar sus capas, dimensiones, mecanismos y adaptación de tarea con las fuentes, probarla y registrar el cambio antes de usar su exención. Los resultados anteriores no se le atribuirían automáticamente.

Se mantienen **1.000 arquitecturas NAS-1 por familia/origen**, k=40, cinco semillas NAS-2, las 18 configuraciones HPO con dos semillas y los presupuestos de 50/200 épocas con la parada registrada. El máximo sigue en **23.202 intentos neuronales de selección** para los tres orígenes, sin incluir el resto de E2. Ningún requisito de robustez, inferencia o reproducción se elimina.

## Optimización funcional de TimesNet

Las convoluciones procesan rejillas muy pequeñas de ocho tokens. Con padding de ceros, un coeficiente situado a más de H−1 o W−1 del centro nunca se encuentra con un valor real de una rejilla H×W: multiplica ceros en todas las salidas. El nuevo cálculo recorta sólo esa zona inactiva antes de fusionar los kernels y usa el padding equivalente.

**Se conservan todos los parámetros**, sus gradientes completos, los bloques, anchos, patrones de dropout, frecuencias, semillas, pérdida y optimizador. Las posiciones inactivas reciben gradiente cero y siguen sujetas a la misma política de AdamW. Esto optimiza la implementación; no reduce el espacio NAS ni convierte el modelo en réplica de un artículo.

Las pruebas comparan salidas y gradientes de todas las rejillas, varios números de kernels y top-k, y modelos completos con y sin dropout durante tres actualizaciones AdamW. La igualdad algebraica no significa identidad bit a bit de trayectorias FP32: se declaran tolerancias y procedencia de implementación. Los reajustes finales deberán usar una implementación común declarada; no se presentan resultados antiguos como nuevas ejecuciones.

## Medición y activación selectiva

El [microperfil preliminar de CPU](../_static/implementation_optimization_20261111/cpu_microprofile.json) usa cinco repeticiones, batch8 y seis kernels. Excluye AdamW, carga de datos, procesos y sincronización remota; **no estima la aceleración total ni el tiempo de T4**.

| Ancho del bloque | Razón tiempo original / recortado, CPU |
|---|---:|
| 16 | 0,93× |
| 32 | 1,07× |
| 64 | 1,61× |
| 128 | 0,96× |
| 256 | 1,33× |

Una razón menor que uno significa que el recorte fue más lento en ese caso. Por eso no se habilita globalmente.

Antes de nuevos ajustes, el notebook mide **en la GPU real** las cinco anchuras y batches16/32/64, con seis kernels, cinco repeticiones de forward/backward y orden alternado. Deben pasar las comprobaciones numéricas. Sólo se usa el recorte en los pares ancho/batch cuya mediana mejore al menos 5 %. Los demás pares, los lotes finales no perfilados y los casos de dos/cuatro kernels conservan el cálculo original. Si ningún caso mejora, todo TimesNet conserva el cálculo original; el nuevo orden de ejecución sigue siendo útil. El microperfil es representativo, no una garantía de acelerar cada corrida o su coste completo.

No se habilitan AMP, TF32 ni menos semillas o épocas. Si falla la comprobación numérica, el controlador se bloquea antes de entrenar mercado. El perfil de admisión queda sellado dentro del nuevo plan y se reutiliza al reanudar, sin repetirlo innecesariamente.

## Orden de ejecución y reutilización

Primero se recorren los ocho sistemas con cuerpo fijo o de referencia —FCN-1D, ResNet-1D, Inception-1D, TSMixer, iTransformer, MLP16, LSTM32 y GRU32— en los tres orígenes. Después se recorren las seis familias NAS. No se cambia el número de oportunidades de ningún modelo; se evita que una búsqueda larga monopolice el avance inicial.

El controlador utiliza la fuente científica A3 inmutable y una **campaña separada de implementación**, `campaigns/e2-cuda-crop-v1`. El snapshot anterior se valida y debe descender del cierre de R4. Se reutiliza un resultado sólo si coinciden trabajo original, configuración, semilla, soporte preparado y payload. Los recibos antiguos se conservan en su rama original.

`completed_jobs` en la nueva campaña cuenta **sólo ajustes nuevos**. `reused_jobs` cuenta resultados anteriores efectivamente recorridos; `total_available_unique_jobs` distingue la disponibilidad histórica más los nuevos. No se suman sus pasos como nuevo trabajo GPU ni se duplican entrenamientos ya confirmados. Ambos planes y la procedencia de código permanecen identificados.

## Presupuesto y límites

Se transfiere **únicamente el saldo sin gastar del límite original de 48 horas**, calculado desde el diario validado, no desde un `status.json` posiblemente antiguo. No se conceden otras 48 horas. La admisión y el perfil que concluyen con un plan sellado, el replay de caché y los ajustes consumen el reloj de la nueva sesión. Un preflight rechazado puede consumir recursos externos sin crear un registro científico; la publicación final conserva la frontera contable del diario original.

El saldo puede no alcanzar para todos los trabajos y las reservas completas pueden provocar una pausa aunque quede algo de tiempo. No se amplía automáticamente. Esta optimización no garantiza terminar las miles de corridas dentro de ese saldo ni modifica el dimensionamiento pendiente de R.

## Cómo usarlo

El [notebook de optimización](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_optimizacion_tiempo.ipynb), enlazado desde la portada, prepara el mismo entorno A3 y sus datos de desarrollo, comprueba que no haya otro controlador y verifica la fuente fijada del nuevo ejecutor. Ejecuta primero sus pruebas y el perfil CUDA; sólo después inicia el calendario nuevo. No debe ejecutarse simultáneamente con el notebook anterior. La publicación del libro no cambia la sesión activa.

[Registro de la implementación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e2_implementation_optimization_20261111/registration.json) · [Ejecutor versionado](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_e2_optimized_selection.py).
