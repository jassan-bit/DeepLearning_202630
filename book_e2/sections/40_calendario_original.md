# 40. Calendario original y optimizaciones de implementación

**Enmienda posterior (12/11/2026):** el [catálogo obligatorio de §43](43_catalogo_obligatorio.md) se reduce a **16 sistemas: 13 redes y tres comparadores**. Híbrido adicional y Ridge son opcionales. Este capítulo conserva el plan sellado anterior de catorce redes y su techo de 85.074 intentos como antecedente; la publicación no cambia la sesión de Colab ni elimina resultados. No se modifican los once orígenes ni las demás políticas en este paso.

**Decisión registrada el 11/11/2026, anterior a la enmienda de catálogo:** conservar el calendario original de **once orígenes**, todas las arquitecturas y las exigencias de E2. La reducción a tres orígenes de §37 queda como **propuesta no operativa, pendiente de justificar suficiencia temporal e inferencial**. Que la guía no imponga once no demuestra que tres basten.

## Qué conserva el plan sellado anterior

- Selecciones mensuales de enero a octubre de 2024 y selección de enero de 2025: once orígenes comunes para todos los modelos.
- Top 10, candidato adicional, tres referencias neuronales E1 y cuatro comparadores; las cuatro familias del Top 10 se documentan en §38.
- Seis familias NAS, 1.000 arquitecturas NAS-1 por familia y origen, k=40 más referencia cuando corresponda, cinco semillas de confirmación; HPO de 18 configuraciones y dos semillas por red/origen.
- Contrato D7×5, objetivo σ₇, horizonte, primaria, δ, embargo, límites de épocas y parada registrados. El test conocido sigue siendo exploratorio.
- Potencia/R, auditoría, comparadores, inferencia con multiplicidad, robustez, interpretabilidad, ablaciones, incertidumbre, reproducción y entrega final: no se eliminan ni se dan por completados.

| Búsqueda neuronal, once orígenes | Máximo de intentos lógicos |
|---|---:|
| HPO | 5.544 |
| NAS-1 | 66.000 |
| NAS-2 | 13.530 |
| **Total** | **85.074** |

Es el calendario original del proyecto, no un número literal exigido por el PDF. Estos máximos excluyen reajustes y otros análisis. No representan una duración ni un compromiso de completar todo dentro del presupuesto disponible.

## Qué se optimiza

Se conserva la optimización funcional selectiva de TimesNet descrita en [§39](39_optimizacion_tiempo.md): recortar sólo coeficientes que multiplican padding cero, sin eliminar parámetros ni cambiar el espacio NAS. En la GPU real se verifican salidas/gradientes y se perfilan quince casos ancho/batch; sólo los casos con mejora mediana ≥5 % utilizan ese cálculo. El resto mantiene el cálculo original. Los microperfiles no acreditan una aceleración de extremo a extremo.

Se priorizan los sistemas con cuerpo fijo/referencias en los **once** orígenes y después las familias NAS, conservando sus oportunidades de ajuste. Se reutilizan resultados anteriores sólo con identidad de trabajo, configuración, semilla y soporte coincidentes. No se cuentan como nuevos ajustes ni nuevos pasos GPU.

El ejecutor v2 usa una campaña separada, `campaigns/e2-cuda-crop-11-v2`, ligada al código y perfil de recursos. Conserva el historial A3 y el saldo transferido del techo inicial de 48 horas: **no reinicia ni amplía automáticamente el presupuesto**. El 11/11 se registraron las ampliaciones manuales autorizadas a 56 y después **64 horas acumuladas**; [§41](41_avance_optimizacion_11_origenes.md) conserva sus eventos y contabilidad. Si no alcanza para otra reserva completa, pausa. La finalización de E2 depende de completar sus evidencias y no de agotar un presupuesto.

## Uso y estado

**Alcance del supervisor publicado:** la sección 12 del notebook conserva un techo combinado de **56 horas** y sólo reanuda pausas limpias por límite de sesión. Al detectar el nuevo total de 64 horas se detiene; ese supervisor no se ha actualizado en este corte documental y no corresponde activarlo para este estado. No añade tiempo ni evita cierres de Colab. La ampliación histórica a 56 permanece en el [commit de enmienda](https://github.com/jassan-bit/DeepLearning_202630/commit/25fcf0282cc67465056883f03684184ac8b2bdb6).


**Pausa por presupuesto:** volver a ejecutar sin cambiar el saldo puede producir la misma pausa, incluso con tiempo positivo si no alcanza la reserva completa de otro ajuste. Toda enmienda manual exige un NUEVO LÍMITE TOTAL introducido por el usuario, un evento añadido al diario y confirmación de GitHub antes de reanudar. Conserva plan, gasto anterior y resultados; el mismo límite no se suma dos veces. Publicar código no autoriza una ampliación. Los cambios de presupuesto se declaran como desviaciones operativas, conservando arquitecturas, semillas y requisitos de E2.


**Ejecución documentada:** el [corte del 12/11/2026, 02:29:43 de Bogotá (§41)](41_avance_optimizacion_11_origenes.md) confirma 1.484 ajustes HPO nuevos completados en la T4, sin fallos en esos registros. La primera ola de ocho redes/referencias cerró enero–mayo; en junio, FCN/ResNet/Inception/TSMixer completaron 36 cada uno e iTransformer registra 8/36. La campaña quedó pausada por límite de sesión, con cierre limpio. Sigue siendo selección parcial, sin benchmark final ni R cerrado.

**Reanudación de esta pausa limpia:** la siguiente reserva requería 7.500 s y quedaban 7.453,14 s de la sesión. Al cierre hay 14.951,37 s disponibles dentro del techo acumulado de 64 horas. La ruta de continuación es el ejecutor ordinario de la sección 8, con la preparación 1–7 verificada, RUN_SEARCH=True, RUN_OPTIMIZED_SELECTION=True y RETRY_LOST_ONCE=False, tras comprobar la T4 y ausencia de otro controlador. El diario conserva el saldo y los resultados. Este corte no tiene trabajos lógicos pendientes de conciliación ni necesita repetir la recuperación del incidente anterior.

**Recuperación documentada:** tras la desconexión se conservaban 889 resultados y un intento abierto. El [código de recuperación registrado](https://github.com/jassan-bit/DeepLearning_202630/blob/9d70bdad62af54316c9089d9c22e0781c3dd55a2/notebooks/recover_e2_colab_loss_20261112.py) registró el nuevo total introducido por el usuario, recuperó ese mismo trabajo una sola vez y continuó el ejecutor científico congelado. Los recibos de autorización, inicio, observación y cierre están en §41. La asignación completa de ocho horas de la sesión perdida se conserva en la contabilidad; el desenlace y los pasos del intento original siguen desconocidos. Esta actualización del libro no inicia procesos ni cambia la sesión privada.


El notebook vigente es [entregable2_optimizacion_11_origenes.ipynb](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_optimizacion_11_origenes.ipynb), enlazado desde la portada. Debe comprobarse el cierre del controlador anterior antes de iniciarlo; nunca deben correr dos simultáneamente. El código publicado no modifica la sesión privada de Colab. Las versiones de tres orígenes se conservan como antecedentes y no se recomiendan para nuevas ejecuciones.

Las pruebas CPU cubren equivalencia funcional/gradientes, modelos con dropout y AdamW, reutilización sin callbacks nuevos, cobertura de los once orígenes, esquema del diario y conservación del presupuesto. La medición CUDA sintética de esta admisión se realizó en la T4 antes de los nuevos ajustes y figura en el plan sellado de la campaña. Sólo mide los casos del bloque TimesNet declarados; no demuestra una aceleración global del entrenamiento. No se afirma cumplimiento integral de E2 ni se promete una calificación por publicar esta decisión.

[Registro vigente con corrección JSON](https://github.com/jassan-bit/DeepLearning_202630/blob/8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9/experiments/e2_implementation_optimization_11_origins_20261111/registration_json_metadata_fix.json) · [Registro v2 anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/1ca723348f3fd64ad0e48c899b8fa3f0f09ac5c3/experiments/e2_implementation_optimization_11_origins_20261111/registration.json) · [Ejecutor v2](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_e2_optimized_selection_v2.py) · [Matriz de requisitos pendientes](38_cumplimiento_guia.md).
