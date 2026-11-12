# 40. Calendario original y optimizaciones de implementación

**Decisión vigente del 11/11/2026:** conservar el calendario original de **once orígenes**, todas las arquitecturas y las exigencias de E2. La reducción a tres orígenes de §37 queda como **propuesta no operativa, pendiente de justificar suficiencia temporal e inferencial**. Que la guía no imponga once no demuestra que tres basten.

## Qué se conserva

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

El ejecutor v2 usa una campaña separada, `campaigns/e2-cuda-crop-11-v2`, ligada al código y perfil de recursos. Conserva el historial A3 y el saldo transferido del límite inicial de 48 horas: **no reinicia ni amplía automáticamente el presupuesto**. El 11/11 se registró la ampliación manual autorizada a 56 horas acumuladas; [§41](41_avance_optimizacion_11_origenes.md) conserva su evento y contabilidad. Si no alcanza, pausa; una pausa no equivale a cumplir el benchmark. Para acabar podría ser necesario disponer de más recursos autorizados, en lugar de recortar requisitos sin evidencia.

## Uso y estado

**Supervisión opcional dentro del límite autorizado:** la sección 12 del notebook espera al controlador actual y puede reanudar automáticamente sólo las pausas limpias por límite de sesión, con un techo combinado de **56 horas ya autorizadas**. No añade tiempo ni cambia el código científico. Se detiene ante presupuesto global insuficiente para otra reserva, errores, conciliación o falta de progreso; no evita desconexiones ni cierres de Colab. Las 56 horas son acumuladas, no adicionales. La ampliación manual a 56 quedó registrada en el [commit de enmienda](https://github.com/jassan-bit/DeepLearning_202630/commit/25fcf0282cc67465056883f03684184ac8b2bdb6).


**Pausa por presupuesto:** volver a ejecutar la sección 8 sin cambiar el saldo puede producir la misma pausa, incluso con tiempo positivo si no alcanza la reserva completa de otro ajuste. La sección 11 del notebook permite una **enmienda manual**, sólo mediante un NUEVO LÍMITE TOTAL introducido por el usuario. Registra un evento append-only y confirma GitHub antes de reanudar; conserva plan, gasto anterior y resultados, y el mismo límite no se suma dos veces. Abrir o publicar la celda no autoriza ninguna ampliación. El límite inicial de 48 horas permanece en el historial; una ampliación efectiva deberá reportarse como desviación operativa, sin cambiar arquitectura, semillas o requisitos de E2.


**Ejecución documentada:** el [corte del 11/11/2026, 19:32:18 de Bogotá (§41)](41_avance_optimizacion_11_origenes.md) confirma 881 ajustes HPO nuevos completados en la T4, sin fallos en esos registros. La preparación ya dio paso a entrenamiento de mercado; sigue siendo selección parcial, sin benchmark final ni R cerrado.


El notebook vigente es [entregable2_optimizacion_11_origenes.ipynb](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_optimizacion_11_origenes.ipynb), enlazado desde la portada. Debe comprobarse el cierre del controlador anterior antes de iniciarlo; nunca deben correr dos simultáneamente. El código publicado no modifica la sesión privada de Colab. Las versiones de tres orígenes se conservan como antecedentes y no se recomiendan para nuevas ejecuciones.

Las pruebas CPU cubren equivalencia funcional/gradientes, modelos con dropout y AdamW, reutilización sin callbacks nuevos, cobertura de los once orígenes, esquema del diario y conservación del presupuesto. La medición CUDA sintética de esta admisión se realizó en la T4 antes de los nuevos ajustes y figura en el plan sellado de la campaña. Sólo mide los casos del bloque TimesNet declarados; no demuestra una aceleración global del entrenamiento. No se afirma cumplimiento integral de E2 ni se promete una calificación por publicar esta decisión.

[Registro vigente con corrección JSON](https://github.com/jassan-bit/DeepLearning_202630/blob/8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9/experiments/e2_implementation_optimization_11_origins_20261111/registration_json_metadata_fix.json) · [Registro v2 anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/1ca723348f3fd64ad0e48c899b8fa3f0f09ac5c3/experiments/e2_implementation_optimization_11_origins_20261111/registration.json) · [Ejecutor v2](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_e2_optimized_selection_v2.py) · [Matriz de requisitos pendientes](38_cumplimiento_guia.md).
