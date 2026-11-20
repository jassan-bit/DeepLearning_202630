# Revisión independiente del contrato de las nueve redes históricas

Revisión automática asistida por IA. Se comprobaron los bytes ONNX aceptados y los metadatos originales de las nueve redes adicionales. No se ejecutó entrenamiento, recalibración ni reexportación. El recibo `contract_review.json` distingue esta comprobación estructural de la paridad funcional, del ensayo de recursos y de la aceptación pública, que requieren sus propios recibos.

## Contrato común verificado

Las nueve entradas ONNX son `windows`, `tensor(float)` de forma `[1,7,5]` y la salida única es `normalized_prediction`, con inversión **una sola vez** mediante `0.020953875259434797` para obtener sigma7 poblacional, diaria y no anualizada. Los representantes usan semilla 271801 y el escalador congelado de 655 ventanas, terminado el 2024-10-31. La transformación correcta es logaritmo de los cuatro precios relativo al último cierre observado, `log1p` del volumen, estandarización con media/desviación congeladas y conversión a float32. No se normaliza con el CSV recibido.

| Modelo | q original, sigma7 | Bytes ONNX | OOD en metadato original | Referencia de explicación en metadato original |
|---|---:|---:|---|---|
| FCN-1D | 0.0065703122088813862 | 1092005 | No | No |
| Inception-1D | 0.0075870616069819742 | 2070967 | No | No |
| LSTM-autoencoder con predictor | 0.0092055694529858435 | 41397 | No | No |
| PatchTST | 0.016008127390891344 | 107745 | Sí | No |
| ResNet-1D | 0.0072527258390185645 | 2156012 | No | No |
| TiDE | 0.011361150287153492 | 664848 | Sí | No |
| TimeMixer | 0.010437534506757353 | 172380 | Sí | No |
| TimesNet | 0.0089159080467011743 | 4805411 | Sí | No |
| iTransformer | 0.0063839239889132912 | 3201493 | Sí | No |

Los nueve cuantiles están vinculados al SHA del mismo checkpoint y a 53 fechas históricas hasta el **2024-12-30**. Deben aplicarse únicamente a objetivos posteriores a esa fecha. Para fechas anteriores o coincidentes, se conserva el pronóstico puntual y se deshabilita el intervalo con explicación visible. La cobertura nominal 0,90 no garantiza cobertura futura, por subgrupo ni bajo cambio temporal.

## Extensión del contrato sin mutar metadatos científicos

Los cuatro exportados estándar (FCN, ResNet, Inception y autoencoder) carecen de límites OOD y referencia de explicación en sus metadatos originales. Los cinco temporales tienen OOD, pero no referencia cruda. Estos campos deben agregarse en un descriptor nuevo y versionado; los metadatos originales se conservan byte por byte.

Es admisible usar los rangos por celda de `training_X` del cohorte cuyo SHA256 es `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`, únicamente tras verificar coincidencia exacta del escalador, fechas y dimensiones. La referencia de explicación es la primera ventana cruda de ajuste, objetivo 2023-01-08; coincide con la referencia congelada E2 ya usada por MLP16. No se utilizan fechas de evaluación para fijar OOD ni referencia. No se modifica un calibrador para compensar una exportación.

La explicación por oclusión debe transformar de nuevo cada ventana perturbada con el mismo escalador y referencia. Es sensibilidad local, no atribución causal ni prueba de fidelidad global. El aviso OOD describe los rangos de ajuste y no certifica fiabilidad cuando no se exceden.

## Diferencias que el cargador debe respetar

- Los cinco modelos originales conservan su metadato y sus binarios. La aditiva tiene salida sigma7 directa (escala 1), escalador propio de 532 ventanas e input adicional `variance_floor` float64 `[1]`.
- El piso observable se calcula con seis retornos conocidos como `sum((r-r.mean())**2)/7`; no se sustituye por una desviación de siete retornos ni por la etiqueta futura.
- El cuantíl posterior de la aditiva procede de su perfil versionado ya existente, ligado a modelo, semilla y checkpoint exactos; no se toma el q de otro predictor.
- V3 `atom_crps` usa otro contrato (`values`, `A`), carece de calibrador y permanece fuera del perfil histórico de 14. Tener un ONNX público no autoriza incorporarlo silenciosamente al servicio.
- TimesNet se ejecuta con lote uno: su selector espectral y orden en empates ya están dentro del grafo aceptado. No sustituirlo por periodos fijos ni hacer batch de usuarios.
- Una sesión ORT residente por proceso, carga diferida y bloqueo serial conservan un límite concreto. Debe medirse el proceso completo (incluyendo explicación y HTTP) antes de alegar que cabe en una instancia Render determinada.

## Por qué los tres clásicos no caben en el mismo CSV de siete barras

El contrato actual recibe barras completas `t-7,…,t-1`, pero la etiqueta sigma7 de una fecha es la desviación poblacional de **siete retornos**, que requiere **ocho cierres**. Esto se verifica en `scripts/e2_temporal_controls.py:sigma7`.

| Clásico congelado | Entrada científica que falta | Política del perfil de siete barras |
|---|---|---|
| Persistence, lag 1 | sigma7(t−1), cierres t−8,…,t−1: ocho barras diarias y calidad explícita | Conservar resultado histórico; no inventar octavo cierre ni usar el piso de seis retornos como sustituto |
| SeasonalNaive, lag 7 | sigma7(t−7), cierres t−14,…,t−7: si se entrega historia cruda continua hasta t−1 se necesitan al menos 14 barras | Conservar resultado histórico hasta tener interfaz de historia fechada; no cambiar el lag |
| ARIMA congelado | Parámetros, escala y estado filtrado correspondientes a su fecha exacta; actualización diaria de sigma7 observada y pronóstico persistido antes de incorporar cada etiqueta | Conservar resultado histórico; requiere servicio estatal o reconstrucción auditada desde checkpoint e historia, no un refit sobre siete barras |

`FrozenARIMA.forecast` exige estado en t−1 y `observe` exige avanzar exactamente un día; una fecha arbitraria no puede saltarse esa secuencia. La elección original de orden y parámetros debe mantenerse. Un contrato alternativo con historia/endógenas sería una capacidad nueva versionada, con pruebas propias; no forma parte de esta extensión de nueve redes.

## Resultado

**GO técnico limitado para probar un perfil histórico de 14 redes**: las nueve identidades y los contratos ONNX coinciden con los aceptados. Este resultado no habilita automáticamente Render ni demuestra R3. La habilitación exige además paridad de extremo a extremo, manejo de intervalos/fechas, OOD, explicaciones, errores, recursos y aceptación del SHA efectivamente desplegado.
