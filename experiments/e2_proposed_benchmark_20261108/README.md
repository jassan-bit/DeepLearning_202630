# Prototipo aislado del siguiente benchmark E2

**08/11/2026: implementación preliminar no registrada, no benchmark ejecutado.**
No cambia el catálogo de veinte configuraciones, el runner, los candidatos,
los recibos ni los resultados históricos. No abre precios ni test. No ejecuta HPO.
No hay afirmación de superioridad ni de reproducción exacta de artículos.

## Alcance implementado

- `scripts/e2_proposed_models.py`: FCN1D (tres convoluciones 3/3/1),
  ResNet1D (dos bloques, convoluciones 3/3 con proyección de atajo),
  Inception1D (bottleneck, ramas 1/3/5 y max-pooling), LSTM-AE predictivo
  (encoder, decoder de latente repetido, cabeza escalar), corrector HAR residual.
- Son adaptaciones originales compactas de regresión. Ancho predeterminado 16,
  ReLU y pooling temporal promedio en convoluciones. Sin batch normalization,
  dilataciones ni dropout. Estos detalles son **supuestos de prototipado**, no
  configuraciones elegidas del profesor ni réplicas FCN/ResNet/InceptionTime.
- Las cuatro redes reciben exactamente `(batch,7,5)`, días cerrados OHLCV y
  entregan `(batch,)` con softplus. El padding simétrico usa sólo la ventana ya
  observada: no es una convolución causal intraventana ni se presenta como tal.
- LSTM-AE ofrece reconstrucción y predicción por separado. La intención es
  pretrain de reconstrucción seguido por finetune supervisado. No implementa
  scheduler, selección de épocas ni política de congelado. No afirma que la
  fuente de 2019 use pérdida conjunta. El test sintético de gradientes combina
  objetivos sólo para verificar conexiones, no para elegir la receta.
- HAR residual calcula `max(0, HAR_raw + corrección_firmada(D7x5))`. HAR es una
  predicción externa: **todavía no se implementa ajuste HAR/OLS ni pipeline de
  cross-fitting**. La salida no negativa es una propuesta, no una regla heredada
  automáticamente. **No listo para entrenar con datos reales:** `clamp_min`
  produce gradiente cero cuando la suma es negativa. La prueba adicional fuerza
  ese caso y acredita la limitación. La reducción de pérdida sintética anterior
  no resuelve este problema. Una futura receta puede entrenar la corrección
  firmada contra residuos OOF crudos, aplicando truncamiento sólo al predecir;
  esta alternativa está pendiente de diseño/registro, no implementada como
  pipeline ni aprobada retrospectivamente.

## Dos soportes de información, no once entradas equivalentes

El soporte neural sigue D7×5 sin rezagos del objetivo. La utilidad
`har_features` propone medias de sigma7 en 1/7/22 **días calendario** anteriores
al origen. 22 no es una elección validada para Bitcoin 24/7; 30 debe discutirse
antes de fijar el diseño, no probarse hasta obtener ventaja. No son sesiones
bursátiles ni volatilidad realizada intradía.

Una sigma7 fechada t-22 usa cierres t-29..t-22; el soporte HAR alcanza t-29,
frente a siete barras de la red. Se rechaza cualquier hueco y nunca se comprimen
fechas. Se debe definir cohorte común tras calentamiento y contar exclusiones
por modelo/fecha antes de comparar. Red sola D7×5 es un control útil, pero no
identifica una ganancia exclusivamente arquitectónica del híbrido: faltan
controles de soporte equiparable y un HAR solo con idéntico calendario y link.

## Prevención de fuga, pendiente de integrar en runner nuevo

`validate_training_scope` rechaza orígenes fuera del train explícito de cada
fold. Debe llamarse **antes** de ajustar escaladores, preentrenar encoder y
entrenar de forma supervisada. No se permite preentrenar en todo desarrollo
para luego hacer CV interna, aunque no use etiquetas. Cada fold reinicia pesos,
escaladores y estado; no se reciclan checkpoints posteriores hacia el pasado.

`residual_training_targets` valida registros temporales OOF del componente
base: corte de etiquetas al menos ocho días antes del origen (siete días
intermedios de embargo), y verdad disponible al ajuste del corrector. Esto
replica conservadoramente la separación heredada, sin registrar un nuevo
calendario. Debe generarse realmente cada predicción sin entrenar sobre su
etiqueta, no inventar metadatos. La función no puede demostrar que un llamador
haya declarado procedencia verdadera. No basta para auditar cross-fitting.

Para el futuro pipeline: separar dentro de train un periodo inicial HAR y
bloques temporales OOF; entrenar corrección sólo en sus residuos disponibles;
para validación, reajustar HAR con train permitido y congelar ambos componentes.
La diferencia entre HAR OOF y HAR reajustado, transformaciones, datos faltantes,
escalado del residual y calibración requieren reglas previas. Ningún residuo de
validación/test puede convertirse en objetivo de entrenamiento de ese fold.

## Decisiones pendientes antes de entrenar datos reales

1. Validar fuentes/configuraciones del curso y familias exigidas; no equivale
   esta implementación a completar la revisión bibliográfica.
2. Congelar arquitectura, objetivos/pérdidas, pretraining y finetune, optimizador,
   transformaciones y escalado train-only, NAS/HPO, épocas, semillas y costes.
   El pretraining cuenta dentro del presupuesto del LSTM-AE.
3. Congelar fechas y embargo; soportes HAR y controles comparables; política de
   negativos, métricas en unidades originales y ablations. Diez propuestas más
   híbrido no fija aún el número total de configuraciones.
4. Incorporar seis adaptadores retenidos y anclas pertinentes sólo en un runner
   **nuevo**, preservando identidades históricas. Registrar hash, entorno,
   manifiesto completo y flujo preflight/register/fit/score separado.
5. Revisar análisis, incertidumbre, potencia, multiplicidad y criterio práctico
   antes de resultados. El test ya observado sigue siendo exploratorio; no se
   recupera independencia cambiando el catálogo o el nombre del experimento.
6. HAR + red no es novedad genérica. La revisión bibliográfica identifica
   antecedentes y resultados desfavorables; no prometer ventaja.

## Verificación realizada (sólo sintética)

Comando desde raíz: `python tests/test_e2_proposed_benchmark.py`.
Diez pruebas pasaron el 08/11/2026 con Python 3.12.14, torch 2.15.0+cpu,
numpy 2.3.5, un hilo CPU. No se afirma coincidencia completa con el lock histórico
(numpy allí tiene otra versión); falta fijar un entorno del nuevo diseño.

Cobertura: formas/finitez, no negatividad, gradientes, invariancia de batch,
roundtrip de pesos en memoria, rechazo de formas inválidas y NaN, reducción de
pérdida de predicción en ejemplos artificiales y de reconstrucción del AE,
HAR sin lectura del presente/futuro, huecos/calendario, rechazo de pretraining
fuera de fold y de residuos no OOF/no disponibles (incluidos generadores), y
gradiente cero documentado bajo saturación negativa del híbrido. Ninguna prueba abre datos de
mercado. La memorización sintética verifica que el código aprende, no precisión,
robustez o superioridad. No se ejecutó el benchmark ni su suite integral.

Corrección de QA: se materializan los registros OOF antes de recorrerlos dos
veces, evitando que un generador agotado omita la validación por registro.
