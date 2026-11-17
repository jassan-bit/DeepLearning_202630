# Estabilidad de la CNN-LSTM con atención y cabeza aditiva v2

Esta campaña evalúa la variación entre diez inicializaciones nuevas de una receta ya elegida. Se prepara **después** de conocer el resultado histórico del checkpoint original (`c27f892c7743a6a80f6b49b3a6930b18e2e20943`); es un análisis retrospectivo de estabilidad, no un preregistro confirmatorio ni una acreditación de R1 o R3.

## Receta fijada

- Semillas nuevas `271801` a `271810`, todas conservadas. La semilla histórica seleccionada `20261116` queda fuera de la media nueva.
- Misma CNN causal, LSTM y atención de contenido; cabeza aditiva, 1210 parámetros.
- 532 ejemplos de ajuste hasta 2024-06-30 y 116 fechas de validación entre 2024-07-08 y 2024-10-31; mismo escalador guardado de v2, sin reajuste.
- Inicialización nueva e igualación inicial del exceso mediante la bisección original, usando únicamente ajuste.
- AdamW, tasa `0.001`, `weight_decay=0`, batch 32, clip 1, máximo 40 épocas por semilla y paciencia 8. La mejor época se determina exclusivamente por RMSE de validación; los empates conservan la primera.
- Codificador y atención FP32, cabeza FP64, sin AMP ni TF32.

Las épocas se intercalan entre semillas. Se guardan pesos, optimizador, RNG, permutación y cursor de lote. La publicación normal se hace cada 50 épocas de modelo, es decir, aproximadamente cinco épocas de cada semilla activa.

## Entrenamiento, validación y test

El proceso de entrenamiento abre sólo los tres miembros `training_*` del NPZ auditado. La evaluación de las 608 fechas históricas de E2 se ejecuta en una fase separada, después de publicar y confirmar los checkpoints de las diez semillas. El trabajador comprueba que los hashes de esos diez archivos coincidan con la confirmación remota antes de abrir los miembros `evaluation_*`.

Los resultados incluyen cada semilla y la media de los diez RMSE individuales, la desviación estándar muestral (`ddof=1`) y un intervalo percentil por bootstrap de semillas con 10000 remuestras. Este intervalo describe variación de optimización condicionada al calendario observado; no es incertidumbre entre conjuntos de datos. No se utiliza el RMSE del ensemble ni se elige una semilla por el test. La semilla representativa para una exportación posterior queda fijada en `271801`.

Los controles analíticos son el piso conocido y el piso más el exceso medio del ajuste. La tabla de E2 se conserva completa. E2 tiene seis semillas por red y 655 ejemplos de ajuste; esta campaña tiene diez semillas de E3 y 532 ejemplos. No se declara igualdad de presupuesto, significancia, novedad ni superioridad confirmatoria.

## Presupuesto y almacenamiento

El cierre anterior de momentos permanece en `650.3773435349993` segundos contabilizados. Esta campaña asigna como máximo 500 segundos adicionales bajo el límite global heredado de 1200; no reinicia los contadores ni reabre los pilotos anteriores. El contador incluye inicialización, igualación del sesgo, entrenamiento, validación, evaluación y guardado local. Instalación, descargas y publicación añaden tiempo de reloj. Se registra cualquier operación iniciada que sobrepase el límite, sin ocultar su coste.

Cada bloque científico exige una reserva activa confirmada en GitHub. No comienza otro bloque hasta publicar el anterior. Una reconexión reutiliza el mismo descriptor, corrida y presupuesto. La recuperación conserva avances y costes; un registro divergente o parcialmente inconsistente se conserva y rechaza. Una campaña agotada o fallida queda cerrada. Si faltan semillas, se informa una campaña incompleta y no se presenta una media de diez.

El destino es la rama `e3-estabilidad-aditiva-v1`, prefijo `runs/estabilidad_aditiva_v1`. El secreto `GITHUB_TOKEN` permanece en el coordinador; el trabajador recibe un entorno sin credenciales. Se usa el runtime aislado de v2 y se reutiliza la instalación de momentos cuando su identidad coincide. Los pesos y resultados históricos permanecen conservados en sus ramas.

## Verificación

Las pruebas usan precios, etiquetas y entrenamiento artificiales en CPU, junto con GitHub simulado. Cubren reanudación con pesos e historia idénticos, interrupción a mitad de época, conservación del presupuesto, acceso restringido a miembros NPZ, confirmación previa de las diez semillas, recuperación de publicaciones y conflictos de escritura. Una prueba de integración conecta el coordinador con el trabajador real sobre datos artificiales, completa las diez semillas y comprueba que repetir la celda no ejecuta otra campaña.

Desde la raíz del repositorio, con PyTorch 2.8.0 y NumPy 2.3.5:

```bash
python -m unittest discover -s notebooks/e3_estabilidad_aditiva_v1 -p 'test_*.py' -v
```

La preparación de este software no ejecutó nuevos entrenamientos ni evaluaciones sobre los datos reales. El código se elaboró con asistencia de IA, que debe declararse en el informe. El estudiante debe comprender y poder defender la implementación.

## Recuperación del transporte Git

Una ejecución avanzó hasta el segundo bloque y recibió `HTTP 422; Object does not exist` al actualizar la referencia por la API de GitHub. El último estado remoto confirmado contiene cinco épocas por semilla y 51.85614561499959 segundos adicionales contabilizados. El estado HTTP no permite identificar por sí solo la causa exacta del rechazo.

`repair_git_transport.py` conserva los archivos originales del coordinador, trabajador, referencia y plan, comprobando sus hashes. Sustituye en memoria únicamente el transporte de lectura y publicación por Git nativo, reutilizando el transporte con `askpass` de los pilotos anteriores. La rama y el prefijo siguen fijados; la publicación no utiliza fuerza y comprueba la referencia esperada antes y después de guardar.

El coordinador original reconcilia primero el avance local con el remoto. Así, un bloque calculado cuyo guardado falló se publica antes de continuar el entrenamiento. Se mantienen corrida, pesos, optimizador, cursores, semillas, sesiones y consumo. La reparación se registra como metadato de transporte y no modifica los parámetros científicos ni el presupuesto. Las credenciales permanecen en el entorno privado de Git y se eliminan al cerrar el transporte.
