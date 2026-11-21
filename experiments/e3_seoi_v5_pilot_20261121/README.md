# V5 SE-OI: piloto de escala causal para volatilidad solapada

**Preparado e implementado el 21 de noviembre de 2026.** Candidata fija:
`seoi_mse`. Se entrega código y un piloto; no hay entrenamientos o resultados
de mercado nuevos en este paquete. Asistencia de IA declarada.

La adaptación divide las coordenadas logarítmicas de OHLC por una escala
calculada exclusivamente con seis retornos ya conocidos. Aprende una ley
atómica de innovación y restituye la escala **antes** de aplicar
`sqrt(A+(6/49)*U²)`. La pérdida de la candidata es MSE en las unidades de la
métrica primaria. No es el antiguo control MSE promovido por su posición en
una tabla: cambian la representación y la reconstrucción del pronóstico.

La normalización reversible, las leyes finitas, CNN, LSTM, atención y la
homogeneidad tienen antecedentes. La contribución defendida y sus límites
están en [la comparación bibliográfica](docs/LITERATURE_AND_CONTRIBUTION.md)
y [la formulación y demostración](docs/THEORY.md). La propiedad distintiva
es equivariancia local de amplitud con referencias de ajuste fijas cuando
el piso numérico no actúa; no prueba mejora predictiva ni calibración.

## Cómo ejecutar en Colab: GitHub, sin montar Drive

Abre [el notebook actualizado](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/e3-seoi-v5-pilot-20261121/experiments/e3_seoi_v5_pilot_20261121/pilot_colab.ipynb).
Usa un runtime CPU. Esta revisión elimina `drive.mount` y el error de
propagación de credenciales de Google Drive.

1. Ejecuta la preparación: descarga fuentes verificadas, instala el entorno
   CPU aislado y corre las pruebas sintéticas. No entrena.
2. Introduce de forma privada el token GitHub y la clave de respaldo. El token
   necesita `Contents: Read and write` sobre `DeepLearning_202630`. Puedes
   guardarlos en secretos Colab: `GITHUB_TOKEN` y `E3_V5_BACKUP_KEY`.
   Conserva la clave de al menos32 caracteres: es necesaria para descifrar los
   checkpoints al recuperar el caso. No la publiques ni la pegues en el chat.
3. La preparación registra o restaura el caso en una rama de ejecución nueva,
   `e3-seoi-v5-run-01`, y verifica el primer guardado antes de permitir ajustes.
4. Ejecuta el apartado 3 para entrenar. Cada trabajo terminado se guarda en
   GitHub antes del siguiente; también hay respaldos de época si transcurre
   un minuto desde el último. Un fallo de persistencia detiene nuevos ajustes.
5. Ejecuta el apartado 4 para evaluar y guardar las tablas; el último apartado
   muestra el estado y el enlace de resultados en GitHub.

Para reanudar, usa la misma rama, versión y clave. La recuperación verifica
cifrado, hashes, fuente y plan antes de cargar checkpoints; descarga de nuevo
el NPZ público por su hash. No sobrescribe otra ejecución o progreso remoto
más reciente. No abras dos runtimes sobre la misma rama.

Los resultados y recibos son públicos. Pesos, Adam y RNG se guardan como
un respaldo AES-256-GCM cifrado; los datos brutos se excluyen. El código
transmite las credenciales al proceso por stdin y las excluye de argumentos,
variables de entorno del proceso y archivos del caso. Más detalle en
[GITHUB_STORAGE.md](docs/GITHUB_STORAGE.md).

## Experimento fijado

| Receta | Escala causal en entrada | Escala causal en salida | Papel |
|---|---:|---:|---|
| `seoi_mse` | Sí | Sí | Candidata fijada, MSE |
| `input_only_mse` | Sí | No | Ablación de reconstrucción |
| `output_only_mse` | No | Sí | Ablación de normalización |
| `global_mse` | No | No | Control atómico estático MSE |
| `global_crps` | No | No | Control estático MSE+0,1CRPS_U |
| `additive_mse` | No | No | Referencia aditiva |

Cinco semillas por receta:30 ajustes, sin HPO. Además se calculan una sola
vez los controles analíticos `floor_only` y `floor_plus_mean_excess`.
Se conservan532 ejemplos de ajuste,54 de selección y55 de evaluación de
desarrollo **ya expuesta**. Hay dos purgas de siete fechas,14 en total.
Sólo se deserializan tres miembros `training_*` del NPZ fijado; nunca
los arrays de calibración o prueba final.

La continuación requiere, simultáneamente, terminar los 30 ajustes, reducir
al menos2 % la **media de RMSE por semilla** frente al mejor de todos los
controles, ganar en≥4/5 semillas al control neuronal de mejor RMSE media y
tener MAE media≤1,05veces la MAE de ese mismo control neuronal. El2 % es una
regla de este piloto, no un requisito universal de R3. Si una ablación gana,
no reemplaza automáticamente a la candidata.

Cada ajuste usa AdamW, LR 0,003, batch 32, hasta40 épocas y paciencia 8.
El presupuesto nominal es120 s activos por ajuste,3600 s para los 30,
además de instalación, preparación, guardados en GitHub y evaluación.
Los controles atómicos tienen1262 parámetros y la aditiva1210; no se afirma
igual capacidad. No se ajustan por separado los hiperparámetros de V5.

## Reanudación y coste

Por época se guarda un snapshot con pesos actuales/mejores, Adam, RNG,
orden del muestreador, contadores y hashes. Si se pierde una época parcial,
se restaura la última época completa y se carga su reserva de5 s al
presupuesto. Los costes observados adicionales se conservan cuando existen.
El límite es cooperativo: una operación en curso puede excederlo; el código
no promete un límite duro de tiempo. Se conservan los intentos fallidos.

La referencia histórica V4 consumió54,93 s de fase para20 ajustes en esta
máquina. Eso **no es una medición de V5 o de Colab**. El máximo nominal del
piloto es una hora activa CPU; el tiempo real y las escrituras a GitHub se
medirán al ejecutarlo. La primera instalación descarga aproximadamente
200–350MB. No se cobran recursos GPU por decisión del programa.

## Resultados y evidencia

Después de una ejecución completa se generan `metrics.csv`,
`recipe_summary.csv`, `predictions.csv`, `resources.csv`, curvas por trabajo,
inventario de checkpoints, fallos y `pilot_result.json` con la decisión.
Las1760 filas previstas son1650 predicciones neuronales y110 analíticas.
La exportación usa una lista permitida y excluye NPZ, pesos, Adam y RNG.

Las comprobaciones realizadas al preparar este paquete se guardan en `qa/`.
Se verificaron36pruebas y una instalación CPU limpia descargando las fuentes
públicas fijadas, seguida de preparación real y repetición idempotente con el
mismo caso. No se ejecutó entrenamiento. Esa comprobación histórica corresponde a la revisión original. La revisión
GitHub añade pruebas de persistencia con transporte simulado; no se inventa
una autenticación o guardado con credenciales del usuario.
No se mezclan con entrenamientos, evaluaciones o una reproducción humana.
Los796trabajos de la aditiva,30 ajustes GPU de V3 y20 CPU de V4 permanecen
intactos; los resultados adversos anteriores y sus decisiones no cambian.

La [continuación confirmatoria](docs/CONFIRMATION_HANDOFF.md) explica qué
habilitaría un piloto favorable y qué evidencia adicional exige R3. No hay
un dictamen docente o una victoria experimental generados por escribir código.

## Procedencia y derechos

`seoi_v5/frozen_original.py` es una copia byte a byte del archivo propio
`src/original_model.py` de la base9fb198a, con su SHA256 en el manifiesto.
Se reutilizan su encoder e inicialización; no se atribuyen como invenciones.
La cabeza atómica sigue la formulación V3 propia, citada en el diseño.
No se copió código de RevIN, DAIN o Non-stationary Transformers.
Se conservan los derechos y condiciones aplicables del proyecto y los datos;
este suplemento no inventa una licencia nueva para archivos ajenos.
PyTorch y NumPy se instalan como dependencias y mantienen sus licencias.
