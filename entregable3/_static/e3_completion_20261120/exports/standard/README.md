# Exportaciones congeladas E2: cuatro modelos estándar

Se recuperaron los checkpoints originales de **FCN-1D, ResNet-1D,
Inception-1D y LSTM-autoencoder con predictor**, semilla fija **271801**.
La regla fue usar la menor semilla final registrada, sin selección por resultados
de prueba. Los SHA256 y tamaños de los cuatro checkpoints coincidieron con
el archivo original; se cargaron sus estados estrictamente.

**No se entrenó, reajustó ni recalibró ningún modelo.** Este paquete crea un
perfil de exportación CPU fuera de línea. La ejecución no modifica el catálogo
del dashboard, el master de los 796 trabajos ni los 30 ajustes del piloto V3.
Tampoco extiende la evidencia de esos modelos a V3 ni acredita R3.

## Verificación ejecutada

Para cada modelo se verificaron el preprocesamiento original, la inferencia en
memoria CPU y ONNX sobre **608 ventanas reales de evaluación** y **53 de
calibración**, comparándolos además con las predicciones guardadas. La tolerancia
absoluta se fijó antes de evaluar en **1e-6 de sigma7 poblacional original**.
Se comprobó el grafo con `onnx.checker` y la inferencia utilizó solamente
`CPUExecutionProvider`, un hilo intra-operador y uno inter-operador.

| Modelo | Parámetros | Bytes ONNX | Máximo error ONNX vs. congelado, evaluación | Máximo error ONNX vs. congelado, calibración |
|---|---:|---:|---:|---:|
| FCN-1D | 268929 | 1092005 | 1.7803e-8 | 1.7797e-8 |
| ResNet-1D | 522955 | 2156012 | 1.7015e-8 | 1.6420e-8 |
| Inception-1D | 490561 | 2070967 | 2.7452e-8 | 2.9865e-8 |
| LSTM-autoencoder con predictor | 8739 | 41397 | 3.0993e-7 | 6.0808e-8 |

Todos pasaron. El mayor error de cualquiera de las comparaciones ejecutadas,
incluida memoria CPU contra las predicciones congeladas, fue **3.1493e-7** en
SAE. Las diferencias CPU/GPU históricas no se corrigieron cambiando pesos,
normalizadores ni tolerancia.

Los archivos originales `scaler.json`, `calibrator.json` y
`selection_receipt.json` se conservaron **byte por byte**. Se verificaron el
hash interno del calibrador, su identidad de checkpoint, el q original y los
hashes de las 53 predicciones y objetivos de calibración. Ninguno de los cuatro
intervalos cambió la clasificación de cobertura de las 608 fechas por efecto
del redondeo ONNX. El calibrador sigue siendo histórico de BTCUSDT: no garantiza
cobertura futura, independencia temporal ni cobertura por subgrupos.

Los avisos del trazador se conservan en `roundtrip.json`. Las dimensiones,
validaciones de finitud y modos de evaluación son fijos en el grafo; las
operaciones predictivas se verificaron sobre 661 ventanas diferentes.
`predict_export.py` valida forma, finitud, positividad de precios, volumen y
consistencia OHLC antes de llamar al grafo. Las comprobaciones de fechas,
procedencia y disponibilidad de cada ventana corresponden al cliente que lo
invoque. El contrato no admite longitudes ni tamaños de lote distintos.

## Contrato y archivos

Entrada ONNX `windows`: `float32 [1,7,5]`, ya transformada. Para precios,
`log(precio_t)-log(último cierre observado)`; para volumen, `log1p(volumen_t)`;
después se aplican la media y desviación de entrenamiento originales. Salida
`normalized_prediction`: sigma7 dividida por `output_scale`; la inversión se
aplica **una vez**. Estos cuatro modelos no reciben el piso observable A.

Cada carpeta de modelo contiene el ONNX, metadatos, prueba de ida y vuelta,
procedencia y los tres recibos originales. `export_summary.csv` resume los
resultados. `runtime_registry.json` permite integrar y medir el perfil sin
inferir nombres de entradas, escalas ni calibradores. `input_manifest.json`
fija hashes, URLs de checkpoints y predicciones, protocolo y fuentes.
No se redistribuyen datos OHLCV brutos ni código completo de terceros.

Las latencias locales de `roundtrip.json` son **provisionales**: miden por separado
forward y preprocesamiento + forward + inversión de escala, con lote uno,
10 calentamientos y 100 llamadas por modalidad. Su RSS es el pico del proceso
completo que cargó Torch y ORT, **no** memoria incremental del modelo.
La comparación final de costes necesita el benchmark común secuencial con
procesos ORT equivalentes; este paquete no presenta esos tiempos provisionales
como una comparación justa entre sistemas.

## Reproducción

Entorno utilizado: Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5,
ONNX 1.19.1 y ONNX Runtime 1.23.2. No se usó CUDA.

Se requiere un checkout del proyecto que coincida con los hashes de fuentes
del manifiesto y el NPZ original autorizado con SHA256
`186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
Ese NPZ incluye las ventanas brutas y se mantiene fuera del paquete publicable.
Los checkpoints y las predicciones guardadas se recuperan con:

```bash
python download_inputs.py --manifest input_manifest.json --destination /tmp/e3-export-inputs
```

El script comprueba los hashes y prepara los scalers, calibradores y recibos
incluidos, sin solicitar credenciales. Con `$E3_REPO` apuntando al checkout
compatible y `$E3_COHORT` al NPZ autorizado, el comando para FCN es:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python export_standard.py \
  --model FCN-1D --repo "$E3_REPO" --checkpoints /tmp/e3-export-inputs \
  --npz "$E3_COHORT" --predictions /tmp/e3-export-inputs/combined_predictors_133.json \
  --protocol "$E3_REPO/book_e2/_static/search_protocol_20261109/protocol.json" \
  --selections /tmp/e3-export-inputs/selection_receipts \
  --freeze /tmp/e3-export-inputs/freeze --output /tmp/e3-export-results
```

Los otros valores de `--model` son `ResNet-1D`, `Inception-1D` y
`LSTM-autoencoder con predictor`. El script interrumpe la ejecución si cambia
un hash, la receta, la semilla, el calibrador o la tolerancia de paridad.
La construcción de los cuerpos inicializa una instancia vacía y luego restaura
estrictamente el estado guardado; no ejecuta fases de entrenamiento.

La prueba ligera de empaquetado y guardas se ejecuta con:

```bash
python verify_exports.py --payload . --npz "$E3_COHORT" --output guard_validation.json
```

`guard_validation.json` distingue esta comprobación ligera de la paridad
completa ya ejecutada. Estos resultados constituyen reproducción asistida por
IA de la **exportación e inferencia**; no una reproducción humana independiente
ni una reproducción integral de entrenamiento.

**Asistencia de IA:** OpenAI Codex ayudó a recuperar, exportar, comprobar y
documentar los artefactos. Los resultados adversos previos y las limitaciones
del proyecto se mantienen.
