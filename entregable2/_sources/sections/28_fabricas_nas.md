# 28. Fábricas NAS implementadas y verificadas sintéticamente

**Integración posterior:** [§29 conecta entrenamiento por etapas, selección, reajuste y replay de18 sistemas](29_entrenamiento_temporal.md), con datos artificiales y perfilado CPU acotado. Las limitaciones que siguen corresponden al corte de fábricas de este capítulo.

**Corte del 09/11/2026.** Se implementan las seis variantes nuevas de [§27](27_nas_hpo_presupuesto.md) y su enlace con las configuraciones registradas. Los cuerpos y recibos históricos de §§18–26 permanecen intactos.

**Resultado:** 6.000 casos numéricos de predictores, las mil combinaciones del espacio para cada una de las dos fases auxiliares SAE, 108 enlaces HPO sobre referencia y una comprobación adicional del extremo grande de TimesNet. **Cero pasos de optimizador, cero lectura de mercado y ninguna búsqueda HPO/NAS ejecutada.** Son pruebas de mecanismos, no entrenamientos ni resultados de pronóstico.

[Resumen y hashes](../_static/nas_factories_20261109/summary.json) · [108 casos de referencia y extremo TimesNet](../_static/nas_factories_20261109/reference_and_maximum.json).

## 28.1 Qué se implementó

- `e2_nas_common.py`: contrato no vacío [B,7,5], configuración inmutable W³/bits, momentos poblacionales y dropout por bloque.
- `e2_nas_factory.py`: construcción con IDs de familia, arquitectura y trial. `build_from_plan` rechaza configuraciones no incluidas entre las18 registradas, IDs mezclados, anchos inválidos o factores ajenos a la receta.
- `e2_nas_autoencoder.py`: SAE con dos preentrenamientos y tercer grupo denso; TiDE con tres grupos residuales, sentinel futuro cero y cabeza escalar.
- `e2_nas_spectral.py`: tres TimesBlocks con mezcla FFT/períodos 2D; tres bloques PDM con mezcla estacional/tendencia y FMM por escala.
- `e2_nas_attention.py`: PatchTST de tres anchos con cabezas w/16 y FF8w, RevIN previo al padding y atención residual entre bloques desactivada; híbrido CNN–LSTM–atención con estado reiniciado.

Todos los predictores retornan σ₇/a positiva. El ajuste de a y de los escaladores queda fuera de la fábrica y debe usar sólo entrenamiento; la inferencia restaurará unidades originales. El readout conserva diez momentos para TimesNet, TiDE, TimeMixer y PatchTST, y ninguno para SAE o híbrido. La asimetría de momentos declarada en §22 no se elimina.

Las proyecciones entre grupos son aprendidas incluso si los anchos coinciden. Los nuevos LSTM usan la inicialización uniforme nativa de PyTorch; SAE/TiDE usan Dense Xavier con bias cero, TimesNet usa inicialización Kaiming de sus convoluciones de períodos y PatchTST/híbrido conservan las inicializaciones declaradas en su código. Estas reglas son fijas dentro de cada espacio; no se afirma equivalencia numérica con las publicaciones ni con las recetas anteriores.

## 28.2 Cobertura numérica exhaustiva del espacio W³/bits

Cada familia recorrió las1.000 combinaciones, sin omisiones ni poda: batch artificial de una ventana, modo entrenamiento para activar dropout/normalización, forward y backward. Se verificaron forma [B], salida positiva finita, gradiente de entrada y conexión/finitez del gradiente de cada parámetro entrenable.

| Familia nueva | Casos numéricos | Parámetros del predictor en ese recorrido | Trial estructural utilizado |
|---|---:|---:|---|
| LSTM-autoencoder con predictor | 1.000/1.000 | 3.937–861.697 | `SAE-D7-NAS-v1:hpo-000` |
| TiDE | 1.000/1.000 | 3.945–171.225 | `TIDE-D7-NAS-v1:hpo-000` |
| PatchTST | 1.000/1.000 | 17.205–4.084.245 | `PT-D7-NAS-v1:hpo-000` |
| CNN-1D + LSTM + atención temporal | 1.000/1.000 | 2.897–599.297 | `HIB-D7-NAS-v1:hpo-000` |
| TimesNet | 1.000/1.000 | 32.747–8.011.307 | `TN-D7-NAS-v1:hpo-000` |
| TimeMixer | 1.000/1.000 | 4.647–926.487 | `TM-D7-NAS-v1:hpo-000` |

**Alcance del trial:** se aplican sus ejes estructurales, no learning rate, weight decay, scheduler ni batch de entrenamiento. No hay optimizador y el batch de comprobación es1. No se presentan como seis ganadores HPO ni como18×1.000 combinaciones probadas.

En particular, TimesNet usa en este recorrido el trial registrado con **top-k3 y dos kernels**. Dos kernels es una opción del espacio HPO ya publicado, no una reducción oculta de anchos o de profundidad. Los seis kernels se verificaron adicionalmente como se explica abajo.

Los gradientes conectados y finitos pueden ser cero en ciertos componentes o ejemplos. La escala de longitud1 de TimeMixer pierde contraste temporal por su normalización; eso permanece documentado como limitación, no como observación adicional o señal útil demostrada.

## 28.3 Extensión a HPO, geometría máxima y fases SAE

**HPO:** se construyó la referencia (64,64,64), patrón (0,1,0), con cada una de las18 configuraciones registradas por familia: **108 forward/backward numéricos**. Esto comprueba el enlace de IDs y las opciones estructurales, incluidos top-k, número de kernels, media móvil y kernel causal. El entrenamiento AdamW y sus reglas aún requieren integración separada.

**TimesNet con seis kernels:** las1.000 combinaciones se construyeron además en dispositivo `meta`, comprobando geometría y conexiones de autograd con el cuerpo de seis kernels. Meta no calcula valores ni demuestra finitez. La arquitectura (256,256,256), patrón(1,1,1), también pasó una prueba numérica con un trial registrado de seis kernels: **225.075.755 parámetros**. El recibo registra ~28.29s para construcción+forward+backward y un pico de proceso de 2.23GiB en esa sesión CPU. No incluye AdamW, épocas completas ni coste de búsqueda, y no sirve como estimación de horas de mercado.

**SAE auxiliares:** AE1 y AE2 recorrieron cada una las mil combinaciones y pasaron forward/backward numéricos, sin actualizar pesos. Las fábricas auxiliares no dependen de todos los ejes del predictor: cambiar sólo w₃/b₃ puede repetir el mismo subgrafo auxiliar. Se reportan2.000 comprobaciones, no2.000 diseños auxiliares distintos. La entrada de AE2 de ese recorrido es artificial; no se afirma que provenga de un AE1 entrenado.

Las pruebas específicas de etapas verifican escala s ajustada sólo al train copiado, reconstrucción de X/s sin escalar la entrada del encoder, AE1 en evaluación/sin gradientes al preparar AE2 y transferencia de ambos encoders sin decoders ni alias. El predictor transferido vuelve a permitir ajuste supervisado de encoders.

## 28.4 Evidencia completa descargable

Los diarios conservan cada caso y su alcance. Los hashes del resumen vinculan fuente, protocolo, datos artificiales y archivos comprimidos; no certifican resultados predictivos.

- {download}`Predictores SAE/TiDE: 2.000 casos <../_static/nas_factories_20261109/predictors_sae_tide.jsonl.gz>`.
- {download}`PatchTST e híbrido: 2.000 casos <../_static/nas_factories_20261109/predictors_attention.jsonl.gz>`.
- {download}`TimesNet y TimeMixer: 2.000 casos <../_static/nas_factories_20261109/predictors_spectral.jsonl.gz>`.
- {download}`SAE auxiliares: 2.000 comprobaciones <../_static/nas_factories_20261109/sae_auxiliary.jsonl.gz>`.
- {download}`TimesNet seis kernels: 1.000 casos meta <../_static/nas_factories_20261109/timesnet_six_kernel_meta.jsonl.gz>`.

La batería nueva contiene **35 pruebas unitarias e integridad de evidencia, sin omisiones**. La ejecución de CI incluye además seis pruebas previas de escenarios de coste: **41 en total**. Cubre rechazo de entradas y trials alterados, aislamiento de batch en evaluación, normalización congelada, causalidad sobre tensores transformados, atención normalizada, transferencia SAE y equivalencia de salidas/gradientes entre convoluciones Inception literales y su fusión lineal exacta. Esa fusión conserva todos los kernels y sus parámetros.

La validación CPU se añade a CI en un job propio con dependencias fijadas. La publicación depende de esa batería y de la construcción/verificación de enlaces de los tres libros. La evidencia exhaustiva se conserva con hashes; CI no repite toda la batería de miles de casos en cada publicación.

## 28.5 Reproducción sin mercado

Desde un checkout del repositorio:

```bash
python -m venv work/nas-venv
work/nas-venv/bin/python -m pip install -r requirements-nas-cpu.lock.txt
work/nas-venv/bin/python -m pip check
work/nas-venv/bin/python -m unittest discover -s tests -p 'test_e2_nas_*.py' -v
work/nas-venv/bin/python scripts/check_e2_nas_factory.py \
  --protocol book_e2/_static/search_protocol_20261109/protocol.json \
  --mode numeric --output work/nas-numeric-nuevo --threads 1
work/nas-venv/bin/python scripts/check_e2_nas_factory.py \
  --protocol book_e2/_static/search_protocol_20261109/protocol.json \
  --mode meta --model TimesNet --output work/nas-meta-nuevo --threads 1
work/nas-venv/bin/python scripts/check_e2_nas_auxiliary.py \
  --protocol book_e2/_static/search_protocol_20261109/protocol.json \
  --output work/nas-aux-nuevo --threads 1
```

Los directorios de recibos deben ser nuevos; no se sobrescriben. No se aceptan archivos de mercado como entrada a esos checkers. Se generan tensores artificiales dentro del programa; no se instalan ni ejecutan los scripts históricos de entrenamiento.

## 28.6 Qué falta antes de entrenar

La fábrica ya construye las seis familias y acepta las configuraciones estructurales registradas. **No es aún el ejecutor completo de selección y entrenamiento de18 sistemas.** Faltan:

1. Integrar AdamW, scheduler, batches, parada, escalado y las tres etapas SAE en el ciclo de selección interna/reajuste con estado nuevo por origen y semilla.
2. Auditar la integración temporal completa, almacenamiento durable de checkpoints y pronósticos y tratamiento de fallos.
3. Cerrar el protocolo del perfilado/pilotos y medir tiempo, memoria y almacenamiento con esas etapas completas en Codex. El caso TimesNet grande exige presupuestarse explícitamente.
4. Completar k/R, inferencia, calibración, controles y preregistro antes de la campaña de mercado.

Las 608 fechas conocidas siguen sin certificación de soporte común y conservan carácter exploratorio. Esta implementación no abre esa reserva, no modifica los recibos anteriores ni declara ganador.

