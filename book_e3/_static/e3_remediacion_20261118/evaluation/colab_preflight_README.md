# Preflight sintético de Colab

El cuaderno `notebooks/e3_remediacion_preflight_colab.ipynb` prepara la revisión del entorno y de las 17 recetas reales. Abrirlo en Colab, seleccionar **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU → T4** (o la disponible) y luego **Ejecutar todas**. No pegar comandos en la última celda vacía del cuaderno histórico de innovación.

El cuaderno consulta `main`, guarda el HEAD realmente obtenido y verifica siete fuentes mediante `colab_source_lock.json`. Para los 13 rivales E2 descarga exclusivamente el ZIP D02 público de 7.84 MB, fijado por commit, tamaño y SHA-256. Su inventario verifica 12 constructores, cuatro metadatos, 13 recetas y 76 archivos de autores con sus licencias. No descarga datos de mercado, pesos entrenados ni exige credenciales. El despacho real `_builder` conserva la variante publicada TiDE v2.

Cada neuronal ejecuta un forward y backward con ventanas artificiales normalizadas `[4,7,5]`, comprueba parámetros, salida finita no negativa `[4]`, gradientes finitos y guardado/recarga de `state_dict` con salida idéntica en el mismo dispositivo. No crea optimizadores ni actualiza pesos. Los 13 rivales devuelven sigma7 dividida por la escala objetivo; el candidato devuelve unidades originales. No se comparan esos valores como rendimiento. El candidato es exactamente `InnovationModel('additive_positive_head')`, 1210 parámetros, con A no negativa sintética y escalas sintéticas explícitas; no se transfieren propiedades de otra arquitectura.

Persistence y SeasonalNaive ejercitan sus rezagos 1 y 7 sobre una historia fechada artificial. ARIMA usa el verdadero `FrozenARIMA`, orden `[2,0,1]`, un ajuste sintético, siete pasos de filtrado, el rechazo de observar antes de pronosticar y una recarga de su estado. Ese ajuste se registra por separado: **un ajuste ARIMA artificial y cero ajustes con datos de mercado**.

## Entorno y bloqueos

Se conservan los perfiles revisados en los locks existentes, sin instalarlos automáticamente: Python 3.12; Torch 2.8.0+cpu para la comprobación local o Torch 2.8.0+cu126 y CUDA 12.6 para Colab; NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, statsmodels 0.14.5 y einops 0.8.1. Los imports autores de iTransformer también requieren `reformer-pytorch` y sus dependencias del perfil E2 existente. Un paquete ausente conserva el fallo concreto de su receta, nunca una sustitución silenciosa.

El cuaderno prepara por defecto un **venv CUDA separado** mediante `colab_runtime.prepare_runtime`: reutiliza las funciones de preparación del helper E2 publicado y el lock E2 completo, fijados a commit y SHA-256. Obtiene su propio Python 3.12.12 aunque el kernel sea Python 3.13, instala las dependencias en `/content/e3-preflight-gpu-runtime-v1` y devuelve el intérprete y entorno que ejecutan los 17 contratos. El Python y Torch base de Colab no cambian; no se reinicia el kernel. La primera descarga de wheels puede sumar varios GB y tardar varios minutos. No se descargan wheels CUDA si no hay GPU visible.

Un venv completo se verifica y reutiliza. Un entorno parcial o cuyo lock cambió se conserva y se rechaza: debe elegirse otra carpeta local del runtime, por ejemplo `v2`. Si falla la preparación se guarda ese bloqueo en el recibo y siguen bloqueados los modelos y la campaña. La prueba local es CPU: **la instalación y ejecución CUDA completa de Colab todavía no se han ejecutado aquí**.

## Guard de campaña

`campaign_gate` falla de forma cerrada si no existe un manifiesto final. Para una futura revisión exige identidad del candidato, los 17 sistemas, tres datasets realmente admitidos y contratos sigma7/h1 con reloj explícito; manifiestos de procedencia, licencias, auditoría técnica y exposición atados a archivos y hashes; tratamiento común, presupuesto con cargos de fallos y potencia individual/conjunta de toda la batería; y un tag Git anotado cuyos bytes coincidan con el manifiesto. Verifica que tag y congelamiento precedan al primer ajuste/evaluación, que el congelamiento ya exista y que el primer trabajo siga planificado en el futuro.

La comprobación tipada de archivos y fechas ayuda a revisar el protocolo; no sustituye la revisión humana de las decisiones científicas, las licencias, la novedad ni los cálculos. Este cuaderno no incluye un despachador de entrenamientos reales. El estado actual sigue siendo **0 datasets admitidos y campaña bloqueada**, incluso después de pasar los contratos sintéticos. No acredita R1 ni R3.

## Comandos

Pruebas negativas del guard, sin Torch ni red:

```bash
python -B -m unittest discover \
  -s experiments/e3_remediacion_20261118/evaluation \
  -p test_colab_preflight.py -v
```

Preflight real sobre el perfil CPU ya instalado y un destino nuevo:

```bash
python experiments/e3_remediacion_20261118/evaluation/colab_preflight.py \
  --device cpu --output /tmp/e3-preflight-CPU
```

La ejecución de Colab usa `--device cuda:0`; su propio recibo debe demostrar GPU disponible y los contratos pasados. Se puede añadir `--package-root /ruta/D02` para reutilizar un paquete local que supere el mismo inventario. Cada modelo tiene un proceso aislado, un límite de 180 segundos y un log conservado. Los destinos ya existentes no se sobrescriben.

Preparación aislada desde una celda de Colab que ya haya cargado el módulo publicado:

```python
gpu_python, gpu_env, setup_receipt = colab_runtime.prepare_runtime(
    "/content/e3-preflight-gpu-runtime-v1"
)
subprocess.run(
    [str(gpu_python), str(preflight_script), "--device", "cuda:0", "--output", str(new_output)],
    env=gpu_env,
    check=True,
)
```

`test_colab_runtime.py` verifica en CPU el lock completo, la eliminación de configuración del kernel y credenciales del entorno hijo, el rechazo previo a red si falta GPU, la selección del intérprete aislado y la conservación de un runtime con recibo incompatible. Sus fixtures de rutas no simulan una instalación o ejecución GPU real.

La verificación local entregada contiene 17 perfiles pasados, 14 backwards neuronales, un ajuste ARIMA sintético, guard bloqueado y cero métricas o predicciones de mercado. Las cuatro pruebas negativas cubren cambios de fuentes, manifiestos inválidos, tags/orden temporal y extracción insegura. El cuaderno se entrega sin salidas que simulen una ejecución GPU.
