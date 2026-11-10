# 35. Selección CUDA con presupuesto y continuación en GitHub

**Preparación del 09/11/2026:** se incorporó un ejecutor HPO → NAS-1 → NAS-2 para los catorce sistemas neuronales, con resultados recuperables entre sesiones de Colab. Conserva los dominios de [§27](27_nas_hpo_presupuesto.md), sus semillas, la separación temporal y las reglas de selección. El usuario fijó **8 horas por sesión y 24 horas acumuladas** como límites iniciales y eligió guardar las campañas en [su repositorio de GitHub](https://github.com/jassan-bit/DeepLearning_202630).

Ese corte entregó software y validación sintética; todavía no acreditaba una búsqueda científica GPU ejecutada. El avance posterior sobre desarrollo se documenta en [§36](36_avance_campana_cuda.md). Los [29 perfiles de recursos y 872 pasos conocidos de §§33–34](34_perfil_gpu_ampliado.md) conservan su evidencia histórica separada. La comparación final de dieciocho sistemas, los controles pendientes y el R final siguen abiertos.

:::{warning}
**Campaña en curso al corte del 10/11/2026.** Para recuperar la vista del entrenamiento activo, vuelve a conectar la sesión existente de Colab. No pulses **Ejecutar todas**, no inicies otra máquina ni autorices otro reintento por una desconexión del navegador. Las instrucciones siguientes documentan la puesta en marcha inicial; no deben repetirse sobre esa ejecución. Consulta el [estado parcial y sus límites en §36](36_avance_campana_cuda.md). Los límites iniciales 8/24 se conservan como antecedente; el diario posterior registra 8 horas por sesión y una ampliación global a 36 horas, descrita con su evento en §36.
:::

## Puesta en marcha inicial del ejecutor

El [notebook de búsqueda](https://github.com/jassan-bit/DeepLearning_202630/blob/72e3525adce1956eaba57b30f4a780da5aea10c1/notebooks/entregable2_busqueda_gpu.ipynb) puede [abrirse en Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/72e3525adce1956eaba57b30f4a780da5aea10c1/notebooks/entregable2_busqueda_gpu.ipynb). Utiliza el código congelado en `564ba4762f84a800b02fa4c734f5752a169f5223` y el registro `experiments/e2_cuda_search_20261109/registration.json`, con SHA256 `78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9`.

1. **Mostrar el plan.** Ejecutar primero con `RUN_SEARCH=False`. Se recuperan código y protocolos y se enumeran modelos, orígenes, oportunidades y límites. Este modo no necesita GPU ni credenciales, no instala CUDA, no abre los CSV ni crea una campaña remota.
2. **Preparar la credencial de Colab.** Crear en GitHub un token de acceso personal de alcance fino, limitado al repositorio `jassan-bit/DeepLearning_202630`, con permiso **Contents: Read and write**. En Colab, abrir **Secretos**, añadirlo con el nombre `GITHUB_TOKEN` y habilitar el acceso de este notebook. El valor no se pega en las celdas, en el chat ni en archivos del repositorio. La conexión de GitHub de esta conversación no se transfiere a la máquina de Colab.
3. **Activar la búsqueda.** Seleccionar la T4 y conservar la configuración siguiente en la **primera celda de código**. Cambiar `RUN_SEARCH=True`, **ejecutar esa celda** para aplicar el valor y después pulsar **Ejecutar todas**. Antes de cargar valores de desarrollo se comprueban la fuente, el entorno y los recorridos sintéticos CUDA. La sincronización inicial confirma la escritura del plan remoto antes de los ajustes.

```python
RUN_SEARCH = True
MODELS = None
ORIGINS = None
SESSION_HOURS = 8
CAMPAIGN_HOURS = 24
EXTEND_CAMPAIGN_HOURS = None
STORAGE_GIB = 10
CAMPAIGN_NAME = "e2-cuda-selection-v1"
GITHUB_REPOSITORY = "jassan-bit/DeepLearning_202630"
```

`MODELS=None` conserva los catorce sistemas neuronales y `ORIGINS=None` los once cortes del plan. Un subconjunto se registra como cobertura parcial. El intérprete del piloto se reutiliza: CPython 3.12.12 aislado, PyTorch 2.8.0 + CUDA 12.6 y el lock dedicado. La fuente de búsqueda tiene su propio checkout para no reemplazar el del piloto de recursos.

## Dominio científico y presupuesto operativo

Los cortes son los diez orígenes mensuales de enero a octubre de 2024 y el origen de selección final `2025-01-01`. Este último sólo extiende el calendario: sus valores de mercado no se cargan. La admisión utiliza los derivados fijados de 2023–2024; cada callback recibe únicamente sus ventanas interiores de entrenamiento y validación, con escalado ajustado al entrenamiento correspondiente.

| Etapa | Dominio conservado | Tope de épocas y selección |
|---|---|---|
| HPO | 18 configuraciones × 2 semillas por modelo y origen, sobre la referencia arquitectónica cuando corresponde | 200 épocas por etapa; menor RMSE medio de las semillas y desempate por ID estable |
| NAS-1 | 1000 arquitecturas por cada una de las seis familias y por origen, con una semilla distinta de HPO | 50 épocas por etapa; sin poda oculta del espacio |
| NAS-2 | `k=40` y la referencia si no estaba entre los finalistas: 40 o 41 arquitecturas, con cinco semillas distintas | 200 épocas por etapa; regla registrada de margen/intervalos y parsimonia |

El k40 mantiene la decisión conservadora cuando la correlación de rankings no es estimable; los perfiles de recursos no lo convierten en una medición de ruido. Se conservan paciencia 20, las tres etapas SAE, los valores HPO congelados antes de NAS y el margen heredado. Los ocho cuerpos neuronales sin NAS realizan HPO con sus arquitecturas fijadas.

Para los catorce sistemas y once orígenes, el plan enumera **5544 oportunidades HPO**, **66.000 NAS-1** y **hasta 13.530 NAS-2**: **hasta 85.074 intentos de configuración y semilla**. Son conteos de oportunidades, no ajustes ejecutados ni etapas de optimización: SAE tiene tres etapas por intento. Tampoco representan el benchmark completo de dieciocho sistemas; los cuatro comparadores restantes mantienen sus implementaciones separadas.

| Control | Valor o alcance |
|---|---|
| Sesión del ejecutor | 8 horas como asignación inicial |
| Campaña acumulada | 24 horas inicialmente; ampliación explícita y registrada |
| Etapa científica corta / completa | 1800 / 7200 segundos, por separado para las fases SAE |
| Admisión de un trabajador | Reserva de sus topes por etapa y margen operativo antes de iniciarlo |
| Ejecución numérica | CUDA explícito, FP32, sin AMP ni TF32, determinismo fijado |
| Persistencia de selección | Metadatos científicos y resultados; sin pesos de cada candidato |
| Almacenamiento lógico configurado | `STORAGE_GIB=10`; no amplía las cuotas del proveedor |

Las 24 horas son un techo operativo inicial: **no predicen la duración completa ni prometen terminar el dominio**. El reloj de una sesión comienza al admitir su primer trabajo nuevo y contabiliza los ajustes y la gestión posterior. La instalación, la conexión al proveedor y la preparación previa no constituyen duración observada de búsqueda ni una estimación del coste total de Colab. Los topes científicos de 1800/7200 segundos se conservan separados de los antiguos límites del piloto de recursos.

Si no queda asignación suficiente para reservar el siguiente trabajador, la sesión se pausa antes de ese intento. Una pausa no permite elegir un ganador entre candidatos o semillas pendientes. La interfaz general contempla ampliar el límite global conservando `CAMPAIGN_HOURS=24` y fijando `EXTEND_CAMPAIGN_HOURS` a un valor mayor: exige añadir una enmienda antes del trabajo adicional, sin reescribir el plan original ni modificar candidatos y semillas. **La API por sí sola no concede una ampliación. En el corte posterior de §36, el diario sí registra una extensión de 24 a 36 horas; se conserva su evento y la configuración inicial como antecedentes.**

La reserva puede producir una pausa antes de consumir todas las horas disponibles. Un intento SAE de presupuesto completo reserva tres etapas de 7200 segundos más el margen operativo: 6 horas y 5 minutos. Esa reserva permite admitir el trabajo entero; no predice que necesite ese tiempo ni divide automáticamente un ajuste entre sesiones.

## Qué se guarda en GitHub

La campaña predeterminada utiliza la rama dedicada `campaigns/e2-cuda-selection-v1` y el prefijo del mismo nombre. El cliente publica una rama de metadatos, sin modificar `main`, `gh-pages` ni el código de los libros. El plan identifica fuente, hashes de implementación, datos, dominios, presupuesto inicial, destino remoto y política CUDA. Al continuar, también se comprueba la identidad informada del dispositivo y del entorno; una configuración diferente no reutiliza silenciosamente resultados.

Se sincronizan únicamente JSON científicos admitidos: plan y controles, eventos de sesión e intento, resultados de validación, épocas, costes conocidos, identidades y hashes. `status.json` ofrece el estado derivado y la cobertura completada. Los registros de intentos y resultados confirmados se conservan; el resumen derivado puede actualizarse. El snapshot verificado permite recuperar esos archivos en otra sesión.

El inicio de cada intento debe quedar confirmado en GitHub **antes de entrenar**. Su resultado local debe quedar confirmado remotamente **antes del siguiente ajuste**. La sincronización predeterminada utiliza `git fetch` y `git push` exclusivamente sobre la rama de campaña. Una actualización remota concurrente, un límite de GitHub o un fallo de red puede producir una pausa; el cliente no fuerza la rama ni repite ajustes para sortear el problema. Los tiempos de sincronización forman parte de la gestión de ejecución y pueden consumir asignación mientras la GPU está reservada.

La credencial se utiliza sólo en el proceso de sincronización. No forma parte del plan, de los argumentos, de los recibos ni del entorno de los trabajadores de entrenamiento. Los comandos, las rutas internas, los diarios crudos y los archivos de pesos quedan fuera de la publicación de campaña. El perfil máximo de §34 tenía unos 0,839 GiB de estado; guardar un archivo de ese tamaño por candidato no es la política de esta selección.

## Continuar una campaña

Este apartado describe el procedimiento general de continuación entre sesiones, no una indicación de relanzar la campaña activa de [§36](36_avance_campana_cuda.md). Cuando una sesión anterior ya terminó y se ha comprobado su estado, se conserva el mismo notebook fijado, `CAMPAIGN_NAME`, fuente, repositorio, modelos, orígenes y controles iniciales. El cliente puede recuperar el estado de GitHub, validar sus hashes y reconstruir los resultados completados para continuar la selección sin volver a entrenarlos. Un inicio sin resultado recuperable requiere la reconciliación indicada a continuación.

| Estado encontrado | Tratamiento |
|---|---|
| Intento completado y confirmado | Se reutiliza su resultado; no se vuelve a ajustar |
| Pausa por presupuesto, límite remoto o red antes del próximo intento | Se conservan los pendientes y no se declara completo el dominio |
| Resultado local terminado, todavía sin confirmación remota | En el mismo entorno puede sincronizarse sin repetir el ajuste; conservar esa caché local |
| Inicio confirmado, sin resultado recuperable | Se bloquea la campaña para reconciliar el intento; no se transforma en un ensayo no realizado ni se reintenta automáticamente |

La recuperación es **entre trabajos completados**. Los checkpoints de inferencia no contienen el estado del optimizador para continuar una época interrumpida. Si una sesión no tiene cierre confirmado, su asignación puede contabilizarse de forma conservadora; esa carga no se etiqueta como tiempo de entrenamiento observado. La pérdida del runtime antes de confirmar el resultado remoto puede exigir reconciliación manual.

## Comandos y componentes

En la fuente y el entorno fijados, el comando de planificación no necesita el secreto:

```bash
python scripts/run_e2_cuda_search.py --plan-only \
  --source-commit 564ba4762f84a800b02fa4c734f5752a169f5223 \
  --expected-registration-sha256 78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9 \
  --session-hours 8 --campaign-hours 24 --storage-gib 10 \
  --github-repository jassan-bit/DeepLearning_202630 \
  --github-branch campaigns/e2-cuda-selection-v1 \
  --github-prefix campaigns/e2-cuda-selection-v1
```

El notebook prepara las fuentes de iTransformer y los tres archivos de desarrollo verificados. Con ese entorno y `GITHUB_TOKEN` ya disponible en el entorno del proceso padre, el lanzamiento tiene esta forma:

```bash
python scripts/run_e2_cuda_search.py --execute \
  --source-commit 564ba4762f84a800b02fa4c734f5752a169f5223 \
  --expected-registration-sha256 78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9 \
  --data-root work/development-source \
  --source-root-itransformer work/author_sources/iTransformer \
  --output work/campaign-cache/e2-cuda-selection-v1 \
  --session-hours 8 --campaign-hours 24 --storage-gib 10 \
  --github-repository jassan-bit/DeepLearning_202630 \
  --github-branch campaigns/e2-cuda-selection-v1 \
  --github-prefix campaigns/e2-cuda-selection-v1
```

| Componente nuevo | Función |
|---|---|
| `e2_cuda_bindings.py` | Selección e inferencia CUDA; estados CPU y almacenamiento acotado para reajustes separados |
| `e2_cuda_search_plan.py` | Plan estándar sin GPU, datos ni autenticación; identidad de dominios y controles |
| `e2_search_session.py` | Presupuesto, bloqueo y resultados durables; recuperación entre trabajos |
| `e2_github_campaign.py` | Esquema público de JSON, inventario Merkle, validación y recuperación de la campaña |
| `e2_github_git_transport.py` | Transporte predeterminado mediante Git para la rama de campaña, sin escrituras REST por objeto |
| `run_e2_cuda_search.py` | Orquestación registrada y trabajadores aislados con dispositivo explícito |

El transporte Git entrega la credencial mediante un `ASKPASS` temporal y un entorno privado; no la coloca en la URL remota, los argumentos ni la configuración de Git. Conserva las mismas comprobaciones de JSON científico e integridad del cliente de campaña.

La descripción de hardware CPU del protocolo histórico se preserva. El registro de admisión nuevo declara la ejecución CUDA y sus controles sin reescribir esa evidencia ni alterar los dominios científicos.

El límite lógico local de metadatos es de 10 GiB. Como guardas de transferencia, el cliente acota por separado cada registro JSON a 8 MiB, el snapshot agregado a 2 GiB y la descarga comprimida a 1 GiB. Son controles del cliente y no reservas de capacidad ni una ampliación de las cuotas de GitHub.

## Validación histórica de preparación: 09/11/2026

El [recibo público de preparación](../_static/cuda_search_preparation_20261109/validation.json), con SHA256 `e573207de91bcb6b31d22ccab167accc983ca5f4dc544997039078c4290a130b`, reúne los identificadores de fuente, los conteos de pruebas y el resultado de la recuperación pública del código y su plan.

El corte histórico de preparación reunió **123 casos: 122 aprobados y un caso CUDA real omitido** por ausencia de GPU en ese ejecutor. Son 106 casos de código (105 aprobados y uno omitido) y diecisiete del notebook aprobados. Cubren selección y reajustes sintéticos, fases SAE, contabilidad de fallos y pasos desconocidos, presupuestos, recuperación de resultados, sincronización simulada con GitHub, transferencia Git contra repositorios locales de prueba, cambios concurrentes y exclusión de credenciales y rutas internas. El notebook comprueba el modo plan sin dependencias GPU y las guardas antes del lanzamiento. El caso CUDA omitido se mantiene identificado; los perfiles T4 de §§33–34 no se presentan como ejecución de este nuevo orquestador. Estos conteos de pruebas no se suman a los resultados científicos posteriores.

También se recuperó realmente la fuente pública fijada, se verificaron los 61 hashes de implementación y el registro, y se ejecutó su plan estándar con los paquetes de sitio desactivados. El recorrido confirmó los catorce modelos y once orígenes, sin materializar CSV, crear una campaña, pedir credenciales de campaña, instalar paquetes ni inicializar CUDA. Esta prueba comprueba la descarga pública y la planificación; no publica resultados de entrenamiento.

El corte del 09/11/2026 dejó preparada la búsqueda científica GPU en Colab, sin aportar todavía nuevos ajustes de mercado ni ganadores NAS/HPO. La ejecución posterior y su cobertura incompleta se describen en [§36: avance de la campaña CUDA](36_avance_campana_cuda.md). El reajuste final registrado conserva la guarda que exige fijar primero el R del protocolo de potencia; los valores finales de 2025–2026 y `final_R` permanecen fuera del lanzamiento de selección. Ninguno de estos avances demuestra superioridad del candidato 11.
