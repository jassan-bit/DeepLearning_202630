(e2-reproducibilidad)=
# 8. Reproducibilidad, ética y comunicación científica

**Vigencia documental (12/11/2026):** la selección HPO/NAS ejecutada parcialmente corresponde al plan histórico de catorce redes. El catálogo obligatorio vigente es de **16 sistemas**, con las identidades de [§43](43_catalogo_obligatorio.md), las arquitecturas de [§44](44_arquitecturas_publicadas.md) y el protocolo registrado de [§45](45_enmienda_protocolo.md). Esos registros históricos no cierran el benchmark vigente ni acreditan las cuatro recetas restauradas. R y el cierre inferencial siguen pendientes. [Corte verificable de resultados (§41)](41_avance_optimizacion_11_origenes.md).

La tabla histórica de §8.4 conserva su ámbito. La auditoría actual añade [las 21 filas del Apéndice B](../_static/guide_audit_20261111_181b9e5/appendix_B.csv), sin considerar aprobados sus pendientes. El comando P0 no reproduce por sí solo los 16 sistemas obligatorios vigentes; los 18 sistemas pertenecían al diseño anterior.

**Vigencia al 11/11/2026:** La guía exige reproducción integral en entorno limpio y Apéndice B completado. La compilación del libro acredita construcción y enlaces, no reproducción del benchmark. El contenedor es opcional en E2 (p. 41). La lista vigente de artefactos y condiciones de cierre está en §38. [Matriz de exigencias y evidencia](38_cumplimiento_guia.md) · [Enmienda de calendario](37_ajuste_guia.md).


La guía completa exige un entorno fijado, semillas/configuraciones recuperables y reproducción independiente de la tabla principal (PDF pp40–41); su lista de comprobación está en el Apéndice B (p63). Esta sección distingue el **catálogo obligatorio vigente de 16 sistemas** y el protocolo ya registrado en §§43–45 de la evidencia histórica conservada. El híbrido adicional y Ridge son opcionales/históricos; las enmiendas registradas no acreditan ejecución integral. El procedimiento P0/OOF de desarrollo está ejecutado: los recibos verifican 220 trayectorias y 440 ajustes cerrados; los resultados se presentan en la [sección 9](09_protocolo_registrado.md).

## 8.1 Fuente registrada y recibos de ejecución

La fuente experimental de P0 se fijó en el commit [`599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0). El [recibo de registro](../_static/remediation_preflight/registration.json), fechado **06/11/2026, 14:52:32 UTC**, vincula plan, código, dos fuentes de desarrollo y archivos de los autores mediante hashes. Registra cero ajustes de datos previos **en ese directorio de salida**; no afirma que nunca existieran experimentos de E1 ni que su test vuelva a ser desconocido.

Se conserva un checkout separado de la fuente registrada y las salidas experimentales quedan fuera del repositorio: checkpoints, curvas, cierres, logs y predicciones no sobrescriben resultados de E1. Las mejoras documentales posteriores al registro no cambian los bytes con los que se inició P0. Los JSON de diseño conservan sus contadores iniciales: el estado observado se acredita con recibos de ejecución, no editando un plan cerrado.

El [recibo de la auditoría de soporte](../_static/remediation_preflight/data_support_audit.json) se generó antes del registro, a las **14:52:31 UTC**. Comprueba los once calendarios de desarrollo; no es un certificado completo de los modelos futuros. El [protocolo y estado de P0](09_protocolo_registrado.md) y la [auditoría de datos](10_auditoria_datos.md) explican esos alcances.

## 8.2 Entorno observado y aleatoriedad

El [recibo de hardware](../_static/remediation_preflight/hardware_receipt.json) se tomó a las **14:37:37 UTC**, antes de los ajustes. Son datos del entorno, sin confundirlos con costes medidos por modelo.

| Elemento | Evidencia del entorno P0 |
|---|---|
| Sistema | Debian GNU/Linux13, versión13.6; kernel6.18.57, x86_64 |
| Python | 3.12.14 |
| Framework y paquetes principales registrados | PyTorch2.15.0+cpu, NumPy2.5.4, pandas3.1.0, einops0.8.1, reformer-pytorch1.4.4 |
| Dispositivo | CPU; modelo reportado AMD EPYC9V74 80-Core Processor |
| Recursos efectivos | Cinco CPU lógicas visibles; cuota cgroup400000/100000, equivalente a cuatro CPU, sin atribuir ochenta núcleos disponibles al proceso |
| Paralelismo de trabajo | Un thread numérico por ajuste; máximo dos procesos concurrentes |
| Memoria | Límite cgroup34.359.738.368bytes, 32GiB; no es memoria pico medida de una red |
| Dependencias fijadas | `requirements-e2-remediation.lock.txt`, que incluye `requirements-paper-guide.lock.txt` |

Los entornos se identifican por etapa. El archivo del entorno EDA fija statsmodels0.14.6; el archivo de remediación fija statsmodels0.15.0. Cada reproducción de las pruebas sintéticas de §7 debe identificar la versión utilizada; aprobarlas no equivale a una ejecución ARIMA sobre Bitcoin. El piloto P0 histórico ajustó únicamente sus cuatro recetas registradas; no aporta ejecuciones de los modelos incorporados al catálogo vigente.

Las semillas P0 son **660101–660105**, emparejadas entre TimeMixer, iTransformer, PatchTST y GRU-c03. El driver fija `random`, NumPy y Torch en cada ajuste, usa un generador explícito para el orden de minibatches y activa `torch.use_deterministic_algorithms(True)`. El recibo inicial indica `deterministic_algorithms=false` **al registrar**, antes de llamar a esa configuración: se conservan ambos hechos en sus etapas correspondientes. Las semillas de refit y de los demás procedimientos tienen registros propios; ninguna repetición del cronómetro cuenta como nueva observación de mercado. No se promete identidad numérica entre hardware o versiones distintos sin una reproducción medida.

## 8.3 Comandos de reproducción del procedimiento disponible

Los siguientes comandos reproducen exclusivamente el **P0/OOF histórico**. No ejecutan el catálogo vigente ni constituyen instrucciones para continuar el plan sustituido. Se parte del repositorio clonado, de Python3.12 y de una ruta nueva de salida. Los dos CSV diarios de desarrollo están versionados en la fuente registrada; los hashes exigidos por el plan detectan modificaciones. No es necesario descargar precios nuevos para ese piloto.

Crear un checkout y un entorno separados permite conservar E1 y la ejecución original:

```bash
git worktree add --detach ../e2-repro-registered 599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0
cd ../e2-repro-registered
python3.12 -m venv ../e2-repro-env
../e2-repro-env/bin/python -m pip install -r requirements-e2-remediation.lock.txt
```

Las dos fuentes oficiales se recuperan en sus commits fijados; el runner verifica los archivos concretos contra el plan antes de usarlos:

```bash
git clone https://github.com/kwuking/TimeMixer.git ../e2-author-TimeMixer
git -C ../e2-author-TimeMixer checkout --detach e24610583b36fdd8c76cc17a8df4e65759a5f460
git clone https://github.com/thuml/Time-Series-Library.git ../e2-author-controls
git -C ../e2-author-controls checkout --detach 4e938a1767106324dd753b2a44832bf870a0252e
```

La validación documental no entrena. El preflight añade controles de los adaptadores con entradas sintéticas y una auditoría de desarrollo sin ajustes de modelos:

```bash
../e2-repro-env/bin/python scripts/e2_protocol_plan.py validate
../e2-repro-env/bin/python scripts/execute_e2_development_pilot.py --stage preflight --source-root ../e2-author-TimeMixer --control-source-root ../e2-author-controls --output-dir ../e2-repro-output --workers 2
```

El flujo disponible de una sola invocación registra esa nueva reproducción, ajusta las trayectorias programadas y puntúa desarrollo **después de sus cierres**; posteriormente analiza las predicciones guardadas:

```bash
../e2-repro-env/bin/python scripts/execute_e2_development_pilot.py --stage all --source-root ../e2-author-TimeMixer --control-source-root ../e2-author-controls --output-dir ../e2-repro-output --workers 2
```

La salida debe estar fuera del repositorio. El registro rechaza intentos anteriores sin registro y cambios en entradas cerradas; no se reutiliza un directorio para ocultar una ejecución fallida. El flujo no acepta precios de evaluación posteriores a 2024 y no cambia las recetas con P0. Una reproducción completa de la **tabla principal final** aún no está demostrada: no se equipara este comando de desarrollo, un build del libro o una prueba sintética con ese requisito de las páginas41/63. Su discrepancia máxima deberá medirse cuando exista la tabla y una reproducción independiente válida.

### 8.3.1 Implementación archivada y requisitos de la nueva versión

La [descarga de implementación archivada](../_static/completion_download/completion_bundle.zip), su [diseño](../_static/completion_protocol/design.json), su [enmienda](../_static/completion_protocol/analysis_additions.json) y su [registro](../_static/completion_preflight/registration.json) conservan la procedencia del trabajo previo. Esos archivos y `scripts/run_e2_final_benchmark.py` **no son el ejecutor ni el registro del catálogo vigente**. Se mantienen sin alterar para que sus bytes sigan siendo verificables.

El [recibo parcial del 06/11/2026, 19:00:35 UTC](../_static/completion_preflight/execution_progress.json) acredita **doce trayectorias y veinticuatro ajustes cerrados de TSMixer** bajo el registro anterior. Es una instantánea histórica de HPO incompleto, no un proceso cuya actividad actual esté demostrada, un resultado final ni una ejecución del nuevo catálogo. Los recibos sintéticos tampoco se convierten en resultados de mercado.

La nueva enmienda deberá identificar fuentes y adaptaciones de los diez modelos, la receta CNN-1D + LSTM + atención temporal, controles, dependencias, presupuesto, selección temporal, semillas, dimensionamiento y orden de cierre. MLP, LSTM y GRU de E1 siguen siendo referencias separadas. No se transfieren automáticamente sus recetas o métricas ni las del P0 a la comparación nueva.

Cada nueva ejecución requerirá fuente identificable, entorno verificable, datos e índices con hashes y una salida propia. Deben conservarse curvas, tiempos, estados de cierre y procedencia, incluidos intentos fallidos. Un build del libro, un prototipo o una prueba sintética no acredita HPO, calibración o evaluación de mercado. El número de repeticiones y el almacenamiento se planificarán antes de evaluar; no se cambia R después de observar resultados para ajustar el experimento a los recursos disponibles.

## 8.4 Lista de comprobación frente al Apéndice B

Los estados distinguen una especificación registrada de evidencia experimental terminada.

| Ítem de p63 | Estado verificable |
|---|---|
| Catálogo del corte histórico | Diez modelos de literatura/adaptaciones y CNN-1D + LSTM + atención temporal como candidato 11, sin superioridad demostrada; sustituido por los 16 sistemas obligatorios de §43 |
| Registro antes de los nuevos ajustes | Nueva enmienda pendiente; los recibos anteriores son históricos |
| Datos, índices y particiones | Procedencia heredada identificada; auditoría específica de los nuevos soportes pendiente |
| Selección y ajuste causal | Reglas metodológicas descritas; implementación común del catálogo vigente por validar |
| Presupuesto, configuraciones y R | Pendientes de definir y dimensionar; no se heredan del catálogo sustituido |
| Referencias E1 | MLP, LSTM y GRU separadas del catálogo; recetas y cohortes comparables por identificar |
| Software, hardware y determinismo | Evidencia P0 conservada; cada ejecución nueva necesita sus propios recibos |
| Resultados individuales y agregados | P0 ejecutado e instantánea parcial de HPO histórico enlazados; sin resultados finales del catálogo vigente |
| Hipótesis, multiplicidad, efectos e IC | Nuevas familias y decisiones pendientes de registrar |
| Robustez, interpretabilidad e incertidumbre | Métodos y límites descritos; validación del nuevo diseño pendiente |
| Evaluación desconocida | Los periodos heredados ya se conocen y no recuperan independencia por cambiar los modelos |
| Réplica del paper guía | Evidencia TimeMixer/ETTh1 en §5, separada de Bitcoin |
| Reproducción independiente de tabla principal | Pendiente; ninguna discrepancia final inventada |
| Reproducción de un comando | Disponible para el P0 histórico; no acreditada para el catálogo vigente |

## 8.5 Ética, licencias y comunicación

Se mantiene el uso académico y no comercial y la procedencia de Binance Vision descritos en §1, con sus términos conservados en E1. El código de los autores conserva sus licencias y commits identificados; usar su implementación no significa reutilizar pesos preentrenados ni reproducir todas las condiciones originales. P0 inicializa sus redes, con una adaptación escalar explícita a σ₇; el resultado externo de ETTh1 no se presenta como rendimiento Bitcoin.

Las velas no contienen identificadores de personas, y no se intenta reconstruir identidades de operadores. El mercado BTCUSDT de un exchange no representa otras criptomonedas, exchanges o cotizaciones: las diferencias de error por volatilidad y periodo se comunicarán con tamaños e incertidumbre. Las consecuencias de errores de pronóstico no se traducen automáticamente en rentabilidad, reducción de riesgo o recomendación de inversión.

Las tablas y figuras deben identificar unidades, muestra, escala, fuente y alcance; los IC marginales se distinguen de límites simultáneos. No se presenta cero como sustituto de una medición pendiente ni un promedio sin dispersión como prueba de superioridad. Las afirmaciones bibliográficas y DOI se conservan trazables; indexación o editorial no sustituyen comprobar un original.

**Archivo histórico:** [implementación y enmienda anteriores](../_static/completion_download/completion_bundle.zip), [paquete unificado de protocolo, código y evidencias](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip), [registro](../_static/remediation_preflight/registration.json), [soporte de datos](../_static/remediation_preflight/data_support_audit.json) y [entorno/hardware](../_static/remediation_preflight/hardware_receipt.json).
