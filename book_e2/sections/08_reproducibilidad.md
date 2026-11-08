(e2-reproducibilidad)=
# 8. Reproducibilidad, ética y comunicación científica

La guía completa exige un entorno fijado, semillas/configuraciones recuperables y reproducción independiente de la tabla principal (PDF pp40–41); su lista de comprobación está en el Apéndice B (p63). Esta sección documenta **lo que ya existe para el procedimiento P0/OOF de desarrollo** y lo distingue del benchmark final de veinte configuraciones, todavía pendiente. P0 está ejecutado: los recibos verifican 220 trayectorias y 440 ajustes cerrados; los resultados se presentan en la [sección 9](09_protocolo_registrado.md).

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

Los entornos se identifican por etapa. El archivo del entorno EDA fija statsmodels0.14.6; el archivo de remediación fija statsmodels0.15.0. Cada reproducción de las pruebas sintéticas de §7 debe identificar la versión utilizada; aprobarlas no equivale a una ejecución ARIMA sobre Bitcoin. El piloto P0 actual solo ajusta las cuatro redes registradas, sin atribuirle resultados de ARIMA ni de sus otras referencias futuras.

Las semillas P0 son **660101–660105**, emparejadas entre TimeMixer, iTransformer, PatchTST y GRU-c03. El driver fija `random`, NumPy y Torch en cada ajuste, usa un generador explícito para el orden de minibatches y activa `torch.use_deterministic_algorithms(True)`. El recibo inicial indica `deterministic_algorithms=false` **al registrar**, antes de llamar a esa configuración: se conservan ambos hechos en sus etapas correspondientes. Las semillas de refit y de los demás procedimientos tienen registros propios; ninguna repetición del cronómetro cuenta como nueva observación de mercado. No se promete identidad numérica entre hardware o versiones distintos sin una reproducción medida.

## 8.3 Comandos de reproducción del procedimiento disponible

Los siguientes comandos reproducen el flujo **P0/OOF**, no ejecutan HPO, P1 ni el benchmark final. Se parte del repositorio clonado, de Python3.12 y de una ruta nueva de salida. Los dos CSV diarios de desarrollo están versionados en la fuente registrada; los hashes exigidos por el plan detectan modificaciones. No es necesario descargar precios nuevos para ese piloto.

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

### 8.3.1 Implementación del benchmark final

La [descarga nueva](../_static/completion_download/completion_bundle.zip) incorpora el runner `scripts/run_e2_final_benchmark.py`, adaptadores finales, estadística, incertidumbre, diagnósticos, contador de operaciones, referencia de predictibilidad y pruebas. El [diseño](../_static/completion_protocol/design.json) y la [enmienda](../_static/completion_protocol/analysis_additions.json) identifican los insumos del [registro previo](../_static/completion_preflight/registration.json) que habilitó sus ajustes reales. Se conservan el código/registro P0 y sus recibos sin reescribirlos.

El programa separa inventario/preflight, registro, HPO, P1, coste, potencia, cierre de pesos primarios y A, calibración, scoring y análisis. Debe ejecutarse desde un checkout limpio con fuentes de autores fijadas y salida privada nueva. No basta instalar paquetes: se verifican los bytes de todas las dependencias importadas, datos de desarrollo, candidatos y revisión arquitectónica. Los comandos y opciones exactas están en el paquete; el flujo aún no acredita una reproducción independiente de la tabla final.

Las pruebas del software usan fixtures sintéticos. La sanidad de memorización de los diez adaptadores usa únicamente entradas sintéticas y no se suma al contador de ajustes de Bitcoin. Los resultados reales requieren curvas, tiempos, cierres y recibos separados. Si un reinicio cambia el entorno, se reinstala/verifica el lock y se repite su preflight antes de continuar; no se reutilizan mediciones de hardware de P0 como costes actuales.

El HPO puede podar pesos sólo después de cerrar sus predicciones de desarrollo, conservando configuración, curva, preprocesamiento, índice, hash original y recibo de poda. Los pesos primarios/A se conservan durante sus cierres globales y puntuación. Las ablaciones DEV53 pueden cerrar/podar secuencialmente sin usar test; se retienen cuatro variantes de la primera semilla. Se estima almacenamiento antes de entrenar y no se cambia R después de ver resultados para ajustarlo al disco.

## 8.4 Lista de comprobación frente al Apéndice B

Los estados distinguen una especificación registrada de evidencia experimental terminada.

| Ítem de p63 | Estado verificable |
|---|---|
| Registro previo al entrenamiento nuevo | Recibo de P0 enlazado; el calendario histórico sigue conocido |
| Hash de datos/test | Fuente e índices identificados; el hash no restaura desconocimiento del test |
| Auditoría completa de fuga | Soporte P0/OOF aprobado; verificaciones del pipeline final pendientes, véase §10 |
| Particiones y semillas recuperables | Plan y once calendarios de desarrollo registrados |
| Pipeline y ajuste solo en train | P0 disponible; diez adaptadores finales revisados, preflight y registro publicados; recibo parcial HPO del 06/11, sin cierre experimental final publicado |
| Presupuesto común y configuraciones | AsignaciónT12 conservada; enmienda de once aleatorios+c12 fuente preparada; HPO sin ejecutar |
| Repeticiones y resultados individuales | Cinco semillas previstas por receta P0; ejecución completada y analizada en §9, sin afirmar benchmark final completo |
| Potencia y número final | P1/dimensionamiento posteriores a HPO; P0 no acredita potencia |
| Baselines y tres bases E1 | Panel de veinte registrado; comparación final con HPO equitativo pendiente |
| Software, hardware y determinismo | Recibos y reglas por etapa documentados arriba |
| Pruebas, efectos e IC finales | Familias registradas; resultados de benchmark pendientes |
| Familias y multiplicidad | Treinta contrastes frente a cada GRU, α0,02 por familia; 199 diagnósticos, α0,01; véase §9 |
| Techo y margen normalizado | No certificados por este preflight; análisis específico pendiente |
| Arquitectura no definida/NAS | Diez casos definidos y revisados individualmente; regla condicional conservada, sin NAS ejecutada |
| Ganador y sesgo de selección | Ningún ganador final declarado |
| Robustez e interpretabilidad | Diseños documentados; ejecución final pendiente |
| Incertidumbre Top3 | A53 predefinida para veinte modelos, cinco miembros fijos antes de P1; implementación probada, calibración/cobertura pendiente |
| Acceso a prueba con pesos fijados | P0 restringido a desarrollo; test E1 conocido y benchmark final aún pendiente |
| Réplica del paper guía | Evidencia externa TimeMixer/ETTh1 en §5; separada de Bitcoin |
| Reproducción independiente de tabla principal | Pendiente; sin discrepancia final inventada |
| Ejecución completa de un comando | Disponible para P0/OOF; aún no demostrada para todo E2 |

## 8.5 Ética, licencias y comunicación

Se mantiene el uso académico y no comercial y la procedencia de Binance Vision descritos en §1, con sus términos conservados en E1. El código de los autores conserva sus licencias y commits identificados; usar su implementación no significa reutilizar pesos preentrenados ni reproducir todas las condiciones originales. P0 inicializa sus redes, con una adaptación escalar explícita a σ₇; el resultado externo de ETTh1 no se presenta como rendimiento Bitcoin.

Las velas no contienen identificadores de personas, y no se intenta reconstruir identidades de operadores. El mercado BTCUSDT de un exchange no representa otras criptomonedas, exchanges o cotizaciones: las diferencias de error por volatilidad y periodo se comunicarán con tamaños e incertidumbre. Las consecuencias de errores de pronóstico no se traducen automáticamente en rentabilidad, reducción de riesgo o recomendación de inversión.

Las tablas y figuras deben identificar unidades, muestra, escala, fuente y alcance; los IC marginales se distinguen de límites simultáneos. No se presenta cero como sustituto de una medición pendiente ni un promedio sin dispersión como prueba de superioridad. Las afirmaciones bibliográficas y DOI se conservan trazables; indexación o editorial no sustituyen comprobar un original.

**Descargas:** [implementación final y enmienda](../_static/completion_download/completion_bundle.zip), [paquete unificado de protocolo, código y evidencias](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip), [registro](../_static/remediation_preflight/registration.json), [soporte de datos](../_static/remediation_preflight/data_support_audit.json) y [entorno/hardware](../_static/remediation_preflight/hardware_receipt.json).
