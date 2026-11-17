# Piloto exploratorio de momentos en Colab

**16 de noviembre de 2026. Software preparado; ejecución real en Colab pendiente.** El usuario solicita comprobar el modelo en desarrollo antes de completar los requisitos finales. Esta decisión permite un piloto exploratorio posterior al dictamen interno, sin convertirlo en preregistro, evidencia de novedad o comparación confirmatoria.

Se conserva **CNN causal → LSTM unidireccional → atención temporal**. El notebook guarda automáticamente la corrida, checkpoints, curvas, predicciones y presupuesto en GitHub, en `e3-desarrollo-resultados-momentos`, prefijo `runs/momentos_v1`.

## Ejecutar

1. Abrir [el notebook en Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_momentos_colab.ipynb).
2. Seleccionar **GPU** y habilitar el secreto **`GITHUB_TOKEN`**, con permiso de escritura de contenidos en este repositorio. El código registra el dispositivo realmente asignado; no presupone que sea una A100.
3. Pulsar **Ejecutar todas**. La instalación inicial puede tardar varios minutos y utiliza un intérprete aislado; no requiere reiniciar el kernel. La celda de inicialización ya ajusta estadísticas y consume presupuesto.

Si la sesión se interrumpe, volver al mismo notebook, las mismas carpetas y el mismo repositorio. Un registro remoto único conserva la identidad y el contador; no se crea otra corrida al repetir las celdas. Si otro proceso conserva un arrendamiento activo, se espera su liberación o expiración; no se fuerza ni se reinicia el presupuesto. La recuperación conservativa de una interrupción puede consumir saldo y quedar documentada como agotamiento.

El código y cada fuente se fijan a un commit y SHA-256 antes de importar o ajustar. El token queda en memoria del coordinador y se filtra del entorno del trabajador. El destino de resultados se crea al inicializar la primera corrida; antes de ese paso la rama aún puede no existir. No se usa Google Drive ni se abren controladores de entrenamiento de E2.

## Qué se compara

| Variante | Función | Parámetros |
|---|---|---:|
| `moment_joint` | Cabeza por momentos, MSE y dos scores auxiliares con pesos 0,1. | 1236 |
| `moment_mean_only` | Misma cabeza con MSE sola; aísla la supervisión auxiliar. Sus salidas `m,q` no se presentan como momentos identificados. | 1236 |
| `direct_moment_aux` | Media positiva directa y las mismas dos cabezas/scores auxiliares; aísla el vínculo de salida. | 1236 |
| `frozen_v2_additive` | CSV verificado del control aditivo v2, con las mismas 116 fechas y objetivos. Sin nuevo ajuste o inferencia de esa red. | 1210 en la receta original |
| Piso observable / piso más exceso medio de ajuste | Dos referencias analíticas; el incremento se obtiene sólo de ajuste. | 0 / 1 |

Las tres redes nuevas tienen 1197 parámetros comunes de codificador/atención y 39 de tres lecturas activas. El readout anterior se elimina; no se suman parámetros inactivos. Las lecturas y el operador son FP64, mientras codificador y atención usan FP32. Se comparte semilla y estado inicial del tronco. Los pesos de lectura empiezan en cero, `b=√8`, `d=1` para las cabezas de momentos; la media directa iguala el exceso medio inicial usando sólo ajuste. Las predicciones iniciales por muestra pueden diferir entre tipos de cabeza.

La formulación, loss y campos exactos están en [plan.json](plan.json). Se atribuyen los componentes conocidos y se mantienen los límites del [dictamen anterior](../e3_dictamen_momentos_20261116/README.md). Esta prueba no incluye el control adaptado de Faithful ni la campaña completa de E3; su resultado no resuelve por sí solo la sustantividad de la contribución.

## Datos y criterio fijados antes de ejecutar

Sólo se indexan `training_raw`, `training_y_original` y `training_dates` del NPZ inmutable de E2, verificado por SHA-256. Se conservan **532 casos de ajuste**, siete purgados y **116 de validación** entre el 8 de julio y el 31 de octubre de 2024. Las estadísticas y la inicialización usan ajuste exclusivamente. La validación ya orientó el desarrollo anterior y vuelve a elegir la época: no es una confirmación independiente. No se cargan arrays de calibración o prueba.

Se fija una semilla, AdamW con `lr=0,001`, batch 32, hasta 40 épocas por red, paciencia 8 y épocas intercaladas. No hay búsqueda de hiperparámetros, selección de semillas o extensión para obtener un resultado favorable. La salida de momentos **no es un intervalo de predicción**. Una inestabilidad numérica detiene la corrida y conserva el último checkpoint válido; no se eliminan filas ni se ocultan mediante clipping de las salidas.

La heurística exploratoria exige al candidato **al menos 2 % menos RMSE que el mejor de los cinco controles**, con las tres recetas completas. Si se agota el presupuesto antes, la comparación queda incompleta y no se declara éxito. Se conservan RMSE, MAE, sesgo, errores por régimen retrospectivo, scores y curvas de todas las recetas. Los grupos por objetivo realizado no son entradas disponibles al pronosticar. El margen confirmatorio heredado permanece inalterado.

## Presupuesto y reanudación

El consumo anterior es **556,2126593619992 s** entre v1 y v2. Se asignan **600 s adicionales de cálculo contabilizado**, manteniendo el límite global de 1200 s y una reserva aritmética de 43,7873406380008 s tras esa asignación. El contador comprueba límites entre operaciones; una operación ya iniciada puede añadir sobremarcha, que se registra sin recortar el gasto real. Instalación, descargas y publicaciones añaden tiempo de reloj fuera de ese contador cuando no hay una reserva activa pendiente.

Se confirma un claim remoto antes del primer ajuste y de cada bloque. Se publica y confirma el avance antes de continuar. El presupuesto incluye estadísticas de ajuste, inicialización, entrenamiento, validación, guardado local y cargos de recuperación. Los checkpoints preservan optimizador, RNG, permutación, cursor de batch, mejor época y procedencia; intercambiar variantes de igual forma o retroceder progreso se rechaza. Un estado terminado, fallido o agotado se conserva sin nuevas épocas o recetas.

Las ramas previas deben seguir selladas en sus commits fijados. De v2 se descargan sólo ocho textos y su manifiesto desde el commit inmutable, con verificación de bytes, fechas y objetivos; no se descargan ni cargan pesos antiguos. Los textos incluyen el ledger para conciliar el consumo histórico. El proceso que maneja credenciales está separado del trabajador científico.

## Qué se ha comprobado y qué falta ejecutar

Las pruebas locales usan exclusivamente datos artificiales y CPU. Incluyen gradientes y fórmulas, parámetros activos, estadísticas sólo de ajuste, reanudación exacta, recuperación de interrupciones, corrupción/intercambio de checkpoints, agotamiento y guardado transaccional. La integración recorre `preflight → run → resume` con tres redes y un GitHub simulado. El notebook se publica sin salidas fabricadas. El código todavía debe ejecutarse en la GPU real de Colab para obtener resultados de mercado.

```bash
python -m pytest tests/test_moment_model.py tests/test_moment_training.py tests/test_moment_github.py tests/test_moment_notebook.py -q
python scripts/e3_build_moment_notebook.py --check
```

La conversación actual no ofrece una herramienta para controlar el navegador o la sesión de Colab del usuario. El entorno de trabajo `ccarenv_b64_Y2NhcmVudl84MTlhYWJiMmEyMmM4MTkxOTBlYjQ3ODlmYTJkODM0YQ` declara escritorio remoto, pero no expone aquí la herramienta JavaScript de escritorio ni el visor. Por eso el inicio real corresponde al usuario mediante el enlace anterior; no se afirma haber usado su A100 o iniciado una corrida real.

La documentación final incorporará los resultados, incluidos fallos o un desenlace desfavorable, sin retrofechar protocolo o revisión externa. Se mantiene la estructura del libro con **capítulos 1–16**, ampliando los capítulos 6 y 7 y los anexos existentes. La asistencia de ChatGPT/Codex incluye implementación, pruebas, coordinación de almacenamiento, notebook y documentación; no constituye verificación humana externa ni defensa del estudiante.
