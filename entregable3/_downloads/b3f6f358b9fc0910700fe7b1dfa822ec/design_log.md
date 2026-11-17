# Bitácora consolidada del diseño de E3: corte posterior del 17 de noviembre de 2026

Fecha real de esta consolidación: **2026-11-17**, zona America/Bogota. Fuente del libro revisado: `15f6f664217dc9a88dc12c6ce22ad1516ff9e384`. Esta es una revisión documental posterior a los resultados conocidos; **no es un preregistro, no reinicia presupuesto y no acredita R1, defensa de R2 o R3**. Los archivos históricos se conservan sin cambiar sus fechas, estados originales o métricas.

El índice canónico es `experiments/design_log.md`. El CSV `design_inventory.csv` ofrece filas verificables y este documento explica cómo leerlas. Los registros futuros deberán añadirse con identificador nuevo y su fecha real; una corrección conservará la fila original, el motivo y un nuevo registro, en lugar de ocultar un desenlace.

## Conteos y alcance

Se recuperaron **ocho recetas neuronales entrenadas en desarrollo**: dos de v1, tres de v2 y tres por momentos. La fase posterior inicializó diez réplicas de la misma receta aditiva congelada: son **diez ajustes adicionales de una receta existente**, no diez diseños. El inventario separa también cuatro intervenciones artificiales iniciales y tres controles analíticos/reutilizados, sin sumarlos como ocho recetas neuronales nuevas.

El **Nd global sigue sin conciliar**. Debe incluir propuestas y parametrizaciones consideradas, modificaciones de software, controles de salida, intentos, fallos y decisiones que influyeron en el diseño. No se calcula un sesgo de selección a partir del subtotal ocho ni de los dieciocho ajustes neuronales observados. La lista no afirma enumerar todos los intentos informales. Los análisis, la bisección de sesgo, la cuadratura, las variantes artificiales y las réplicas conservan clases distintas en el inventario; su impacto en la selección debe justificarse antes del cierre.

`recorded_on` es la fecha de esta consolidación. `historical_plan_date` indica sólo la fecha que contiene un plan de diseño, cuando existe. Las columnas de inicio/fin de ejecución quedan **vacías cuando no hay timestamps en el resumen consultado**. El nombre de una carpeta o el valor de una semilla no se usan como fecha de entrenamiento. Los planes de diseño y los planes incorporados al ejecutor pueden tener bytes distintos: los hashes registrados en los resúmenes no se sustituyen por el hash de otro plan local.

## Recetas y resultados de desarrollo

Todas estas recetas usan una semilla de desarrollo, 20261116, 532 ejemplos de ajuste y 116 de validación tras siete purgados. La validación selecciona época y parada y ya fue expuesta a decisiones. Las RMSE están en unidades originales de σ7; **no son las métricas de prueba de diez semillas**.

| Etapa | Receta | Parámetros | RMSE de desarrollo | Mejor época / épocas completadas |
|---|---|---:|---:|---:|
| v1 | `full` | 1211 | 0.006045314272 | 35 / 40 |
| v1 | `M00` | 1210 | 0.005376766517 | 39 / 40 |
| v2 | `additive_positive_head` | 1210 | 0.003144614784 | 38 / 40 |
| v2 | `innovation_mean_only` | 1210 | 0.003173926388 | 39 / 40 |
| v2 | `innovation_density_mean` | 1210 | 0.003198296979 | 39 / 40 |
| moments | `moment_joint` | 1236 | 0.003258920057 | 1 / 9 |
| moments | `moment_mean_only` | 1236 | 0.003259444189 | 1 / 9 |
| moments | `direct_moment_aux` | 1236 | 0.003234074349 | 29 / 37 |

V1 fue desfavorable al mecanismo completo frente a M00; no se oculta ni se convierte en una validación de robustez de mercado. El diagnóstico del piso observable fue posterior a ese piloto. En v2, la cabeza aditiva obtuvo la menor RMSE entre las tres nuevas recetas. La variante con NLL no superó la heurística exploratoria del 2 %. El piloto por momentos completó sus tres recetas y tampoco superó esa heurística: `moment_joint` quedó un 3,635 % peor que la aditiva congelada. El 2 % fue una regla de continuación de desarrollo, **no el margen confirmatorio δ** ni una decisión sobre novedad.

Los controles `floor_only`, `floor_plus_fit_mean` y `frozen_M00_projected` se identifican por separado. Calcular un promedio sólo con ajuste es un ajuste estadístico, aunque no sea otro entrenamiento neuronal. Proyectar predicciones heredadas cambia la salida y debe registrarse; no equivale a entrenar nuevamente M00. La lectura aditiva usa atención temporal por contenido convencional; «aditiva» describe el exceso positivo de la salida.

## Réplicas y análisis posteriores

Las semillas **271801–271810** repiten la receta aditiva de 1210 parámetros y se conservan todas, con sus mejores épocas por validación. La semilla histórica 20261116 no forma parte de su media. En las 608 fechas de prueba históricas del 1 de enero de 2025 al 31 de agosto de 2026, la media de RMSE individuales es **0.002939432419**, con DE muestral **0.000047125418** y MAE media **0.001892827303**. No es la RMSE de un ensamble ni selección de la mejor semilla.

El análisis posterior reutiliza esas predicciones y las de 16 rivales. Sus 48 rechazos tras Holm global y el MCS auxiliar de diez supervivientes, incluida E3, tienen el alcance de los procedimientos declarados en el capítulo 8. No son una familia de ablación ni una demostración de equivalencia, ganador único, potencia suficiente o causalidad de la arquitectura. E2 se ajustó con 655 ejemplos y calibró con 53; E3 usó 532 de ajuste y 116 de validación, sin calibración. Diez semillas y fechas comunes no igualan esos tratamientos.

## Consumo conservado y esfuerzo pendiente

| Etapa E3 | Consumo adicional registrado (s) | Acumulado E3 (s) |
|---|---:|---:|
| V1 | 369,477959639 | 369,477959639 |
| V2 | 186,734699723 | 556,212659362 |
| Momentos | 94,164684173 | 650,377343535 |
| Estabilidad de la aditiva | 298,062281507 | 948,439625042 |

Los resúmenes mantienen los cargos originales, incluidos los cargos conservadores de recuperación que correspondan. El acumulado **948,439625042 s incluye etapas anteriores**; no es el coste exclusivo de las diez réplicas. El límite histórico global era 1200 s y no se reinicia al consolidar la bitácora. Instalación, descargas y publicación quedaron fuera del contador activo descrito por esos planes; estos segundos no son el tiempo total de trabajo, el consumo energético ni una estimación completa de horas GPU.

La ampliación de E2 añadió 52 ajustes bajo un contador **separado**, que cerró en 5130,970719594 s sobre 7200 s. No se suma al saldo E3 ni prueba presupuesto equivalente. No se ha conciliado el esfuerzo humano/IA de revisión, formulación, pruebas y diseño frente al esfuerzo de cada benchmark. Las búsquedas de HPO y sus presupuestos deben documentarse por separado y hacerse equivalentes en una campaña futura.

Se preservan los desenlaces interrumpidos y las reparaciones de infraestructura en sus fuentes: la evaluación histórica recuperó resultados locales tras fallos GitHub sin repetir inferencia, la estabilidad continuó con transporte Git nativo tras un fallo de actualización de referencia y la extensión E2 corrigió metadatos antes de completar sus ajustes. **Esta consolidación no reconstruye una lista exhaustiva ni tiempos individuales de todos esos incidentes**. Tampoco los cuenta como nuevos mecanismos o elimina su consumo del historial. El cierre de Nd y del esfuerzo debe enlazar sus registros detallados y los intentos restantes.

## Condiciones para una confirmación futura

- Precisar qué mecanismo se defenderá como contribución y qué limitación BTC cuantificada pretende resolver. La hipótesis de penalización/saturación no se transfiere a la referencia aditiva convencional.
- Fijar, antes de sus experimentos, la razón del mecanismo, la métrica, el subconjunto, la magnitud, las predicciones secundarias y los criterios de éxito, fracaso y resultado no concluyente. Las observaciones conocidas se etiquetan posteriores.
- Separar desarrollo y confirmación de manera real. Los 608 días ya evaluados y la reserva histórica expuesta no recuperan cegamiento al volver a puntuarlos.
- Elegir al menos tres conjuntos adicionales antes de conocer su desempeño; fijar derechos, tarea, fuentes, particiones y análisis. Si guían el diseño, se reportan como desarrollo.
- Conciliar Nd global y esfuerzo, cerrar HPO/política final comparables, dimensionar potencia por contraste y conjunta y fijar semillas, hipótesis y familias antes de congelar. Diez réplicas observadas no prueban potencia.
- Registrar fecha y tag reales de la versión final. Un cambio posterior de arquitectura, pérdida o espacio de ajuste requiere nueva versión y conserva el historial.
- Completar la ablación del mecanismo defendido: retirar/sustituir componentes, base sin todos, cada componente añadido solo, efectos e interacciones con IC, control de tamaño si cambia capacidad y Holm en una familia separada de la principal. M10/M01 del primer diseño todavía no tienen evaluación aislada de mercado; retirar atención convencional no crea una tercera innovación.
- Vincular la propiedad formal con evidencia numérica del código correspondiente y con un beneficio predictivo medible. Las propiedades de los diseños históricos no se trasladan automáticamente a otro modelo, y una ablación favorable apoyaría contribución sin demostrar por sí sola por qué funciona.

Estas condiciones son un registro de pendientes fechado ahora, **no decisiones supuestamente fijadas antes de los pilotos**. El archivo `requirements_review.csv` relaciona D1–D5 y 7.2–7.4 con la evidencia y el cierre que falta.

## Fuentes inmutables verificadas

Los resúmenes se verificaron por SHA256 antes de transcribir sus métricas. No se modificaron los archivos fuente, no se deserializaron todos los checkpoints y no se entrenó ni infirió sobre mercado para producir este registro.

| Etapa | Resumen por commit | SHA256 del resumen |
|---|---|---|
| v1 | [runs/desarrollo_v1/summary.json](https://github.com/jassan-bit/DeepLearning_202630/blob/f7826d120f3a6eb8e9a0952dd9842cd77f152f34/runs/desarrollo_v1/summary.json) | `499247e464e356364038035d48ee67a3d9ccf5791c28d5f40fe3b51adb160f90` |
| v2 | [runs/innovacion_v2/summary.json](https://github.com/jassan-bit/DeepLearning_202630/blob/1ca029e869a517cb8de704e573c78ca1e0cbad3c/runs/innovacion_v2/summary.json) | `4f0a08f68b0d4da2bd43cf1ad72b75fb503a565afc21a825e02ffd784bd3cb23` |
| moments | [runs/momentos_v1/summary.json](https://github.com/jassan-bit/DeepLearning_202630/blob/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1/summary.json) | `94e820f4bfcf3117303cb1068265231e71e7c2b6e1a316d5d7b97390019f4f1c` |
| estabilidad | [runs/estabilidad_aditiva_v1/summary.json](https://github.com/jassan-bit/DeepLearning_202630/blob/510c3bddfdc9926322e3210ac61ee1f5eeffad2e/runs/estabilidad_aditiva_v1/summary.json) | `7c90c800adcca2218ea94553081cbcf744f3b9d9ff657ece17402df40be8f176` |

Otros registros preservados: `experiments/e3_bloque_c_20261115/design_log.md` conserva el corte inicial del 15 de noviembre (cuatro variantes artificiales y Nd mercado=0 en ese corte); los planes de v2 y momentos conservan su estado original previo a las corridas; `book_e3/_static/e3_revision_bc_20261117/c_software_review.json` identifica las 47 pruebas CPU de modelo y entrenamiento sintético, no 47 pruebas exclusivas de la aditiva. Los capítulos 6–8 distinguen las arquitecturas de 1211, 1210 y 1236 parámetros.
