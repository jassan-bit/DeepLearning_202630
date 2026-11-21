# Resultado del diagnóstico V4 de desarrollo

Se implementó una revisión nueva y separada: `OutcomeCRPSDevelopmentModel`,
versión `e3-overlap-sigma-crps-v4-development`. Conserva el forward y los
1262 parámetros de V3, con idénticos tensores iniciales al mismo seed. Cambia
el CRPS auxiliar de U a sigma y su normalización de sU a sY. El control de
normalización y el control U originales permanecen explícitos; MSE-only no se
promueve a candidata por su resultado anterior.

## Ejecución real y procedencia

El plan local quedó fijado el 21 de noviembre de 2026 a las 00:39:20 UTC
(20 de noviembre a las 19:39:20 de Bogotá). La enmienda matemática de empates
quedó fijada a las 00:46:35 UTC. Ambos se anclaron públicamente en el commit
[`f7def09fc4283cc2d5429c58a372d7cf636bc1b5`](https://github.com/jassan-bit/DeepLearning_202630/commit/f7def09fc4283cc2d5429c58a372d7cf636bc1b5)
antes de ejecutar la cuadrícula. El registro es de desarrollo, no un protocolo
confirmatorio. Las guardas de software se ejecutaron entre el plan local y
el diagnóstico; no se afirma que todas fueran posteriores al anclaje público.

El diagnóstico fijo ejecutó **144 casos a las 00:51:19 UTC**, en CPU con
Python 3.12.14, Torch 2.8.0+cpu y dos hilos, en aproximadamente **0,294 segundos**.
No hubo sorteos Monte Carlo, optimizador, ajustes neuronales o datos de mercado.
Las expectativas de las dos leyes finitas se evaluaron exactamente mediante
sus pesos. Las pruebas de software sí incluyen forwards con entradas artificiales;
no son predicciones de mercado ni una reproducción humana.

El CSV conserva las **144 filas**, todos los scores y gradientes crudos de
cinco parámetros. **72 casos están marcados como sensibles a empates** y sus
valores siguen disponibles; no se usan para atribuir diferencias angulares.
Los otros **72 son diferenciables según la regla previa** y tienen gradientes
puntuales/auxiliares definidos en esta cuadrícula.

## Qué se observó

| Comprobación o comparación fija | Resultado |
|---|---:|
| Identidad de scores normalizados para A=0: error absoluto máximo | 2,78e-16 |
| Identidad de gradientes para A=0 en casos diferenciables: error máximo | 1,39e-16 |
| Casos diferenciables con A>0 y dirección U/sigma no colineal | 50 de 60 |
| Diferencias de dirección atribuibles sólo a normalización positiva | 0 |
| Mejor alineación del gradiente auxiliar sigma con la MSE | 49 de 72 |
| Peor alineación del gradiente auxiliar sigma con la MSE | **11 de 72** |
| Igual alineación dentro de la tolerancia fijada | 12 de 72, todos A=0 |

Por A, las comparaciones mejor/peor/igual fueron: cero, 0/0/12;
1e-8, 6/6/0; 0,001, 8/4/0; 0,1, 11/1/0; 1, 12/0/0; y 100, 12/0/0.
Los denominadores son doce casos diferenciables por A, con otros doce marcados
como sensibles a empates. La tabla completa permite revisar también cada
gradiente y la pendiente puntual asociada al objetivo conjunto.

La diferencia angular distingue el efecto no lineal de la coordenada del mero
cambio de norma por la escala. El conteo 49/11 no es una prueba de superioridad:
son celdas de una cuadrícula elegida antes, dependientes entre sí, no una muestra
aleatoria de tareas. No produce p-valor, potencia, calibración o RMSE de mercado.
Los once casos adversos impiden inferir dominancia local universal incluso en
este ejercicio. La ausencia de otros casos dañinos tampoco demostraría una cota.

## Decisión que respalda este resultado

La hipótesis de **diferente geometría auxiliar** pasa este control acotado. Se
deja una implementación ejecutable y controles, no una candidata seleccionada
por los conteos. Un eventual piloto de mercado debe tener otra decisión y un
plan previo de desarrollo con presupuesto, recursos, fuentes y controles comunes.
Este resultado no lo autoriza automáticamente, y no abre confirmación.

La diferenciación bibliográfica de V4 requiere revisión propia: puntuar una
distribución de la etiqueta con CRPS es conocido y puede acercar esta formulación
a una regresión de densidad/mezcla ordinaria con soporte restringido. Cambiar
el score no acredita por sí solo una adaptación sustantiva. No se hereda el
dictamen R1 de V3. Una eventual mejora de mercado tampoco resolvería esa objeción
por sí sola.

La receta V3 original conserva **NO-GO**, los 30 ajustes y su deterioro de
2,7043119888 % frente a `atom_mse_only`. La referencia aditiva conserva sus
796 trabajos y la transferencia sus 51 trayectorias/20.400 filas. **R3 sigue
no acreditada.** Ninguna firma o ejecución humana se ha añadido.

## Software y artefactos

Las **14 guardas finales aprobaron**, incluidas inicialización/forward idénticos,
soporte, gradcheck, escalas fijas, properness en comparaciones finitas declaradas,
ausencia de la etiqueta futura en forward y manejo de empates. El primer intento
de doce guardas tuvo un fallo de precisión de una fixture float32; sus fuentes
y log permanecen bajo `before_tie_amendment/`. La corrección fijó float64 sin
relajar tolerancias. Son comprobaciones de software/identidades, no validación
predictiva de una red entrenada.

`synthetic_results/all_scenarios.csv`: **208.875 bytes**, SHA256
`92dbc41b86a26f696568732d36c7d3051bb5b11dcab312f3039ae1aeffe05171`.
`synthetic_results/diagnostic_receipt.json` fija versión, tiempos, hashes,
entorno, denominadores y límites. El programa sólo acepta otro directorio nuevo;
no sobrescribe resultados previos. La ejecución offline lee el recibo del anclaje
verificado por el orquestador y no afirma una comprobación de red propia.
