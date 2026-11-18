# Hipótesis y controles futuros de la adaptación de soporte

**Borrador posterior, creado el 18 de noviembre de 2026. No es un preregistro, no está congelado y no habilita una campaña.** Especifica una pregunta de utilidad; un resultado favorable no demostraría por sí solo novedad. No modifica modelos, pesos, métricas, los 48 contrastes ni el calendario histórico.

## Limitación identificada

El soporte observable del objetivo no se trató de la misma manera en los rivales históricos y la cabeza aditiva. El control posterior ya publicado muestra que esa diferencia explica descriptivamente una parte importante de la brecha; no permite cuantificar una contribución causal. La comparación propuesta debe dar a todos los tratamientos el mismo acceso al piso, al escalado y al conjunto de ajuste.

## Hipótesis falsable propuesta

Una cabeza aditiva entrenada dentro del soporte podría reducir el riesgo RMSE frente al mismo codificador con cabeza libre y proyección final común, bajo igual información, capacidad, particiones, presupuesto y regla de parada. El estimando sería la diferencia de RMSE del control menos la del candidato, agregada por una regla fijada antes de confirmar. El margen práctico `delta_RMSE` queda **sin fijar** hasta el diseño y piloto de potencia. No se reutiliza automáticamente la heurística del 2 % del desarrollo como margen confirmatorio.

Predicciones secundarias propuestas: el beneficio no debe desaparecer al igualar el piso de los controles; debe mantenerse en una métrica secundaria fijada y no venir acompañado de degradación inaceptable en escenarios observables al pronosticar. Se propone un estrato de piso alto definido con el cuantil75 de `b/a_train` del ajuste, no con objetivos futuros. Umbrales, métricas, severidades y tolerancias siguen pendientes de congelar antes de observar resultados confirmatorios.

## Controles que separan las explicaciones

| Control futuro | Regla | Pregunta aislada |
|---|---|---|
| C0: sólo soporte | `b` sin aprendizaje. | ¿La complejidad neuronal añade utilidad frente a la información ya conocida? |
| C1: exceso constante | `b+d_fit`, donde `d_fit` se calcula sólo con ajuste. | ¿Se necesita un exceso condicionado por X? |
| C2: cabeza libre y proyección común | Mismo codificador y lectura; ajustar la salida libre y evaluar `max(pred,b)`. | ¿Hay beneficio de entrenar la restricción frente a imponer el mismo soporte al final? |
| C3: guarda exponencial conocida | Mismo codificador, `b+a*epsilon+a*exp(z)`, con inicialización de exceso comparable. | ¿El resultado depende del enlace positivo, cuyo rango admisible es compartido? |
| C4: referencia aditiva | Mismo codificador, `b+a*(epsilon+softplus(z))`. | Tratamiento candidato que se conserva. |
| C5: contexto sin atención aprendida | Cabeza aditiva con una agregación temporal predeterminada; inventariar la diferencia de parámetros. | ¿El pooling aprendido añade utilidad sobre el soporte y el codificador? |

Estos controles no son modelos nuevos ya entrenados. La capacidad y el coste deben medirse: no se declarará igualdad exacta si se altera el número de parámetros o el trabajo de optimización. Conservar parámetros inactivos no iguala capacidad efectiva. Las comparaciones de enlaces requieren registrar saturación, desbordamientos, gradientes y fallos, con reglas previas que no favorezcan un enlace después de conocer el resultado.

## Condiciones antes de ejecutar

1. Fijar identidad del mecanismo que se defenderá y resolver su revisión bibliográfica.
2. Declarar exposición a calendarios conocidos y reservar una evaluación futura admisible.
3. Fijar particiones, escaladores, información del piso, capacidades, presupuesto total, semillas, paradas e inicialización de cada control.
4. Registrar margen práctico, tres procedimientos pertinentes, familia de contrastes, dependencia temporal, multiplicidad, potencia por contraste y conjunta y número de ejecuciones derivado del piloto.
5. Congelar escenarios y análisis secundarios; registrar todas las variantes en la bitácora con fechas reales.
6. Mantener la evaluación científica distinta de la revisión de novedad. Si la diferencia sólo consiste en una guarda conocida aplicada al objetivo, una mejora predictiva no la transforma automáticamente en adaptación sustantiva.

La comprobación de la identidad escalar y la consulta de publicaciones realizadas en este expediente no consumen el conjunto de confirmación ni acreditan esas condiciones. El entrenamiento histórico se conserva y no se repite como trámite.
