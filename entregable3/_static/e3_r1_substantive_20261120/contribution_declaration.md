# Declaración precisa de contribución y límites

IA declarada; revisión documental del 20 de noviembre de 2026. Esta declaración
describe el código congelado y su contraste actual. No se atribuye al diseño
una lectura que sólo se realizó después, ni se convierte en preregistro.

## Pretensión que se defiende

Frente a la estimación neuronal de densidades financieras por mezclas
de Schittenkopf et al. (1998), las redes de regresión de desviación móvil de
De Stefani et al. (2017), las posiciones y masas aprendidas de FQF
(Yang et al., 2019) y la mezcla neuronal GARCH de Feng et al. (2023), se adapta
la lectura y el objetivo al componente todavía desconocido de una etiqueta
de volatilidad solapada. Se aprende una ley finita de la magnitud
`U=|r_futuro-m6|`, se transforma cada átomo por
`T_A(u)=sqrt(A+(6/49)u²)` y sólo después se promedia. El ajuste combina MSE
de la media de sigma7 con CRPS de U normalizado, con peso 0,1 fijado en el plan.

El fenómeno es pronóstico de volatilidad financiera; la tarea es la desviación
**poblacional** de siete retornos, seis ya observados y uno futuro. No se reclama
novedad por Bitcoin, por siete en lugar de otra longitud o por `ddof=0`.
El aporte defendido es el acoplamiento computacional y de entrenamiento,
ausente en las partes de los antecedentes cercanos examinadas y delimitadas
en las matrices. La revisión no prueba ausencia de cualquier antecedente mundial.

## Diferencia verificable

Con `A=sum_{i=1}^6(ri-m6)²/7`, el grafo de V3 produce posiciones positivas
ordenadas `a_k`, masas `p_k` y `mu=sum_k p_k T_A(a_k)`. La aditiva produce
un exceso escalar sobre `sqrt(A)` y usa MSE. FQF aprende fracciones y valores
mediante objetivos de Bellman/cuantiles; no es un antecedente inexistente ni
se le atribuye el objetivo supervisado de V3. Las mezclas financieras
comparadas estiman otras densidades/volatilidades; no se supone que su
varianza condicional sea la misma etiqueta móvil con seis retornos conocidos.

La ley finita y CRPS son herramientas existentes. Además, cambiar de U a Y
permite escribir CRPS_U como un CRPS ponderado sobre la ley de Y: la revisión
no reclama una regla de puntuación inédita. Es una adaptación de salida,
representación y pérdida para esta tarea, no una demostración de que una
red aditiva flexible sea incapaz de representar la misma media puntual.

## Limitación abordada y evidencia que la distingue de utilidad

Una predicción puntual no identifica una ley del componente futuro; tampoco
su segundo momento determina en general la media de la transformación
no lineal `T_A`. Esa limitación matemática motiva la lectura distribucional.
No significa que la referencia aditiva sea incorrecta o incapaz de pronosticar.
U se reconstruye de Y y A sólo como etiqueta: no aporta información futura
independiente y no se suministra al forward.

Se demuestra soporte y esperanza exacta para la ley finita aprendida. Las
cotas de Jensen/Taylor exigen sus hipótesis; la identidad de riesgo sólo
compara con el plug-in bajo la ley verdadera. No se demuestran calibración
de mercado, convergencia del optimizador o superioridad frente a la aditiva.

El piloto de treinta ajustes, con tres semillas finales por receta, tuvo un
resultado adverso para la receta completa: **2,70 % peor que atom_mse_only**.
No superó la regla de continuación del 2 %. Eso no refuta novedad, pero sí
impide presentar como confirmada la utilidad predictiva prevista del conjunto
completo. No se sustituye el objetivo primario por una ablación ganadora.

## Alcance de la acreditación documental

La revisión independiente por IA puede apoyar R1 de esta adaptación concreta
conforme a la guía, con cobertura y objeciones visibles. Ese concepto no
acredita R1 de la aditiva, no convierte los 796 trabajos en evidencia de V3,
no observa la comprensión del estudiante y no acredita R3. No es un acta
del profesor ni una asignación de calificación.
