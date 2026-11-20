# Declaración propuesta para revisión y defensa del autor

Texto asistido por IA. Se formula una contribución verificable; el dictamen
bibliográfico debe resolver separadamente su novedad para la aplicación.

> Frente a la referencia CNN–LSTM con atención y cabeza escalar positiva,
> sustituimos la predicción directa de exceso de sigma7 por una ley condicional
> finita de la magnitud del retorno que falta en la ventana solapada. La red
> aprende tres átomos positivos ordenados y sus masas, transforma cada átomo
> mediante `T_A(u)=sqrt(A+(6/49)u²)` y después promedia. Añadimos al MSE de la
> predicción de volatilidad un CRPS de la magnitud no observada al origen,
> reconstruida de la etiqueta sólo durante el ajuste. La modificación aborda
> que el segundo momento de esa magnitud no determina la media de la
> volatilidad transformada y proporciona una representación distribucional que
> la cabeza escalar no identifica. No afirmamos que una red aditiva flexible
> no pueda representar esa misma media puntual.

La diferencia se demuestra en los operadores y en sus gradientes; no se
atribuye novedad a CNN, LSTM, atención, distribución discreta, softmax, CRPS,
Jensen o actualización de varianza. Tampoco se atribuye una nueva fuente de
información a la etiqueta auxiliar U, que deriva del mismo Y y A.

La pretensión limitada de adaptación consiste en **cómo se define la ley a
aprender, cómo se conecta con esta etiqueta solapada y cómo se entrena esa
conexión**. Debe precisarse frente a los antecedentes que ya hagan esos tres
pasos para la misma tarea, no sólo frente a modelos con distinto nombre.
Haber cambiado el código no resuelve por sí solo ese contraste.

La propiedad formal asegura soporte y una esperanza exacta para la ley finita
aprendida, con cotas parciales y propiedades oracle bajo hipótesis explícitas.
No certifica calibración, convergencia o superioridad general.

El piloto conservado de 30 ajustes es evidencia de desarrollo: `atom_crps`
obtuvo 2,70 % más RMSE que `atom_mse_only` y no superó la heurística previa.
El resultado se declara sin promocionar la ablación ganadora. La contribución
puede evaluarse como propuesta y análisis crítico; su utilidad predictiva
prevista no queda acreditada por ese piloto.

## Afirmaciones a retirar si aparecen en una defensa

- “Inventamos CRPS/mezclas neuronales/Jensen/la actualización de varianza”.
- “El segundo momento jamás puede dar una predicción útil”.
- “V3 es más expresiva que cualquier red aditiva”.
- “U es una etiqueta independiente o el retorno firmado futuro”.
- “El CRPS de U es exactamente el CRPS de sigma7”.
- “La media es exacta para el mercado porque la suma de tres términos es exacta”.
- “La prueba oracle garantiza menor RMSE que la aditiva entrenada”.
- “El fracaso del piloto prueba que no existe novedad”.
- “Los 796 trabajos acreditan V3” o “la mejor ablación se convierte en la hipótesis original”.

## Decisiones mínimas del autor

1. Confirmar que ésta es la formulación de contribución que entiende y defiende.
2. Identificar las fuentes efectivamente usadas al diseñar, distinguiéndolas
   de antecedentes encontrados después. No reescribir su fecha como preregistro.
3. Precisar el bosquejo ambiguo Faithful mediante ecuación, pérdida, código o
   declaración de que no llegó a implementarse; no inventar la respuesta.
4. Mantener o limitar la pretensión de novedad según el contraste específico
   final, sin exigir entrenamiento nuevo para redactar una afirmación honesta.
