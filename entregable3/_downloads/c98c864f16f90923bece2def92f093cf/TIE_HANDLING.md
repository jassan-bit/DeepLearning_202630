# Enmienda matemática anterior al diagnóstico fijo

La revisión independiente identificó una dificultad antes de ejecutar la
cuadrícula: CRPS contiene valores absolutos y no es diferenciable cuando un
átomo coincide con una observación. Transformar y normalizar en punto flotante
puede mover esa coincidencia a uno u otro lado por errores del orden de 1e-16.
No es una diferencia de geometría atribuible a la coordenada en un punto
diferenciable. La convención cero de PyTorch en una igualdad exacta tampoco
selecciona automáticamente el mismo subgradiente en dos expresiones redondeadas.

La enmienda previa `tie_amendment_before_results.json` conserva la cuadrícula
de 144 casos, todas sus filas, scores y gradientes de cinco parámetros. Su SHA256
es `6e42edc846898310ad1badab06dd64dfe12bf5e31d3716f5f2946406714076c4`.
El plan original no se reescribe. Su código inicial y el primer log de software
quedan bajo `before_tie_amendment/`.

La regla fijada marca como sensible a empates cualquier par átomo/observación,
en las tres coordenadas normalizadas, cuya separación sea a lo sumo
`64 * epsilon_float64 * max(1,abs(atomo),abs(observacion))`. No altera el score,
la pérdida de la candidata o el cálculo autograd. Sólo impide atribuir una
diferencia angular a la coordenada en esos puntos; informa sus denominadores
por separado. Los resultados crudos de esos casos siguen en el CSV.

Se verifica la identidad A=0 de scores sobre todos sus casos. La identidad de
gradientes A=0 y la colinealidad entre las dos normalizaciones sigma se juzgan
sólo donde el diagnóstico está separado de esos empates. Los puntos marcados
no se interpretan como una selección favorable ni una evidencia adversa de V4.

Un primer guard de software usó `torch.ones` float32 en una comprobación de
coseno a trece decimales. Falló por precisión de esa fixture; los otros once
guards pasaron. La corrección hace explícito float64, la precisión declarada
del diagnóstico. Conserva el fallo y no relaja tolerancias ni modifica objetivos.

El programa exige un recibo del anclaje público previo facilitado por el
orquestador y liga sus hashes al plan/enmienda. La ejecución offline lee ese
recibo: la verificación pública efectiva corresponde al orquestador, no a una
supuesta autenticación de red realizada por el script. Es registro de un
diagnóstico de desarrollo, no preregistro confirmatorio ni acreditación de R3.
