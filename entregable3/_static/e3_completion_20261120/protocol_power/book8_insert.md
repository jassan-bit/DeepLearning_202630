## V3: decisión de continuación y preparación prospectiva

El piloto V3 de treinta ajustes conserva su resultado adverso: `atom_crps`
fue **2,70 % peor que `atom_mse_only`** en la media de RMSE individuales de
tres semillas sobre 55 fechas de desarrollo. La heurística original de
mejora del 2 % no se superó. La decisión científica actual es **NO-GO para
confirmación**; no se cambia la regla ni se promociona automáticamente λ=0.
El dictamen documental favorable de R1 no transfiere a V3 los 796 trabajos
de la aditiva y no acredita R3.

Se preparó por separado una pregunta pequeña: cómo transfieren las cinco
recetas y los dos controles del piloto a D01–D03 usando sólo los quince
pesos finales, las escalas BTC y las particiones originales. La decisión
diagnóstica debe quedar fijada antes de sus nuevas predicciones. Es un
diagnóstico de transferencia sin ajustes ni calibración; los calendarios
ya expuestos impiden llamarlo confirmación independiente. Su ejecución y
sus resultados se documentan en el recibo específico, no se presuponen aquí.

El [protocolo preparatorio](../_static/e3_completion_20261120/protocol_power/protocol.DRAFT.md)
define la familia futura completa: **20 sistemas, 19 rivales y 57 componentes
de prueba**, incluyendo E2, aditiva y controles físicos. Las tres ablaciones
se mantienen en familia separada. Es un DRAFT posterior al piloto: no tiene
tag de congelamiento, datos confirmatorios admitidos, presupuesto autorizado
ni motor V3 listo para una nueva campaña.

La guía §8.7.2 pide al menos cinco semillas piloto para estimar variabilidad
por rival. Aquí hay tres y, entre los seis controles del piloto, sólo aditiva
y dos analíticos pertenecen a la familia principal; faltan los dieciséis
rivales E2. Por ello **no se fija R final ni se afirma que diez semillas
basten**. El [cálculo CPU ejecutado](../_static/e3_completion_20261120/protocol_power/results/planning_receipt.json)
lee exclusivamente pronósticos V3 ya guardados: 162 escenarios de semillas
y 324 temporales, con separación de inicialización y calendario. No hubo
entrenamiento, nuevas predicciones ni potencia observada posterior.
Son **486 evaluaciones analíticas y cero remuestreos Monte Carlo**.

Las alternativas matemáticas `1,25δ`, `1,5δ` y `2δ` se definen sobre el margen
heredado como referencia, con distancia de potencia **g1−δ** y nivel
conservador `.05/57`. La incertidumbre con tres semillas es amplia; las
sensibilidades no establecen la potencia de tres pruebas temporales ni
convierten un número de fechas hipotético en observaciones independientes.
Se informa la conjunta bajo supuestos explícitos y sus límites; la guía
exige reportarla, no exige conjunta 0,90.

Antes de una confirmación se necesita una nueva decisión científica fechada,
datos genuinamente no expuestos, justificación de márgenes por escala,
batería F2 validada y alineada con la media individual de RMSE, potencia de
la familia completa, presupuesto comparable y un tag anotado real previo a
resultados. Bootstrap estacionario, DM por influencia y una extensión GW
son propuestas sujetas a revisión: nombrarlas no acredita su admisibilidad.
Holm se propone global sobre 57 y el máximo de los tres p ajustados gobierna
cada rival; los diecinueve deben pasar. La adecuación, calibración propia,
robustez y costes conservan sus requisitos.

El [preflight CPU](../_static/e3_completion_20261120/protocol_power/preflight.py)
se verificó con cinco pruebas de bloqueo, hashes y reanudación. Devuelve
`BLOCKED_PREPARATION_ONLY` con los pendientes explícitos y no tiene una ruta
que entrene. La [matriz de requisitos](../_static/e3_completion_20261120/protocol_power/requirements_matrix.csv)
permite resolver esas decisiones sin repetir las campañas anteriores.
Se declaró la asistencia de IA en esta preparación.
