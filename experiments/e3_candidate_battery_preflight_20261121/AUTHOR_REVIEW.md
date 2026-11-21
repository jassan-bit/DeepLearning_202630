# Material para revisión auténtica del autor y de metodología

Estado: preparado, **sin decisión firmada ni aceptación imputada**. Fecha del material:21 de noviembre de 2026, America/Bogota. Asistencia de IA declarada. Completar campos describe una actuación futura; este formulario no la acredita por sí solo.

## A. Decisión del autor sobre la contribución

**Recomendación técnica preparada:** conservar V3 como objeto documental de la adaptación y sus límites y adoptar el informe alternativo de benchmark/resultados adversos. No hay candidata habilitada para confirmación. La alternativa está permitida por guía §14.5.9p.47 y no acredita R3.

Material de revisión: CANDIDATE_DECISION.md, las tablas originales del piloto y el dictamen específico§6.18. No se solicita autorizar entrenamientos.

- Persona que decide y relación con el trabajo: pendiente.
- Fecha real de revisión: pendiente.
- Decisión y fundamento: pendiente.
- Si propone otra adaptación: identidad, antecedente más próximo, limitación, ecuaciones, predicción falsable y qué evidencia motivaría abandonarla: pendiente.
- Reconocimiento explícito de que no hereda los resultados/dictámenes de otra versión: pendiente.

## B. Revisión estadística y correspondencia con la guía

El material concreto es STATISTICAL_BATTERY.md y BIAS_BOUND_DERIVATION.md. La identidad exacta de Jensen está separada de su uso estadístico.

El concepto debe responder, con fundamentos y cambios trazables:

1. ¿El blanco Psi=E_S sqrt(E_Z q|S), condicionado al entrenamiento fijado, corresponde a la afirmación pretendida? Si se prefiere un blanco condicionado al calendario, delimitar qué conclusión deja de cubrir.
2. ¿Qué supuestos permiten una cota útil de B_m y de su incertidumbre? ¿Cómo se controlan medias cercanas a cero, la esperanza sobre semillas y las autocovarianzas omitidas?
3. ¿Qué aproximación justifica el intervalo para tau=E(thetahat), su covarianza cruzada y el resto? La ampliación [L−B_c,U+B_j] no prueba esa aproximación.
4. ¿Cómo se reparte alpha_root+eta y se derivan p conservadores/IC coherentes para toda la familia? No basta reutilizar p históricos.
5. ¿Dos procedimientos pareados más un MCS que cubra todos los óptimos y retenga sólo la candidata satisfacen la guía, o hace falta un tercer procedimiento por contraste? Fundamentar con§8.4,Cuadro 4 y§8.7.1.
6. ¿El plan de validación tiene verdades analíticas, escenarios, precisión, presupuesto y criterios suficientes para el alcance delimitado? Una respuesta positiva no certifica el desempeño de mercado.

- Identidad, formación/rol y fecha del revisor: pendiente.
- Objeciones, correcciones y alcance aceptado: pendiente.
- Estado de método y tercer procedimiento: pendiente; no se completa por defecto.

## C. Contrato previo de un eventual piloto separado

Sólo se usa si existe una propuesta concreta motivada; no reactiva V3/V4. Antes de resultados debe fijar identidad/versión, datos de desarrollo y exposición, comparadores, hipótesis, métrica, regla de continuación/abandono, semillas, presupuesto y registro. Un piloto puede servir para estimar dispersión: no se le exige potencia final antes de producirla. Una mejora observada no acredita R3.

Actualmente propuesta, datos, piloto y presupuesto permanecen sin seleccionar. No hay código Colab de entrenamiento habilitado por este documento.

## D. Lista para una futura autorización de campaña

Antes de solicitarla deben existir: candidata con continuidad defendible y evaluación R1/R2 correspondiente; familia completa; batería admitida y traducción de la regla; panel principal y tres adicionales admitidos; piloto compatible y diseño suficiente R/T; potencia por contraste y conjunta reportada; hipótesis/márgenes/decisiones; registro público anterior a definitivos; paquete de trabajos, coste y persistencia concretos.

Todos esos actos deben tener fechas y evidencias reales. El estado aquí es **pendiente, sin autorización de campaña solicitada o concedida**. La sustentación 4,5 indicada por el autor no constituye una defensa observada.
