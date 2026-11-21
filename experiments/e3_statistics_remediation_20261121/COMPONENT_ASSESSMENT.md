# Valoración orientativa del componente del 15 %

**Propuesta: 3,0/5, frente al 2,5 anterior. Continúa en desarrollo.** Es una evaluación técnica asistida por IA del componente «Pruebas de hipótesis y adecuación», no una nota del docente, una equivalencia universal del nivel cualitativo o una recalificación global del proyecto.

| Aspecto evaluado | Evidencia obtenida | Consecuencia para la valoración |
|---|---|---|
| Hipótesis, estimando y familias | Riesgo medio por semilla separado de ensemble/MSE, nulo de igualdad separado de δ; Holm38, Holm6 y Holm88 explícitos. | Mejora metodológica verificable. No crea tres procedimientos primarios acreditados. |
| Validación del método global | 640 réplicas frescas después de publicar plan y código. MCS bootstrap-t satisface los tres filtros nulos acotados, con controles pareados conservados. | Avance científico parcial; no certificación de exactitud nominal o validez en mercado. |
| Error familiar de la batería | Semillas: 14/128 rechazos falsos del subconjunto de dos métodos, IC95 % 6,11–17,67 %, nominal 5 %. | La batería primaria sigue sin aprobarse. Impide una valoración de cierre satisfactorio o sobresaliente. |
| Potencia | Se distinguen 112/128 rechazos globales, 39/128 conjuntos con candidato único y 18/128 intersecciones de dos p-valores. Auditoría de 71 fuentes/486 escenarios, contrato y vacíos de varianza explícitos. | No existe dimensión final o potencia ≥0,90 por contraste bajo la regla completa; reportar la conjunta no exige universalmente 0,90. |
| Adecuación | 88 diagnósticos y 44 IC condicionales de sesgo desde predicciones guardadas; 13 rechazos MZ, curvas y sensibilidad documentadas. | Se conoce mejor la inadecuación o incertidumbre pendiente. No rechazar LB/ARCH con T55 no acredita adecuación. |
| Alcance y transparencia | Se conservan resultados adversos, controles, fuentes antes de ejecutar y límites de inferencia. Sin nueva candidata, entrenamientos o predicciones de mercado. | Se reconoce rigor y aprendizaje verificables, sin premiar el volumen de código ni la significancia favorable. |

La recomendación independiente está en [qa/COMPONENT_GRADE_REVIEW.json](qa/COMPONENT_GRADE_REVIEW.json). Su [adenda de evidencia](qa/COMPONENT_GRADE_EVIDENCE_ADDENDUM.json) vincula la aclaración posterior del alcance de `18/128` sin cambiar números, fuentes de validación o valoración. La calificación propuesta no ignora la inflación detectada ni convierte diagnósticos retrospectivos en confirmación.

No justificaría 3,5–5,0 con estos resultados: falta una batería primaria válida y pertinente, potencia de toda la regla, datos admitidos y adecuación de una candidata viable. El NO-GO V3 y el abandono V4 permanecen.

Si esta valoración se incorpora manteniendo los otros componentes fijos, su aporte sería `3,0 × 0,15 = 0,45` sobre cinco; antes era `2,5 × 0,15 = 0,375`. La diferencia aritmética de **0,075** no promete un incremento de la nota final ni modifica los topes condicionales R1/R2/R3.

**La sustentación se conserva en 4,5 como dato proporcionado por el autor. No se afirma haber observado la defensa. R3 no se acredita.**
