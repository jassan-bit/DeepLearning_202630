> Actualización: tras confirmar el contexto académico se definió un criterio operativo futuro del 5 % frente a ambas referencias. Véanse academic_margin_README.md y academic_margin_policy.json. El texto siguiente conserva el alcance histórico de los primeros escenarios; no es una política previa al test ya observado.

# Escenarios para un modelo avanzado futuro

La referencia neuronal es MLP/D7, elegida con CV antes del test. Se usan su RMSE media y DE entre semillas de 2024 como anclas históricas. La persistencia se muestra por separado porque tiene menor error observado; superar solo la referencia neuronal no basta para atribuir utilidad frente a esa referencia sencilla.

Los márgenes relativos de 1, 2,5, 5 y 10 % son escenarios operativos, no una tolerancia financiera validada ni un margen seleccionado retroactivamente. Debe documentarse qué diferencia tiene valor para la aplicación. No se ha entrenado aquí un modelo avanzado.

Cada escenario declara la DE de ese modelo como 0,5/1/2 veces la DE histórica del baseline y correlación pareada 0/0,5/0,9. Son supuestos ilustrativos, no estimaciones de un modelo desconocido ni un conjunto exhaustivo de correlaciones posibles. Se calcula R para detectar una diferencia δ frente a cero con prueba bilateral t de diferencias normales, potencia objetivo 0,80 y α de planificación 0,025 (Bonferroni conservador para dos comparaciones futuras). La inversión de t no central añade la corrección de tamaño finito que omite la aproximación normal.

Detectar una diferencia δ frente a cero no equivale a demostrar que la mejora excede δ. Para contrastar una frontera práctica debe definirse una alternativa por encima de esa frontera y planificar con su separación. Tampoco puede deducirse potencia alcanzada de la tabla: faltan la variabilidad/correlación reales, un piloto prospectivo y una evaluación nueva independiente.

Los resultados son condicionales a aleatoriedad de semillas sobre un calendario; no representan automáticamente incertidumbre entre mercados. El margen aplicado permanece pendiente mientras no exista una justificación documentada del objetivo o decisión que lo exige.
