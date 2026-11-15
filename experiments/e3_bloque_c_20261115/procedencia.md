# Procedencia del candidato de E3

El candidato desarrolla la formulación de `book_e3/sections/07_diseno_preliminar.md`: CNN causal y LSTM unidireccional con una cabeza que combina penalización de discordancia respecto a un promedio causal y saturación radial de valores. La implementación de `src/original_model.py` se escribió con asistencia de IA desde esas ecuaciones, usando operaciones tensoriales y capas primitivas de PyTorch. No importa una arquitectura predefinida ni usa pesos preentrenados, código de autores o una descarga como núcleo del mecanismo.

La atención por contenido, CNN, LSTM, las cotas Lipschitz y la saturación no se presentan como invenciones. La saturación es el gradiente de una función con forma pseudo-Huber radial; la identidad algebraica está explicitada en la revisión, sin atribuir una fuente histórica que no se haya verificado. La matriz de siete trabajos cercanos conserva fuentes y niveles de lectura. Kim et al. (2021) y Dasoulas et al. (2021) se examinaron focalmente como originales desde el repositorio editorial oficial; no se copiaron sus implementaciones ni se redistribuyen los PDF.

La diferencia candidata es una regla de atención específica, no sólo aplicar CNN-LSTM a otro dataset. Su novedad exacta sigue pendiente: las consultas nuevas a tres índices fueron bloqueadas y no hay verificación humana externa. No se redacta una declaración de ausencia de antecedentes.

Las pruebas sobre datos artificiales comprueban software y una propiedad de la cabeza latente. No son entrenamiento en Bitcoin, HPO, ablaciones de desempeño, confirmación de superioridad ni validación en tres datasets adicionales. La prueba formal no incluye todavía una cota del codificador ni del preprocesamiento desde OHLCV crudo.

El autor del proyecto debe revisar, comprender y poder modificar cada parte del código y la demostración. La asistencia de IA y el alcance de las lecturas deben declararse en la entrega. La existencia del archivo no certifica por sí sola R1, la defensa académica de R2 o R3.
