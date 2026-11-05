# Nueva fase D7: diseño previo al HPO, piloto y evaluación reservada

`design.json` define una nueva fase del proyecto GRU–LSTM–TCN. Se publica antes de ejecutar su búsqueda de hiperparámetros. No modifica ni atribuye carácter prospectivo a los resultados históricos anteriores.

La búsqueda exhaustiva evalúa ocho configuraciones por familia, dos semillas y los mismos diez bloques forward chaining de enero a octubre de 2024. Sus 960 ajustes tienen reglas compartidas. Se eligen tres recetas y un baseline de desarrollo antes del piloto.

El piloto utiliza diez semillas distintas y 60 días de desarrollo, del 1 de noviembre al 30 de diciembre de 2024. Sus modelos se ajustan antes de noviembre y permanecen fijos durante esos 60 días. El piloto precede a las ejecuciones definitivas de esta fase; el calendario de desarrollo ya era conocido en el proyecto.

El dimensionamiento calcula el menor R común, como mínimo diez, con potencia marginal objetivo de 0,80 para cada comparación pareada bajo los supuestos registrados. Utiliza t no central, alfa de planificación 0,05/3, margen absoluto 0,00025772842664981106 y un límite superior simultáneo del 80 % de las desviaciones pareadas del piloto. Se reportan sensibilidades y los supuestos de normalidad y transferencia de dispersión. No se atribuye potencia temporal de 0,80 a los 60 días ni potencia conjunta de 0,80.

La reserva nueva es el 1 de septiembre–30 de octubre de 2026, 60 días. Sus archivos ya fueron adquiridos y recorridos para verificar integridad, timestamps y esquema; no se habían utilizado para construir el objetivo, seleccionar modelos ni calcular resultados del proyecto. Esta fase asigna explícitamente la reserva. La evaluación histórica nueva no es una predicción emitida en vivo ni otro mercado independiente.

Las decisiones estadísticas, comparaciones, multiplicidad, margen y reglas de parada quedan fijadas antes de derivar objetivos o evaluar esta reserva. Todos los checkpoints y escaladores definitivos deben estar cerrados antes de su acceso. Un resultado adverso o inconcluso se conserva y se publica.

El código activo es `scripts/run_gru_lstm_tcn_planned.py`, con estadísticas y datos en los módulos nuevos `planned_study_statistics.py` y `planned_study_data.py`. El registro guarda hashes de código, diseño, datos de desarrollo y metadatos de adquisición. Los archivos privados de ejecución se escriben fuera del repositorio. Los agregados se publican después, con procedencia y cronología.

El piloto, HPO y dimensionamiento se refieren a **D7**. Las entradas D14/21/28 del benchmark anterior conservan su alcance exploratorio y no heredan los resultados de esta fase.
