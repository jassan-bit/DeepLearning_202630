# Entregable 3 de Deep Learning

**Jassan Arteta · Deep Learning 2026-30**

El informe está organizado exclusivamente en los **capítulos 1 a 16 de la guía**, con el desarrollo disponible hasta ahora. La navegación principal reúne los bloques A–G, diseño, evaluación, entrega, sustentación, integridad y conclusiones. Las ampliaciones se integran en esos capítulos; los anexos A–G reúnen plantillas y detalle técnico de apoyo, junto con referencias y trazabilidad. El último anexo conserva reproducibilidad, autoría y bibliografía.

| Sección | Contenido |
|---|---|
| [1 Introducción](sections/01_introduccion.md) | Problema, objetivos, antecedentes y alcance actual. |
| [2 Estándar de profundidad exigido](sections/02_estandar_profundidad.md) | Requisitos teóricos, potencia, semillas y compromisos previos. |
| [3 Panorama general del entregable](sections/03_panorama_general.md) | Bloques A–G, dependencias y estado de avance. |
| [4 Bloque A Correcciones del Entregable 2](sections/04_correcciones_e2.md) | Trazabilidad y auditorías vigentes, con sus pendientes. |
| [5 Bloque B Selección de configuraciones y exportación](sections/05_configuraciones_exportacion.md) | Selección por validación, checkpoints y tres exportaciones verificadas; requisitos de cierre. |
| [6 Bloque C El modelo original](sections/06_modelo_original.md) | Arquitectura evaluada, antecedentes, formulación y evidencia específica para R1/R2/R3. |
| [7 Diseño riguroso del modelo original](sections/07_diseno_riguroso.md) | Hipótesis, disciplina de diseño, sustento teórico y ablación. |
| [8 Bloque D Verificación estadística y evaluación ampliada](sections/08_verificacion_estadistica.md) | Comparación posterior de diez semillas, alcances de incertidumbre, MCS y requisitos aún pendientes de R3. |
| [9 Bloque E Dashboard predictivo en Dash Plotly](sections/09_dashboard_predictivo.md) | Diseño de Contexto, EDA y Modelos, contratos de inferencia y requisitos de calidad. |
| [10 Bloque F Despliegue en Render](sections/10_despliegue_render.md) | Perfil de producción, exportación ONNX pendiente, prueba de humo y contingencia. |
| [11 Bloque G Informe técnico en Jupyter Book](sections/11_informe_tecnico.md) | Estructura científica, mapa del Cuadro 7 y requisitos de reproducción. |
| [12 Lista de entregables](sections/12_lista_entregables.md) | Trece productos, evidencia existente y pendientes de cierre. |
| [13 Sustentación](sections/13_sustentacion.md) | Guion de 20 minutos, demostración y preparación de preguntas. |
| [14 Evaluación y calificación](sections/14_evaluacion_calificacion.md) | Rúbrica de 100 %, topes condicionales y matrices de evidencia, sin nota actual. |
| [15 Integridad académica](sections/15_integridad_academica.md) | Atribución, novedad, uso de IA y trazabilidad de decisiones y resultados. |
| [16 Conclusiones](sections/16_conclusiones.md) | Avances acreditados, límites y capacidades pendientes de demostrar. |

## Estado del proyecto

E2 conserva su benchmark de 16 sistemas y ahora reúne diez semillas para cada una de sus trece redes, más una trayectoria por comparador determinista: **133 predictores sobre las mismas 608 fechas**. La extensión conserva las seis semillas neuronales históricas y añade cuatro sin repetir la selección de recetas. E3 cuenta con diez semillas del candidato **CNN causal → LSTM unidireccional → atención temporal, con cabeza positiva aditiva y 1210 parámetros**. Su RMSE media individual es **0,00293943** en ese calendario conocido; la [tabla de métricas](_static/e3_estadistica_posterior_20261117/metrics.csv) conserva los valores completos.

Los capítulos [6](sections/06_modelo_original.md) y [7](sections/07_diseno_riguroso.md) distinguen ese candidato de los pilotos anteriores. En el piloto v2, el control aditivo obtuvo la menor RMSE de desarrollo y la variante con pérdida auxiliar no superó la heurística del 2 %. El [anexo E](sections/20_cierre_v2.md) conserva ese corte histórico y sus limitaciones en alta volatilidad. La elección posterior de la cabeza aditiva para la comparación **no acredita por sí misma un aporte original**.

La [sección 8](sections/08_verificacion_estadistica.md) incorpora el análisis posterior ya completado: los **48 contrastes primarios** rechazan sus nulas y los 16 rivales cumplen la regla conjunta de los tres procedimientos. Son tres procedimientos distintos con ámbitos de incertidumbre declarados; su pertinencia y combinación deben justificarse para la ruta F2a. La guía no exige literalmente que sus ámbitos sean idénticos. E2 usa 655 ejemplos de entrenamiento y 53 de calibración; E3 usa 532 de ajuste y 116 de validación para parada, sin calibración adicional. No se declara igualdad de presupuestos. El MCS al 90 % conserva diez sistemas, incluido el candidato, y no identifica un ganador único.

**La novedad R1, la defensa académica de R2 y la superioridad R3 siguen sin acreditarse.** La comparación utiliza un calendario histórico conocido y no constituye una campaña confirmatoria ciega ni demuestra potencia o reproducción en mercados independientes. Los bloques A y B conservan sus pendientes de cierre, al igual que el dashboard Render. Las plantillas no se convierten retrospectivamente en preregistros ni firmas. La bibliografía inicial y las revisiones focales posteriores distinguen antecedentes, procedencia y alcance de lectura; se conservan las 37 referencias sugeridas por la guía. El análisis estadístico reutiliza predicciones congeladas, sin nuevos ajustes ni inferencias de modelos.

La [concordancia y los anexos](sections/concordancia_secciones.md) permiten localizar cada evidencia y su capítulo principal. Se conservan las direcciones de las páginas anteriores, ahora identificadas como anexos, para mantener sus enlaces.

La cabeza de momentos y su dictamen se conservan como una línea histórica de desarrollo en los capítulos 6 y 7 y los anexos [F](sections/21_propuesta_momentos.md) y [G](sections/22_dictamen_momentos.md). Sus formulaciones, instrucciones de Colab y cortes de admisibilidad no describen el sistema de la comparación vigente: ésta evalúa la cabeza positiva aditiva. Ese historial tampoco selecciona la adaptación por momentos como aporte original.

El [anexo final](sections/cierre_reproducibilidad.md) distingue conclusiones provisionales, evidencia de software, reproducción científica pendiente y declaración de autoría sin firmas. La rúbrica no se convierte en una calificación automática del proyecto.

El [Apéndice A](sections/apendice_a_plantillas.md) ofrece las seis plantillas y su paquete descargable; las [referencias sugeridas](sections/referencias_sugeridas.md) conservan los 37 números de la guía y aclaran qué métodos son pertinentes para F2a.

## Revisión frente a la guía

La [introducción](sections/01_introduccion.md) explicita los cuatro productos obligatorios y el alcance actual. El [estándar](sections/02_estandar_profundidad.md) conserva los cinco requisitos de profundidad, distingue diez semillas de potencia suficiente y enlaza su declaración con fecha actual y registro prospectivo pendiente. El [panorama](sections/03_panorama_general.md) relaciona los productos con los bloques A–G. Las [correcciones](sections/04_correcciones_e2.md) separan observaciones internas de comentarios docentes, la auditoría histórica de la posterior y el ranking de seis semillas de su ampliación a diez.

La revisión del [Bloque B](sections/05_configuraciones_exportacion.md) contrasta selección, ajuste final, checkpoints, metadatos, exportación y costes con la evidencia guardada. El [Bloque C](sections/06_modelo_original.md) identifica la formulación de la referencia aditiva realmente evaluada y distingue sus pruebas de las del primer mecanismo y de la cabeza de momentos. Las matrices de requisitos y los registros de reproducibilidad reflejan las diez semillas y el análisis posterior ya terminado.

El [diseño riguroso del capítulo 7](sections/07_diseno_riguroso.md) distingue las hipótesis de los mecanismos históricos de la receta aditiva evaluada, y relaciona bitácora, separación de datos, sustento teórico y ablaciones con las exigencias de la guía. El árbol del apartado 6.6 conserva las reglas condicionales de R1/R2/R3 sin convertirlas en una calificación del avance.

Para avanzar al cierre faltan, entre otros productos, justificar una contribución admisible con revisión de literatura y defensa del autor, preparar la evaluación comparable con potencia y tres conjuntos adicionales y exportar/verificar el catálogo para el dashboard. La revisión no cambia predicciones, p-valores o recetas, ni convierte pendientes en requisitos cumplidos. Los archivos y enlaces históricos se conservan.

## Libros del curso

- [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html).
- [Entregable 2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html).
- [Selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html).
