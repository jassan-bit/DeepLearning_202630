# Entregable 3 de Deep Learning

**Jassan Arteta · Deep Learning 2026-30**

El informe está organizado exclusivamente en los **capítulos 1 a 16 de la guía**, con el desarrollo disponible hasta ahora. La navegación principal reúne los bloques A–G, diseño, evaluación, entrega, sustentación, integridad y conclusiones. Las ampliaciones se integran en esos capítulos; los anexos A–G reúnen plantillas y detalle técnico de apoyo, junto con referencias y trazabilidad. El último anexo conserva reproducibilidad, autoría y bibliografía.

| Sección | Contenido |
|---|---|
| [1 Introducción](sections/01_introduccion.md) | Problema, objetivos, antecedentes y alcance actual. |
| [2 Estándar de profundidad exigido](sections/02_estandar_profundidad.md) | Requisitos teóricos, potencia, semillas y compromisos previos. |
| [3 Panorama general del entregable](sections/03_panorama_general.md) | Bloques A–G, dependencias y estado de avance. |
| [4 Bloque A Correcciones del Entregable 2](sections/04_correcciones_e2.md) | Trazabilidad y auditorías vigentes, con sus pendientes. |
| [5 Bloque B Selección de configuraciones y exportación](sections/05_configuraciones_exportacion.md) | Selección por validación, checkpoints, tres TorchScript históricos y cuatro ONNX CPU verificados; requisitos de cierre. |
| [6 Bloque C El modelo original](sections/06_modelo_original.md) | Arquitectura evaluada, antecedentes, formulación y evidencia específica para R1/R2/R3. |
| [7 Diseño riguroso del modelo original](sections/07_diseno_riguroso.md) | Hipótesis, disciplina de diseño, sustento teórico y ablación. |
| [8 Bloque D Verificación estadística y evaluación ampliada](sections/08_verificacion_estadistica.md) | Comparación posterior de diez semillas, alcances de incertidumbre, MCS y requisitos aún pendientes de R3. |
| [9 Bloque E Dashboard predictivo en Dash Plotly](sections/09_dashboard_predictivo.md) | Aplicación Contexto, EDA y Modelos, 17 sistemas y cuatro artefactos ONNX CPU con contratos y evidencia. |
| [10 Bloque F Despliegue en Render](sections/10_despliegue_render.md) | Servicio Docker/Free creado con URL asignada; funcionamiento y humo públicos por verificar. |
| [11 Bloque G Informe técnico en Jupyter Book](sections/11_informe_tecnico.md) | Estructura científica, mapa del Cuadro 7 y requisitos de reproducción. |
| [12 Lista de entregables](sections/12_lista_entregables.md) | Trece productos, evidencia existente y pendientes de cierre. |
| [13 Sustentación](sections/13_sustentacion.md) | Guion de 20 minutos, demostración y preparación de preguntas. |
| [14 Evaluación y calificación](sections/14_evaluacion_calificacion.md) | Rúbrica de 100 %, topes condicionales y matrices de evidencia, sin nota actual. |
| [15 Integridad académica](sections/15_integridad_academica.md) | Atribución, novedad, uso de IA y trazabilidad de decisiones y resultados. |
| [16 Conclusiones](sections/16_conclusiones.md) | Avances acreditados, límites y capacidades pendientes de demostrar. |

## Estado del proyecto

E2 conserva su benchmark de 16 sistemas y ahora reúne diez semillas para cada una de sus trece redes, más una trayectoria por comparador determinista: **133 predictores sobre las mismas 608 fechas**. La extensión conserva las seis semillas neuronales históricas y añade cuatro sin repetir la selección de recetas. E3 cuenta con diez semillas del candidato **CNN causal → LSTM unidireccional → atención temporal, con cabeza positiva aditiva y 1210 parámetros**. Su RMSE media individual es **0,00293943** en ese calendario conocido; la [tabla de métricas](_static/e3_estadistica_posterior_20261117/metrics.csv) conserva los valores completos.

Los capítulos [6](sections/06_modelo_original.md) y [7](sections/07_diseno_riguroso.md) distinguen ese candidato de los pilotos anteriores. En el piloto v2, el control aditivo obtuvo la menor RMSE de desarrollo y la variante con pérdida auxiliar no superó la heurística del 2 %. El [anexo E](sections/20_cierre_v2.md) conserva ese corte histórico y sus limitaciones en alta volatilidad. La elección posterior de la cabeza aditiva para la comparación **no acredita por sí misma un aporte original**.

La [sección 8](sections/08_verificacion_estadistica.md) incorpora el análisis posterior ya completado: los **48 contrastes primarios** rechazan sus nulas y los 16 rivales cumplen la regla conjunta de los tres procedimientos. Sus ámbitos de incertidumbre, supuestos y relación con la Tabla 4 temporal se explican por separado; disponer de tres métodos no acredita automáticamente las tres pruebas exigidas para R3. E2 usa 655 ejemplos de entrenamiento y 53 de calibración; E3 usa 532 de ajuste y 116 de validación para parada, sin calibración adicional. No se declara igualdad de presupuestos. El MCS al 90 % conserva diez sistemas, incluido el candidato, y no identifica un ganador único.

La [tabla ampliada de los 17 sistemas](_static/e3_revision_bloque_d_20261117/metrics_extended.csv) conserva sMAPE medio **11,10 %** y R² medio **0,9118**, y los intervalos heredados de E2. Su corte original no calculó intervalos E3. El [diagnóstico posterior separado del capítulo 8](sections/08_verificacion_estadistica.md) añade diez calibradores: cobertura histórica **94,49 %** para nominal 90 %, pero **86,22 % en 2026Q1**. No están activados en el dashboard ni acreditan cobertura uniforme temporal. Las métricas puntuales y los 48 contrastes históricos no cambian.

**La novedad R1, la defensa académica de R2 y la superioridad R3 siguen sin acreditarse.** La comparación utiliza un calendario histórico conocido y no constituye una campaña confirmatoria ciega ni demuestra potencia o reproducción en mercados independientes. Los bloques A y B conservan sus pendientes de cierre, al igual que el despliegue en Render. Las plantillas no se convierten retrospectivamente en preregistros ni firmas. La bibliografía inicial y las revisiones focales posteriores distinguen antecedentes, procedencia y alcance de lectura; se conservan las 37 referencias sugeridas por la guía. El análisis estadístico reutilizó predicciones congeladas sin nuevos ajustes o inferencias; la etapa del dashboard sí ejecuta inferencia CPU para comprobar los artefactos, sin entrenar ni recalibrar.

El [dashboard del capítulo 9](sections/09_dashboard_predictivo.md) implementa las tres páginas exigidas y conserva el catálogo de 17 sistemas. MLP16, LSTM32, GRU32 y la CNN-LSTM aditiva tienen artefactos ONNX verificados sobre las 608 ventanas, con semilla representativa fija 271801 y tolerancia de 10⁻⁶ en σ₇. Los otros trece sistemas muestran resultados precalculados. La aplicación se ejecuta con un servidor Python; GitHub Pages publica este informe y sus evidencias. Render ya asignó [la URL del dashboard](https://deeplearning-e3-dashboard.onrender.com); su funcionamiento y prueba de humo públicos siguen sin verificar.

La [concordancia y los anexos](sections/concordancia_secciones.md) permiten localizar cada evidencia y su capítulo principal. Se conservan las direcciones de las páginas anteriores, ahora identificadas como anexos, para mantener sus enlaces.

La cabeza de momentos y su dictamen se conservan como una línea histórica de desarrollo en los capítulos 6 y 7 y los anexos [F](sections/21_propuesta_momentos.md) y [G](sections/22_dictamen_momentos.md). Sus formulaciones, instrucciones de Colab y cortes de admisibilidad no describen el sistema de la comparación vigente: ésta evalúa la cabeza positiva aditiva. Ese historial tampoco selecciona la adaptación por momentos como aporte original.

El [anexo final](sections/cierre_reproducibilidad.md) conserva autoría sin firmas y distingue la reproducción automática ya ejecutada de la humana pendiente. El nuevo [contenedor científico](sections/11_informe_tecnico.md) reconstruyó **11 de 11 CSV idénticos**, desde 143 trayectorias guardadas, sin entrenar ni inferir. Su archivo versionado descargable tiene SHA-256; el depósito persistente con DOI continúa pendiente. La rúbrica no se convierte en una calificación automática del proyecto.

El [Apéndice A](sections/apendice_a_plantillas.md) ofrece las seis plantillas y su paquete descargable; las [referencias sugeridas](sections/referencias_sugeridas.md) conservan los 37 números de la guía y aclaran qué métodos son pertinentes para F2a.

## Revisión frente a la guía

La [introducción](sections/01_introduccion.md) explicita los cuatro productos obligatorios y el alcance actual. El [estándar](sections/02_estandar_profundidad.md) conserva los cinco requisitos de profundidad, distingue diez semillas de potencia suficiente y enlaza su declaración con fecha actual y registro prospectivo pendiente. El [panorama](sections/03_panorama_general.md) relaciona los productos con los bloques A–G. Las [correcciones](sections/04_correcciones_e2.md) separan observaciones internas de comentarios docentes, la auditoría histórica de la posterior y el ranking de seis semillas de su ampliación a diez.

La revisión del [Bloque B](sections/05_configuraciones_exportacion.md) contrasta selección, ajuste final, checkpoints, metadatos, exportación y costes con la evidencia guardada. El [Bloque C](sections/06_modelo_original.md) identifica la formulación de la referencia aditiva realmente evaluada y distingue sus pruebas de las del primer mecanismo y de la cabeza de momentos. Las matrices de requisitos y los registros de reproducibilidad reflejan las diez semillas y el análisis posterior ya terminado.

El [diseño riguroso del capítulo 7](sections/07_diseno_riguroso.md) distingue las hipótesis de los mecanismos históricos de la receta aditiva evaluada, y relaciona bitácora, separación de datos, sustento teórico y ablaciones con las exigencias de la guía. El árbol del apartado 6.6 conserva las reglas condicionales de R1/R2/R3 sin convertirlas en una calificación del avance.

La revisión del [Bloque D](sections/08_verificacion_estadistica.md) incorpora los campos del protocolo, las seis condiciones simultáneas de R3, la pertinencia de las pruebas, las ilustraciones de potencia y los estudios de robustez y reporte exigidos. `experiments/protocolo.md` es un borrador consolidado posterior, con decisiones abiertas y sin tag confirmatorio; no registra retrospectivamente los resultados conocidos.

Para avanzar al cierre faltan, entre otros productos, justificar una contribución admisible con revisión de literatura y defensa del autor, preparar la evaluación comparable con potencia y tres conjuntos adicionales y exportar/verificar el catálogo para el dashboard. La revisión no cambia predicciones, p-valores o recetas, ni convierte pendientes en requisitos cumplidos. Los archivos y enlaces históricos se conservan.

La revisión de los [bloques F](sections/10_despliegue_render.md) y [G](sections/11_informe_tecnico.md) incorpora un cliente de humo para las tres páginas y los cuatro modelos habilitados, probado contra Gunicorn y Docker locales, con estados separados de la comprobación pública pendiente. El informe añade un resumen de 217 palabras, numeración y referencias de ecuaciones, generadores por producto y un procedimiento de reproducción humana con discrepancia por campo. La [lista de trece entregables](sections/12_lista_entregables.md) fija evidencia y condiciones de cierre; la [sustentación](sections/13_sustentacion.md) conserva 18 minutos de contenido y dos de navegación/transiciones, con preguntas y práctica técnica. La disponibilidad y el humo públicos de Render, la reproducción humana, la licencia propia, las firmas y el archivo persistente conservan sus pendientes.

La revisión de las [secciones 14](sections/14_evaluacion_calificacion.md), [15](sections/15_integridad_academica.md) y [16](sections/16_conclusiones.md) conserva los once componentes de la rúbrica —100 %—, las 45 condiciones para aspirar a 5,0 y las cinco capacidades del cierre. La fórmula y los topes son reglas de la guía, no una nota asignada al avance. El [Apéndice A](sections/apendice_a_plantillas.md) reúne las seis plantillas, los 17 ítems de reproducibilidad y los 16 errores frecuentes; sus archivos históricos se conservan como tales. Las [37 referencias sugeridas](sections/referencias_sugeridas.md) mantienen numeración y procedencia, separadas de los antecedentes del candidato y de la evidencia de lectura. La [URL asignada al dashboard](https://deeplearning-e3-dashboard.onrender.com) consta como servicio creado, con último estado aportado Building; la aceptación pública sigue sin verificarse.

## Libros del curso

- [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html).
- [Entregable 2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html).
- [Selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html).

## Correctivos posteriores del 18 de noviembre de 2026

La revisión de [literatura](sections/06_modelo_original.md) compara once trabajos y ocho lecturas focales de texto completo. Identifica antecedentes de atención, estabilidad y restricciones de salida; **R1 sigue sin acreditarse** y la cobertura efectiva de tres bases no está cerrada.

El [control de soporte de la etiqueta](sections/08_verificacion_estadistica.md) modifica la interpretación del resultado: aplicar el mismo piso observable a los rivales reduce la ventaja RMSE descriptiva de **24,10 % a 10,33 %**; el mejor rival cambia de SAE a TSMixer. Un control sin aprendizaje que usa sólo ese piso obtiene **3,70 % menos MAE que E3**, aunque mayor RMSE. Ambos resultados se conservan, sin atribuir causalmente la diferencia al mecanismo.

La planificación de potencia y el cuaderno de preparación para **Colab con GPU** están en el [capítulo 8](sections/08_verificacion_estadistica.md). Hay **tres fuentes candidatas y cero datasets admitidos**: todavía faltan derechos, hashes y contratos de calendario/objetivo, además del protocolo y presupuesto común. El cuaderno comprueba el entorno y las recetas; no inicia la nueva campaña con esos requisitos abiertos. La verificación pública de Render y la reproducción humana siguen pendientes.
