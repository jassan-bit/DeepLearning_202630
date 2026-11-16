# Entregable 3 de Deep Learning

**Jassan Arteta · Deep Learning 2026-30**

El informe está organizado exclusivamente en los **capítulos 1 a 16 de la guía**, con el desarrollo disponible hasta ahora. La navegación principal reúne los bloques A–G, diseño, evaluación, entrega, sustentación, integridad y conclusiones. Las ampliaciones se integran en esos capítulos; los anexos A–G reúnen plantillas y detalle técnico de apoyo, junto con referencias y trazabilidad. El último anexo conserva reproducibilidad, autoría y bibliografía.

| Sección | Contenido |
|---|---|
| [1 Introducción](sections/01_introduccion.md) | Problema, objetivos, antecedentes y alcance actual. |
| [2 Estándar de profundidad exigido](sections/02_estandar_profundidad.md) | Requisitos teóricos, potencia, semillas y compromisos previos. |
| [3 Panorama general del entregable](sections/03_panorama_general.md) | Bloques A–G, dependencias y estado de avance. |
| [4 Bloque A Correcciones del Entregable 2](sections/04_correcciones_e2.md) | Trazabilidad y auditorías vigentes, con sus pendientes. |
| [5 Bloque B Selección de configuraciones y exportación](sections/05_configuraciones_exportacion.md) | Tabla de 16 sistemas y tres exportaciones verificadas. |
| [6 Bloque C El modelo original](sections/06_modelo_original.md) | Candidato, antecedentes, implementación y estados R1/R2/R3. |
| [7 Diseño riguroso del modelo original](sections/07_diseno_riguroso.md) | Hipótesis, disciplina de diseño, sustento teórico y ablación. |
| [8 Bloque D Verificación estadística y evaluación ampliada](sections/08_verificacion_estadistica.md) | Borrador de protocolo, equidad, potencia, familia completa y criterios de R3. |
| [9 Bloque E Dashboard predictivo en Dash Plotly](sections/09_dashboard_predictivo.md) | Diseño de Contexto, EDA y Modelos, contratos de inferencia y requisitos de calidad. |
| [10 Bloque F Despliegue en Render](sections/10_despliegue_render.md) | Perfil de producción, exportación ONNX pendiente, prueba de humo y contingencia. |
| [11 Bloque G Informe técnico en Jupyter Book](sections/11_informe_tecnico.md) | Estructura científica, mapa del Cuadro 7 y requisitos de reproducción. |
| [12 Lista de entregables](sections/12_lista_entregables.md) | Trece productos, evidencia existente y pendientes de cierre. |
| [13 Sustentación](sections/13_sustentacion.md) | Guion de 20 minutos, demostración y preparación de preguntas. |
| [14 Evaluación y calificación](sections/14_evaluacion_calificacion.md) | Rúbrica de 100 %, topes condicionales y matrices de evidencia, sin nota actual. |
| [15 Integridad académica](sections/15_integridad_academica.md) | Atribución, novedad, uso de IA y trazabilidad de decisiones y resultados. |
| [16 Conclusiones](sections/16_conclusiones.md) | Avances acreditados, límites y capacidades pendientes de demostrar. |

## Estado del proyecto

E2 conserva su benchmark de 16 sistemas y 608 fechas, con auditorías y suplementos posteriores de alcance explícito. E3 tiene inventario de artefactos, configuraciones verificadas y tres exportaciones CPU sobre calibración histórica. La evidencia neuronal incluye formulación, demostraciones con alcance explícito, pruebas de software y dos pilotos de desarrollo sobre la misma base CNN-LSTM-atención.

Los capítulos [6](sections/06_modelo_original.md) y [7](sections/07_diseno_riguroso.md) integran el diseño, la revisión de admisibilidad y los resultados de los pilotos. El control aditivo v2 obtuvo la menor RMSE global de desarrollo; el candidato con pérdida auxiliar no superó la heurística del 2 %. Se conserva como referencia predictiva y **no se promueve a modelo original por ese resultado**. El [anexo E](sections/20_cierre_v2.md) conserva el cierre detallado y sus limitaciones en alta volatilidad.

**La novedad R1, la defensa académica de R2 y la superioridad R3 siguen pendientes.** Los bloques A y B aún no están cerrados; tampoco existe una campaña confirmatoria del candidato o un dashboard Render terminado. El protocolo sigue en borrador. Las plantillas preparan registros futuros y no son preregistros, resultados o firmas. La bibliografía inicial y las revisiones focales posteriores distinguen antecedentes, procedencia y alcance de lectura; se conservan las 37 referencias sugeridas por la guía. El cierre incorpora resultados ya obtenidos en Colab y los audita sin nuevos ajustes, inferencias o cambios de las métricas y recetas originales.

La [concordancia y los anexos](sections/concordancia_secciones.md) permiten localizar cada evidencia y su capítulo principal. Se conservan las direcciones de las páginas anteriores, ahora identificadas como anexos, para mantener sus enlaces.

La propuesta por momentos y su dictamen forman parte de esos capítulos 6 y 7. Los anexos [F](sections/21_propuesta_momentos.md) y [G](sections/22_dictamen_momentos.md) conservan formulación completa, verificaciones artificiales y paquete para revisión externa. La decisión interna es mantener la propuesta como adaptación experimental, sin seleccionarla todavía como aporte original ni lanzar otro piloto. La revisión bibliográfica completa y la verificación humana externa siguen pendientes.

El [anexo final](sections/cierre_reproducibilidad.md) distingue conclusiones provisionales, evidencia de software, reproducción científica pendiente y declaración de autoría sin firmas. La rúbrica no se convierte en una calificación automática del proyecto.

El [Apéndice A](sections/apendice_a_plantillas.md) ofrece las seis plantillas y su paquete descargable; las [referencias sugeridas](sections/referencias_sugeridas.md) conservan los 37 números de la guía y aclaran qué métodos son pertinentes para F2a.

## Libros del curso

- [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html).
- [Entregable 2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html).
- [Selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html).
