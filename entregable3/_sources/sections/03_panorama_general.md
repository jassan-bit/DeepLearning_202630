# 3 Panorama general del entregable

El trabajo se organiza en siete bloques. El desarrollo de software puede avanzar en paralelo, pero ningún producto definitivo se cierra sin sus prerrequisitos documentados. Implementar el candidato no cierra la auditoría ni congela una comparación justa.

| Bloque | Producto | Dependencia | Estado actual |
|---|---|---|---|
| A | Correcciones de E2, trazabilidad, auditoría y congelamiento. | Evidencias y artefactos de E2. | Inventario y auditorías parciales ejecutados; cierre integral pendiente. |
| B | Mejores configuraciones y exportaciones de inferencia. | Resultados y recetas válidos de A. | Recetas de los 16 sistemas conservadas; E2 reúne 133 predictores. Hay cuatro ONNX CPU verificados sobre 608 ventanas, además de tres TorchScript históricos; trece exportaciones y la política final común siguen pendientes. |
| C | Mecanismo neuronal, formulación, código y demostración. | Limitación sustentada y trabajo cercano. | El candidato de la comparación vigente es CNN causal → LSTM → atención temporal, con cabeza positiva aditiva y 1210 parámetros, evaluado con diez semillas. Los pilotos y la cabeza de momentos se conservan como historia de desarrollo, sin acreditar R1/R2. Estado integrado en 6/7. |
| D | [Comparación estadística y evaluación ampliada](08_verificacion_estadistica.md). | Predicciones congeladas y alcance de inferencia explícito. | Cálculo retrospectivo completado, 48/48 contrastes favorables; MCS conserva diez sistemas. Controles posteriores de piso común y calibración documentados, con resultados adversos. Planificación de potencia condicional y tres fuentes candidatas, cero admitidas; R3 pendiente. |
| E | [Dashboard Dash Plotly con Contexto, EDA y Modelos](09_dashboard_predictivo.md). | Artefactos y tabla maestra consistentes. | Aplicación local con 17 sistemas, cuatro modelos ONNX en vivo y trece precalculados; aceptación de navegador y contenedor registrada. Interpretabilidad global, calibración E3 y otros requisitos conservan su estado parcial. |
| F | [Despliegue público en Render y prueba de humo](10_despliegue_render.md). | Dashboard probado, perfil de producción y límites de recursos medidos. | Aplicación, locks, ONNX/ORT, contenedor de servicio local, blueprint y guion de humo preparados; servicio Docker/Free creado con URL asignada y construcción observada; disponibilidad y humo públicos pendientes. Las medidas locales no certifican capacidad de Render. |
| G | [Informe científico y reproducibilidad](11_informe_tecnico.md). | Evidencias de todos los bloques. | Informe 1–16 y contenedor científico: once CSV idénticos desde 143 trayectorias; archivo versionado con hashes. No reproduce entrenamiento ni sustituye la revisión humana. DOI, derechos, firmas y cierre confirmatorio pendientes. |

## Relación de los cuatro productos con los bloques A–G

Los **cuatro componentes obligatorios** de la introducción de la guía se desarrollan mediante los bloques A–G. No son cuatro opciones entre las que se elige:

| Componente obligatorio | Bloques que lo desarrollan | Condición para su cierre |
|---|---|---|
| Correcciones del Entregable 2 | A y sus dependencias en B | Observaciones aplicables resueltas, auditoría y trazabilidad con impacto consistente; ausencia de comentarios docentes no equivale a aprobación |
| Dashboard predictivo Dash Plotly en Render | B → E → F | Artefactos de inferencia o precálculos justificados, Contexto/EDA/Modelos, integración coherente, URL funcional y prueba de humo |
| Modelo neuronal original o transferido | C → D | Contribución sustantiva atribuida, implementación, propiedad pertinente, revisión y evaluación completa bajo el estándar |
| Informe científico reproducible | G documenta A–F | Fuentes y compilación consistentes, resultados verificables, entorno y reproducción independiente, declaraciones y evidencias finales |

Las flechas expresan dependencias de cierre, no un preregistro de la historia ya ejecutada. El desarrollo de software puede avanzar en paralelo; los capítulos [1](01_introduccion.md) y [2](02_estandar_profundidad.md) distinguen el estado documental y los productos finales pendientes.

## 3.1 Orden de trabajo

Para una futura campaña confirmatoria se mantiene este orden, sin presentarlo como el origen previo del análisis histórico ya ejecutado:

1. Cerrar las correcciones aplicables y las auditorías de datos; conservar fallos, limitaciones e impacto.
2. Fijar el conjunto de comparación y los artefactos que puedan reutilizarse, con sus hashes.
3. Desarrollar y revisar el mecanismo sólo con información de desarrollo; registrar variantes y esfuerzo.
4. Elegir evidencia adicional y dimensionar el presupuesto con piloto antes del protocolo final.
5. Congelar diseño, hipótesis, particiones, semillas, pruebas y criterios de decisión.
6. Ejecutar y reportar la comparación, incluyendo resultados adversos; después integrar y publicar el producto final.

El inventario de exportación y el diseño del dashboard pueden avanzar durante la investigación. Los resultados precalculados o las pruebas artificiales deberán seguir identificados como tales.

Las solicitudes posteriores de ejecución en Colab permitieron los pilotos y las repeticiones de desarrollo documentados en los capítulos 6 y 7. La comparación histórica de 48 contrastes reutiliza esas predicciones congeladas, sin añadir entrenamiento ni inferencia de modelos. El diagnóstico posterior de intervalos tiene su propio recibo de 530 inferencias CPU en calibración. Su carácter posterior no se transforma en confirmación por fechar después de los resultados un protocolo como si hubiera sido previo.

## 3.2 Procedencia y versiones

Las evidencias de los bloques A/B están en `experiments/e3_bloques_ab_20261115/`; las del candidato, en `experiments/e3_bloque_c_20261115/`. Cada conjunto conserva su corte y sus fuentes. La [ampliación posterior de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html) aporta auditorías y diagnósticos adicionales sin cambiar la A2 original.

La [sección 8](08_verificacion_estadistica.md) y su [tabla de métricas](../_static/e3_estadistica_posterior_20261117/metrics.csv) reúnen la comparación posterior con diez semillas neuronales. E2 conserva 655 ejemplos de entrenamiento y 53 de calibración; E3, 532 de ajuste y 116 de validación para parada, sin calibración adicional. Las 608 fechas comunes y el número de semillas no igualan esos protocolos ni constituyen múltiples conjuntos independientes.

Los capítulos principales 1–16 integran el estado actual. Los capítulos [6](06_modelo_original.md) y [7](07_diseno_riguroso.md) reúnen el diseño y la historia de desarrollo. Su detalle está en los anexos B–G, junto con las plantillas y la [concordancia](concordancia_secciones.md); se conservan las direcciones de las páginas y los informes guardados con sus hashes.

La revisión de los capítulos [10](10_despliegue_render.md) a [13](13_sustentacion.md) distingue el servicio por desplegar, la estructura científica del informe, el registro de trece entregables y la preparación de una exposición de 20 minutos —18 de contenido y dos de margen—. El guion y las pruebas técnicas no se registran como despliegue, reproducción humana o defensa realizados.

Los correctivos del 18 de noviembre separan [literatura y mecanismo](06_modelo_original.md), [controles posteriores y preparación Colab](08_verificacion_estadistica.md) y [reproducción científica automática](11_informe_tecnico.md). El diagnóstico de calibración ejecutó 530 inferencias CPU exclusivamente en las ventanas de calibración; la reconstrucción de tablas no ejecutó modelos. Ninguno cambia los 48 contrastes históricos ni convierte las tres fuentes candidatas en datasets admitidos.
