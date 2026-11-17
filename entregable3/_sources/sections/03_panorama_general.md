# 3 Panorama general del entregable

El trabajo se organiza en siete bloques. El desarrollo de software puede avanzar en paralelo, pero ningún producto definitivo se cierra sin sus prerrequisitos documentados. Implementar el candidato no cierra la auditoría ni congela una comparación justa.

| Bloque | Producto | Dependencia | Estado actual |
|---|---|---|---|
| A | Correcciones de E2, trazabilidad, auditoría y congelamiento. | Evidencias y artefactos de E2. | Inventario y auditorías parciales ejecutados; cierre integral pendiente. |
| B | Mejores configuraciones y exportaciones de inferencia. | Resultados y recetas válidos de A. | Recetas de los 16 sistemas conservadas; E2 reúne 133 predictores, con diez semillas por red y una trayectoria por clásico. Tres referencias exportadas; resto y cierre pendientes. |
| C | Mecanismo neuronal, formulación, código y demostración. | Limitación sustentada y trabajo cercano. | El candidato de la comparación vigente es CNN causal → LSTM → atención temporal, con cabeza positiva aditiva y 1210 parámetros, evaluado con diez semillas. Los pilotos y la cabeza de momentos se conservan como historia de desarrollo, sin acreditar R1/R2. Estado integrado en 6/7. |
| D | [Comparación estadística y evaluación ampliada](08_verificacion_estadistica.md). | Predicciones congeladas y alcance de inferencia explícito. | Análisis posterior completado sobre 608 fechas comunes: 48/48 contrastes primarios rechazan, con ámbitos de incertidumbre distintos. MCS al 90 % conserva diez sistemas incluido el candidato. Potencia, campaña confirmatoria y R3 no quedan acreditadas. |
| E | [Dashboard Dash Plotly con Contexto, EDA y Modelos](09_dashboard_predictivo.md). | Artefactos y tabla maestra consistentes. | Diseño y requisitos documentados; aplicación, integración y pruebas de aceptación pendientes. |
| F | [Despliegue público en Render y prueba de humo](10_despliegue_render.md). | Dashboard probado, perfil de producción y límites de recursos medidos. | Plan documentado; aplicación, ONNX/ORT, URL y prueba de humo pendientes. |
| G | [Informe científico y reproducibilidad](11_informe_tecnico.md). | Evidencias de todos los bloques. | Informe con análisis posterior y bibliografía configurada; cierre confirmatorio, contenedor y reproducción independiente pendientes. |

## 3.1 Orden de trabajo

Para una futura campaña confirmatoria se mantiene este orden, sin presentarlo como el origen previo del análisis histórico ya ejecutado:

1. Cerrar las correcciones aplicables y las auditorías de datos; conservar fallos, limitaciones e impacto.
2. Fijar el conjunto de comparación y los artefactos que puedan reutilizarse, con sus hashes.
3. Desarrollar y revisar el mecanismo sólo con información de desarrollo; registrar variantes y esfuerzo.
4. Elegir evidencia adicional y dimensionar el presupuesto con piloto antes del protocolo final.
5. Congelar diseño, hipótesis, particiones, semillas, pruebas y criterios de decisión.
6. Ejecutar y reportar la comparación, incluyendo resultados adversos; después integrar y publicar el producto final.

El inventario de exportación y el diseño del dashboard pueden avanzar durante la investigación. Los resultados precalculados o las pruebas artificiales deberán seguir identificados como tales.

Las solicitudes posteriores de ejecución en Colab permitieron los pilotos y las repeticiones de desarrollo documentados en los capítulos 6 y 7. El análisis estadístico vigente reutiliza esas predicciones congeladas; no añade entrenamiento ni inferencia de modelos. Su carácter posterior no se transforma en confirmación por fechar después de los resultados un protocolo como si hubiera sido previo.

## 3.2 Procedencia y versiones

Las evidencias de los bloques A/B están en `experiments/e3_bloques_ab_20261115/`; las del candidato, en `experiments/e3_bloque_c_20261115/`. Cada conjunto conserva su corte y sus fuentes. La [ampliación posterior de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html) aporta auditorías y diagnósticos adicionales sin cambiar la A2 original.

La [sección 8](08_verificacion_estadistica.md) y su [tabla de métricas](../_static/e3_estadistica_posterior_20261117/metrics.csv) reúnen la comparación posterior con diez semillas neuronales. E2 conserva 655 ejemplos de entrenamiento y 53 de calibración; E3, 532 de ajuste y 116 de validación para parada, sin calibración adicional. Las 608 fechas comunes y el número de semillas no igualan esos protocolos ni constituyen múltiples conjuntos independientes.

Los capítulos principales 1–16 integran el estado actual. Los capítulos [6](06_modelo_original.md) y [7](07_diseno_riguroso.md) reúnen el diseño y la historia de desarrollo. Su detalle está en los anexos B–G, junto con las plantillas y la [concordancia](concordancia_secciones.md); se conservan las direcciones de las páginas y los informes guardados con sus hashes.
