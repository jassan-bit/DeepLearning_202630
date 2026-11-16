# 3 Panorama general del entregable

El trabajo se organiza en siete bloques. El desarrollo de software puede avanzar en paralelo, pero ningún producto definitivo se cierra sin sus prerrequisitos documentados. Implementar el candidato no cierra la auditoría ni congela una comparación justa.

| Bloque | Producto | Dependencia | Estado actual |
|---|---|---|---|
| A | Correcciones de E2, trazabilidad, auditoría y congelamiento. | Evidencias y artefactos de E2. | Inventario y auditorías parciales ejecutados; cierre integral pendiente. |
| B | Mejores configuraciones y exportaciones de inferencia. | Resultados y recetas válidos de A. | Tabla de 16 sistemas y tres referencias exportadas; resto y cierre pendientes. |
| C | Mecanismo neuronal, formulación, código y demostración. | Limitación sustentada y trabajo cercano. | Candidato con teoría y pruebas artificiales; novedad y evaluación real pendientes. |
| D | [Comparación estadística y evaluación ampliada](08_verificacion_estadistica.md). | Diseño, presupuesto y protocolo congelados. | Borrador y requisitos organizados; piloto, congelamiento y campaña confirmatoria pendientes. |
| E | Dashboard Dash Plotly con Contexto, EDA y Modelos. | Artefactos y tabla maestra consistentes. | Pendiente de construcción como producto predictivo. |
| F | Despliegue público en Render y prueba de humo. | Dashboard probado y límites de recursos medidos. | Pendiente; el Jupyter Book publicado no lo sustituye. |
| G | Informe científico y paquete de reproducibilidad. | Evidencias de todos los bloques. | Informe parcial publicado; entrega integral pendiente. |

## 3.1 Orden de trabajo

1. Cerrar las correcciones aplicables y las auditorías de datos; conservar fallos, limitaciones e impacto.
2. Fijar el conjunto de comparación y los artefactos que puedan reutilizarse, con sus hashes.
3. Desarrollar y revisar el mecanismo sólo con información de desarrollo; registrar variantes y esfuerzo.
4. Elegir evidencia adicional y dimensionar el presupuesto con piloto antes del protocolo final.
5. Congelar diseño, hipótesis, particiones, semillas, pruebas y criterios de decisión.
6. Ejecutar y reportar la comparación, incluyendo resultados adversos; después integrar y publicar el producto final.

El inventario de exportación y el diseño del dashboard pueden avanzar durante la investigación. Los resultados precalculados o las pruebas artificiales deberán seguir identificados como tales.

## 3.2 Procedencia y versiones

Las evidencias de los bloques A/B están en `experiments/e3_bloques_ab_20261115/`; las del candidato, en `experiments/e3_bloque_c_20261115/`. Cada conjunto conserva su corte y sus fuentes. La [ampliación posterior de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html) aporta auditorías y diagnósticos adicionales sin cambiar la A2 original.

Las secciones siguientes integran ese estado actual. Las páginas de trabajo anteriores siguen accesibles en los [anexos y concordancia](concordancia_secciones.md), sin modificar los informes guardados o sus hashes.
