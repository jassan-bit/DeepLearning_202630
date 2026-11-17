# 3 Panorama general del entregable

El trabajo se organiza en siete bloques. El desarrollo de software puede avanzar en paralelo, pero ningún producto definitivo se cierra sin sus prerrequisitos documentados. Implementar el candidato no cierra la auditoría ni congela una comparación justa.

| Bloque | Producto | Dependencia | Estado actual |
|---|---|---|---|
| A | Correcciones de E2, trazabilidad, auditoría y congelamiento. | Evidencias y artefactos de E2. | Inventario y auditorías parciales ejecutados; cierre integral pendiente. |
| B | Mejores configuraciones y exportaciones de inferencia. | Resultados y recetas válidos de A. | Tabla de 16 sistemas y tres referencias exportadas; resto y cierre pendientes. |
| C | Mecanismo neuronal, formulación, código y demostración. | Limitación sustentada y trabajo cercano. | Dos pilotos de desarrollo completos y aditiva v2 como referencia. Cabeza de momentos implementada, con pruebas artificiales y notebook exploratorio preparado; sin ejecución real nueva ni selección como aporte original. Estado integrado en 6/7. |
| D | [Comparación estadística y evaluación ampliada](08_verificacion_estadistica.md). | Diseño, presupuesto y protocolo congelados. | Borrador y requisitos organizados; piloto de potencia, congelamiento y campaña confirmatoria pendientes. Los pilotos de desarrollo no los sustituyen. |
| E | [Dashboard Dash Plotly con Contexto, EDA y Modelos](09_dashboard_predictivo.md). | Artefactos y tabla maestra consistentes. | Diseño y requisitos documentados; aplicación, integración y pruebas de aceptación pendientes. |
| F | [Despliegue público en Render y prueba de humo](10_despliegue_render.md). | Dashboard probado, perfil de producción y límites de recursos medidos. | Plan documentado; aplicación, ONNX/ORT, URL y prueba de humo pendientes. |
| G | [Informe científico y reproducibilidad](11_informe_tecnico.md). | Evidencias de todos los bloques. | Informe parcial publicado y bibliografía configurada; campaña, contenedor y reproducción independiente pendientes. |

## 3.1 Orden de trabajo

1. Cerrar las correcciones aplicables y las auditorías de datos; conservar fallos, limitaciones e impacto.
2. Fijar el conjunto de comparación y los artefactos que puedan reutilizarse, con sus hashes.
3. Desarrollar y revisar el mecanismo sólo con información de desarrollo; registrar variantes y esfuerzo.
4. Elegir evidencia adicional y dimensionar el presupuesto con piloto antes del protocolo final.
5. Congelar diseño, hipótesis, particiones, semillas, pruebas y criterios de decisión.
6. Ejecutar y reportar la comparación, incluyendo resultados adversos; después integrar y publicar el producto final.

El inventario de exportación y el diseño del dashboard pueden avanzar durante la investigación. Los resultados precalculados o las pruebas artificiales deberán seguir identificados como tales.

La solicitud posterior de comprobar el modelo en Colab habilita un piloto de desarrollo antes del cierre de todos los requisitos académicos. Su configuración operativa se fija antes de ejecutar y se documenta en 7.8. Esa exploración no sustituye el protocolo confirmatorio ni permite fechar requisitos después de los resultados como si hubieran sido previos.

## 3.2 Procedencia y versiones

Las evidencias de los bloques A/B están en `experiments/e3_bloques_ab_20261115/`; las del candidato, en `experiments/e3_bloque_c_20261115/`. Cada conjunto conserva su corte y sus fuentes. La [ampliación posterior de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html) aporta auditorías y diagnósticos adicionales sin cambiar la A2 original.

Los capítulos principales 1–16 integran el estado actual. Los capítulos [6](06_modelo_original.md) y [7](07_diseno_riguroso.md) reúnen el diseño y los resultados posteriores de desarrollo. Su detalle está en los anexos B–G, junto con las plantillas y la [concordancia](concordancia_secciones.md); se conservan las direcciones de las páginas y los informes guardados con sus hashes.
