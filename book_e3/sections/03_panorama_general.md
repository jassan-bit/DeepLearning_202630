# 3 Panorama general del entregable

El trabajo se organiza en siete bloques. El desarrollo de software puede avanzar en paralelo, pero ningún producto definitivo se cierra sin sus prerrequisitos documentados. Implementar el candidato no cierra la auditoría ni congela una comparación justa.

| Bloque | Producto | Dependencia | Estado actual |
|---|---|---|---|
| A | Correcciones de E2, trazabilidad, auditoría y congelamiento. | Evidencias y artefactos de E2. | Inventario y auditorías parciales ejecutados; cierre integral pendiente. |
| B | Mejores configuraciones y exportaciones de inferencia. | Resultados y recetas válidos de A. | Recetas de los 16 sistemas conservadas; E2 reúne 133 predictores. El perfil vigente habilita cinco ONNX CPU; nueve representantes neuronales adicionales ya tienen paridad offline en 608 ventanas de prueba y 53 de calibración. Son catorce representantes neuronales ONNX verificados en total, además de tres TorchScript históricos. Las vistas originales conservan cinco predictores y `/perfil-14` habilita las catorce redes, con aceptación pública de ambos clientes en {ref}`10.10 <e3-render-public-acceptance-20261120>`. Los tres clásicos requieren historia/estado adicional a siete barras y conservan precálculos. La política científica final común sigue pendiente; recibos de exportación en el {ref}`cierre técnico <e3-completion-20261120>`. |
| C | Mecanismo neuronal, formulación, código y demostración. | Limitación sustentada y trabajo cercano. | El candidato de la comparación vigente es CNN causal → LSTM → atención temporal, con cabeza positiva aditiva y 1210 parámetros, evaluado con diez semillas. Los pilotos y la cabeza de momentos se conservan como historia de desarrollo, sin acreditar R1/R2. Estado integrado en 6/7. |
| D | [Comparación estadística y evaluación ampliada](08_verificacion_estadistica.md). | Predicciones congeladas y alcance de inferencia explícito. | Comparación histórica conservada: 48/48 contrastes favorables y MCS con diez sistemas. La {ref}`auditoría exploratoria completa de 8.18 <e3-final-extended-audit>` añade **796 trabajos, 19 sistemas en cuatro conjuntos, 76 paneles y 1.874.560 filas**, con análisis terminado y 16 figuras. El protocolo y presupuesto se conservan; potencia confirmatoria, justificación de independencia y R3 siguen pendientes. |
| E | [Dashboard Dash Plotly con Contexto, EDA y Modelos](09_dashboard_predictivo.md). | Artefactos y tabla maestra consistentes. | Aplicación con catálogo de 17 sistemas, cinco ONNX en las vistas originales y catorce en `/perfil-14`, públicamente aceptados; los tres clásicos conservan precálculos. E3 adopta un perfil posterior individual, con cobertura histórica 94,08 % y 85,56 % en 2026Q1; sólo se aplica después del 30-12-2024. Interpretabilidad global, validación de calibración y otros requisitos siguen parciales. |
| F | [Despliegue público en Render y prueba de humo](10_despliegue_render.md). | Dashboard probado, perfil de producción y límites de recursos medidos. | Render H (`feb83dbc`) aceptado públicamente: cliente original de tres páginas/cinco modelos y cliente14 por API/navegador, con intervalos, huellas, guardas y seis capturas; {ref}`10.10 <e3-render-public-acceptance-20261120>`. Las medidas locales no certifican capacidad remota o disponibilidad sostenida. |
| G | [Informe científico y reproducibilidad](11_informe_tecnico.md). | Evidencias de todos los bloques. | Informe 1–16 y contenedor científico: once CSV idénticos desde 143 trayectorias; archivo versionado con hashes. No reproduce entrenamiento ni sustituye la revisión humana. DOI del suplemento documental y ZIP exacto publicados y verificados en {ref}`11.10 <e3-zenodo-publication-verified-20261120>`, con licencias propias seleccionadas. Pendientes: metadatos públicos, archivo integral, derechos fuera de ese alcance, firmas y cierre confirmatorio. |

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

La [sección 8](08_verificacion_estadistica.md) y su [tabla de métricas](../_static/e3_estadistica_posterior_20261117/metrics.csv) reúnen la comparación posterior con diez semillas neuronales. E2 conserva 655 ejemplos de entrenamiento y 53 de calibración; la receta de E3, 532 de ajuste y 116 de validación para parada, sin usar ese bloque para entrenar o seleccionar época. El servicio adopta después el cuantil residual existente de la semilla 271801: `q = 0.005402463770651868`, calibración del 8 de noviembre al 30 de diciembre de 2024, sólo aplicable a objetivos posteriores al último día. Conserva pesos, escalador y puntos; las cifras individuales 94,08 % y 85,56 % no son la media de diez semillas. Las 608 fechas comunes y el número de semillas no igualan tratamientos ni constituyen múltiples conjuntos independientes.

Los capítulos principales 1–16 integran el estado actual. Los capítulos [6](06_modelo_original.md) y [7](07_diseno_riguroso.md) reúnen el diseño y la historia de desarrollo. Su detalle está en los anexos B–G, junto con las plantillas y la [concordancia](concordancia_secciones.md); se conservan las direcciones de las páginas y los informes guardados con sus hashes.

La revisión de los capítulos [10](10_despliegue_render.md) a [13](13_sustentacion.md) distingue la aceptación pública del servicio, la estructura científica del informe, el registro de trece entregables y la preparación de una exposición de 20 minutos —18 de contenido y dos de margen—. El guion y las pruebas técnicas no se registran como aceptación remota, reproducción humana o defensa realizados. El corte histórico de cuatro ONNX y sus mediciones se conserva; TSMixer añade un quinto con recibo separado, sin mezclar perfiles de latencia.

Los correctivos del 18 de noviembre separan [literatura y mecanismo](06_modelo_original.md), [controles posteriores y preparación Colab](08_verificacion_estadistica.md) y [reproducción científica automática](11_informe_tecnico.md). El diagnóstico de calibración ejecutó 530 inferencias CPU exclusivamente en calibración; la reconstrucción de tablas no ejecutó modelos. Ninguno cambia los 48 contrastes históricos. El corte de 8.13 cerró primero la admisión de **D01 WIKI/AAPL para σ₇ RAW de siete sesiones y horizonte próxima sesión**: un archivo de tres admitido, ninguno evaluado por modelos y dos pendientes. Esa descarga, auditoría de calendario y decisión de política no ejecutaron entrenamientos reales nuevos.

La auditoría posterior conserva el corte parcial de 251 trabajos de [8.17](08_verificacion_estadistica.md) y documenta su {ref}`cierre en 8.18 <e3-final-extended-audit>`. D00 y los tres conjuntos adicionales D01–D03 completaron los 19 sistemas: **796 trabajos**, incluidos 776 ajustes neuronales y cuatro ARIMA, además de dieciséis controles. Los 76 paneles, las tablas de análisis y las 16 figuras quedaron [archivados en GitHub](https://github.com/jassan-bit/DeepLearning_202630/tree/16a4504f6fb9208c7d6f89b65017194fa20470da/runs/e3_extended_audit/20261119T234320461006Z) bajo el mismo master y protocolo, tras recuperar las interrupciones. Este cierre exploratorio no transforma las cronologías conocidas y dependientes en confirmación independiente ni certifica la potencia completa de R3.
