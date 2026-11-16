# 2 Estándar de profundidad exigido

La guía establece un único estándar: **profundidad teórica con evaluación rigurosa**. Los requisitos siguientes se mantienen como condiciones de cierre; no se consideran cumplidos por disponer de código o por publicar el libro.

| Requisito | Exigencia de la guía | Evidencia actual y trabajo restante |
|---|---|---|
| Sustento teórico | Al menos una propiedad formal, con supuestos explícitos y verificación numérica. | La cabeza latente tiene demostración y comprobaciones artificiales. La garantía no cubre automáticamente el pipeline OHLCV ni acredita un beneficio de mercado. |
| Evaluación amplia | Tres conjuntos adicionales de la misma tarea, de naturaleza distinta y elegidos antes de ver sus resultados. | No se han seleccionado ni evaluado para E3. |
| Familia completa | Original frente a todo el Top 10 y todos los baselines fuertes, con Holm. | El conjunto de trabajo conserva los 16 sistemas de E2; el protocolo definitivo debe enumerar todos los rivales y sus estimandos. |
| Potencia y repeticiones | Potencia por contraste de al menos 0,90, conjunta reportada y al menos diez semillas o el número mayor que requiera el análisis. | E2 tiene seis semillas por red y una limitación de potencia documentada. E3 requiere piloto y dimensionamiento propios antes del cierre. |
| Pruebas por contraste | Al menos tres pruebas distintas, adecuadas a la tarea y con control familiar. | No se han ejecutado contrastes confirmatorios del candidato. |

## 2.1 Compromiso previo al protocolo

La guía exige `experiments/estandar.md`, con fecha y tag Git anteriores al congelamiento del protocolo. Los documentos de desarrollo actuales **no sustituyen ese registro** ni se retrofechan como preregistro. El estándar no se reduce después de conocer resultados.

El número de semillas se fija antes de confirmar. Diez es un mínimo, no una garantía de potencia. El tratamiento de comparadores deterministas deberá quedar explícito: repetir la misma trayectoria no crea semillas independientes. Las fuentes de incertidumbre entre fechas, series e inicializaciones deben distinguirse.

## 2.2 Métrica, márgenes y afirmaciones

Se conserva la RMSE en unidades originales y el margen práctico heredado **δ=0,00025772842664981106**. El margen de brecha **δg** es otro parámetro y no se supone igual a δ. Los análisis posteriores de E2 conservan sus fechas, unidades y alcance; no se transforman en pruebas confirmatorias de E3.

La superioridad requiere además relevancia práctica, tamaño de efecto, adecuación del modelo, robustez de la conclusión y consistencia en los conjuntos adicionales. Mejorar una métrica puntual, demostrar una cota o no rechazar una diferencia no establece por sí solo superioridad o equivalencia.

La [matriz inicial de requisitos](01_alcance_requisitos.md) se conserva como anexo de trazabilidad. El estado vigente se desarrolla en las secciones 4 a 16. La [sección 8](08_verificacion_estadistica.md) organiza el protocolo propuesto y sus decisiones pendientes; no sustituye el registro previo exigido. Las secciones [9](09_dashboard_predictivo.md) y [10](10_despliegue_render.md) especifican el dashboard y su despliegue. La [sección 14](14_evaluacion_calificacion.md) mantiene los criterios y condiciones más restrictivas para la nota máxima, y la [15](15_integridad_academica.md) conserva las reglas de integridad, sin asignar una nota actual. La plantilla A.1 del [apéndice](apendice_a_plantillas.md) prepara la declaración del estándar; no la registra ni la congela.
