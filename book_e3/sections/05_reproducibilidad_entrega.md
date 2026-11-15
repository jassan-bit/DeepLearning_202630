# 5 Reproducibilidad y entrega

## Procedencia del arranque

La revisión usa una instantánea del repositorio en el commit `0bdbc42bd724c6ffbbb1dec67564327adedb9883` y la guía *Entregable3DeepLearningProject.pdf* aportada en la conversación. Su SHA-256 es:

```text
d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364
```

El manifiesto `experiments/e3_inicio_20261115/fuentes.json` identifica las fuentes revisadas. Las tablas editables de requisitos y correcciones están en esa misma carpeta. El arranque contiene decisiones de planificación; no genera tag de preregistro, métricas nuevas, pesos o exportaciones.

## Evidencia ejecutable de los bloques A y B

El [avance A y B](08_bloques_a_b.md) enlaza inventarios, tabla maestra y paridad de tres artefactos TorchScript heredados. Sus verificaciones offline están automatizadas en `.github/workflows/verify-e3-handoff.yml`; el código y las fuentes de cada informe se conservan en `experiments/e3_bloques_ab_20261115/`. Los controles de fuga completos, la prueba formal de δg y la exportación de los demás sistemas siguen pendientes.

## Estructura final del informe

La organización actual conserva las direcciones existentes y añade diagnóstico y diseño preliminar. Antes de la entrega se reorganizará según el Cuadro 7 de la guía (p. 58), conservando enlaces o concordancia:

| Capítulo | Contenido final |
|---|---|
| 0 | Introducción, resumen ≤250 palabras, contribuciones y URL del dashboard. |
| 1 | Trazabilidad de correcciones. |
| 2 | Datos, EDA y auditoría. |
| 3 | Modelos benchmark y exportación. |
| 4 | Revisión de literatura y novedad. |
| 5 | Diseño del modelo y demostración formal. |
| 6 | Implementación, pruebas y ablaciones. |
| 7 | Protocolo y validación estadística. |
| 8 | Evaluación ampliada y robustez. |
| 9 | Dashboard y despliegue. |
| 10 | Conclusiones, limitaciones, autoría y uso de IA. |
| 11 | Reproducibilidad y bibliografía. |

## Construcción del libro

Desde la raíz del repositorio, en el entorno fijado por `requirements-book.lock.txt`:

```bash
.venv/bin/jupyter-book build book_e3 --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py --book-dir book_e3
```

Estos comandos construyen y comprueban documentación. La ejecución de notebooks permanece desactivada en la configuración. La construcción no entrena modelos ni descarga mercado. El reporte de verificación del PR deberá distinguir revisión estructural de una compilación real.

## Reproducibilidad experimental pendiente

Se requieren versiones exactas, semillas, configuración de determinismo, tablas maestras, tests y contenedor Docker probado. La reproducción de E2 se conserva con su alcance; no sustituye reproducción del modelo E3, sus tres datasets adicionales, exportaciones y dashboard.

La entrega final incluirá `MODEL_CARD.md`, `DATA_STATEMENT.md`, fuentes y artefactos autorizados, registro con DOI, URL Render y prueba de humo con contingencia. No se copiarán a la publicación pesos o datos cuya distribución no esté autorizada.

## Uso de herramientas de IA

Este arranque se preparó con asistencia de IA para revisar documentación, organizar requisitos, redactar el diagnóstico, formular una hipótesis técnica preliminar y programar las herramientas de auditoría y exportación heredada de los bloques A y B. La propuesta no se presenta como invención verificada del autor ni como experimento ejecutado. Se mantendrá una declaración actualizada de los usos efectivamente realizados; el autor deberá comprender y poder defender el código, la formulación y las demostraciones que finalmente entregue, conforme a §§6.5 y 15 de la guía.
