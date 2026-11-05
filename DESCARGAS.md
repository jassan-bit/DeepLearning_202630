# Descargas del Entregable 1 · GRU–LSTM–TCN

[Descargar desde la unidad 8 del Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1).

[Descargar los siete productos juntos](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/entregable1_completo.zip) · [Carpeta de descargas](https://github.com/jassan-bit/DeepLearning_202630/tree/entregables)

| Ítem | Archivo | Contenido |
|---|---|---|
| 8.1 | [Informe HTML](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_1_informe_html.zip) | Libro navegable, figuras y agregados de fase I y fase II D7 |
| 8.2 | [Código](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_2_codigo.zip) | EDA, arquitecturas, ejecutores de ambas fases, diseño canónico, estadística, pruebas, locks y BUILD.md |
| 8.3 | [Obtención del dataset](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_3_dataset_instrucciones.zip) | Instrucciones, descargador oficial, calendario y manifiestos |
| 8.4 | [Literatura y flujo](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_4_literatura.zip) | Siete estudios cripto, 40 referencias y selección de familias con sus límites |
| 8.5 | [Benchmark e inferencia](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_5_benchmark.zip) | Agregados de cuatro entradas exploratorias y estudio D7 con nueva reserva preespecificada |
| 8.6 | [Configuraciones](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_6_configuraciones.zip) | Locks, semillas, búsqueda, piloto, dimensionamiento y configuraciones por fase |
| 8.7 | [Declaración](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_7_declaracion.zip) | Métrica, margen, independencia, partición y registros públicos previos de fase II; cronología original de fase I |

Abrir los ZIP descomprimidos y leer `LEEME.md`. El HTML se inicia en `informe/intro.html`; si el navegador restringe recursos, ejecutar `python -m http.server 8000 --directory informe` y abrir `http://localhost:8000`. MathJax y los enlaces externos pueden necesitar internet.

Para trabajar con código, reunir los productos **8.2, 8.3 y 8.5** en una raíz común, conservando sus subdirectorios y guardando aparte los LEEME/manifiestos de cada producto. Si se usa el ZIP completo, primero extraer ese contenedor y después los ZIP internos. [BUILD.md](BUILD.md) explica entornos, preparación EDA y comandos de cada fase.

Los paquetes contienen código, metadatos, agregados y figuras. Las ilustraciones temporales fijan sus semillas —4101 en fase I y 8101 en fase II— antes de comparar errores; no se redistribuyen sus vectores tabulares. Los datos brutos, predicciones individuales, escaladores y pesos completos permanecen fuera de la exportación pública. Los derivados diarios de desarrollo 2023–2024 están versionados en el repositorio, pero se omiten de estos ZIP y se documenta su generación.

La **fase I** conserva su test ya observado de enero de 2025–agosto de 2026 y su inferencia exploratoria. Su GRU/D7 tiene 32 unidades y 3.777 parámetros; el R² 0,854173 pertenece a esa receta y periodo.

La **fase II D7** tiene diseño y código públicos anteriores al HPO, ocho candidatos y dos semillas por familia, un piloto de diez semillas nuevas por modelo y R=465 calculado antes de sus ajustes definitivos. Su baseline por desarrollo es GRU/D7-c03 de 16 unidades y 1.121 parámetros. La evaluación usa una reserva histórica de **01/09–30/10/2026**, sesenta días. Los datos se habían auditado por integridad/timestamps/esquema, sin uso previo para objetivos o resultados según los registros. No se afirma que los archivos nunca se abrieron ni que se emitieron pronósticos en vivo.

La potencia objetivo 0,80 es marginal entre semillas, bajo supuestos de normalidad y transferencia de DE; no es potencia temporal o conjunta. Las métricas de dos cohortes distintas no se comparan como una mejora causada por HPO. TCN conserva respaldo bibliográfico limitado y no se afirma utilidad financiera.

[Catálogo de versión](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/catalogo.json) · [SHA256SUMS.txt](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/SHA256SUMS.txt).

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt
```
