# Entregables independientes E2, versión 3

Cada ZIP se extrae y ejecuta por separado. No requiere el sitio web, Git ni archivos de otro entregable. Las instrucciones detalladas están en el `LEEME.md` de cada carpeta.

```sh
python run.py
```

Once ejecuciones predeterminadas usan sólo la biblioteca estándar de Python. D02 requiere instalar `requirements_cpu.txt` y comprueba trece puertos con entradas sintéticas en CPU. Su prueba breve no vuelve a entrenar el estudio completo. Los ajustes reales de CPU y sus recetas están incluidos como evidencia y código revisable; no se presenta el controlador histórico con rutas absolutas como una ejecución portátil sin cambios.

| Archivo | Contenido | MB |
|---|---|---:|
| [01_informe_jupyter_book.zip](01_informe_jupyter_book.zip) | Informe Jupyter Book completo y fuentes | 93.89 |
| [02_codigo_modelos.zip](02_codigo_modelos.zip) | Código y ejecución de las trece redes | 11.49 |
| [03_datos_contrato.zip](03_datos_contrato.zip) | Datos, Data Card y contrato | 0.58 |
| [04_preregistro_auditoria.zip](04_preregistro_auditoria.zip) | Preregistro y auditoría de datos | 42.03 |
| [05_ficha_ruta_F2a.zip](05_ficha_ruta_F2a.zip) | Ficha metodológica F2a y evidencia estadística | 27.53 |
| [06_benchmark_comparaciones.zip](06_benchmark_comparaciones.zip) | Benchmark, métricas y comparaciones | 18.00 |
| [07_registro_experimentos.zip](07_registro_experimentos.zip) | Registro experimental y fichas de modelo | 22.25 |
| [08_techo_incertidumbre.zip](08_techo_incertidumbre.zip) | Techo, incertidumbre y errores | 0.96 |
| [09_arquitecturas_no_aplica.zip](09_arquitecturas_no_aplica.zip) | Justificación de NAS no aplicable | 0.05 |
| [10_paper_guia_replica.zip](10_paper_guia_replica.zip) | Paper guía y réplica parcial | 0.50 |
| [11_reproducibilidad.zip](11_reproducibilidad.zip) | Lista de reproducibilidad y evidencia | 64.55 |
| [12_presentacion_defensa.zip](12_presentacion_defensa.zip) | Presentación de defensa y notas del orador | 28.34 |

D01 materializa el libro y MathJax locales. Sus dos ZIP científicos opcionales están catalogados por SHA256 y enlace HTTPS inmutable; D04 y D11 incluyen ambos íntegros. D03 incluye los CSV reales y los 21 arrays de las cohortes 655/53/608. Cada carpeta conserva las limitaciones de sus estudios y las generaciones anteriores de evidencia.

Las reproducciones numéricas opcionales requieren `requirements_scientific.txt` en D05/D06, o `requirements_controls_replay.txt` en D04. Desde la carpeta extraída correspondiente:

```sh
# D05
python scientific_reanalysis/run.py --program period_comparison --output replay_period
# D06
python scientific_reanalysis/run.py --program cpu_test_iut --output replay_cpu_IUT
# D04
python control_replay/run.py --output replay_controls
```

## Verificación de esta entrega

Las doce ejecuciones predeterminadas pasaron desde los ZIP extraídos fuera del repositorio. También se reprodujeron dos análisis estadísticos guardados y 28 comprobaciones aritméticas de los catorce controles publicados. Los informes adjuntos conservan los alcances y límites científicos; estas comprobaciones de entrega no cambian los resultados originales. No se efectuaron nuevos ajustes ni inferencias de modelos de mercado durante el empaquetado y su QA.

Fuente exacta: `78271c46a1fe7a9544e8d1ec46e096dafb7cf9e9`. Los cuatro comandos finales de QA consumieron 71.111 segundos, registrados una sola vez en el presupuesto existente.

Verificación de archivos: `SHA256SUMS` contiene los doce ZIP y `manifest.json` incluye las huellas de los archivos de esta entrega. `execution_report.json` registra las doce ejecuciones; los otros tres informes de replay registran sus comparaciones numéricas. Las versiones v1/v2 y el suplemento C3 permanecen intactos.
