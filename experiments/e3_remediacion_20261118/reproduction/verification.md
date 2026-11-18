# Verificación automática del paquete científico v1

El 18 de noviembre de 2026 se extrajo `reproduction_bundle_v1.tar.gz` en una
carpeta nueva, se descargaron desde los commits públicos fijados sus once
insumos y se construyó la imagen Docker científica desde ese archivo extraído.
La ejecución CPU terminó con código **0**, sin OOM, en **4,84 s** de tiempo
medido por el cliente. Este tiempo corresponde a reconstruir agregados; no
incluye descarga, construcción de imagen o entrenamiento.

**Resultado comprobado:** 17 sistemas, 143 trayectorias históricas guardadas,
608 fechas comunes; **11 de 11 CSV idénticos byte a byte**, con 431 filas y
3.963 pares numéricos comprobados. Las 101 combinaciones de campo y unidad
tienen discrepancia absoluta máxima **0** y cero incumplimientos de tolerancia.
Los valores ausentes, textos y decisiones también se conservaron. No se
combinaron métricas de distinta unidad en un máximo global.

La imagen usó Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 y Matplotlib 3.10.8;
se comprobaron las doce versiones del lock y la ausencia de Torch. El proceso
corrió con red deshabilitada, raíz e insumos de sólo lectura y usuario sin
privilegios. Los límites configurados fueron 1 CPU, 1 GiB, 128 procesos y 64 MiB
de `/tmp`; son límites, no mediciones de consumo máximo.

Las ocho pruebas de contrato pasaron desde el paquete extraído. Incluyen
controles negativos de desalineación, pérdida de columnas/filas, duplicados,
sustitución de ausencia por cero, no finitos e intervalos alterados. Reconstruir
el archivo desde sus 31 entradas extraídas produjo exactamente el mismo SHA-256.

| Evidencia | Archivo |
|---|---|
| Estado de Docker, imagen, versiones y alcance | [container_verification.json](container_verification.json) |
| Recibo completo, hashes y comparación de cada tabla | [reproduction_report.json](reproduction_report.json) |
| Diferencias por campo y unidad | [field_discrepancies.csv](field_discrepancies.csv) |
| Registro de construcción y ejecución | [container_build.txt](container_build.txt), [container_run.txt](container_run.txt) |
| Código y comandos para repetir | [README.md](README.md), [reproduce.py](reproduce.py) |
| Archivo de 70.337 bytes | [reproduction_bundle_v1.tar.gz](reproduction_bundle_v1.tar.gz) |
| Entradas, tamaños y SHA-256 del archivo | [reproduction_bundle_v1.manifest.json](reproduction_bundle_v1.manifest.json) |

SHA-256 del archivo v1:
`c08265b7ff75793883359dc5bd84b2493975003c23848ec21d9f3de21d2509bd`.

Se ejecutaron **0 ajustes de modelos, 0 inferencias y 0 recalibraciones**. Se
recalculó el análisis estadístico posterior existente y su suplemento desde
predicciones congeladas; no se reprodujo el entrenamiento de los 143
predictores. Este resultado tampoco recupera el cegamiento del calendario ni
certifica novedad o R1/R2/R3. La reproducción humana independiente con acta,
el depósito persistente/DOI y la verificación final de derechos de
redistribución continúan pendientes. No se fabricó una firma ni un DOI.
