# Cierre documental del piloto V3 con predicciones guardadas

Se declara asistencia de IA en revisión, código, análisis y redacción. Este paquete verifica resultados ya calculados; **no entrena, reanuda un optimizador, importa modelos, ejecuta inferencia o genera nuevos remuestreos**. Tampoco cambia la regla de continuación del piloto ni acredita R1/R3.

## Fuente y conservación

- Repositorio: `jassan-bit/DeepLearning_202630`.
- Código del piloto: `8f65eac34aaf49293924ae8328c235cedbdd2866`.
- Ejecución: `20261120T031424176339Z`.
- Resultados inmutables: `8eaa581bc34b615a8507c9e384a3e4be29084720`.
- Ruta original: `runs/e3_v3_development_pilot/20261120T031424176339Z/`.
- Snapshot: secuencia 93, manifiesto con 259 archivos de payload.

`archive/` conserva **19 archivos originales**: el manifiesto del snapshot y un subconjunto de 18 recibos, metadatos y CSV. Se preservan sus bytes, incluidos los finales CRLF de los CSV. `source_objects.json` registra sus SHA de objeto Git. La verificación comprueba esos objetos, los tamaños y SHA-256 del manifiesto, cinco hashes de artefactos del recibo, los enlaces entre plan/admisión/launcher y el estado de los 30 ajustes. **No se descargaron checkpoints ni se reprodujeron los 259 archivos o los 93 snapshots.** Las huellas de checkpoints presentes en el recibo se conservan como metadatos originales, sin afirmar una nueva comprobación de sus bytes.

Los 796 trabajos de la auditoría aditiva son otro estudio y se conservan con su propio alcance. Ningún archivo de este paquete los sustituye o los presenta como ajustes V3.

## Resultado que se conserva

El piloto completó 15 ajustes HPO y 15 finales, con cinco recetas neuronales, tres semillas finales y dos controles analíticos. La evaluación comprende 17 trayectorias y 935 filas, pero sólo **55 fechas distintas de desarrollo ya expuesto**, del 7 de septiembre al 31 de octubre de 2024. No se abrieron miembros de calibración/prueba del NPZ ni se usaron D01–D03. La selección empleó otras 54 fechas, separadas por siete purgadas.

La regla original usa la **media de las RMSE individuales de tres semillas**. No usa la RMSE de una predicción media entre semillas. `atom_crps` obtiene 0,0021299157684633078, frente a 0,002073832857860484 de `atom_mse_only`: **2,7043 % peor**. El límite para mejorar al mejor control en al menos 2 % era 0,0020323562007032742. El veredicto original y el recalculado coinciden: **no se supera la heurística**. La completa ocupa el quinto puesto entre las siete recetas por ese criterio; la mejora del 1,2685 % frente a la aditiva no cambia la decisión.

La menor RMSE descriptiva de la variante sin CRPS no la promueve automáticamente a modelo final. El efecto empírico no acredita prioridad de la composición, ni el fallo de la heurística refuta por sí solo la novedad. Los controles de masas fijas tienen 1236 parámetros, frente a 1262 en las variantes de masas aprendidas; esa diferencia impide interpretar su contraste como capacidad perfectamente igualada.

## Análisis posterior y límites

`generated/` añade una recomputación de RMSE/MAE, comparaciones de todas las recetas, errores firmados, correlaciones de residuos con desfases de uno a siete días y los dos cortes mensuales completos: septiembre (24 fechas) y octubre (31). Son diagnósticos **especificados después de conocer el resultado**, sin valores p, intervalos, bootstrap, prueba confirmatoria o selección de un mes favorable.

Los errores de ventanas solapadas y las tres semillas no son 165 observaciones independientes. Una correlación pequeña en algún desfase no demuestra independencia. La figura presenta dispersión entre semillas y correlaciones descriptivas, no intervalos de confianza o pruebas.

Se conservan los CRPS_U originales y se comprueba su agregación entre semillas. **No se recalcula CRPS_U desde la ley atómica**, porque este subconjunto no contiene las leyes por observación. No hay intervalos predictivos guardados para evaluar cobertura probabilística. Tampoco hay predicciones de ajuste suficientes para cerrar las Tablas A/B o calcular la brecha ajuste/validación. Las puntuaciones de selección y evaluación pertenecen a fechas y volatilidades distintas y no sustituyen esos riesgos.

Las cinco recetas eligieron la tasa máxima de la cuadrícula prefijada (0,003). Se reporta ese límite de sensibilidad; no se amplía la búsqueda usando el resultado de evaluación. Tres semillas y este calendario pueden informar un diseño futuro, pero no certificar retrospectivamente potencia prospectiva, margen formal o R3. `adequacy_scope_matrix.csv` separa lo comprobado de lo que permanece abierto.

## Reproducir el reanálisis

Requiere Python, NumPy y, para la figura, Matplotlib. No necesita Torch, GPU, credenciales ni red:

```bash
python analyze_saved_pilot.py
python verify_saved_pilot.py
# Verificación completa de archivo y aritmética sin escribir ni recrear figuras:
python analyze_saved_pilot.py --verify-only
```

`--no-figures` permite ejecutar sólo verificación y tablas. `--output /tmp/e3-pilot-derived` dirige las tablas a un directorio externo; `--root` fija explícitamente la ruta del paquete si se desea. Los valores por defecto se resuelven junto al script y no dependen del directorio de trabajo. `--verify-only` necesita NumPy pero no importa Matplotlib; verifica 19 archivos de origen y llama al verificador aritmético de sólo lectura. `verify_saved_pilot.py` usa únicamente la biblioteca estándar y una segunda formulación aritmética (`math.fsum`) para comprobar 17 trayectorias, siete medias por receta, 34 cortes mensuales y 119 correlaciones. Ninguno de los dos scripts escribe sobre `archive/`.

`verification.json` y `manifest.json` registran el entorno, los alcances y las huellas de esta entrega derivada. Las políticas y derechos de la fuente de los datos siguen su declaración histórica; conservar resultados derivados no modifica sus condiciones de distribución.
