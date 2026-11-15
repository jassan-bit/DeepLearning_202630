# Efectos estandarizados y contraste de erosión por régimen

El plan se guardó antes de este cálculo, después de conocer A2. Es una ampliación
exploratoria; conserva los pronósticos, A2 y las veinte decisiones RQ1 originales.

`standardized_effects_20.csv` aporta d de Cohen pareado e IC percentiles entre
las seis ganancias brutas por semilla. Su incertidumbre es de optimización,
condicionada al calendario; no es un efecto de 608 fechas independientes.
Persistence conserva una trayectoria fija: no se crean seis ajustes clásicos.
`paired_seed_gains_120.csv` conserva los valores que permiten auditarlo.

`rq2_primary_Holm10.csv` contrasta erosión mayor que δ frente a GRU32. Calcula
RMSE por trayectoria dentro de cada régimen, después promedia las seis RMSE.
Se remuestrea todo el calendario en bloques móviles no circulares, conservando
las etiquetas causales de régimen y los predictores fijos. La decisión corresponde
sólo al bloque 14 y Holm10; 7/28 son sensibilidad descriptiva, sin elegir el p menor.
`causal_regime_labels_608.csv` y `rq2_support.json` documentan disponibilidad,
soporte y bloques solapados; esos bloques no son observaciones independientes.

Los IC son marginales. La aproximación requiere dependencia suficientemente
corta y una longitud de bloque adecuada; no se afirma calibración exacta ante
cualquier cambio de régimen. No hay potencia RQ2 demostrada. No rechazo no
significa robustez, equivalencia ni ausencia de erosión. Los campos no estimables
se conservan; no se sustituyen por cero ni se generan sorteos nuevos para ocultarlos.

Reproducción: desde el repositorio, con Python y NumPy instalados, registre en
una carpeta NUEVA y ejecute sin modificar los bytes registrados:

```sh
python scripts/e2_effects_rq2_closure.py --output /tmp/e2-effects-rq2-repeat --register-plan
python scripts/e2_effects_rq2_closure.py --output /tmp/e2-effects-rq2-repeat --execute
```

Plan, recibo, hashes de entradas/programa y versiones permiten auditar la ejecución.
Los tiempos de registro cambian al repetir; las semillas y resultados son reproducibles.
