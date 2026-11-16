# Prueba CPU e intersección–unión completas

Se ejecutaron 78 predictores congelados sobre las 608 fechas admitidas del test:
47.424 predicciones nuevas, con el cargador eager archivado, CPU FP32, batch1,
dos hilos y MKLDNN desactivado. No se ajustó ningún peso ni hiperparámetro.

Train655 y calibración53 se reutilizan sólo tras verificar identidad exacta de
inputs FP32, targets, fechas, escalador, checkpoints, estados, cargador y runtime.
El test se transforma con ese mismo escalador de entrenamiento. El plan se
registró antes de las nuevas predicciones; costes anteriores ya conocidos y
periodo de E1 conocido hacen de este análisis una ampliación exploratoria.

Se calcularon realmente ambos componentes para diez candidatos frente a GRU32:
ganancia de RMSE CPU mayor que δ y diferencia de latencia CPU menor que cero.
La precisión usa media de RMSE individuales, seis trayectorias fijas y 20.000
remuestreos móviles no circulares de 14 días. El coste usa los 12.000 tiempos CPU
ya guardados, remuestreando 20.000 veces sus 100 bloques completos con seis pares
de semillas. Se forma max(p_ganancia,p_coste) y después Holm10. Los IC son
marginales. Ganancias y RMSE están en σ₇ original; diferencias de coste e IC, en
segundos. No se mezclan precisión GPU y costes CPU.

La nueva IUT CPU está ejecutada: los diez p de coste, IUT y Holm son 1, y ninguna
mejora conjunta queda establecida. SAE alcanza RMSE CPU 0,003803430626 frente a
0,004001161981 de GRU32; su ganancia 0,000197731356 es menor que δ. Esto no
prueba equivalencia ni imposibilidad de mejora en otro hardware o periodo.

La IUT del antiguo plan714 no se renombra como ejecutada y la puerta original
de paridad GPU sigue sin acreditarse. A2 y RQ1 originales se conservan intactos.
La calibración sigue siendo calibración; no se convierte en validación HPO
independiente. Bootstrap temporal y de mediciones requieren sus supuestos; seis
semillas de una misma cronología no son seis mercados.

`analysis_plan.json` fija métodos, fuentes y reutilización; `inference/` conserva
predicciones, arrays, estados y runtime. `statistics/` contiene los dos
componentes, IUT, ajustes, recibo y resumen. `statistics_environment.json` registra
el entorno NumPy/SciPy antes del cálculo; `verification_report.json` coteja las
47.424 claves únicas y las diez identidades de intersección–unión sin repetir
estadística. El programa ejecutado es `scripts/e2_cpu_test_iut.py` y su SHA figura
en el plan y los recibos. La ejecución requiere planes y hashes inmutables,
entradas originales, pesos y librerías indicados; no es sólo un informe guardado.
