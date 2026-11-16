# Comparación de los periodos originales de calibración y prueba

Análisis exploratorio registrado después del benchmark y antes de este nuevo
cálculo. Usa exclusivamente las trayectorias originales guardadas, emparejadas
por modelo, semilla y checkpoint: RMSE_test608 − RMSE_calibración53.

La familia principal aplica t bilateral a seis diferencias por red (5 gl), IC95 %
marginal y Holm13. Se condiciona a ambos calendarios y a la aproximación de las
diferencias entre inicializaciones. No se emparejan fechas ni se tratan como iid.
La media de RMSE individuales no se sustituye por la RMSE de un ensemble.

La sensibilidad enumera 64 asignaciones de signos bajo simetría, con familia Holm13
separada. Su resolución bilateral mínima 2/64=0,03125 no permite rechazo tras Holm13
a 0,05; no se escoge el procedimiento con p menor. Los tres clásicos R=1 permanecen
descriptivos: no se inventan seis semillas ni IC de inicialización para ellos.

La calibración ya se usó para piloto e intervalos y NO se renombra validación HPO
independiente. Una diferencia de periodos no identifica sobreajuste; no rechazo
no demuestra consistencia ni equivalencia. No se ejecuta el antiguo plan714 ni
se mezclan riesgos CPU/GPU. No hay nuevos ajustes, predicciones o cambios en A2/RQ1.

Las identidades y verificaciones del exportador original se heredan del JSON
fijado por SHA; sus comprobaciones sobre los brutos originales no se repiten aquí.
La política eval/FP32/batch1 y los metadatos originales disponibles se documentan
en source_admission.json, sin fingir una observación nueva del hardware.

Repetición desde el repositorio con NumPy/SciPy y una carpeta nueva:

```sh
python scripts/e2_original_period_comparison.py --output /tmp/e2-period-repeat --register-plan
python scripts/e2_original_period_comparison.py --output /tmp/e2-period-repeat --execute
```
