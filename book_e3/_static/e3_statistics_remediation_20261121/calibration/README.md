# Desarrollo estadístico v2 y plan anterior a la validación nueva

Componente preparado con asistencia de IA. No entrena modelos, genera predicciones
de mercado, habilita confirmación ni acredita R3. V3 conserva NO-GO y V4 permanece
abandonada. Las variantes bootstrap no son tres procedimientos primarios distintos.

## Diagnóstico del resultado anterior

Se releen los 1.024 registros ya existentes, sin repetir su Monte Carlo. En el
caso EPA con **12 semillas, 128 fechas y 20 modelos**, el MCS anterior rechazó
27/128 veces al nivel 10 %: 21,09 %, IC95 % individual 14,38–29,19 %. El rango
incluye 190 pares dependientes. `OLD_MCS_DIAGNOSIS.json` conserva conteos y hashes.

No se detectó un error algebraico evidente de signo, centrado o eliminación. El
uso de una dispersión bootstrap fija también aparece en el MCS estándar: no se
denomina un fallo de programación por sí mismo. Bajo EPA los modelos del DGP son
intercambiables y el sesgo finito de la raíz es común; no explica esa inflación.
Las colas finitas con pocas semillas, la dimensión del máximo y la extensión
cruzada/no lineal son mecanismos plausibles que no quedaron identificados como
causa única. Un ejemplo analítico AR gaussiano al cuadrado muestra además la
subestimación que puede producir un ancho de banda finito, especialmente con
phi=0,85; no constituye la varianza exacta del funcional cruzado.

## Propuesta y controles

`crossed_rmse_v2.py` mantiene el estimando **E_semilla[raíz(E_fecha(error²))]**.
No lo sustituye por RMSE de ensemble o MSE promediada. Implementa una modificación
bootstrap-t propuesta del MCS funcional: estima la covarianza completa entre
riesgos por semilla y por fecha, y la vuelve a estimar dentro de cada réplica.

El plug-in usa covarianza de las RMSE entre semillas más covarianza HAC de la
influencia temporal media. La doble contabilización de interacción sólo tiene
una interpretación conservadora dentro de una linealización cruzada aditiva;
con ancho de banda finito o dependencia arbitraria no es una cota garantizada.
La configuración conserva **bloque medio 12 y lag 8**, incluso en el estrés.

Se comparan, sobre las **mismas muestras bootstrap**:

1. El MCS v1, extraído literalmente y fijado por hash, con su dispersión original.
2. La covarianza plug-in nueva y studentizador fijo.
3. La covarianza nueva y studentizador recalculado bootstrap-t.

La segunda comparación permite aislar la re-studentización dentro de la nueva
construcción de covarianza. La comparación v1/v2 incluye dos intervenciones y
no identifica por sí sola una causa única. `LEGACY_REFERENCE.json` fija las
funciones originales; no se modifica ningún archivo del componente anterior.

Los dos contrastes propuestos por rival usan bootstrap básico y delta-HAC con
factor h=1 HLN/t(R−1), respectivamente. La referencia cruzada no es un teorema
de DM-HLN clásico. Aquí se reportan **nulos de igualdad bilaterales**; dirección
y margen práctico se examinan por separado. El MCS sigue siendo global y su
exclusión no se convierte en un tercer p-valor por rival.

Las guardas numéricas nuevas usan escala relativa. Una varianza degenerada no
produce un intervalo de certeza ni evidencia de calibración: se marca como no
soportada, se conserva p=1 como reserva y se informa un resultado inconcluso.

## Ejecución predeclarada

`VALIDATION_PLAN.json` fija cuatro escenarios nuevos: EPA ordinaria, EPA con
dependencia fuerte, EPA sólo entre semillas y una alternativa. Son **640
réplicas**, **999 remuestreos por réplica**, máximo **300 s** y dos hilos. La
semilla de validación 2026112191 no es la semilla anterior 2026112107. El control
v1 reutiliza los riesgos bootstrap ya calculados y no añade remuestreos.

El criterio de cribado de desarrollo es, por caso nulo, un límite superior del
IC95 % central Clopper–Pearson de rechazo global menor o igual a 0,13. Es una
tolerancia exploratoria de 0,03, no una certificación de nivel exacto 0,10 ni una
garantía simultánea. El plan fija también diagnósticos de FWER/intervalos,
seis comparaciones pareadas de mecanismos con Holm, potencia de un subconjunto
de dos métodos y tratamiento de resultados no soportados. La precisión limitada
puede dejar el resultado inconcluso. No se retocan métodos o bloques después.

Antes de ejecutar se requiere el commit público real de plan/fuentes y un
`PUBLIC_ANCHOR.json` verificado por el proceso principal. `CODE_MANIFEST.json`
congela ocho archivos de ejecución y pruebas. Una carpeta de salida existente
se rechaza; un límite de recursos conserva los registros parciales y no abre
otra corrida para buscar un resultado favorable.

Desde la carpeta `experiments/e3_statistics_remediation_20261121` de un checkout
completo, usando las dependencias fijadas:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python calibration/run_validation.py \
  --root calibration --anchor calibration/PUBLIC_ANCHOR.json
```

La ejecución original se lanza una sola vez, una vez verificado el anclaje:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python calibration/run_validation.py \
  --root calibration --anchor calibration/PUBLIC_ANCHOR.json \
  --output calibration/validation_fresh --execute
```

Para reproducir posteriormente, preservar la salida original y usar un directorio
nuevo con el mismo plan y semillas; no seleccionar entre réplicas. Los resultados
y su decisión posterior se documentarán en `RESULTS_DECISION.md`, separados de
este plan previo. La calibración de tres métodos primarios y la potencia final
de mercado permanecen sin acreditar aunque algún cribado resulte favorable.
