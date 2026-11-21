# Adecuación retrospectiva del piloto CPU V4

Asistencia de IA declarada. Este componente amplía los diagnósticos a las
1.210 predicciones ya guardadas de V4 y sus controles: cuatro recetas neuronales
por cinco semillas y dos controles analíticos ejecutados una vez. No modifica
el resultado adverso ni la decisión de abandonar V4. No vuelve a ejecutar los
diagnósticos anteriores de la aditiva, V3 GPU o la transferencia de 20.400 filas.

El plan se escribe después de conocer esos calendarios y resultados. Su anclaje
público anterior al nuevo cálculo fija su alcance, pero no recupera independencia
ni constituye un preregistro de confirmación. El programa exige ese anclaje,
fuentes congeladas y hashes de 46 entradas antes de evaluar los datos.

## Qué se calcula

Cada una de las 22 trayectorias conserva las mismas 55 fechas consecutivas,
del 7 de septiembre al 31 de octubre de 2024. Los residuos son objetivo menos
predicción. No se agrupan 5×55 celdas como observaciones independientes ni se
cuentan cinco copias de los controles analíticos. Los dos cortes mensuales se
reportan completos **dentro del calendario disponible**: septiembre7–30, 24
fechas; octubre1–31, 31. Septiembre no es un mes calendario completo.

La familia fija tiene **88 posiciones**: Ljung–Box a 7 y 14 rezagos, ARCH-LM a
7 y Mincer–Zarnowitz conjunto con HAC Bartlett a 14 para cada trayectoria.
Los rezagos son días/observaciones. Holm se calcula sobre las 88 posiciones
completas. Una estadística no soportada conserva su posición con p=1 únicamente
para la corrección y su p marginal queda ausente; no se llama adecuación aprobada.

Todos los valores p son **asintóticos y condicionados a los pesos de esa
trayectoria**, con supuestos adicionales. Con 55 fechas no se certifica su
calibración finita. Ljung–Box clásico no es robusto a heterocedasticidad; ARCH
requiere regularidad del LM y no convierte heterocedasticidad en inutilidad de
la media pronosticada. Mincer–Zarnowitz usa OLS auxiliar, HAC Bartlett/HC1 y
contrasta conjuntamente intercepto cero y pendiente uno; no reentrena el modelo.

MZ se calcula en coordenadas de predicción estandarizadas para evitar que la
unidad pequeña de sigma7 elimine artificialmente un rango por `pinv`. El nulo y
la covarianza se transforman de forma coherente. Covarianza degenerada, predictor
constante o ajuste exacto a precisión numérica se declaran no soportados.

El solapamiento de objetivos móviles motiva examinar dependencia, **pero no la
impone en errores óptimos de predicción a un paso**: si el pronóstico fuera la
esperanza condicional dada la información disponible, sus errores serían una
diferencia de martingala. No se explica automáticamente un rechazo de Ljung–Box
por compartir retornos. Tampoco la ausencia de rechazo acredita el modelo.

Se añaden 44 intervalos puntuales asintóticos para el **sesgo medio residual**
condicional, con HAC a 7/14. No son intervalos de RMSE poblacional entre semillas,
no son simultáneos y su cobertura finita no ha sido validada. Sirven para
sensibilidad a la ventana HAC, sin aportar nuevos p de superioridad. La forma
residual incluye asimetría, curtosis y Jarque–Bera como estadístico, sin interpretar
un p iid de normalidad ni exigir residuos normales para un pronóstico útil.

## Qué permanece bloqueado

La media de RMSE individuales, su desviación entre semillas y el RMSE de la
predicción promedio se muestran separados: son estimandos distintos. Los tres
contrastes neuronales pareados describen las cinco diferencias por semilla sobre
un solo calendario expuesto. Cohen dz es descriptivo; no se calcula una prueba
t o un certificado de potencia, ni se ignora la incertidumbre del calendario.

Se mantiene la guarda de diez semillas de la inferencia cruzada primaria. Cinco
semillas no la satisfacen. Estos diagnósticos auxiliares no son la batería R3.

Las curvas de veinte trabajos registran el objetivo de entrenamiento y la MSE
de parada. El objetivo CRPS compuesto no se divide por MSE de validación. La
razón RMSE de evaluación/parada compara calendarios distintos y un modelo cuya
época se seleccionó con la parada; no demuestra ausencia de sobreajuste. No hay
predicciones guardadas de ajuste/parada que permitan reconstruir el contraste
completo, y no se fabrican mediante nuevas inferencias.

V4 no tiene un calibrador de intervalos publicado ni distribución predictiva
guardada utilizable para PIT/cobertura. No se hereda el calibrador E2/aditivo ni
se inventan intervalos. La calibración lineal MZ es otro objeto.

## Ejecutar después del anclaje público

```bash
python audit_v4_saved.py \
  --input-root /ruta/al/repositorio/experiments/e3_next_evidence_20261121/development/cpu_pilot_results \
  --anchor PUBLIC_ANALYSIS_ANCHOR.json --out results
python audit_v4_saved.py \
  --input-root /ruta/al/repositorio/experiments/e3_next_evidence_20261121/development/cpu_pilot_results \
  --out results --verify-only
python -m unittest -v test_audit_v4_saved
```

`--verify-only` verifica hashes y miembros; no repite regresiones, remuestreos,
entrenamientos o predicciones. Las pruebas usan datos sintéticos y cotejos
numéricos con statsmodels; prueban software y no suponen calibración estadística.
Se conserva el primer error de un valor esperado manual en una fixture ACF y su
corrección anterior a cualquier ejecución real en el historial de pruebas.
