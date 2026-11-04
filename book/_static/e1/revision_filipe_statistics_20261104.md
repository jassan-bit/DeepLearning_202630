# Auditoría independiente del complemento estadístico de CV de Filipe — 04/11/2026

**Resultado: PASS, sin discrepancias numéricas ni bloqueadores detectados.**

El verificador independiente reconstruyó los resultados desde las 21.960 predicciones de la CV actual y las diez semillas por modelo/representación. No importó el analizador principal ni su helper numérico para recalcular; no entrenó, leyó mercados brutos o abrió test. La fecha del protocolo antecede a estos cálculos y el commit registrado es `db77f783377f42f9297d2a849aa771f16bcda4f2`; la selección y los datos de 2024 ya estaban expuestos.

## Comprobaciones

- Nueve contrastes pareados por semillas: medias, DE, IC t marginales, estadísticas, p-valores, d_z y Shapiro; ajuste global de doce al incluir los tres controles de preprocesamiento.
- Dieciocho DM calculados directamente con pérdidas cuadradas medias sobre semillas por fecha, Bartlett-HAC 14/7/28 y factor HLN de h=1; Holm en los seis primarios.
- Quince diagnósticos: Ljung–Box calculado desde autocorrelaciones y referencia χ², BDS mediante la implementación estándar sobre los residuos originales; Holm-15.
- Dieciocho intervalos de bootstrap por bloques de fechas y 21 frecuencias de ranking, recreados con 2.000 sorteos por longitud (14, 7 y 28 días), sin remuestrear semillas o reentrenar.
- Seis sensibilidades MDE/R orientativas y seis probabilidades de superioridad con intervalos binomiales exactos cuando no hay empates.
- Ochenta y siete valores ACF; 25 hashes de entradas/salidas, más las firmas de analizador/helper.
- Tres curvas reales del corte interno de diciembre de 2024, semilla 4101: MLP 25/37 épocas seleccionadas/registradas; LSTM 24/36; CNN1D 99/100. Se verificó la inversión de escala de Y y las fechas.

## Resultados interpretativos comprobados

| Comparación por semillas | Diferencia RMSE | p ajustado global Holm-12 |
|---|---:|---:|
| MLP − LSTM | -0.0002863406 | 0.1033521914 |
| MLP − CNN1D | -0.0001805249 | 1 |
| MLP − Persistencia | 0.0012601042 | 1.336632282e-06 |
| LSTM − CNN1D | 0.0001058156 | 1 |
| LSTM − Persistencia | 0.0015464448 | 9.285397537e-09 |
| CNN1D − Persistencia | 0.0014406291 | 1.956419529e-06 |

Ningún par neuronal se distingue al nivel 0,05 bajo Holm-12. Las tres diferencias neuronales menos persistencia son positivas y rechazadas bajo el alcance de aleatorización de entrenamiento condicionada al calendario. No representan confirmación en otro periodo.

Ninguno de los seis DM primarios rechaza tras Holm-6. Los p-valores de red frente a persistencia son:
- MLP: 0.06901894695.
- LSTM: 0.05111444822.
- CNN1D: 0.05531747708.

Los IC de bootstrap L14 de la diferencia de media de RMSE por semilla frente a persistencia son marginales y positivos:
- MLP: [0.0003342840; 0.0022572606].
- LSTM: [0.0005280828; 0.0026420823].
- CNN1D: [0.0003919079; 0.0024582710].

Esto no contradice DM: cambian el funcional, la referencia y el tratamiento de multiplicidad. No debe escogerse la conclusión favorable entre ambos ni llamar simultáneos a los IC marginales.

Los quince diagnósticos rechazan sus hipótesis de referencia tras Holm. BDS identifica una desviación de i.i.d.; no prueba exclusivamente no linealidad aprovechable ni una mejora futura cuantificada. La dependencia del objetivo móvil y los cambios mensuales de modelo limitan la calibración.

La sensibilidad del diseño procede de dispersión observada retrospectivamente. Los R orientativos frente a un efecto de tamaño 5 % de la RMSE de persistencia son 19/9/28 para MLP/LSTM/CNN1D bajo la aproximación registrada. No acreditan un piloto previo o potencia prospectiva 0,80 ni dimensionan la prueba de superar 5 % sin un efecto alternativo definido.

Las probabilidades de superioridad se refieren a aleatorizaciones de entrenamiento en el calendario fijo. Sus IC exactos son marginales, no afirmaciones adicionales ajustadas dentro de la familia de doce. Para la ganancia >5 %, la decisión utiliza su prueba unilateral contra el borde; el IC bilateral y d_z sin centrar en el borde son resúmenes descriptivos.

## Reproducción

Con el entorno de entrenamiento fijado y los archivos por corrida restaurados:

```bash
python scripts/verify_filipe_statistics.py --output /tmp/filipe_statistics_independent_audit.json
```

Las fuentes, resultados y protocolo permanecen intactos. El informe JSON contiene recuentos y alcance de todas las verificaciones.
