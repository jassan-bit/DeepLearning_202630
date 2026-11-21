# Especificación previa de un piloto CPU nuevo de desarrollo

Esta preparación se autorizó tras revisar la hipótesis matemática y el diagnóstico
fijo, bajo el encargo de avanzar experimentalmente con presupuesto acotado.
El diagnóstico no selecciona por sí mismo una receta. El piloto tiene identidad
propia y requiere que **su plan completo y sus fuentes** se anclen públicamente,
con QA aprobada, antes del primer ajuste o predicción de evaluación.

## Hipótesis y controles

Se contrasta si la revisión sigma-CRPS conserva una mejora descriptiva de RMSE
frente al mejor control en el calendario de desarrollo ya expuesto. Cambia
coordenada y normalización del score; no se atribuirá un eventual resultado
solamente a una de ellas. No se afinan el coeficiente 0,1 ni otra receta a partir
de este resultado.

| Receta | Parámetros | Papel |
|---|---:|---|
| `sigma_crps_v4` | 1262 | Nueva revisión de desarrollo, sin R1 heredada. |
| `u_crps_control` | 1262 | Pérdida y forward originales de V3, reajustados en la cohorte CPU común. |
| `mse_only_control` | 1262 | Control de ausencia de auxiliar, sin promoción automática. |
| `additive_positive_head` | 1210 | Control neuronal aditivo, con encoder/atención comunes. |
| `floor_only` | 0 | Control analítico `sqrt(A)`, una trayectoria determinista. |
| `floor_plus_mean_excess` | Un escalar de ajuste | `sqrt(A)+d_fit`, con exceso medio estimado sólo en los 532 ejemplos de ajuste. |

Las tres leyes atómicas tienen tensores iniciales idénticos por seed. El aditivo
no se declara igualado en capacidad ni en salida inicial. Los resultados previos
de GPU no son una comparación justa con estos nuevos ajustes CPU y no se mezclan
en sus medias.

## Datos efectivamente admitidos antes del plan

El NPZ existente tiene 460.090 bytes y SHA256
`186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
No se descargó una cohorte nueva. Se verificó el transporte completo y se
deserializaron exclusivamente `training_raw`, `training_y_original` y
`training_dates` mediante el loader congelado. Las matrices de calibración y
evaluación de 608 fechas no se abrieron ni puntuaron.

`CPU_DATA_ADMISSION.json`, SHA256
`185eb346a93ed28bd325589418c68a4403621c572b63ef1ce569fba723027d0e`,
comprueba esta geometría y las escalas de ajuste:

- 532 ejemplos de ajuste: 2023-01-08 a 2024-06-30.
- 54 fechas de parada: 2024-07-08 a 2024-08-30.
- Siete fechas de purga.
- 55 fechas de evaluación de desarrollo: 2024-09-07 a 2024-10-31.
- sU = 0,019540729290595266; sY = 0,02063308428641038.

Todas esas fechas pertenecen a desarrollo ya expuesto. Repetir el corte no
recupera independencia. El acceso anterior a las etiquetas se declara; el piloto
no se presenta como confirmación ni como el piloto de planificación de toda la
familia de rivales.

## Oportunidad común y decisión fijada

Se fijan cinco seeds nuevos pareados: 20261121–20261125, cuatro recetas por
seed, **20 ajustes nuevos**. Tasa común 0,003, transferida como elección común
del desarrollo previo; no se declara óptima para V4 ni constituye una búsqueda
de hiperparámetros nueva. AdamW, weight decay cero, batch 32, hasta 40 épocas,
paciencia ocho, clip uno; MSE sobre las mismas 54 fechas para la parada.

Cada fase tiene 120 segundos de pared con deadline inmutable y checkpoint del
optimizador/RNG. Cada proceso dispone de un límite externo de 150 segundos y
el ensayo completo de 3300 segundos, sin resetear sus relojes al reanudar. Los
2400 segundos suman oportunidades nominales de ajuste; no son consumo CPU real
ni tiempo total. Se registran tiempos efectivos, fases fallidas y sobrepasos.

La cohorte común es CPU, dos hilos, Python 3.12.14, NumPy 2.3.5 y Torch 2.8.0+cpu,
con determinismo y TF32 deshabilitado. No se pide otra GPU ni se reanuda el master
original. Las fuentes históricas se copian y fijan por hash; sólo se escriben
directorios nuevos del piloto.

La evaluación de las 55 fechas se hace **después de verificar los veinte ajustes**:
1100 salidas neuronales y 110 analíticas, **1210 salidas nuevas** si todos
completan. Los controles analíticos se evalúan una vez, no se replican en cinco
seeds. Son seis recetas y 22 trayectorias. Se calculan RMSE/MAE por seed y media
de RMSE entre cinco seeds. El criterio nuevo de continuación de desarrollo es
**al menos 2 % de reducción de la media RMSE de V4 frente al mejor de los tres
controles neuronales y frente al mejor de los cinco controles totales**, incluidos
los dos analíticos. Estos controles fuertes se añadieron durante la revisión
previa, antes de congelar el plan o producir resultados. Los controles de masas
fijas y lectura plug-in del piloto viejo no se repiten en este ensayo acotado;
esa omisión está declarada y no se presenta como familia confirmatoria completa.
No es el margen confirmatorio δ ni una prueba estadística. Si falla,
se registra el abandono de esta receta; no se promueve otro control, no se cambia
la regla ni se inicia otra modificación dentro del mismo ensayo.

Si algún ajuste falla o su checkpoint no está validado, el piloto queda incompleto
y no se aplica esa regla como resultado válido. Los trabajos completos se
verifican y se omiten al reanudar; los parciales conservan su oportunidad original.

## Alcance de lo que puede cerrarse

Este piloto podría aportar evidencia de desarrollo nueva y su incertidumbre
descriptiva entre seeds. No satisface diez semillas finales, potencia de una
familia completa, tres conjuntos adicionales no expuestos o la batería
confirmatoria. No acredita R1/R3, calibración, reproducción humana o dominio del
autor. Los 796 trabajos, 30 ajustes V3 y 20.400 filas retrospectivas conservan
sus identidades, y los veinte ajustes/1210 salidas se cuentan aparte.

El resultado de la receta V3 original y su NO-GO permanecen intactos. La decisión
de un futuro protocolo debe revisar la familia de rivales de V4, incluidos los
controles añadidos; no se heredan automáticamente 19 rivales/57 componentes.
