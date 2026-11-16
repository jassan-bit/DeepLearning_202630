# Primer ensayo de desarrollo de E3

Este ensayo compara el candidato CNN-LSTM causal con penalización de discordancia
y saturación radial (`full`) con el control `M00`, que retira conjuntamente esas
dos modificaciones. Es una comprobación de aprendizaje y coste de desarrollo,
con una sola semilla; no es el protocolo confirmatorio ni una prueba de R3.
El control no permite atribuir por separado el efecto de cada modificación.

## Datos y separación temporal

La entrada es `current_cohort_arrays.npz` de la auditoría versionada de E2,
fijada por commit y SHA-256 en `plan.json`. El cargador sólo abre
`training_raw`, `training_y_original` y `training_dates`. Las particiones de
calibración y evaluación de E2 se reservan fuera de este ensayo.

Se ajusta con 532 objetivos entre 2023-01-08 y 2024-06-30. Se purgan los siete
objetivos de 2024-07-01 a 2024-07-07 y se usa como validación interna el periodo
2024-07-08 a 2024-10-31 (116 objetivos). La representación causal es
`log(precio) - log(último cierre observado)` y `log1p(volumen)`. Sus medias y
desviaciones, y la media de la variable objetivo usada como escala de salida,
se calculan exclusivamente sobre los 532 ejemplos de ajuste. Los objetivos
auditados se leen del archivo; no se reconstruyen a partir de una ventana que
no contiene el cierre futuro.

La separación evita compartir los retornos que forman los objetivos a ambos
lados del corte. Las ventanas dentro de cada periodo siguen siendo dependientes.
La validación guía la parada temprana y no constituye una prueba independiente.
Sus métricas no se comparan directamente con las métricas publicadas de E2,
obtenidas con otra partición y otro número de ejemplos de ajuste.

## Ejecución y reanudación

El notebook `notebooks/entregable3_desarrollo_colab.ipynb` prepara un entorno
aislado de Colab, comprueba la GPU disponible y guarda la corrida en Drive.
El límite de 1.200 segundos corresponde al total de entrenamiento de ambos
modelos. No incluye la descarga de dependencias y no amplía el presupuesto de E2.
Una reanudación conserva el saldo consumido, los pesos, el optimizador, la
semilla y el estado aleatorio. Se conservan por separado el último punto de
reanudación y los mejores pesos según validación. Un cierre abrupto puede
cargar conservadoramente el tiempo transcurrido hasta agotar el saldo.

El plan usa una semilla, AdamW, un máximo de 40 épocas por modelo y parada
temprana con paciencia de ocho. Las épocas se alternan entre ambos modelos
para repartir el tiempo. Las métricas de validación y las curvas son evidencia
exploratoria. No se calculan pruebas de superioridad, calibración final ni
resultados del conjunto de evaluación.

## Decisión posterior

Revisar si ambos modelos aprenden, si la inversión de escala es correcta, si
aparecen inestabilidades y cuánto cuesta cada época. Una mejora en este ensayo
justifica estudiar el candidato; no autoriza declararlo superior al Top 10.
La revisión de novedad, las ablaciones individuales, los controles de capacidad,
el análisis de potencia y el protocolo confirmatorio siguen pendientes.

Las pruebas automatizadas de este paquete usan datos artificiales. No equivalen
a haber ejecutado este notebook en la GPU del autor ni a una reproducción humana
independiente. Los resultados reales se producirán al ejecutar el notebook.
