# Artefactos ONNX para inferencia CPU de E3

Este catálogo habilita cuatro predictores, todos de la semilla representante
**271801**, fijada por el menor identificador, no por su error de prueba:
MLP16, LSTM32, GRU32 y CNN-LSTM con atención temporal y salida aditiva positiva.
Los otros trece sistemas siguen disponibles como resultados históricos; no se
afirma que excedan memoria sin haberlos medido.

`manifest.json` identifica cada carpeta y la huella de sus metadatos.
`metadata.json` conserva el SHA256 de `model.onnx`, escaladores, fuente de
pesos, esquema, calibrador existente cuando corresponde, rangos OOD de ajuste
y reporte de ida y vuelta. No se acepta un modelo serializado del usuario.

La entrada del grafo es `windows`, float32 de forma **1×7×5**, ya transformada
con las estadísticas guardadas. E3 añade `variance_floor`, float64 de forma
**1**, calculado sólo con los seis retornos conocidos. Las tres referencias
devuelven sigma7 normalizada y requieren la escala indicada en metadatos;
E3 ya devuelve sigma7 original. El módulo `e3_dashboard.inference` resuelve
esas diferencias, valida las siete barras CSV y carga una sesión ORT CPU a
la vez. La aplicación no importa Torch ni código de entrenamiento.

Se verificaron las **608 ventanas originales de prueba** contra el modelo
en memoria CPU y sus predicciones congeladas. La tolerancia declarada es
**10⁻⁶ en unidades originales de sigma7**. Es paridad de un predictor fijo,
no la métrica media de las diez semillas ni una prueba de superioridad nueva.
El reporte exacto y la reproducción están en
`experiments/e3_dashboard_20261117/`.

La salida doble de E3 representa Softplus mediante
`max(z,0)+log(1+exp(-abs(z)))`, equivalente en aritmética real, porque ORT
1.23.2 no implementa el operador Softplus float64. Se contrastó contra el
módulo científico sin cambiar sus pesos, incluidos sus redondeos numéricos.

MLP16/LSTM32/GRU32 conservan sus cuantiles residuales de calibración de E2;
no se ajustan de nuevo. **E3 no tiene intervalo calibrado.** Las advertencias
OOD comparan rangos de ajuste por rezago/canal, no certifican fiabilidad.
La oclusión de barras es una sensibilidad exacta del predictor, no una
atribución causal ni una explicación cuya fidelidad y estabilidad estén
validadas. La memoria y los tiempos del dashboard completo requieren su
propia comprobación; las mediciones del exportador no prueban cabida en Render.
