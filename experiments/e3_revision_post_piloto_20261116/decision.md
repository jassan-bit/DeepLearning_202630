# Decisión de desarrollo después del primer piloto

Fecha: 16 de noviembre de 2026. Ámbito: decisiones exploratorias anteriores a
un protocolo confirmatorio todavía no congelado.

## Decisión actual

Conservar `full` y `M00` como antecedentes de desarrollo. El primer piloto no
respalda adoptar `full` como receta final: su RMSE fue 12.43 % mayor que el del
control. Esta comparación conjunta tampoco identifica cuál modificación explica
la diferencia. No se amplían sus épocas ni se selecciona otra semilla para buscar
un resultado favorable.

La revisión prioriza ahora la **coherencia con la parte ya observada del objetivo
y el aprendizaje de la innovación futura**. El piso algebraico identifica una
limitación real de ambas redes. Su proyección constituye un control común y
reproducible. Las métricas posteriores se mantienen separadas de las originales.

## Qué fuentes sostienen esta dirección

- Chan, Golub y LeVeque (1983) proporcionan la identidad de actualización de la
  dispersión. La especialización a siete retornos no es una nueva identidad.
- ConstraintNet (Brosowsky et al., 2021) ya parametriza salidas con restricciones
  por muestra, incluida una salida escalar por encima de un umbral. Cambiar la
  activación a Softplus no basta para sustentar R1.
- Spulbar y Ene (2026) distinguen la parte observada y la futura de un objetivo
  móvil financiero. Su objetivo sin centrar difiere de nuestra desviación σ7.
- HARNet, GARCH-NN/GARCH-LSTM y DeepVol aportan estructura financiera e
  inicializaciones o arquitecturas específicas. Sus objetivos, rezagos o datos
  intradía impiden adoptarlos como réplicas equivalentes de D7×5.
- Los dos trabajos de 2026 sobre correcciones residuales y atención anclada
  muestran antecedentes y límites de esa línea: ni una corrección acotada ni
  simplificar un residuo garantizan una mejora predictiva.

Las ecuaciones, localizaciones de lectura y fuentes originales están en
[theory.md](theory.md) y en [la matriz](literature/review_matrix.csv).

## Condiciones para formular la próxima variante

1. Mantener D7×5, σ7 centrada con `ddof=0`, horizonte uno, fechas y escalado
   comparables. Utilizar la parte conocida del objetivo de forma causal.
2. Especificar una modificación algorítmica concreta frente a los antecedentes.
   Un piso, una cabeza positiva o cambiar el activo a BTC no acreditan R1.
3. Alinear la función de pérdida con el estimando: para MSE sobre σ, el óptimo
   es la media condicional de σ. La raíz de la varianza esperada no la sustituye
   en general.
4. Declarar hipótesis y controles antes del próximo ensayo. Incluir el piso
   analítico y la misma regla de coherencia para los controles. Una mejora sobre
   un rival que incumple una restricción disponible no basta para atribuir una
   ventaja al componente neuronal.
5. Registrar cada nueva variante y su coste de forma acumulativa. La selección
   seguirá usando desarrollo; estas 116 fechas no se reclasifican como prueba
   independiente. La campaña final necesita su propio protocolo y potencia.

## Estado de la selección y la novedad

El modelo final de E3 **permanece pendiente de selección**. Esta revisión elige
una limitación que merece estudiarse; no certifica otra arquitectura como nueva.
La matriz contiene lecturas focales de fuentes originales. Los intentos actuales
de acceso a tres API bibliográficas no devolvieron búsquedas utilizables; la
búsqueda sistemática en tres bases y la verificación humana externa siguen
abiertas. No se declara ausencia de antecedentes ni cumplimiento de R1/R3.

No se realizaron nuevos entrenamientos durante esta revisión. Se conservan
los pesos, las métricas, el saldo y el commit original del piloto.
