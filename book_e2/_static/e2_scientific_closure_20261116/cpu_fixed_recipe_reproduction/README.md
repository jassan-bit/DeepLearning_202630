# Nueva réplica científica CPU con objetivos verdaderos

El adaptador `scripts/e2_cpu_fixed_recipe_reproduction.py` implementa ajustes
nuevos de78redes y tres clásicos, calibración53, freeze81, admisión de20archivos
raw, pronósticos608 y A2/inferencia sobre esos pronósticos nuevos. El optimizador
es `phase_cpu` del motor registrado de controles, sin modificarlo. En esta rama
se usan **los objetivos verdaderos de train655**; no se permutan etiquetas ni se
cargan pesos/pronósticos antiguos. No usa ningún flag de entrenamiento para tests.

Es un estudio CPU posterior, con backend y plan propios. No acredita paridad
numérica A100, no sustituye la A2 original ni vuelve a ejecutar el HPO histórico.
El driver y plan CUDA originales permanecen intactos.

## Estado previo a los ajustes

`plan_v2/analysis_plan.json` fija fuentes, recetas/épocas, seis semillas, datos,
runtime y contabilidad antes del primer ajuste verdadero CPU. Los405inputs
científicos y los árboles scripts/experiments/third_party se conservaron en
`/workspace/work/e2-true-target-cpu-source-v1-20261116`. El snapshot omite el script
de presupuesto y cualquier ledger: el padre y los workers siguen usando la
cuenta real del proyecto y su ancla histórica, sin clonarla ni reiniciarla.

`admission_v2/timing_admission.json` verifica los13controles completos, sus
recetas, checkpoints y tiempos. Con suma971,200119s y política fijada de
`1,5 × 6 × suma13 + 1200`, la proyección es **9940,801075s** frente al límite
solicitado de10000s. Es un margen operativo conservador; no es un IC ni una
garantía. La dependencia del cálculo de TimesNet respecto a pesos/periodos,
la carga del sistema y las semillas pueden cambiar los tiempos verdaderos.

La ejecución todavía exige **QA independiente del plan exacto**. Este documento
no constituye esa aprobación ni afirma que los78ajustes se hayan realizado.

## Comando de ejecución después de QA

Crear un recibo de revisión independiente con `plan_sha256` igual al SHA256 del
plan y `approved_for_new_CPU_replication: true`, además de sus controles y autor.
Después, desde el proyecto:

```sh
python scripts/e2_cpu_fixed_recipe_reproduction.py --run --plan book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/plan_v2/analysis_plan.json --admission book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/admission_v2/timing_admission.json --qa RUTA_QA.json --source-root /workspace/work/e2-true-target-cpu-source-v1-20261116 --python /workspace/work/e2-opt-venv/bin/python --output NUEVA_CARPETA --cap-seconds 10000
```

No añadir otra envoltura del ledger. Se comprueban otra vez el saldo libre,
las fuentes y los archivos reales. El proceso padre sólo coordina. Cada worker
tiene su propia reserva y cargo de reloj de pared, incluidos fallos/timeouts.
El límite global compara la **suma de los tiempos de workers**, no la pared del
padre ni la mitad de tiempo que pueda conseguir el paralelismo.

Se ejecutan hasta dos procesos, cada uno con dos hilos Torch y afinidades
separadas `[0,1]` / `[2,3]`, dentro de los núcleos admitidos por el host. Scoring y
análisis se ejecutan después de los81predictores, con reservas propias. No se
cobra el padre otra vez ni se suman sobres de tiempo a sus componentes.

## Matemática y fronteras

Las fases conservan AdamW, grupos de decay, clipping, lotes sembrados y el
scheduler coseno/warmup con horizonte de200épocas; el número de épocas queda
fijado por las selecciones históricas. SAE ejecuta AE1, AE2 con encoder1 congelado,
transferencia y supervisión. Se guardan registros por fase, pesos nuevos,
escalador train-only y calibrador53 antes de abrir numéricamente los raw del test
**dentro de esta ejecución**. El calendario/test ya era conocido históricamente:
no se recupera cegamiento prospectivo.

El predictor CPU batch1 es propio del adaptador. Durante `score_final` reemplaza
sólo el frontend de inferencia CUDA; los archivos originales no se editan.
Los tres clásicos usan el worker clásico original con sus configuraciones fijas.
Se producen nuevos pronósticos, A2, análisis4999 y refinamiento20000. No se
alimentan reporteros que exijan hashes de pronósticos antiguos.

`--resume` sólo reutiliza trabajos nuevos completos con artefactos y cargos
verificados. Un trabajo parcial o un intento sin cierre contable requiere
reconciliación: no se reinicia el optimizador ni se cambia la semilla. Si falla
un worker, se conservan el fallo y el coste, se esperan los workers ya activos y
se interrumpe la cadena con estado parcial.

Incluso al completar esta cadena, `full_original_campaign_passed` permanece
false: HPO completo, estudios de robustez/explicaciones/ablaciones y cualquier
IUT de coste con los nuevos pesos tienen alcances separados. Los tiempos CPU de
los pesos anteriores no se atribuyen automáticamente a estos nuevos pesos.

La revisión de ingeniería v2 conserva fuente/plan/admisión v1. Para reanudar mantiene el límite grupal original y contrasta sólo su saldo no consumido y la proyección de tareas pendientes contra el saldo libre actual. Los cargos previos se verifican contra los finish reales del diario.
