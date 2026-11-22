# Auditoría ejecutada del componente de hipótesis y adecuación (15 %)

Esta auditoría contrasta el PDF original con la evidencia vigente. No entrena,
infiere, descarga mercado, abre reserva, recalibra intervalos ni repite simulación.
La implementación y revisión son asistidas por IA y no constituyen aceptación
académica de R1/R2/R3 ni reproducción humana independiente.

La guía se identifica por SHA256
`d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364`.
Se comprobaron **25 citas con página**, **22 archivos por hash** y **16 filas
solapadas de requisitos**. Las filas no son componentes nuevos ni pesos que se
sumen al 15 %. El Cuadro 9, p. 60, conserva los once componentes y sus pesos.

Archivos principales:

- `results/requirements_matrix.csv`: requisito, evidencia, límite, acción y comando/etapa.
- `results/requirements_matrix.json`: citas verificadas, hashes, estados y guardas.
- `results/execution_receipt.json`: recibo de la auditoría ejecutada.
- `results_v1/`: los tres archivos originales de la auditoría anterior,
  conservados byte a byte antes de incorporar el complemento MCS y los dos
  bloqueos científicos del preflight. La versión actual vincula su hash.
- `retrospective_independent_review.json`: revisión separada del nuevo análisis
  input-only; la aritmética se contrastó con el ZIP original y Holm128 con
  `statsmodels.stats.multitest.multipletests`, sin nuevos sorteos. Su comando
  reproducible es `verify_retrospective_review.py` con el intérprete científico
  fijado y el ZIP original suministrado mediante `--source-zip`.

La auditoría conserva la nota orientativa anterior **2,8/5** y la sustentación
**4,5 aportada por el autor**, sin afirmar defensa observada ni asignar una nota
nueva por generar esta matriz. Una revisión posterior debe justificar cualquier
delta mediante análisis ejecutados, no por documentos o simulaciones copiados.

## Lo ya ejecutado y lo que no se transfiere

La batería temporal moderna tiene admisión **condicional por panel y DGP** para
R10/T768/B8191/HAC48/bloque32/alpha0,025. Se reutiliza su expediente compuesto,
sin sumar el nulo anterior inconcluso ni atribuirle potencia de mercado o una
validación universal de cuatro mercados. Los tres procedimientos son DM-HLN/HAC,
bootstrap estacionario-t y GW condicional; una sensibilidad a bloques no cuenta
como otro procedimiento. El rechazo Wald de GW no es por sí mismo una afirmación
direccional de inferioridad media.

El nuevo análisis de la candidata reutiliza **1760 predicciones, 32 trayectorias,
55 fechas expuestas y cinco semillas**. Sus IC temporales se condicionan a esas
trayectorias; no multiplican fechas por semillas ni acreditan la población de
inicializaciones. Los dos IC RMSE contra `global_mse` incluyen cero. Los
diagnósticos con Holm128 conservan **27 rechazos MZ**, incluidos **2/5 de
input-only**. Los no rechazos de Ljung–Box/ARCH no certifican adecuación. Faltan
predicciones punto a punto de ajuste/selección y cuantiles/ley de pronóstico para
completar Tablas A/B y cobertura/PIT de la candidata.

El MCS histórico corresponde a la aditiva. El complemento nuevo reutiliza el
módulo estándar Tmax para **8 recetas del piloto, 32 trayectorias y 55 fechas**,
con pérdida MSE secundaria sobre cinco semillas fijadas. Los dos MCS90% de
bloques7/14 retienen siete recetas, incluida input-only, y excluyen sólo el piso;
no identifican un ganador único, no usan la primaria media-RMSE ni toda la
familia21, y no acreditan cobertura finita. La fila H12 pasa a **parcial:
complemento secundario ejecutado**, no a cumplida. No revive la extensión que
falló su calibración ni motiva un aumento adicional de nota por inclusión.

El preflight científico independiente conserva dos bloqueos finales: inferencia
sobre MSE/raíz agregada no equivale a inferencia sobre media de RMSE individuales,
y las cotas marginales por panel no certifican por sí solas el error global de la
regla adicional2/3. Con el peor límite CP95 actual `q=0.0478608731`, la cota
conservadora `2q=0.0957217463` no demuestra una cota conjunta0.05; tampoco
demuestra inflación real de la regla. No se modifica alpha, la regla2/3, el
endpoint ni las condiciones para conseguir aprobación. La validación archivada
sigue siendo condicional por panel/DGP, sin presentarse como universal. Las
pruebas entre cuatro paneles necesitan justificar sus supuestos; tres derivados
correlacionados no se convierten en conjuntos independientes por sus nombres.

## Selección de conjuntos adicionales

La p. 23, §8.3, exige que cada adicional difiera del principal en al menos dos
ejes entre tamaño, dominio, ruido, desbalance y dimensionalidad, manteniendo tarea
y métrica y eligiéndose antes de observar desempeño. No exige acciones, Nasdaq,
ni un umbral de tamaño1,5×. Tampoco exige literalmente dos ejes entre cada par
de adicionales.

Spot frente a derivados perpetuos puede constituir una propuesta de dominio
económico si se describen financiación, margen, liquidaciones y colateral. Un
tamaño cronológico fijado y comprobado puede ser el segundo eje. Eso es una
justificación revisable, no una admisión: deben verificarse datos, calendarios,
unidad de volumen, tarea diaria sigma7 de precios de cierre y exposición. Los
perpetuos con subyacente/venue/calendario comunes limitan generalización y no
representan tres mercados independientes. No se confunde volatilidad con ruido
ni se cuenta un ticker como otro eje.

## Repetir la auditoría documental

El PDF debe aportarse localmente; no se redistribuye su contenido completo.
Desde el repositorio y con un paquete del motor ya verificado:

```bash
python experiments/e3_hypothesis_adequacy_20261122/guide_audit/audit_requirements.py \
  --guide-pdf /ruta/Entregable3DeepLearningProject.pdf \
  --repo-root /ruta/DeepLearning_202630 \
  --engine-root /ruta/fuente-motor-verificada \
  --delivery-root /ruta/DeepLearning_202630/experiments/e3_hypothesis_adequacy_20261122 \
  --output /tmp/e3-guide-audit-recheck
```

Requiere Python y `pdftotext`; su cálculo es de biblioteca estándar. Los comandos
de resolución de la matriz son plantillas de etapas futuras, con las variables
de caso y credenciales privadas aportadas por el cliente original. No se
ejecutaron mediante esta auditoría y no sortean las puertas de datos, potencia,
protocolo o activación. La prueba final no se abre para verificar la matriz.
