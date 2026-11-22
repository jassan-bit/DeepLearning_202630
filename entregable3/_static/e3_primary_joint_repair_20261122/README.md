# Primaria media-RMSE, decisión conjunta y criterio académico de redondeo

La entrega conserva la candidata `input-only-overlap-mse-1` y la fuente S2.
Implementa la inferencia de la **media del RMSE por semilla**, completa el
expediente R1/R2 y ejecuta un estudio sintético de cuatro paneles. No entrena
modelos, abre una reserva ni acredita superioridad de mercado.

## Resultado y criterio comunicado por el autor

El estudio V1 terminó **4608/4608 réplicas**. Pasaron los ocho controles de
aceptación conjunta falsa y el control favorable. Dos límites de cobertura,
en la misma frontera práctica, fueron **94,8366053751 %**. El protocolo original
exigía al menos95 % sin redondeo: su resultado queda **`accepted=false`**.

Después del resultado, el autor comunicó que el profesor permite redondear al
porcentaje entero. Aplicada de manera uniforme a los **18 límites de cobertura**,
la regla convierte94,836605 % en95 %: **18/18 cumplen el criterio académico
comunicado**. Los otros89 límites no se redondean. La indicación se atribuye al
reporte del autor; no se inventa una comunicación docente verificada ni una firma.

- [Registro académico y los18 cálculos](results/ACADEMIC_ROUNDING_DECISION.json).
- [Reporte científico original completo](results/VALIDATION_REPORT_V1.json).
- [ZIP íntegro, incluidos los4608 registros](results/validation_v1_checkpoint.zip).
- [Interpretación del resultado y sus límites](joint_validation/RESULTS_INTERPRETATION.md).
- [Derivación y tres procedimientos](methods/DERIVATION.md).

El valor exacto, el protocolo y el registro original permanecen intactos. El
redondeo permite satisfacer el criterio académico comunicado; no transforma un
límite matemático de94,8366 % en una garantía exacta de al menos95 % ni acredita R3.

## Fuentes y ejecución auténtica

El protocolo y las fuentes se publicaron **antes de simular** en el commit
`db5a75bb66a0b28b1d05dacc41b66daf161f9120`. El manifiesto ejecutable de13 archivos
tiene SHA256 `45ad0706f293e487128905d6529d203aea14f8c5fc5a8907020781bd44a1c12f`.
El [recibo de lectura pública](PUBLIC_SOURCE_VERIFICATION.json) separa la
verificación anterior del manifiesto/protocolo de la lectura adicional de blobs.

V1 usa R10/T768/B8191, bloques32, HAC48 y referencia49999. La familia conserva
21 rótulos, Holm120 por panel, pantalla adversa Holm180 y principal más dos de
tres adicionales por rival. Los DGP tienen dependencias y duplicaciones
declaradas; no son21 algoritmos ni diez ajustes independientes de mercado.

Se registraron **4410,373 s CPU**, frente a5400 s de presupuesto blando:
**989,627 s registrados restantes**. La verificación separada recalculó los107
límites y reprodujo tres réplicas prefijadas exactamente, con6,145 s CPU
adicionales. No se cuentan como nuevas réplicas. El coste no es consumo de GPU,
una factura de Colab ni una medición de todo el trabajo editorial.

## Material para la evaluación

- [Expediente R1/R2 y correspondencia con código](r1r2/README.md).
- [Guion de defensa y modificación razonada](r1r2/DEFENSE_GUIDE.md).
- [Recuperación opcional de cinco checkpoints originales](recovery/README.md),
  con clave privada del propietario, sin reentrenar. No se ejecutó esa recuperación.
- El notebook CPU publicado consulta el ZIP completo por defecto, sin repetir
  Monte Carlo ni solicitar credenciales; el indicador original `accepted=false`
  se presenta junto a la interpretación académica posterior.

## Corrección adicional conservada sin otra campaña

Una V2 implementa intervalos por inversión del test fixed-b, conserva p-valores,
niveles95/99, márgenes y regla. Sus18 pruebas propias y34 independientes pasan.
Se archiva en [correction_v2/runtime](correction_v2/runtime/PREPARATION_STATUS.json),
con **cero réplicas formales V2**. Tras la indicación de redondeo no se inició
otro estudio CPU. Esta preparación no se presenta como calibración aprobada.

La nota oficial y los actos académicos de R1/R2 no se asignan aquí. La valoración
orientativa previa no se modifica. R3 mantiene sus requisitos de confirmación
real, potencia, adecuación y robustez. Los796 trabajos aditivos y los resultados
adversos de V3/V4/V5 completo conservan su identidad y decisiones.
