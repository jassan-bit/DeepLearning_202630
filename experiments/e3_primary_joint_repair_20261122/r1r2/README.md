# Expediente R1/R2 preparado y verificado

La candidata `input-only-overlap-mse-1` queda intacta. Se conserva su dictamen técnico específico **SUSTENTADO_ACOTADO**, con el núcleo propio heredado explícitamente atribuido y sus límites. Esta fase añade material concreto para evaluar la implementación y la defensa; no registra un acto docente, no acredita R3 y no cambia notas.

- [Expediente: literatura, ecuaciones, propiedades, hipótesis y procedencia](R1_R2_DOSSIER.md).
- [Guion de defensa con modificación razonada ejecutable](DEFENSE_GUIDE.md).
- [Correspondencia formulación/código con líneas y hashes](code_correspondence.csv).
- [Requisitos de rúbrica, evidencia auténtica y acto restante](REQUIREMENT_EVIDENCE_MATRIX.csv).
- [Recibo de las seis comprobaciones CPU efectivamente ejecutadas](EXECUTION_RECEIPT.json).
- [Manifiesto de fuentes y verificaciones](SOURCE_MANIFEST.json).

La demostración sustituye en una referencia didáctica `sqrt(A+c*u²)` por `hypot(sqrt(A),sqrt(c)*u)`, conserva la función matemática y verifica un caso extremo de overflow de la expresión ingenua. El runtime congelado ya usaba la forma estable y no se modifica. Las pruebas nuevas incluyen un caso adverso para la equivariancia de salida y otro para la identificación distributiva bajo MSE.

Se reutilizan los ocho tests previos de candidata y los recibos de revisión bibliográfica; no se contabilizan como nuevas ejecuciones. No se leen datos de mercado ni checkpoints y no se usan claves. El estudiante conserva la responsabilidad de explicar/defender cada línea. La IA preparó y ejecutó estas verificaciones técnicas, no una sustentación humana.

Repetir la demostración sin sobrescribir su historial:

```bash
python experiments/e3_primary_joint_repair_20261122/r1r2/defense_exercises.py \
  --candidate-root experiments/e3_input_only_candidate_20261121 \
  --output /tmp/e3-defense-repeat.json
```

La ejecución registrada tardó 1,80 s CPU y1,67 s de pared, con seis comprobaciones PASS; el recibo conserva valores precisos, entorno y pico de memoria del proceso importando PyTorch. No es un benchmark de inferencia ni una estimación de entrenamiento. Coste nuevo: cero pasos de optimizador, cero ajustes de pronóstico, cero predicciones de mercado, cero diseños nuevos y cero consultas bibliográficas. Quedan por ejecutar únicamente los actos humanos/académicos que aún no ocurrieron; esta fase no los simula. La corrección de la batería CPU se trabaja por separado y los 340 ajustes de planificación permanecen pausados mientras sus bloqueos no se resuelvan.

Revisión/diseño8 %, implementación/demostración10 % y sustentación7 % son los pesos de la guía p60 afectados por este material, no ganancias de nota prometidas. La valoración orientativa vigente y el tope condicional no se recalculan aquí.
