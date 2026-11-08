# Auditoría de snapshot real E2, sin entrenamiento

Estado: auditoría de desarrollo aprobada; integración real del pipeline E2 todavía no implementada.

## Reproducción

Desde la raíz del repositorio, con el entorno de requirements-e2-current-synthetic.lock.txt:

```
python scripts/audit_e2_real_data.py --output /tmp/e2-real-audit.json
python -m unittest discover -s tests -p test_e2_real_data_audit.py -v
python -m unittest discover -s tests -p 'test_dl_data*.py' -v
```

## Resultado

- Dos derivados diarios existentes verificados contra hashes registrados en experiments/dl_r2_cv_20261104/protocol.json.
- 731 filas/calendario 2023–2024 por fuente, únicas, ordenadas y a medianoche UTC. No faltan filas calendario; se preserva el día incompleto 24/03/2023 sin imputar ni comprimirlo.
- 716 ventanas D7 elegibles y 15 excluidas. Reconstrucción independiente de máscara, ventanas t−7..t−1 y etiquetas de siete retornos poblacionales; diferencia absoluta máxima de etiqueta 9,71445146547012e−17.
- Diez bloques CV enero–octubre 2024: 305 fechas. Piloto noviembre–30 diciembre: 60 fechas. Regla train≤o−8 e inner train≤o−98 comprobada.
- Soporte del piloto PROPUESTO E2: 08/11–30/12/2024, 53 fechas, entrenamiento permitido hasta 31/10/2024. Solo soporte auditado, no ejecutado; no sustituye el piloto histórico E1 de 60 fechas.
- Soporte del ajuste final de §17 reconstruido, sin ajustar: 709 etiquetas hasta 24/12/2024; inner train619 hasta 25/09/2024; validación interna83.
- Estadísticos de precios relativos/log1pV reconstruidos independientemente sobre todas las N×7 posiciones. Media positiva de y, sin centrar. Escaladores separados para inner/refit. Perturbación de todo futuro posterior al último outcome permitido no altera escaladores ni tensores de entrenamiento. Perturbación desde origen no altera su ventana predictora.
- Cero duplicados exactos de ventanas entre train y evaluaciones disponibles. Las ventanas solapadas temporalmente son parte del contrato, no duplicados independientes.
- El pipeline actual comparte exactamente relative_features pero rechaza procedencia real. No se falseó synthetic_only ni se retiraron guardas. No se certifica un runner real E2 completo.
- 12 tests nuevos y 16 regresiones E1 aprobados. Los tests nuevos bloquean construcción de AdamW y llamada train() durante la auditoría; también inyectan hashes erróneos, calendario de ambas fuentes, etiqueta individual alterada, fechas duplicadas/huecos, fechas no UTC, etiqueta muestral, ventana contemporánea, máscara errónea, OHLC inválido y etiqueta futura.

## Límites

No se leen ni recuperan precios de 2025–2026. Sus 608 fechas de evaluación no son necesarias para auditar desarrollo ni se certifican aquí. Tampoco se reagregan archivos de minuto; se verifica la identidad registrada de sus derivados diarios. El soporte final se reconstruye desde el contrato, no desde un runner real E2 integrado. No hay entrenamiento, optimizadores, HPO, predicciones, comparación de rendimiento ni afirmaciones de generalización. No se copian filas de mercado a estos artefactos.

Artefactos públicos: audit.json, plan.json, README.md y manifest.json. El plan es una propuesta no ejecutada; sus presupuestos no acreditan horas disponibles ni preregistro.

El auditor requiere Python sin -O y una ruta de salida nueva. Rechaza modo optimizado y no sobreescribe salidas existentes.
