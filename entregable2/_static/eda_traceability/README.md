# Trazabilidad de decisiones del EDA E2

`hallazgo_decision.csv` se conserva íntegro como registro histórico. Sus campos cubren hallazgo, evidencia, preprocesamiento, impacto en arquitectura/validación y riesgo/fuga de A.1 de la guía completa (PDF p61), más alcance y estado. Las rutas de evidencia parten de `_static/` de E2.

`decisiones_vigentes.csv` complementa, sin sobrescribir, las filas cuyo estado cambió después. Para `RESID`, registra F2a cerrada el 06/11/2026 y enlaza la sección 7. Distingue fecha de decisión y fecha de incorporación documental (08/11/2026), e incluye el SHA-256 del CSV histórico.

Ridge alpha=1 del EDA es una ejecución diagnóstica forward sin HPO. La configuración Ridge del catálogo del benchmark sigue el protocolo registrado y no hereda esas métricas como resultados finales. La frase histórica sobre no integrar el benchmark se refiere a la ejecución diagnóstica, no a excluir la familia Ridge del catálogo vigente.

El JSON de identidad enlaza hashes y cronología de E1. No crea un nuevo test sellado ni modifica el contrato científico. Esta actualización documental no ejecuta análisis ni entrenamiento.
