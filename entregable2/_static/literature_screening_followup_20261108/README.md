# Seguimiento de revisión E2: exclusiones y recuperación dirigida

La lectura asistida por IA del conjunto inicialmente excluido alcanza 337/337 (42 anteriores+295 posteriores). No significa revisión global completa, doble revisión humana ni inclusión definitiva. Las categorías de utilidad secundaria no se armonizaron retrospectivamente entre ambas rondas: no sumar sus clases como una clasificación homogénea de 337. No se estima una tasa estadística general de error.

La recuperación se limita a los 40 candidatos directos de la primera revisión:12 examinados,5 cuerpos revisados en sus secciones empíricas/metodológicas,7 parciales/no recuperados y 28 sin búsqueda individual. E2-0449, rescatado después, no se añade retroactivamente a ese denominador. Las cinco fichas son síntesis propias breves con referencias/localizadores, no originales, citas extensas o reproducción de estudios. NV significa no verificado/no recuperado, nunca ausencia demostrada.

Validación pública: `python scripts/build_e2_screening_followup.py --check`. Regenerar agregados exige `--private-base` fuera del checkout; se comprueba cobertura disjunta/exacta de 337 y correspondencia de la cohorte 40. No se exportan decisiones individuales, resúmenes ni citas privadas. Los expedientes previos, muestras, protocolos, código y resultados experimentales se conservan. No se entrena ningún modelo.
