# Selección bibliográfica de diez modelos

Reconstruir desde la raíz: `python scripts/build_e2_literature_selection.py`, con el entorno de `requirements-eda.lock.txt`. Las decisiones curadas están en references/literature_e2/top10_20261105 y la extracción completa en references/literature_e2/fulltext_20261105. No se ejecutan búsquedas ni se entrenan modelos.

Se leyeron diez originales completos (215 páginas PDF), con versión, hash, métricas de una misma fila, páginas/tablas y evaluación metodológica. El registro primary_source_registry conserva como históricos los estados de la etapa anterior de código sin PDF; no representa la cobertura actual. La matriz actual está en ../literature_fulltext. Los PDF no se redistribuyen.

El puntaje es preliminar: las métricas externas ya están extraídas, pero no son comparables con el contrato de Bitcoin; el componente de desempeño conserva un intervalo de 15 puntos. Se añaden otros 15 puntos de incertidumbre cuando no se encontró el recuento de citas del original en la exportación parcial de Scopus (no equivale a cero citas).

WoS y Scopus aportan 8.802 ocurrencias y 7.833 grupos bibliográficos. Scopus recibido: 7.500 de 7.874 resultados mostrados; faltan 374, que no son exclusiones. El cribado global y el PRISMA de elegibilidad no están cerrados. Los diez modelos son una selección dirigida con adaptación D7 pendiente, no un ranking exhaustivo ni ganadores de Bitcoin. Código público no equivale automáticamente a permiso de redistribución; las licencias observadas constan en el registro.
