# Triaje provisional E2 del corpus exportado, 2026-11-08

## Resultado y alcance real
Se procesaron 1.634 registros: WoS 340, Scopus 828, IEEE 466. La normalización y deduplicación determinista produjo 1.078 grupos de referencias, retirando 556 registros redundantes. Dentro de 2019–2026: 1.605 registros fuente (340, 808, 457), representados en 1.054 grupos únicos; 24 grupos quedan fuera de ventana. El término «grupo» no presupone estudio empírico independiente: hay revisiones y actas agregadas.

Clasificación final de esta corrida provisional: 59 prioridad principal, 520 secundarios transferibles, 138 inciertos conservados, 337 exclusiones propuestas por precio y 24 fuera de período. Se conservan 717 grupos en ventana para evaluación, pero NO son 717 inclusiones definitivas ni 59 estudios elegibles confirmados. Las 337 propuestas de exclusión NO deben convertirse en exclusiones definitivas sin auditoría.

Este producto es **triaje automatizado conservador de títulos y resúmenes, con 21 ajustes focalizados por lectura del asistente**. No es doble cribado humano, lectura individual exhaustiva de los 1.054, revisión de textos completos ni una evaluación de riesgo de sesgo. La columna `targeted_audit` identifica exactamente los ajustes focalizados. Los resúmenes fuente pueden no coincidir entre bases; se conserva el más largo y las señales se buscan en todas sus versiones. Ningún resultado de rendimiento queda verificado por el triaje.

## Criterios
- Principal: Bitcoin + volatilidad como posible objetivo + redes/DL. Incluye candidatos ANN y modelos de riesgo cuya elegibilidad estricta DL debe verificarse después. Coocurrencia de términos no prueba que DL pronostique volatilidad: es prioridad de revisión.
- Secundario: volatilidad cripto/financiera, híbridos financieros con redes o arquitectura transferible; precio, retornos, VaR/ES y volatilidad no se equiparan.
- Incierto: información insuficiente para exclusión segura; conservar. Actas agregadas detectadas durante la revisión se trasladaron aquí para localizar artículos individuales.
- Exclusión propuesta: pronóstico explícito de precio/valor sin señal detectada de objetivo volatilidad o mecanismo híbrido/arquitectura. Es propuesta revisable, no descarte irreversible.
- Ventana: años de publicación exportados 2019–2026. Año early-access e impreso pueden diferir; se conserva lista de años por grupo.

## Reproducibilidad
1. Copiar ambos scripts y ajustes_revision.json a una carpeta privada fuera de cualquier checkout Git. Los scripts rechazan ejecución dentro de Git. Mantener los originales privados en `originales/wos.txt`, `originales/scopus.csv`, `originales/ieee.csv`, junto a `cribar.py` y `ajustes_revision.json`.
2. Verificar tamaño y SHA-256 contra `resumen.json`.
3. Ejecutar `python cribar.py`; genera registros normalizados, cribado completo con resúmenes, relaciones de duplicados, conflictos y resumen.
4. Ejecutar `python crear_derivados_publicos.py`; genera esta vista sin resúmenes ni extractos de evidencia textual.

Se normaliza DOI a minúsculas y se retiran prefijos doi.org; se fusiona por DOI exacto. Si no hay DOI conflictivos, se fusiona título normalizado (Unicode, minúsculas, sin puntuación) + año exacto. No se fusionan similitudes difusas ni variantes título-año dudosas. Hay un conflicto de título/año con DOI distintos, documentado y retenido. El grafo completo de relaciones está en `relaciones_duplicados.csv`. Las fuentes y filas se identifican con WoS-NNNN, Scopus-NNNN, IEEE-NNNN; los grupos E2-NNNN son estables para estos archivos y este orden.

## Límites y sesgos a declarar
- No se verificó la consulta exacta de inicio de Scopus. El export no prueba por sí solo la cadena de búsqueda ni su cobertura.
- Buscar términos puede confundir antecedentes, trabajo futuro, predictores e indicadores de régimen con variables objetivo. La auditoría focalizada corrigió 21 casos, pero no estima precisión ni exhaustividad.
- Exclusiones propuestas pueden contener falsos negativos: términos no cubiertos, volatilidad expresada de otra manera, resúmenes imprecisos, mecanismos transferibles omitidos. Antes de cerrar PRISMA, revisar todas las exclusiones propuestas y los inciertos; no usar una muestra como prueba de ausencia de falsos negativos.
- No se deduplican distintas versiones por inferencia difusa, de modo que podrían quedar duplicados residuales. Tampoco se validaron los DOI contra editoriales en esta fase.
- Las 59 prioridades requieren identificar variable objetivo real, proxy (RV, GK, GARCH u otra), horizonte, temporalidad, entrenamiento/test, métricas y evidencia de BTC. El conjunto no justifica proclamar superioridad de ninguna familia.
- Originales y resúmenes completos son privados. Esta carpeta solo contiene metadatos bibliográficos, decisiones y reglas; no publicar sin autorización.

