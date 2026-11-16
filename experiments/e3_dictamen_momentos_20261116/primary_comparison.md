# Comparación focal de fuentes primarias

**16 de noviembre de 2026.** Se reabrieron los dos PDF oficiales mediante acceso web directo. No hubo nuevas consultas a motores bibliográficos, lectura íntegra declarada, contacto humano ni consulta de mercado. Esta revisión asistida por IA no constituye la verificación humana externa de R1.

## Hallazgos de las fuentes

**Faithful.** La definición 1 obtiene el comparador de media sola eliminando los nodos que no son ancestros de la media. La ecuación (5) y el teorema 1 igualan sus actualizaciones bajo las condiciones indicadas y la misma aleatoriedad; combinan pérdida cuadrática con bloqueos de gradiente. Localizadores leídos: **§2, definiciones 1–2, propuestas 1–2, ecuación (5), teorema 1 y prueba; páginas físicas 2–4**. Se corrige aquí el localizador §3 del registro previo: §3 corresponde a experimentos. [Stirn et al. (2023), PDF PMLR](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf).

**ConstraintNet.** Su guarda parametriza una región dependiente de la muestra. Contempla restricciones procedentes incluso de otra red. La mezcla convexa de vértices de la ecuación (7) se aplica a intervalos; la softmax aproxima sus fronteras sin alcanzarlas exactamente. Localizadores leídos: **páginas editoriales 6813–6817; ecuaciones (3)–(7), discusión de fronteras y aplicación a intervalos**. [Brosowsky et al. (2021), PDF AAAI](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf).

## Inferencias sobre nuestra cabeza, separadas de las fuentes

| Comparación | Diferencia específica | Evidencia que falta |
|---|---|---|
| Frente a Faithful | En `μ=L(m,q)+ρ[H(m)−L(m,q)]`, las lecturas de `m,q` son también ancestros de la media. | Control con la **misma cabeza completa**, retirando únicamente los scores auxiliares. |
| Frente a ConstraintNet | El intervalo procede de una identidad del objetivo y de momentos estimados; se usa el rango semiabierto `[L,H)` y otra coordenada `ρ`. | Justificar qué aporta este acoplamiento frente a una guarda y supervisión auxiliar convencionales. |

Detener los gradientes auxiliares únicamente al tronco no preserva la trayectoria de media sola: los scores todavía actualizan las lecturas de `m,q` y modifican `μ`. Para preservar esa igualdad habría que bloquear sus aportes en todos los ancestros de la media. El control de media directa con auxiliares separados puede inspirarse en Faithful; no transfiere automáticamente su garantía a nuestra cabeza acoplada.

La forma `L+(H−L)ρ` es una guarda de intervalo. Aprender las fronteras no basta como diferencia sustantiva, pero lo leído tampoco demuestra que ConstraintNet publique la receta completa aquí propuesta.

**Conclusión de alcance:** estos antecedentes impiden acreditar originalidad con los argumentos actuales; no demuestran identidad literal de la composición ni ausencia de novedad. La consistencia condicional ya demostrada sigue sujeta a sus supuestos y no implica fidelidad de optimización, mejor RMSE o admisibilidad R1.
