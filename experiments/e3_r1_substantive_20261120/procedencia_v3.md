# Procedencia revisada de V3

Actualización posterior por IA declarada, 20 de noviembre de 2026. El índice
`experiments/procedencia.md` del corte anterior permanece íntegro. Esta revisión
se limita a V3 y remite a sus fuentes, sin inventar inspiración histórica.

## Origen del mecanismo e implementación

La formulación y el núcleo local están registrados en
`src/e3_overlap_model_v3.py` del commit
`8f65eac34aaf49293924ae8328c235cedbdd2866`. Utiliza las primitivas de PyTorch y
el encoder local M00, sin importar un modelo FQF/MDN o su implementación.
La IA intervino en formulación, código, revisión y redacción; el autor conserva
la responsabilidad de comprender y defender lo entregado.

La red aprende tres posiciones ordenadas y dos logits libres para la ley
de U; evalúa `sum p_k sqrt(A+(6/49)a_k²)` y entrena con MSE_Y y CRPS_U.
Los 1262 parámetros y los controles tienen trazabilidad en `code_theory/`.
No se importan a V3 las pruebas de saturación de v1, la calibración de la
aditiva o la comparación de 796 trabajos.

## Antecedentes atribuidos, distintos de fuentes históricas comprobadas

Las mezclas neuronales condicionales, posiciones/masas aprendidas, restricciones
de soporte y reglas de puntuación son antecedentes reconocidos. La revisión
actual los compara en `literature_distribution/` y `literature_application/`.
Reconocer un trabajo encontrado después no permite afirmar que se leyó al
programar. No se presenta la implementación independiente de FQF inspeccionada
como repositorio oficial del paper ni como código utilizado por E3.

La modificación específica para este problema es la composición de ley del
grado de libertad futuro, mapa dependiente del componente observado y ajuste
conjunto de la lectura puntual y la ley auxiliar. Las matrices indican la
diferencia respecto de cada antecedente realmente leído. No se afirma que los
operadores individuales o la regla de scoring sean nuevos.

## Fe de erratas bibliográfica

El PDF institucional `https://research.wu.ac.at/ws/portalfiles/portal/18954278/document.pdf`
es **Identifying Stochastic Processes with Mixture Density Networks**,
Schittenkopf, Dorffner y Dockner, Working Paper 11, 1998;
DOI `10.57938/a6057c38-dbad-42dc-bebc-bd2c71fe537f`.

**Volatility Prediction with Mixture Density Networks**, Working Paper 15,
1998, es otro trabajo: DOI `10.57938/03053498-7a01-4f07-a998-1c1837d98604`.
El acceso actual a su PDF agotó el tiempo. No se le atribuyen ecuaciones o
lecturas del primer documento. Los registros anteriores se conservan como
corte histórico; esta fe de erratas fija la identificación correcta.

## Propiedad, revisión y decisiones

La etiqueta es sigma7 poblacional (`ddof=0`), no la desviación muestral.
La identidad de varianza, Jensen y la proyección de riesgo son resultados
estándar usados con sus supuestos. La revisión de la propiedad está en
`code_theory/theory_audit.md`; la equivalencia con CRPS ponderado, en
`literature_distribution/change_of_variable_note.md`.

El dictamen actual de `independent_review/` identifica V3 y declara su modalidad
IA. La decisión experimental continúa siendo la registrada en el piloto:
heurística fallida, sin promover la ablación mejor situada o iniciar campaña.
Una defensa del estudiante o decisiones históricas no observadas no se rellenan
con una firma de IA.
