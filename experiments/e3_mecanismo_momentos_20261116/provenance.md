# Procedencia y atribución de la propuesta

**16 de noviembre de 2026. Adaptación candidata asistida por IA; novedad pendiente.**

La CNN causal, la LSTM unidireccional y la atención por contenido se conservan del control M00 del proyecto. No se presentan como invenciones. La identidad de varianza móvil se obtiene algebraicamente del objetivo de E2. Se proponen tres lecturas y un operador que combina dos momentos de la innovación con la media de la transformación raíz cuadrada.

| Elemento | Fuente y modificación propuesta |
|---|---|
| Cota inferior `L` | Especialización de Jensen reponderado, Merhav (2023), §2, ec. (8), ejemplo 2 ec. (15). Se condiciona en la información al origen y se utiliza como frontera de una cabeza. No se reclama una desigualdad nueva. |
| Cota superior `H` | Jensen clásico. Se demuestra que es supremo no alcanzado para segundo momento no degenerado mediante una secuencia que conserva momentos. |
| Scores de `m` y `q` | Elicitación de la media por scores Bregman/QLIKE: Gneiting (2011), §3.1. Se aplican a `V` y `V²`, normalizados por escalas de ajuste. No se supone una ley Gamma. |
| Interferencia entre objetivos de media y NLL | Antecedente directo: Stirn et al. (2023), *Faithful Heteroscedastic Regression*. Se especializa un contraejemplo al mapa half-t5 de v2; no se hereda su garantía de trayectoria de entrenamiento. |
| Momentos neuronales y guardas | HNN, GenUT, ConstraintNet y demás fuentes de la matriz. Su existencia impide reclamar como nueva la idea general de aprender momentos o limitar una salida. |
| Adaptación algorítmica a examinar | Estimar `m,q,rho`, producir una media realizable para los momentos reportados y entrenar con MSE y dos scores consistentes dentro de la misma CNN-LSTM-atención. Falta demostrar sustantividad frente a alternativas y completar la revisión de literatura. |

La [matriz](literature/review.md), las [fichas primarias](literature/sources.json) y el [BibTeX](literature/references.bib) conservan localizadores y alcance de lectura. No se importó un modelo o código de terceros para el verificador; se escribió la implementación escalar a partir de las ecuaciones con biblioteca estándar y SciPy/Matplotlib. **No se implementó aún la cabeza neuronal.** Las licencias y versiones de bibliotecas se conservan en los archivos de dependencias del repositorio. No se copian aquí los artículos completos.

Se utilizó ChatGPT/Codex para revisión focal de fuentes, formulación, demostración, crítica matemática, escritura del verificador y pruebas, documentación y preparación de publicación. La revisión entre agentes fue asistencia de IA y no una reproducción o verificación humana externa. No existe declaración firmada del estudiante por este documento. El estudiante deberá comprender, comprobar y poder defender o modificar cada parte entregada; esta asistencia no acredita R2 ni autoría responsable por sí sola.

El objeto transferido es matemático, desde desigualdades de momentos y scores hacia una salida neuronal para σ₇. No se atribuye al proyecto una invención previa ni se afirma que cambiar de aplicación baste para R1. El mecanismo sólo podrá clasificarse como adaptación sustantiva si la revisión completa y la comparación con antecedentes justifican esa diferencia. Hasta entonces, el estado es **propuesta sin originalidad acreditada**.
