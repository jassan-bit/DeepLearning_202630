# Antecedentes distribucionales: revisión focal del cierre de R1

Preparado con asistencia de IA el 20 de noviembre de 2026. Revisa la candidata
V3 ya implementada; no modifica código, pesos, datos, calibradores o decisiones
del piloto. Cero entrenamientos, inferencias, predicciones de mercado y
remuestreos nuevos.

## Hallazgos que cambian la comparación

**FQF de Yang et al. (2019) es un antecedente más cercano que C51 para aprender
simultáneamente posiciones y masas.** Su §3 ya contiene una mezcla de Dirac con
fracciones ajustables. No se puede atribuir a V3 la invención genérica de ese
estado de salida. El retorno de FQF es una suma descontada de recompensas en
RL, no una rentabilidad financiera. V3 sustituye ese objetivo y las actualizaciones
Bellman por supervisión sobre U, transformación por A observable y el objetivo
conjunto CRPS_U/MSE de la media de sigma7. Esas diferencias están programadas;
no se infieren de una aplicación distinta por su nombre.

**Se recuperó el cuerpo primario de Feng, Li y Zhang (2023).** La comparación ya
no depende sólo de su resumen: §3.1–3.2 especifica DeepAR/Gaussian-mixture/GARCH,
muestreo y cuasiverosimilitud negativa. V3 no estima esa misma variable GARCH ni
usa ese procedimiento: su incertidumbre se concentra en U y evalúa una suma
finita transformada. El acceso nuevo no se atribuye al expediente anterior.

**La regla de scoring y la ordenación no son inventos propios.** CRPS tiene una
representación finita estándar. Los coeficientes/valores ordenados también tienen
antecedentes en redes de cuantiles; el material primario de Bremnes consultado
fue una presentación del autor, no el cuerpo del artículo de 2020. El score de
U es además interpretable mediante un cambio de variable como CRPS ponderado
en Y. `change_of_variable_note.md` delimita esta equivalencia y sus supuestos.

**Corrección de identidad bibliográfica.** El PDF WU18954278 y el DOI
a6057c38-dbad-42dc-bebc-bd2c71fe537f corresponden a *Identifying Stochastic
Processes with Mixture Density Networks*, Working Paper11/1998. No se deben
titular como el distinto *Volatility Prediction with Mixture Density Networks*
WP15/1998. El expediente anterior conserva sus bytes; la corrección se declara
en la nueva matriz y se coordina con la revisión de aplicación financiera.

## Diferencia defendible y objeción fuerte

Una declaración estrecha sostenible es: «Se adapta una representación neuronal
de ley condicional finita para la magnitud del único retorno futuro no observado
de una volatilidad móvil solapada; la lectura propaga cada átomo por la identidad
observada de sigma7 antes de obtener su media, y el ajuste combina CRPS de esa
magnitud con MSE de la media transformada».

La objeción fuerte es que todos los operadores aislados y la transformación de
una ley son conocidos. La respuesta no debe llamar nuevos a esos operadores:
la diferencia verificable es **qué variable se representa, dónde se aplica la
transformación dependiente de cada muestra y qué objetivo actualiza conjuntamente
el codificador y la ley**. Va más allá de cambiar ticker, profundidad o activación.
La misma media puntual puede ser representable por una cabeza escalar flexible;
por tanto no se reivindica superioridad universal, nueva capacidad puntual o
calibración por el mero uso de CRPS.

Dentro de los cuerpos y secciones examinados no se identificó una implementación
con el mismo objeto U, mapa g_A y objetivo conjunto. Eso permite defender una
adaptación específica frente a **esos antecedentes**, no afirmar prioridad mundial
ni demostrar por ausencia de resultados que no exista otra publicación. Esta
subrevisión no emite por sí sola la acreditación completa de R1: la matriz global
ha de integrar requisitos de la guía, revisión externa, hipótesis, fuente/código,
propiedad y cronología. Tampoco añade una condición de superioridad empírica a
R1. El fracaso de la heurística y el deterioro de 2,70% se conservan en el piloto;
no son una prueba de ausencia de novedad.

## Archivos y límites de acceso

- `sources.json` y `antecedent_matrix.csv`: nueve registros con mecanismos,
  pérdidas, entrenamiento, aplicación, localizadores y alcance real de acceso.
- `queries.csv`: tres cadenas sustantivas y recuperación focal de fuentes exactas;
  se conservan también dos intentos nativos fallidos. No es una búsqueda exhaustiva.
- `direct_access_receipts.json` y `native_query_attempts.json`: el proxy impidió
  las descargas directas y los API nativos antes de obtener respuesta de origen.
  El proveedor web sí dio cuerpos primarios indicados en cada registro. Estos
  intentos no se cuentan como respuestas bibliográficas de OpenAlex/Crossref.
- `retrieval_response_hashes.json`: hashes de las representaciones textuales del
  proveedor almacenadas sólo en caché temporal, no hashes inventados de PDF.
- La cobertura histórica de tres bases se conserva y no se presenta como una
  nueva búsqueda nativa sobre V3. No se publican cuerpos completos de terceros.

## Seguimiento de citas y repositorio consultado

`citation_followup.json` y `.csv` registran cuatro aristas observadas desde tres
semillas: FQF→C51/IQN; Feng2023→DeepAR; CRPS2007→Matheson/Winkler1976. Cada
arista conserva el localizador bibliográfico de origen y la visita real al
destino, distinguiendo cuerpo focal de resumen. La revisión de aplicación añade
sus dos semillas por separado; no se suman versiones de la misma semilla.

`repository_inspection.json` fija un repositorio pertinente independiente de
FQF/IQN/QR-DQN, su commit, README y licencia MIT observada. No se lo presenta
como código oficial de los autores, no se ejecutó ni se leyó/copió implementación
para incorporarla al proyecto. La licencia de ese repositorio no licencia el
entregable del autor. El paper y la página editorial no proporcionaron un enlace
GitHub observado; la búsqueda de tal enlace también conserva su resultado.
