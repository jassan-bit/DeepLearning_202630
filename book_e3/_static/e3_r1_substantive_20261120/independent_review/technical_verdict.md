# Dictamen técnico independiente por IA sobre R1 de V3

**Dictamen: favorable de forma acotada para V3; no favorable para acreditar la referencia aditiva.** Se recomienda dar por sustentada documentalmente la adaptación específica de V3 bajo la definición de R1 de la guía. Es una revisión técnica por IA declarada conforme a la modalidad autorizada por el docente según el autor; no es una firma humana, una decisión del profesor o una acreditación institucional. No asigna nota ni tope al proyecto.

Corte examinado: `29cedb5228ebcecce2a354a1d0ab89adf3936f12`. La revisión se hizo como tarea separada del diseño y entrenamiento. Se verificaron los manifiestos y payloads de código/teoría, requisitos y dos subrevisiones bibliográficas antes de emitir este concepto. No se modificaron modelos, resultados o registros históricos.

## Criterio aplicado

La guía, §6, pp.7–8, pide novedad para el mismo fenómeno y tarea y diferencia de mecanismo, pérdida o entrenamiento. Acepta transferencia con adaptación algorítmica sustantiva y atribución; no exige invención mundial de cada ingrediente. §6.3 prescribe una declaración respaldada por la búsqueda documentada, no un teorema de ausencia universal. La propiedad formal de §14.5.1 debe ser correcta y verificada; no tiene que ser inédita. R2, R3, sustentación, DOI y disponibilidad del servicio son requisitos diferentes.

Una adaptación no sería sustantiva si sólo cambiase el activo, la longitud de ventana, el número de átomos o un hiperparámetro de un algoritmo intacto. Aquí se cambia la lectura y su objetivo de actualización respecto a los antecedentes comparados. El punto no es declarar más expresiva a V3 que cualquier red escalar.

## Razón del juicio favorable para V3

V3 aprende una ley finita de `U=|r7−m6|`, no el signo de una rentabilidad. Con `A=Σ6(r_i−m6)²/7` observable y `c=6/49`, la lectura es `μ=Σ p_k sqrt(A+c a_k²)`, y el ajuste usa `MSE_Y/sY² + 0,1 CRPS_U/sU`. La desviación es poblacional, `ddof=0`. El código realiza la transformación antes de promediar y actualiza posiciones, masas y codificador con ese objetivo conjunto. U se reconstruye de Y y A sólo durante el ajuste: no es una nueva observación futura independiente.

El contraste no se limita a la aditiva: el MDN financiero y Feng2023 emplean leyes/variables financieras y verosimilitud distintas; MIDAS2017 ya pronostica proxies móviles mediante regresión directa; FQF2019 ya aprende posiciones y masas, pero con actualizaciones de cuantiles/Bellman de recompensas en RL. V3 no reclama esas herramientas. La conexión entre componente faltante, mapa dependiente de A y doble objetivo supervisado es una modificación de representación/propagación y pérdida para la etiqueta solapada, no trasladar esos modelos intactos a Bitcoin.

La revisión focal de los métodos cercanos no encontró esa misma cadena para el problema. Se consideraron también dos trabajos primarios de2026 devueltos por las consultas independientes. Uno utiliza un objetivo de diez retornos íntegramente futuros y lecturas puntuales; otro usa el retorno absoluto futuro como proxy y desviaciones móviles como atributos. Sus tareas y métodos concretos constan en `recent_primary_comparison.json`; no se infiere prioridad por cambiar de ventana o instrumento.

## Objeción más fuerte: composición de piezas conocidas

La objeción es válida como límite de la pretensión. CRPS_U equivale, por cambio de variable, a un CRPS ponderado de la ley transformada de Y; por ello no se inventó una regla propia. La ordenación, softmax, leyes atómicas, transformaciones de distribuciones y la identidad de varianza son conocidas. Tampoco la etiqueta auxiliar añade información respecto a Y y A.

La respuesta es restringida: según la guía, una adaptación de propagación/pérdida puede ser sustantiva aunque sus operaciones sean conocidas. La comparación establece qué objetivo y lectura se modifican frente a los métodos estudiados. No establece una clase puntual universalmente más rica, una incapacidad de la aditiva para aprender la misma media ni una garantía de utilidad. Si un antecedente posterior demuestra que esta misma adaptación ya se había usado para la misma tarea, la declaración debe revisarse.

## Revisión acumulativa y límites reales

Se reutilizan las respuestas históricas válidas de Crossref/arXiv/OpenAlex y las cadenas registradas, sin presentarlas como una nueva búsqueda nativa de V3. La actualización agrega consultas focales, cuerpos primarios pertinentes, seguimiento desde cinco semillas y repositorios inspeccionados con sus límites. El seguimiento bidireccional previo del corpus permanece documentado; las nuevas aristas no se cuentan como un censo completo de citas. Los intentos recientes bloqueados antes de origen no cuentan como respuestas bibliográficas.

Este revisor repitió dos cadenas exactas del subestudio de aplicación mediante el proveedor web y registró treinta resultados agregados. No hay conteos nativos por cadena ni se simula otra base. Los cuerpos no accesibles mantienen esa limitación; sus metadatos no se convierten en lectura completa. La corrección de identidad WP11/WP15 se incorpora sin modificar el expediente antiguo.

## Decisiones separadas y preservación

- **Aditiva de1210 parámetros:** los antecedentes de atención y guarda escalar no permiten acreditar su diferencia sustantiva en esta revisión. La recomendación favorable de V3 no se transfiere a ella.
- **V3 de1262 parámetros:** dictamen técnico favorable a la pretensión restringida de adaptación para el problema, respaldado por la matriz y las propiedades con supuestos. No es aprobación del docente.
- **Utilidad del piloto:** la receta completa fue2,70% peor que `atom_mse_only` y falló la heurística previa. Ese resultado queda intacto; no invalida por sí solo la novedad y no autoriza promoción de la ablación ganadora.
- **R2/R3 y calificación:** no quedan acreditadas por este dictamen. No se trasladan los796 trabajos a V3, no se redefine el experimento como confirmatorio y no se concede un tope4 o5 automáticamente.

No se identifica un entrenamiento nuevo necesario para sostener la diferencia documental. El autor debe confirmar y defender la declaración precisa, distinguir fuentes históricas de antecedentes encontrados después y conservar las limitaciones. La ambigüedad Faithful se mantiene en su formulario específico; no se inventa hoy una receta histórica para cerrar una casilla.
