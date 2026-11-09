# Cribado de la ronda nativa recibida de E1

Se revisan 620 grupos candidatos provisionales de 834 ocurrencias exportadas de WoS/Scopus/IEEE. Los siete resultados IEEE no exportados permanecen fuera del universo recibido. No se ejecutan búsquedas nativas nuevas ni se reconstruyen sus fechas o filtros.

Cada grupo tiene una decisión y razón en decisions.csv. Los resúmenes disponibles se revisan; los candidatos avanzados conservan disposición terminal de lectura primaria, exclusión documentada o informe no recuperado al corte. Acceso al primario, calidad, contexto e inclusión estricta se separan en evidence_matrix.csv y primary_assessments.json. Las revisiones previas enlazadas se reutilizan con identidad, sin presentarlas como lecturas nuevas. records.csv mantiene los campos originales de la unión y añade conteos nativos/históricos separados; el flujo834→620 corresponde únicamente a los exports nuevos.

Se conserva el ledger global de diez enlaces provisionales por título (cinco históricos y cinco entre ocurrencias nuevas); ocho grupos del panel nativo conservan esa marca. Los 620 grupos no se presentan como 620 artículos únicos verificados.

La síntesis es posterior al benchmark cerrado. Ninguna revisión bibliográfica reentrena modelos, selecciona semillas ni cambia sus métricas. El núcleo experimental permanece GRU32–LSTM32–MLP16. Las frecuencias describen la evidencia evaluada y no su prevalencia mundial.

La revisión y reconciliación se realizan mediante agentes; no se inventa doble cribado humano ni acuerdo interevaluador. No se redistribuyen abstracts completos, PDFs, precios, objetivos, pronósticos, pesos o escaladores.

review_inputs/ contiene copias exactas de los juicios y snapshots bibliográficos necesarios para recompilar. Los alias canónicos del ZIP8.4 conservan las rutas del compilador; no convierten el cribado en una consulta o lectura automática nueva.
