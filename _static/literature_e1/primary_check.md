# Comprobación adicional de fuentes primarias

Consulta documental del 2 de noviembre de 2026. No se ejecutó código de modelos ni se procesaron datos de mercado.

## Resultado

No hay evidencia suficiente nueva para contar una MLP no lineal independiente como baseline publicado en los dos estudios comprobados.

### Guo, Bifet y Antulov-Fantulin (2018)

Se identificó el repositorio de autores `weilai0980/mixture_models_crypto`, fijado al commit `7aeaf66bb4438d022ae0cf2af2ffec317a8d068f`. El README cita el artículo exacto. En `mixture_neural.py:924–1018` existe `neural_plain_mlp`, con ReLU, dropout, penalización L1/L2 y salida escalar lineal; en capas adicionales incluye conexiones residuales. Es una clase independiente, no una cabeza de la mezcla. Sin embargo, no se encontró su invocación en los controladores y celdas de código inspeccionados. El controlador neuronal ofrece `lstm-concat` y `lstm-mixture`. No se recuperó el artículo completo ni una tabla publicada de esa MLP.

**Tratamiento:** candidato técnico con código de autores; no incrementar la frecuencia de MLP como comparador publicado ni dar por cerrada su elegibilidad. No afirmar tampoco que el artículo nunca la evaluó: la evidencia es insuficiente. El antecedente de 2018 requiere justificar excepción al periodo 2019–2026.

### Herremans y Low (2025)

El repositorio `dorienh/bitcoin_synthesizer`, commit `8570f4e8bdcfe1dd609e5f094d3d5b0059a810dc`, contiene una LSTM seguida de una capa lineal y variantes Transformer/Synthesizer. Los nombres `dense`, `fac_dense` y `mix_dense` designan variantes de atención; las capas feedforward y cabezas lineales no constituyen por sí solas un MLP comparado de forma independiente.

## Trazabilidad

`primary_check.json` conserva las ubicaciones de evidencia, URLs y hashes. `primary_sources/sanitized_search_log.json` registra once consultas, sin volcar tokens de las páginas. Los totales reportados por el buscador no equivalen a artículos cribados: solo se observó la primera página de cada consulta. Estas búsquedas suplementarias no deben sumarse al flujo anterior sin deduplicación. Los archivos de autores se consultaron como texto, sin importarlos ni ejecutarlos.
