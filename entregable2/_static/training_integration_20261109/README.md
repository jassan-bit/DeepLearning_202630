# Integración y perfilado CPU sintéticos

El resumen enlaza los hashes del código con el fixture publicado y los perfiles
aislados. No se leyó mercado ni se ejecutó la campaña científica T18/1000NAS.

- integration_full.json.gz: intentos, costes por etapa, selección y refit de18sistemas.
- profiles_full.json.gz y profiles_table.json:14referencias+TimesNetmáximo,33pasosAdamW.
- classical_inner_events.jsonl.gz y outer_events.jsonl.gz: diarios con hashes y orden.
- author_source_receipt.json: ocho fuentes públicas fijadas de iTransformer.

Los142checkpoints del fixture se guardaron en el workspace y se recargaron con
weights_only=True; el libro conserva sus recibos/hashes. No contienen estado de
optimizador para reanudar una etapa. Las métricas artificiales no prueban precisión
de mercado. Reproducción y límites en el capítulo29 y BUILD.md.

