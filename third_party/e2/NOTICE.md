# Fuentes y avisos de los adaptadores E2

`e2_final_models_mlp.py` contiene adaptaciones explícitas a PyTorch de TSMixer/TiDE del código fijado de Google Research. Se conserva el aviso de sus fuentes: **Copyright 2026 The Google Research Authors.** Licencia Apache 2.0, incluida en `Apache-2.0.txt`.

Fuentes: `google-research/google-research@d3a4bd8f0eceddda553e47085033b537935b283f`, `tsmixer/tsmixer_basic/models/tsmixer.py` y `tide/models.py`. Las modificaciones incluyen el framework, la representación D7 y la lectura escalar de volatilidad, documentadas mediante metadatos y pruebas. Estas adaptaciones no son réplicas literales de su tarea original.

La recurrencia CPU de Mamba en `e2_final_models_attention.py` adapta la referencia Apache 2.0 de `state-spaces/mamba`, versión v1.2.0, commit `2845255a95f7e4a76fe649ec0532178fe3015dd0`. Se conserva **Copyright (c) 2023, Tri Dao, Albert Gu**. Sus valores y gradientes se contrastan con la referencia fijada.

Los demás árboles de autores se recuperan y verifican por sus commits y hashes fuera del repositorio; este suplemento no redistribuye esos árboles ni sus pesos. Sus licencias originales se conservan con las fuentes recuperadas. El registro histórico documenta MIT para Autoformer e identificaciones Apache 2.0 para Informer/Crossformer; también preserva discrepancias de avisos y la ausencia de una licencia raíz verificada de S-Mamba. No se concede una licencia supuesta a ese repositorio.

Estos avisos se refieren al código señalado, no a la licencia de las velas de mercado, a los artículos científicos ni al documento de la asignatura. Su procedencia y condiciones se identifican por separado.
