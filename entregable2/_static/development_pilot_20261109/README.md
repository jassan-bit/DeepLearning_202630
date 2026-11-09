# Piloto real de desarrollo en Codex Cloud

Registro previo: cf68db00f4d7c3626bdb517bf922265a3b229ea4. Enmienda de análisis secundario previa al replay retenido: d511a8de3b9e9d5fa559d761ce9f2ac732921a80.

40 intentos, 34 completos, 5 fallos de TimeMixer por tiempo y 1 censura externa. Cinco refits bloqueados no fueron intentados. Se registran 50.009 pasos conocidos; falta el número de pasos del extremo censurado.

El resultado primario de tres modelos está incompleto. El análisis secundario sólo usa híbrido y GRU32, cada uno con cinco selecciones y cinco reajustes completos. k=40 es conservador; R=30 es un techo de coste. El R final permanece null.

La simulación temporal no está calibrada bajo la nula (R=30:0.002 frente a alpha/153≈0.000327). Sus tasas de rechazo y MDE son sensibilidades del modelo de planificación, no potencia ni umbrales científicos validados. No se estima la varianza de TimeMixer ni se demuestra superioridad del híbrido.

summary.json conserva los hashes comprimidos y descomprimidos. Los gzip incluyen recibos y diarios verificables; checkpoints.json incluye los 20 hashes de checkpoints de inferencia cargados de forma segura. Los pesos no se hospedan aquí. No se ejecutó HPO/NAS completo ni se accedió a los datos externos 2025–2026.

Reproducción: scripts/prepare_e2_pilot_snapshot.py, run_e2_development_pilot.py, run_e2_reference_variance.py y build_e2_pilot_report.py, con los comandos de la sección 30 del JBook. El snapshot preserva los bytes del protocolo realmente usados; sólo difiere del histórico en un salto final, con JSON canónico idéntico.
