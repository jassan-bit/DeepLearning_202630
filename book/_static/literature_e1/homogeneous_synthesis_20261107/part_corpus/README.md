# Adjudicación homogénea del corpus E1 — 7 de noviembre de 2026

Esta adjudicación aplica la misma regla a las fuentes históricas y a las lecturas posteriores. No modifica N7 ni presume una nueva búsqueda de bases de datos. Incluye registros pendientes para que el universo de acceso sea auditable; su número no equivale a originales leídos.

La unidad es la obra deduplicada por DOI o título/autores/versión. Las frecuencias usan experimento base independiente: el documento probabilístico Dudek2025 no añade otro voto a Dudek2024. DeepVol T3/E1T3 es la misma obra con acceso actualizado.

## Denominadores

```json
{
  "identities_in_documentary_ledger": 47,
  "primary_document_read_identities": 26,
  "complete_mirror_only_identities": 1,
  "crypto_construct_independent_experiments": 12,
  "crypto_reference_extractable_independent_experiments": 10,
  "crypto_reference_reporting_sensitivity_experiments": 8,
  "unresolved_crypto_continuous_construct_identities": 4,
  "historical_N7_unchanged": true,
  "stratum_identities": {
    "crypto_realized_or_historical_continuous_construct": 12,
    "crypto_continuous_volatility_claim_unreconciled": 4,
    "transfer_equity_realized": 3,
    "author_code_only_transfer": 1,
    "partial_or_unobtained_original": 17,
    "crypto_volatility_states": 1,
    "crypto_price_return_volatility_ambiguous": 1,
    "complete_mirror_only_crypto_risk": 1,
    "crypto_multitarget_final_volatility_not_demonstrated": 1,
    "financial_risk_target_unverified": 1,
    "crypto_price_partial_primary": 1,
    "crypto_probabilistic_same_base_experiment": 1,
    "transfer_equity_implied": 1,
    "crypto_implied_volatility_index": 1,
    "author_code_only_crypto": 1
  }
}
```

La matriz confirmada aborda volatilidad realizada/histórica continua de criptomonedas. Incluye otras frecuencias, horizontes, exchanges, futuros y fuentes de datos: no exige reproducir sigma7/OHLCV de E1. CVI implícita cripto conserva estrato propio; NASDAQ, Shanghai y VIX no entran en sus frecuencias.

## Frecuencias comparativas observadas

| Estrato | Denominador | LSTM | GRU | MLP | CNN genérica | TCN | Transformer encoder |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_construct_all | 12 | 10 | 4 | 3 | 1 | 1 | 1 |
| crypto_construct_reference_extractable | 10 | 10 | 4 | 3 | 1 | 1 | 1 |
| crypto_construct_reference_reporting_sensitivity | 8 | 8 | 2 | 2 | 0 | 1 | 1 |
| crypto_declared_target_inclusive_sensitivity | 16 | 12 | 6 | 3 | 1 | 1 | 3 |
| crypto_all_continuous_constructs_including_implied | 13 | 11 | 4 | 3 | 1 | 2 | 1 |
| crypto_construct_publication | 7 | 7 | 3 | 3 | 1 | 1 | 0 |
| crypto_construct_grey | 5 | 3 | 1 | 0 | 0 | 0 | 1 |

Los números son ocurrencias positivas de referencias autónomas, no prevalencias globales ni votos de calidad. El corpus confirmado tiene dos fuentes (S9 y Linde) con modelos propios autónomos cuyo rol de baseline básico no se verificó; se conserva su presencia sin imputar ausencia. El subconjunto extraíble usa como denominador solo estudios con al menos una base verificada. Los conteos por escala, género, CVI, clasificación y transferencias están en `adjudications.json`.

La sensibilidad de documentación §3.1 retira ambos S8 y PSO-GRU por descripción insuficiente de referencia/protocolo. No elimina R01, Dudek2024, Brauneis o Zhou porque falte un ancho definitivo: la receta parcial permite identificar candidatos y exige buscar arquitectura en E2 según §3.2. S7 conserva su ocurrencia, junto con conflictos de dimensión y escalado de muestra completa; la descripción no valida sus resultados.

S6, Sun2025 y CryptoMamba se tratan igual: constructo o proxy no conciliado. Zahid añade solo híbridos y tampoco resuelve el contraste entre fórmula intradía y datos diarios. Su sensibilidad inclusiva nunca se presenta como corpus confirmado. El horizonte numérico no documentado de S8 permanece explícito; contar una familia no vuelve su receta reproducible.

## Implicación para la selección

LSTM tiene más ocurrencias; GRU y MLP siguen y MLP deja de ser una alternativa marginal frente a TCN. En la sensibilidad de descripción GRU y MLP empatan. CNN genérica, TCN y Transformer encoder no se intercambian: una CNN de dimensionalidad desconocida no prueba Conv1D; una DCCNN propia o DCC con atención no prueba la TCN simple de E1. La frecuencia disponible no demuestra que GRU–LSTM–TCN sea la única terna legítima ni satisface por sí sola simplicidad, capacidad y diversidad inductiva.

La guía no exige tres macrofamilias ni un mínimo numérico de artículos TCN. Hace falta combinar recurrencia, complejidad bajo supuestos explícitos, sesgo inductivo y protocolo común, conservando MLP y atención como alternativas. Ninguna frecuencia ni revisión posterior acredita selección previa al entrenamiento o convierte estimaciones en resultados observados.

## Trazabilidad

`protocol.json` fija reglas, fuentes y hashes; `adjudications.json` contiene cada decisión, su evidencia, identidad/dependencia y los conteos reproducibles. `null` significa no verificado; lista vacía, ausencia de ocurrencia positiva bajo la regla, sin negación global. Los textos completos no se redistribuyen. La lectura de DeepVol es del depósito de autor; la de VIX es del cuerpo editorial con tablas no recuperadas.

El procedimiento es adjudicación documental por un agente, con revisión crítica de mecanismos por otro. No hubo dos revisores humanos ni un índice de acuerdo medido.
