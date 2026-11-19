# V3 de desarrollo: ley atómica del componente futuro

Preparado con asistencia de IA el 18 de noviembre de 2026. **No se acredita novedad R1, calibración o superioridad R3. No se entrenó con datos de mercado en esta etapa.**

El módulo nuevo es `src/e3_overlap_model_v3.py`. Conserva el codificador CNN causal–LSTM y pooling por atención de M00, pero sustituye la cabeza escalar por una ley condicional de tres átomos positivos ordenados de la magnitud futura U. Aprende tres incrementos y dos logits; transforma cada átomo mediante la identidad exacta de σ7 y promedia sus valores. La pérdida combina MSE del pronóstico puntual y CRPS de U. La ley no presupone una escala Student fija ni se reduce a estimar únicamente dos momentos.

| Variante de desarrollo | Parámetros activos | Propósito |
|---|---:|---|
| `atom_crps` | 1262 | Tres átomos y masas aprendidas, MSE + 0,1 CRPS normalizado. |
| `atom_mse_only` | 1262 | Misma parametrización e inicio, sin CRPS; aísla la pérdida auxiliar. |
| `fixed_probs` | 1236 | Mismos átomos iniciales, masas uniformes y MSE + 0,1 CRPS; explora aprender las masas. No es una comparación de capacidad idéntica. |

La referencia aditiva histórica tiene 1210 parámetros y se conserva intacta. También se conservan los pilotos del mecanismo de 1211 y de la cabeza por momentos de 1236: el nuevo control `fixed_probs` no identifica ese modelo histórico aunque coincida su conteo. El número de parámetros por sí solo no identifica una arquitectura.

## Interfaz

```python
from src.e3_overlap_model_v3 import OverlapDistributionModel

model = OverlapDistributionModel(
    "atom_crps", u_scale=fit_u_mean, y_scale=fit_sigma_mean, seed=271801
)
outputs = model.forward_components(X_float32, A_raw_variance)
losses = model.loss_components(outputs, target_sigma_raw, target_U_raw)
prediction = model(X_float32, A_raw_variance)
```

`X` es lote×7×5, transformado y estandarizado con estadísticas de ajuste. `A` es la varianza parcial conocida, **no su raíz**. U es una etiqueta calculada de las siete rentabilidades realizadas; nunca entra en `forward`. Las unidades de U y de σ se normalizan por escalas distintas estimadas sólo en ajuste. La suma finita evalúa exactamente la esperanza bajo la ley discreta del modelo, no bajo la verdadera ley condicional desconocida.

El checkpoint debe incluir `configuration()` además de `state_dict()`: el estado tensorial solo no codifica la modalidad de pérdida. El ejecutor debe verificar variante, escalas, datos y plan antes de reanudar. Los tres controles comparten exactamente los parámetros iniciales de codificador y de átomos; el control de masas fijas elimina los logits y se declara su menor capacidad.

Pseudocódigo de un lote de ajuste:

```text
h = atención(codificador(X estandarizado con el tramo de ajuste))
a = acumulada[sU · (epsilon + softplus(lineal_atom(h)))]
p = softmax(lineal_masas(h), 0)   # o masas uniformes declaradas
sigma_atom = sqrt(A + (6/49) a²)
pronóstico = sum(p · sigma_atom)
pérdida = mean(((pronóstico - sigma_real)/sY)²)
          + lambda · mean(CRPS(a/sU, p, U_real/sU))
actualizar parámetros sólo con este lote del tramo de ajuste
```

En inferencia se omiten las dos últimas líneas y no se leen U_real ni sigma_real. La selección, parada y presupuesto de entrenamiento corresponden al ejecutor del piloto y deben ser comunes entre variantes; este módulo no ejecuta entrenamiento por sí mismo.

## Comprobación sin mercado

```bash
python -B -m unittest discover \
  -s experiments/e3_cierre_integral_20261118/mechanism \
  -p test_overlap_v3.py -v
```

Las pruebas usan números artificiales: ley de probabilidades, soporte, conteos, inicio compartido, CRPS por enumeración, riesgo poblacional de una ley finita de juguete, gradientes, escala y serialización. No son experimentos de superioridad ni una campaña confirmatoria. `verification.json` registra únicamente la ejecución efectivamente realizada.

## Decisión siguiente

El piloto de desarrollo debe comparar la referencia histórica y los tres controles con datos, escalado, selección por validación y presupuesto comunes. Después se elegirá conservar o descartar V3, manteniendo todos los resultados, y se fijará el protocolo futuro antes de confirmar en datos adicionales. La pérdida auxiliar de 0,1 y K=3 son decisiones nuevas de desarrollo, no decisiones anteriores a resultados históricos ni un margen confirmatorio.

La hipótesis concreta es que aprender una ley flexible del componente no observado puede aportar frente a una guarda escalar y frente a la misma ley entrenada sólo por MSE. Una mejora de RMSE no prueba novedad; una degradación no se elimina del registro. La procedencia y el alcance matemático están en `theory.md`, `claim_comparison.csv` y `provenance.json`.
