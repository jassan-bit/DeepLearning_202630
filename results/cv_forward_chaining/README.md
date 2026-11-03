# Forward chaining con embargo explícito — regla 2

Generado por `scripts/prepare_forward_chaining.py` con `timeseries-cv` 0.1.5. La firma real es `(sequence, numInputs, numOutputs, numJumps)`. El ejemplo oficial `np.arange(27)`, entrada 4, horizonte 3 y salto 2 conserva sus ocho folds; es una ilustración sintética, no un resultado de Bitcoin.

## Regla de partición de regresión temporal

Para el origen diario UTC `t`, la entrada es `X_t=(z_(t-D),...,z_(t-1))`. Se fija `E_D=D` días de calendario omitidos del ajuste inmediatamente antes del origen: `[t-D,t-1]`, ambos extremos incluidos. Una muestra de entrenamiento con día de resultado `u` solo se admite si `u <= t-D-1`, su etiqueta ya está disponible y su ventana OHLCV y etiqueta son válidas. La distancia entre el último resultado de ajuste y el de validación es al menos `D+1` días.

La biblioteca por sí sola terminaba en `t-D`, omitiendo `D-1` días interiores. Esta versión aplica el filtro adicional explícito de la ruta 6.4 de la guía. En los datos actuales retira una muestra elegible adicional por fold; no elimina días de validación ni comprime el calendario.

El CSV registra el embargo, sus extremos, el último resultado de ajuste permitido, las muestras retiradas, el soporte de retornos del objetivo de validación y el número de retornos que comparte con el último objetivo de ajuste. Los hashes de índices seleccionados se recalculan después de filtrar.

## Alcance y dependencia

La preparación usa únicamente los CSV diarios ya existentes de desarrollo 2023–2024. Se mantienen las cuatro historias D=7/14/21/28 y las cuatro ventanas de volatilidad n=7/14/21/28, horizonte y salto de un día: 16 configuraciones, 366 orígenes de 2024 por configuración y 5.856 filas. El día incompleto del 24/03/2023 y las ventanas inválidas se conservan como ausentes. No se abren ZIP ni archivos de test, no se ajustan modelos y no se ejecutan contrastes de hipótesis.

El embargo de covariables no garantiza que dos objetivos de volatilidad móvil tengan soportes de retornos disjuntos. Con `n>D+1` quedan retornos compartidos: ocurre en 2.196 filas de seis configuraciones de sensibilidad. Esta propiedad queda diagnosticada; las filas no se consideran réplicas independientes ni resultados de entrenamientos. La inferencia posterior debe conservar esta dependencia.

## Recuentos después del embargo

| Historia D | Ventana n | Folds | Muestras de ajuste mín.–máx. | Retornos compartidos: último objetivo de ajuste y validación |
|---:|---:|---:|---:|---:|
| 7 | 7 | 366 | 343–708 | 0 |
| 7 | 14 | 366 | 329–694 | 6 |
| 7 | 21 | 366 | 315–680 | 13 |
| 7 | 28 | 366 | 301–666 | 20 |
| 14 | 7 | 366 | 322–687 | 0 |
| 14 | 14 | 366 | 322–687 | 0 |
| 14 | 21 | 366 | 308–673 | 6 |
| 14 | 28 | 366 | 294–659 | 13 |
| 21 | 7 | 366 | 301–666 | 0 |
| 21 | 14 | 366 | 301–666 | 0 |
| 21 | 21 | 366 | 301–666 | 0 |
| 21 | 28 | 366 | 287–652 | 6 |
| 28 | 7 | 366 | 280–645 | 0 |
| 28 | 14 | 366 | 280–645 | 0 |
| 28 | 21 | 366 | 280–645 | 0 |
| 28 | 28 | 366 | 280–645 | 0 |

## Reproducibilidad y trazabilidad

```bash
/workspace/.venvs/timeseries-cv/bin/python scripts/prepare_forward_chaining.py
```

La ejecución normal utiliza la ilustración oficial cacheada y comprueba su hash; no requiere consultar la red. `--refresh-illustration` permite una actualización explícita del recurso de origen. Los dos CSV de desarrollo no se modifican y sus hashes se conservan en `summary.json`. La generación sincroniza los espejos de `folds.csv` y `summary.json` de `book/_static/cv_forward_chaining/`.

SHA-256 de `folds.csv` de la regla 2: `d46153075ce18231f24ff43b6f80dba555003b963dc5d1c4bd4d8ade90148571`.

La versión anterior, conservada en el historial Git, tenía SHA-256 `dbe66b20b72f28b7e477c79d8b1887fc80b96957fa3fbaab0e211f2a5b398dc5`. No es el manifiesto vigente. El resumen registra el cambio de regla y el hash del generador.

El resumen conserva la procedencia y SHA-256 de la figura oficial del notebook, bajo MIT, copyright 2021 DidierRLopes y crédito a la coautoría de Filipe R. Ramos. El aviso completo está en `book/_static/licenses/timeseries-cv-LICENSE.txt`. Los datos y derivados proceden de Binance Vision/Binance Spot, procesamiento Jassan Arteta, bajo CC BY-NC-SA 4.0 y las condiciones aplicables; véase el capítulo 2 del libro.
