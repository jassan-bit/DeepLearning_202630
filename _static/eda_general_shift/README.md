# Desplazamiento marginal del EDA general: desarrollo 2023–2024

Se reutilizan exclusivamente los dos CSV diarios de desarrollo ya existentes. Se
comparan las cinco covariables OHLCV del día anterior y el objetivo sigma7 del día
de resultado, sobre casos completos comunes: 350 días en 2023 y 366 en 2024. La
cuadrícula de 731 días y el hueco del 24/03/2023 permanecen presentes. No se ajusta
ningún modelo, no se abren ZIP ni archivos de test, no se cambia el preprocesamiento.

## Productos

- `causal_daily_profile.csv`: cuadrícula completa con fechas de covariables,
  resultado y máscara de caso completo, sin comprimir los días ausentes.
- `descriptive_by_period.csv`: distribución y cuantiles por variable y periodo.
- `marginal_distribution_differences.csv`: cambios de medias/medianas, razones,
  efectos estandarizados descriptivos y distancia KS. El signo es 2024 menos 2023.
- `calendar_blocks.csv`: bloques, máscaras, días válidos y decisión de permutación.
- `conditional_block_permutation.csv`: 12 diagnósticos KS por bloques y Holm.
- `summary.json`: métodos, hashes, resultados, comprobaciones y limitaciones.
- Figura: `book/_static/eda_general_shift/marginales_causales_2023_2024.png`.

## Referencia exploratoria de permutación

Los largos 14 y 28 días, 2.000 permutaciones y semilla 42 se fijan antes de esta
ejecución exploratoria. Se permutan solo bloques calendarios completos y plenamente
observados, manteniendo el número original de bloques elegibles en cada periodo.
Las colas de un día de 2023 y dos de 2024 y los bloques con máscaras quedan fijos
en su periodo original. Sus observaciones válidas siguen incluidas en cada KS.
Así, cada comparación conserva 350/366 observaciones y no mueve ni rellena máscaras.
Quedan fijos 28/350 valores de entrenamiento con bloques de 14 días (8 %) y
70/350 con bloques de 28 días (20 %); en validación, 2/366 en ambos casos. Por
ello la sensibilidad cambia también el conjunto permutable. La hipótesis
condicional no es igualdad incondicional de todas las distribuciones anuales:
las diferencias en las partes fijadas se condicionan y no se aleatorizan.

La referencia presupone intercambiabilidad aproximada de los bloques elegibles
condicionada a las partes fijas. No se ha acreditado ese supuesto ni independencia
entre bloques, en particular para niveles de precios con tendencias. Por ello los
valores p son referencias condicionales de sensibilidad, no pruebas exactas ni
confirmatorias. Holm sobre las doce comparaciones no repara una calibración
marginal inadecuada. Las comparaciones son marginales; no cubren toda distribución
conjunta ni todos los rezagos de las cuatro historias de entrada.

Los bloques del diagnóstico no fijan la unidad inferencial definitiva del benchmark.
Un p pequeño no selecciona variables, modelo, objetivo ni ventana; un p grande no
demuestra igualdad. Las diferencias descriptivas sí permiten reconocer cambios de
escala y justificar validación cronológica, cobertura de regímenes y transformaciones
aprendidas solo con entrenamiento permitido, sin consultar test.

Reproducir desde el repositorio:

```bash
/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_general_shift.py
```

Datos: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta.
Derivados y figura bajo CC BY-NC-SA 4.0 y condiciones aplicables de Binance Vision.
