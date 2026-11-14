# 17. Contrato vigente y relación entre E1, E2 y E3

**Estado vigente del 14/11/2026:** E2 está ejecutado para **16 sistemas y 608 fechas**; R=6 para las trece redes y una trayectoria para cada comparador determinista. Se completaron **81 reproducciones y 18 ablaciones**. Esta página fija el contrato actual y distingue el calendario ejecutado de E2 del antecedente E1 y de las propuestas de E3.

## 17.1 Alcance por entregable

| Entregable | Alcance y estado |
|---|---|
| E1 | GRU32, LSTM32 y MLP16, con su benchmark histórico de 608 fechas; mantiene sus resultados propios |
| E2 | Diez candidatos más tres referencias E1 reajustadas y Persistence/SeasonalNaive/ARIMA; evaluación común y suplementos completados en sus alcances documentados |
| E3 | Continuidad temática y propuestas históricas; la guía académica específica y sus resultados propios permanecen separados de E2 |

El híbrido CNN-1D + LSTM + atención y Ridge quedan como antecedentes opcionales. El híbrido no es un candidato obligatorio activo ni una mejora demostrada del benchmark E2. Los archivos históricos `e1_reference.json` y `current_protocol.json` no se renombran como contrato actual.

## 17.2 Objetivo y disponibilidad causal

La serie es **BTCUSDT de Binance Spot, días UTC**. Al comienzo del día t se pronostica

$$
r_t=\log(C_t/C_{t-1}),\quad \bar r_t=\frac{1}{7}\sum_{j=0}^{6}r_{t-j},\quad
\sigma_7(t)=\sqrt{\frac{1}{7}\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

Es desviación **poblacional (`ddof=0`), sin anualización**. Requiere ocho cierres consecutivos. La etiqueta incluye el retorno de t y sólo existe al terminar el día objetivo; h=1. Las entradas son las siete barras anteriores **[t−7,t)**, ordenadas de antigua a reciente, con cinco canales OHLCV. No se añaden rezagos explícitos del objetivo, canales calendario, sentimiento ni historia mensual.

Una barra pasada observada durante evaluación puede entrar en la ventana siguiente porque está disponible en su origen. Esto no permite ajustar pesos, escaladores o umbrales con etiquetas de evaluación. El soporte causal y la máscara común se documentan en [la admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json).

## 17.3 Transformaciones conservadas

1. Cada precio de la ventana se transforma mediante `log(P) − log(C[t−1])`, con el último cierre conocido como ancla.
2. Se aplica `log1p(V)` al volumen.
3. Cada canal se estandariza con media y DE poblacional del entrenamiento permitido sobre sus N×7 posiciones; una DE cero se sustituye por 1.
4. El objetivo se divide por su media positiva de entrenamiento, sin centrar ni usar MinMax; la métrica invierte esa escala.
5. Los escaladores de cada fold y del ajuste final se estiman exclusivamente en su entrenamiento. Los definitivos quedan congelados durante la evaluación.

Se conserva el calendario sin comprimir huecos ni imputar el día incompleto 24/03/2023. Se excluyen las ventanas dependientes de observaciones inválidas mediante las máscaras verificadas. [§1](01_punto_partida.md) desarrolla la formulación; [§10](10_auditoria_datos.md), los controles y sus límites.

## 17.4 Calendario efectivamente ejecutado de E2

| Etapa | Política ejecutada |
|---|---|
| Selección interna | Tres orígenes de desarrollo: enero, julio y octubre de 2024; seis configuraciones por red y embargo causal |
| Ajuste final común | Etiquetas y transformaciones de entrenamiento permitidas hasta **31/10/2024** |
| Calibración | **53 fechas**, separadas del entrenamiento final y posteriores al embargo; cuantiles conservados por predictor |
| Congelación | **81 predictores**, con pesos, escaladores y calibradores, antes de puntuar esta campaña |
| Evaluación | **01/01/2025–31/08/2026**, las mismas **608 fechas** de E1; política de pesos fijos y ventanas de entrada actualizadas causalmente |
| Repeticiones neuronales | **271801–271806**, seis por cada una de las trece redes |

El [protocolo ejecutado de §48](48_protocolo_reducido.md), la [congelación](../_static/e2_benchmark_final_20261113/final_freeze.json) y la [A2 medida](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) son las referencias actuales. Los diez orígenes mensuales de reajuste de planes anteriores no se describen como la política final aplicada.

## 17.5 Métricas y referencias

La primaria es la **media de las RMSE individuales entre semillas**, sobre fechas comunes en escala original. No es la RMSE de un ensamble. Las referencias neuronales son GRU32, LSTM32 y MLP16, reajustadas con su propia selección de validación E2; sus resultados anteriores de E1 no rellenan A2. Los tres comparadores son deterministas y no se duplican para simular incertidumbre entre semillas.

El margen práctico heredado es **δ=0,00025772842664981106** y no se recalcula con una nueva RMSE. La familia primaria comprende veinte pares frente a GRU32 y Persistence. [§56](56_inferencia_estabilidad_final.md) conserva el análisis original y su refinamiento a **20.000 remuestreos**, con los veinte p Holm iguales a 1. Las semillas y R de las antiguas fases E1 siguen perteneciendo a aquellos experimentos.

## 17.6 Límites de independencia y reproducción

Las 608 fechas ya eran conocidas desde E1. La reserva complementaria de septiembre–octubre de 2026 también había sido evaluada; ninguna se vuelve virgen por cambiar modelos. La comparación E2 es retrospectiva. La congelación y la reproducción numérica de [§55](55_reproduccion_ablaciones.md) acreditan trazabilidad, sin recuperar cegamiento ni una nueva confirmación prospectiva.

No se definió previamente δg y el piloto tuvo potencia insuficiente para su efecto planificado. Esas desviaciones permanecen en el [Apéndice B actual](61_reproduccion_consolidada.md). Las comprobaciones aritméticas de la entrega se ejecutan con el comando de [§8](08_reproducibilidad.md), sin volver a entrenar ni generar nuevas predicciones.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/17_contrato_sincronizado.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
