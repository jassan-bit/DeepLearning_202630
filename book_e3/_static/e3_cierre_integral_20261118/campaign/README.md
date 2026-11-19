# Desarrollo común y dimensionamiento de E3

Este código permite ejecutar trabajo nuevo en la GPU de Colab y guardar sus
recibos y pesos en GitHub. Los entrenamientos históricos quedan conservados.
Todavía no hay una campaña confirmatoria congelada ni evidencia nueva de R1/R3.

## Etapa GPU disponible

`colab.run(source_commit, token, suite='controls')` obtiene un checkout separado
del commit exacto, verifica el NPZ publicado y reutiliza el Python 3.12.12 aislado
con Torch 2.8.0+cu126 del preflight. El token permanece en el controlador de Colab;
no se envía al proceso que entrena ni se incorpora en archivos o comandos Git.

Por defecto ejecuta tres semillas nuevas para cinco modelos entrenables:

| Modelo | Parámetros | Función en el piloto |
|---|---:|---|
| `OverlapDistribution-v3` | 1262 | Tres átomos y probabilidades condicionales; MSE de sigma + CRPS de U |
| `OverlapDistribution-v3 atom_mse_only` | 1262 | Misma arquitectura e inicialización; sin CRPS |
| `OverlapDistribution-v3 fixed_probs` | 1236 | Átomos aprendidos y probabilidades uniformes |
| `CNN-LSTM-atencion-aditiva E3` | 1210 | Mecanismo aditivo anterior, reajustado bajo esta política nueva |
| `M00_projected` | 1210 | Salida M00 proyectada sobre el piso observado |

Incluye además `floor_only` y `floor_plus_mean_excess`, sin ajuste numérico ni
réplicas ficticias. Las semillas por defecto son 20261118, 20261119 y 20261120.
No se selecciona automáticamente un ganador ni se certifica novedad a partir
de una diferencia de RMSE.

La geometría permanece en BTC de desarrollo: **532 ejemplos de ajuste, siete
fechas purgadas y 116 de validación**. El cargador abre solamente
`training_raw`, `training_y_original` y `training_dates` del NPZ con SHA-256
`186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
No abre los miembros de prueba o calibración, aunque estén en el mismo archivo.
La validación, ya expuesta, termina el 31 de octubre de 2024. No se utiliza el
conjunto histórico de prueba de 608 fechas para elegir la nueva propuesta.

Todos los modelos neuronales usan OHLCV de siete observaciones, precios
relativos al último cierre conocido y volumen `log1p`. Las medias, desviaciones
y escalas de sigma/U se estiman únicamente en el ajuste. U es una etiqueta y no
entra en `forward`. La selección de época usa el mismo MSE de sigma normalizado
para todos, incluso cuando el entrenamiento de V3 añade CRPS.

El piloto aplica AdamW, tasa 0.001, sin decaimiento, lote 32, máximo 40 épocas,
paciencia ocho y norma de gradiente máxima uno. No realiza búsqueda de
hiperparámetros. La comparación con el mecanismo aditivo es un **nuevo ajuste
de desarrollo** con política común; no reproduce su anterior calibración
inicial de sesgo ni sustituye sus resultados históricos.

## Familia completa opcional y límites de la comparación

`suite='full_family'` ejecuta los 13 cuerpos neuronales E2 verificados por el D02
inmutable, Persistence, SeasonalNaive, ARIMA, el mecanismo aditivo y V3:
18 sistemas y 17 rivales del candidato. Conserva las arquitecturas y sus puntos
registrados de dropout; sustituye el optimizador histórico por la política de
este piloto. No afirma repetir las recetas ganadoras originales de E2.

El autoencoder ejecuta y conserva **AE1, AE2 y ajuste supervisado**, con escala
auxiliar estimada sólo en el ajuste, transferencia de ambos encoders y todas
sus épocas/pasos contabilizados. Esto supone tres fases y un costo distinto al
de otros cuerpos. La igualdad de épocas por fase no constituye igualdad de
presupuesto de cómputo. El HPO, el presupuesto total, los controles de capacidad
y el tratamiento final todavía deben fijarse antes de la confirmación.

Los controles temporales tienen una sola trayectoria por conjunto. Persistence
usa rezago uno y SeasonalNaive rezago siete; ARIMA(1,0,1) estima una vez hasta el
30 de junio, después filtra sin reajustar parámetros. Su pronóstico se escribe
y sincroniza a disco **antes** de revelar la observación siguiente. No se
elimina una fecha ni se sustituye silenciosamente un modelo que falla. Estos
controles acceden al historial de etiquetas rezagadas; el protocolo final debe
explicitar esa información, distinta de una ventana neuronal D7.

Cada fallo aparece en `failure.json` y deja el piloto incompleto. ARIMA
interrumpido requiere conciliación explícita y no se vuelve a ajustar
automáticamente.

## Guardado y reanudación

Antes de iniciar cada fase se sincroniza un marcador inmutable
`phase_started.json`, vinculado a su solicitud y al código de origen. Cuenta el
intento aunque falle la creación del optimizador o el primer checkpoint; no
afirma que ya se haya dado un paso. Después de cada época se guarda de forma
atómica el modelo actual, el mejor,
el optimizador, sus momentos y los estados aleatorios de Python/NumPy/Torch/CUDA.
Una interrupción dentro de una época restaura el límite anterior y repite esa
época. Una fase completa recupera sus mejores pesos; un trabajo completo se
reutiliza verificando su solicitud. La configuración de variante, escalas,
semillas, código y datos se vincula a los recibos.

El tiempo medido suma épocas cerradas; no pretende incluir cómputo perdido en
una interrupción abrupta. No se presenta como presupuesto final de HPO.

El controlador guarda cada tres trabajos terminados en ramas
`e3-closure-pilot-<fecha>`. Copia los `.pt` originales sin volver a serializarlos
y conserva todos los checkpoints locales. Cada parte queda por debajo de
192 MiB y el índice final enlaza todas sus ramas. Esto permite que la familia
completa con diez semillas exceda 256 MiB en total sin exceder el límite por
snapshot. Una interrupción dura de Colab conserva las partes ya subidas.

El archivo `stage_receipt.json` cuenta intentos de ajuste por trabajo en cada
parte a partir de esos marcadores y separa las fases intentadas de los
checkpoints disponibles; no cuenta épocas como modelos nuevos.
El índice final contiene cero ajustes adicionales a los ya guardados en las
partes. Si una subida falla, su snapshot local exacto se conserva para reintento.

Reanudar dentro de la misma sesión usa `resume_output=<ruta anterior>` y la
misma configuración. Restaurar automáticamente todas las partes desde GitHub
en una sesión nueva todavía no está implementado; los pesos y recibos quedan
descargables y verificables en las ramas enlazadas.

## Potencia prospectiva

`power.py` exige predicciones nuevas y completas de los 18 sistemas con al menos
**diez semillas emparejadas**. El piloto pequeño de controles o un preflight
sintético no satisfacen esa entrada. La reducción relativa por semilla es
`(RMSE_rival - RMSE_candidato) / RMSE_rival`.

El planificador recibe por separado el margen de relevancia y el exceso bajo
la alternativa; no sustituye estos valores por la ventaja observada. Simula
la dependencia entre todos los rivales a partir de la covarianza del piloto y
evalúa tres pruebas unilaterales de localización: t emparejada, Wilcoxon de
rangos con signo y signo exacta. Aplica **Holm sobre los 51 valores p juntos**;
el mayor p ajustado de las tres pruebas gobierna cada rival. Informa tanto
potencia por prueba/contraste como potencia conjunta de rechazar los 51.

Se presentan sensibilidad de covarianza, intervalos Monte Carlo simultáneos,
rejilla de semillas y una rejilla explícita de exceso detectable. No se
multiplican potencias marginales para inventar potencia conjunta. Las colas
del supuesto normal que exceden la cota física de reducción del 100 % se
reportan y no se usan para proponer un tamaño.

Esta simulación es **condicional a un modelo normal de variación por semilla
en BTC de desarrollo**. Las tres pruebas no son evidencia independiente. Las
alternativas de media/mediana/localización coinciden bajo sus supuestos; no se
afirma que coincidan para cualquier distribución de errores. La incertidumbre
temporal, la generalización a los tres conjuntos adicionales y la potencia de
la batería entre conjuntos requieren su propio análisis congelado. Por eso
`power_dimensioned` y `real_campaign_allowed` permanecen en `False`, incluso si
se encuentra un tamaño condicional de semillas.

`data.py` prepara una propuesta de particiones por número de observaciones y
una purga de siete soportes. Construir etiquetas de los conjuntos reservados
exige un protocolo final, potencia dimensionada y tag anotado. El piloto BTC
no llama a ese constructor.

## Verificación local realizada

Las pruebas usan datos generados, sin red ni mercados. Cubren la identidad de
varianza, soportes purgados, escala sin contaminación de datos posteriores,
rechazo de acceso prematuro a datos reservados, ajustes reales del optimizador
sobre datos sintéticos, las tres fases SAE, igualdad de trayectoria al reanudar,
reutilización de trabajos, Holm global, bloqueo de potencia incompleta y
conservación exacta de checkpoints en snapshots admitidos por el guardado.

La implementación y la revisión del código utilizaron IA. Esta declaración no
sustituye el dictamen de novedad ni convierte pruebas sintéticas en resultados
experimentales de mercados.
