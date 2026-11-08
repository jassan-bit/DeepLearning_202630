# Integración provisional del catálogo E2

Este paquete es desarrollo de software con datos sintéticos. No es preregistro,
entrenamiento de mercado, búsqueda de hiperparámetros, dimensionamiento ni
benchmark Bitcoin. No modifica los runners, muestras, protocolos o resultados
históricos. El candidato activo es CNN-1D + LSTM + atención temporal; el corrector
HAR anterior permanece como código histórico inactivo.

## Identidad y fidelidad de las recetas

Las diez arquitecturas y el híbrido se adaptan a D7×5 y salida escalar σ₇ a un día.
No son réplicas de los experimentos publicados. FCN/ResNet/Inception proceden de
clasificadores; los prototipos compactos cambian profundidad, anchos, kernels y
normalización. Inception-1D representa una red, no el ensemble de cinco miembros
de InceptionTime. LSTM-autoencoder requiere reconstrucción sólo con entrenamiento
del fold, seguida de ajuste supervisado; disponer de `reconstruct()` no acredita
haber ejecutado ese protocolo con mercado.

Los seis adaptadores existentes conservan sus mecanismos y recetas identificadas,
con cambios D7 explícitos: TimesNet usa extensión aprendida 7→8 y cuatro frecuencias
no-DC; TimeMixer escalas 7→3→1; TiDE salida latente8 que evita LayerNorm(1);
PatchTST replica bordes para patch16/stride8 sin inventar observaciones;
iTransformer usa cinco tokens de variables. No se sustituyen por MLP genéricas
para conseguir que las pruebas pasen. Los seis incorporan diez momentos de la
misma ventana en su cabeza; esa diferencia frente a los prototipos se declara y
no prueba igualdad arquitectónica. No son diez covariables externas.

La nueva composición CNN causal → LSTM unidireccional → atención temporal
conserva siete estados antes de ponderarlos. Su receta inicial es provisional,
no óptima, no receta literal del curso y no innovación acreditada. Las ablaciones
CNN sola, LSTM sola y CNN–LSTM sin atención son controles, no nuevos candidatos.

## Información y límites de las pruebas

Se parte de las transformaciones de E1: logaritmos de precios relativos al último
cierre de cada ventana, log1p(volumen), estandarización por canal entrenada sólo
sobre N×7 posiciones admitidas, y objetivo dividido por su media de entrenamiento
sin centrar. Las etiquetas admitidas terminan hasta origen−8. La causalidad de
prefijo de la CNN se prueba sobre el tensor ya transformado; `relative_window`
usa el cierre final observado como ancla, por lo que no implica invariancia de
prefijo sobre OHLCV crudo.

Las pruebas sintéticas verifican mecánica, no rendimiento de mercado, superioridad
ni benchmark completo. La evaluación futura exige mismo soporte, objetivos,
folds, transformación train-only, selección temporal, presupuesto total incluyendo
preentrenamiento y ablaciones, y métricas en unidades originales. Un periodo ya
observado sigue siendo exploratorio. R=309 de E1 no se transfiere al catálogo.

## Pendientes antes de cualquier entrenamiento de mercado

- Cerrar recetas y diferencias frente a fuentes, licencias y variantes.
- Resolver por receta el requisito de búsqueda estructural de la guía.
- Revisar de forma independiente el pipeline completo y sus controles negativos.
- Registrar folds, exclusiones, selección, presupuesto, piloto, potencia,
  multiplicidad, fallos y congelamiento de pesos antes de evaluación.
- Separar los resultados históricos de E1 de las nuevas ejecuciones.

Los archivos de fuentes de autores y datos sintéticos no se redistribuyen aquí;
se publican identificadores, hashes y recibos agregados cuando estén validados.

## Reproducción sintética

Usar un entorno separado de Python 3.12.14 y las dependencias observadas en
`requirements-e2-current-synthetic.lock.txt`. Las cuatro implementaciones externas
se leen desde árboles fijados fuera del checkout. La fábrica verifica 70 archivos
de esos cuatro árboles; los puertos locales TSMixer/TiDE no requieren importarlos.
El manifiesto de preparación también incluye sus seis archivos opcionales.

```bash
python scripts/download_e2_completion_sources.py --design experiments/e2_current_synthetic_20261108/source_design.json --output-dir /ruta/externa/e2_sources
E2_CURRENT_AUTHOR_SOURCES=/ruta/externa/e2_sources python -m unittest discover -s tests -p 'test_e2_current_models.py' -v
python -m unittest discover -s tests -p 'test_e2_current_pipeline.py' -v
python scripts/smoke_e2_current_catalogue.py --source-root /ruta/externa/e2_sources --steps 2 --pretrain-steps 1 --output /ruta/externa/e2-current-synthetic.json
python scripts/publish_e2_current_implementation.py --check
```

El CLI sólo genera datos sintéticos, limita pasos, tamaños y parámetros y rechaza
sobrescribir un recibo. No admite un CSV de mercado. El recibo público está
saneado: conserva hashes, recetas, controles y pérdidas sintéticas; no incluye
pesos ni predicciones individuales. La fábrica no descarga código por sí sola.

Resultado de la ejecución documentada: 11 modelos, 0 omitidos, 96 días ficticios,
32 ventanas train, 8 de validación, batch 8, 2 pasos supervisados cada uno y 1 paso
previo de reconstrucción AE. Las pérdidas finitas no son un ranking ni prueba
de convergencia. El entorno observado no es el entorno original de los autores.
