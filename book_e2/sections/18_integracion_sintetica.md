# 18. Implementación provisional y verificación sintética

**08/11/2026: los once candidatos están conectados a una fábrica y un flujo común de pruebas sintéticas.** Esto no es entrenamiento Bitcoin, HPO, dimensionamiento, preregistro ni benchmark completo. No hay ganador ni superioridad demostrada. Los runners, protocolos y resultados históricos permanecen intactos.

## 18.1 Código disponible y recetas explícitas

- `scripts/e2_current_models.py`: fábrica exclusiva de los diez modelos y CNN-1D + LSTM + atención temporal; tres ablaciones separadas del catálogo.
- `scripts/e2_current_pipeline.py`: generación sintética, controles de calendario/ventanas, escalado sólo con entrenamiento, ajuste mínimo y predicciones en unidades originales.
- `scripts/smoke_e2_current_catalogue.py`: CLI acotada que genera sus propios datos; no admite archivos de precios, datos de mercado o checkpoints ajenos.

Las [recetas ejecutadas y metadatos](../_static/current_implementation_20261108/recipes.json) registran estructura, parámetros, adaptaciones y fuentes. **Son recetas provisionales de software, no hiperparámetros elegidos por validación ni el diseño experimental definitivo.** El preset `compact_synthetic` reduce el ancho de los cuatro prototipos y del híbrido a ocho; no redimensiona las seis arquitecturas de literatura. El preset por defecto conserva ancho 16 en los cuatro prototipos y CNN/LSTM 32, atención 16 en el híbrido. Ninguno se presenta como configuración óptima o literal del curso.

| Modelo | Parámetros del smoke ejecutado | Identidad y cautela |
|---|---:|---|
| FCN-1D | 409 | Tres convoluciones compactas 3/3/1; adaptación propia, no reproducción del clasificador original |
| ResNet-1D | 785 | Dos bloques residuales compactos; no profundidad original de la fuente |
| Inception-1D | 729 | Ramas 1/3/5 y pooling; una red compacta, no ensemble InceptionTime |
| LSTM-autoencoder con predictor | 1.110 | Encoder/decoder/predictor; preentrenamiento de reconstrucción seguido de ajuste supervisado |
| TimesNet | 1.172.875 | Extensión aprendida 7→8, cuatro frecuencias no-DC y convoluciones 2D conservadas |
| TSMixer | 35.044 | Mezcla temporal y de canales; receta del adaptador existente |
| TiDE | 168.537 | Encoder/decoder denso; salida latente8 evita colapso de LayerNorm(1) |
| TimeMixer | 2.615 | Descomposición y mezcla multiescala 7→3→1 |
| PatchTST | 16.664 | Patch16/stride8 sobre padding de bordes; dos tokens, no nuevas observaciones |
| iTransformer | 795.403 | Cinco tokens de variables; atención y adaptador existentes |
| CNN-1D + LSTM + atención temporal | 793 | CNN causal k3, ancho8; LSTM unidireccional8; atención aditiva8; contexto y Softplus |

Los seis adaptadores conservan **diez momentos de la misma ventana** en sus cabezas. Los cuatro prototipos y el híbrido no los concatenan: es una diferencia declarada de representación, no información externa ni igualdad arquitectónica demostrada. Las fuentes exigidas se verifican por hashes del subconjunto declarado, incluidos archivos de dependencias; esto no certifica el repositorio completo. Las fuentes ausentes o alteradas provocan error: no se reemplazan modelos por otros ni se omiten silenciosamente.

## 18.2 Qué ocurrió realmente

El [recibo sintético](../_static/current_implementation_20261108/synthetic_receipt.json) contiene **11/11 modelos ejecutados y cero omitidos**. El generador produjo 96 días UTC ficticios de 2001; el fold usa 32 ventanas de entrenamiento y ocho de validación. Se hicieron **dos pasos supervisados por modelo**, batch8, Adam LR0,001 y clipping1. El autoencoder recibió además **un paso de reconstrucción previo**, con un optimizador nuevo para el ajuste supervisado.

Todas las salidas y pérdidas fueron finitas. Las cifras de pérdida del recibo sirven para comprobar operaciones, **no para ordenar arquitecturas, medir generalización ni demostrar convergencia**. Dos pasos no entrenan plenamente una red. No se leyeron datos de mercado, no se abrieron las 608 fechas reales y no se produjo un checkpoint del benchmark.

La reconstrucción del autoencoder usa sólo ventanas del entrenamiento. Su decoder participa en esa fase, no en la predicción supervisada escalar; que no tenga gradiente durante el ajuste del predictor es intencional. No se confunde esta secuencia con una pérdida conjunta respaldada por el artículo original.

## 18.3 Controles verificados

- Formas D7×5 y una salida no negativa por ventana; valores finitos, gradientes, roundtrip de estado en memoria y estado recurrente reiniciado.
- Atención temporal sobre **siete estados**, pesos normalizados en el eje temporal y gradientes a CNN, LSTM, atención y cabeza.
- Causalidad de prefijos sobre el tensor ya transformado. El ancla `relative_window` usa C[t−1]; no se afirma causalidad de cada prefijo de OHLCV crudo.
- Calendario UTC diario sin comprimir huecos, ventanas anteriores al objetivo y etiquetas de entrenamiento hasta origen−8; rechazo de solapamientos y fechas incorrectas.
- Media/DE de X sólo del entrenamiento N×7; y dividido por media positiva del entrenamiento, sin centrar. Inversión antes de evaluar salidas sintéticas.
- Validación no usada para gradientes, selección o escaladores; cambios posteriores no alteran el ajuste del preprocesamiento del entrenamiento.
- Invariancia de predicción al acompañamiento de batch en modo evaluación. Las normalizaciones entrenables por batch no implican invariancia durante entrenamiento ni recomiendan batch1.

Los controles no prueban ausencia de toda fuga en un runner de mercado aún no ejecutado. Las tres ablaciones son verificaciones de componentes, no resultados comparativos. CNN sola y LSTM sola usan el último estado; CNN–LSTM sin atención también cambia la agregación respecto del contexto ponderado, por lo que no se interpreta como una intervención causal pura sobre rendimiento.

## 18.4 Reproducibilidad y siguientes pasos

El {download}`paquete documental <../_static/current_implementation_20261108/README.md>`, el [manifiesto público](../_static/current_implementation_20261108/manifest.json) y el [subconjunto de fuentes fijadas](../_static/current_implementation_20261108/source_design.json) permiten reproducir la preparación y los controles. Se mantiene un entorno separado, documentado por versiones; no se afirma reproducir el entorno original de los autores.

Antes de entrenar mercado faltan: adjudicar recetas definitivas y fuentes, resolver NAS según la guía, auditar el pipeline completo, definir HPO y presupuesto incluyendo preentrenamiento/ablaciones, fijar comparadores e inferencia, ejecutar piloto y dimensionamiento propios y publicar el preregistro correspondiente. **R=309 de E1 no se hereda.** La evaluación sobre fechas ya conocidas sigue siendo exploratoria.

HAR residual queda histórico e inactivo. Sus pruebas y limitaciones no se renombran: los nuevos controles y el recibo de esta sección tienen identidad propia.
