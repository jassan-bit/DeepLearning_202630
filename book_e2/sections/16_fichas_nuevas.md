(fichas-bibliograficas-de-las-cuatro-altas-propuestas)=
# Fichas bibliográficas de FCN, ResNet, Inception y LSTM-autoencoder

> **Archivo de antecedentes:** documento **§16** del registro anterior. Conserva su fecha, alcance y resultados históricos. [Ver organización del informe](concordancia_guia.md).

**Lectura vigente:** el protocolo ejecutado es [§48](48_protocolo_reducido.md), los resultados comunes están en [§6](06_benchmark_final.md), la reproducción registrada en [§55](55_reproduccion_ablaciones.md) y el Apéndice B actual en [§61](61_reproduccion_consolidada.md). Las cifras científicas y los recibos históricos conservan su fecha y su alcance.

**Antecedente de planificación del 11/11/2026:** ese corte fue sustituido por la ejecución documentada en [§48](48_protocolo_reducido.md). Su texto exacto se conserva en el archivo enlazado al final; no describe trabajos finales pendientes.

**Fuente y adaptación:** los esquemas/propuestas del 08/11 se conservan como antecedentes bibliográficos. Para la receta ejecutada consultar [§44](44_arquitecturas_publicadas.md) y la actualización de [§48](48_protocolo_reducido.md); §§26–28 conservan decisiones previas: no equiparar el ensemble original de Inception a un miembro, las cabezas lineales propuestas a la Softplus vigente ni confundir fases del autoencoder. Fuentes formalizadas: {cite:p}`e2_wang_fcn_resnet,e2_inceptiontime,e2_lstm_sae`. Wang se identifica como preprint 2016 con DOI arXiv, separado de la referencia IJCNN 2017 cuya verificación editorial sigue parcial.

Las cuatro redes ya forman parte del benchmark ejecutado de [16 sistemas](06_benchmark_final.md): seis inicializaciones por receta neuronal, 608 fechas comunes y predictores congelados. La [extracción vigente](../_static/e2_literature_closure_20261113/current10_extraction.csv) y la [síntesis actual](62_cierre_literatura_focal.md) completan la lectura primaria inicial del 08/11. Los diagramas y las fórmulas de propuesta conservados debajo explican el mecanismo; su cabeza lineal y la pérdida conjunta propuesta para SAE no describen las recetas finales.

La receta ejecutada usa D7×5 OHLCV, salida escalar con Softplus y escalado de entrenamiento. FCN y ResNet mantienen sus cuerpos definidos; Inception es una red individual de seis módulos, no el ensemble de cinco; SAE usa AE1 → AE2 con el primer encoder congelado → transferencia → ajuste supervisado, con encoders 25/26 y sin pérdida conjunta de reconstrucción durante la etapa predictiva. Las identidades, fuentes fijadas y diferencias de tarea están en [§44](44_arquitecturas_publicadas.md) y [§48](48_protocolo_reducido.md). Un kernel mayor que siete se aplica con padding declarado y no aporta días adicionales.

**NR** significa no reportado en la versión y condición examinadas; **NV**, no verificado. Las notas iniciales que agrupaban ambos casos se interpretan con esa distinción y con la matriz actual, sin convertir una falta de verificación en ausencia del dato en el artículo. Las páginas del PDF se cuentan desde 1. Se separan resultado publicado, derivación analítica, propuesta inicial y receta ejecutada.

La [implementación vigente](04_implementacion.md) distingue prototipos y cuerpos publicados. Los parámetros y tiempos medidos de las recetas finales están en [§51](51_eficiencia_costes.md), los perfiles batch 1 A100 en [§54](54_robustez_interpretabilidad.md) y los FLOPs core en [§58](58_flops_predictores.md). Las fórmulas de coste de estas fichas son derivaciones analíticas, no mediciones de los artículos ni sustitutos de esos registros.

## Contexto común de evaluación

FCN, ResNet e InceptionTime nacieron para clasificación de series completas. Sustituir softmax por salida lineal no convierte sus resultados UCR en evidencia de forecasting. El LSTM-SAE identificado sí pronostica, pero no evalúa BTC ni acredita automáticamente un autoencoder de pérdida conjunta. Comparar los cuatro exige la misma variable objetivo, ventana histórica, horizonte, particiones temporales, transformación y presupuesto de búsqueda. No ordenar modelos mezclando MPCE, accuracy y RMSE de distintos datasets.

En la notación histórica de las propuestas, t es el último instante observado. El contrato ejecutado expresa el pronóstico al inicio del día de resultado t: X_t contiene los siete días cerrados t−7,…,t−1 y Y_t incluye el retorno de t. B es tamaño de batch, N número de ventanas, L longitud, C canales y H salidas. La cabeza lineal propuesta ŷ_t = W z_t + b y L_pred = (NH)^−1 Σ_t ||y_t−ŷ_t||² se conservan como antecedente; el ejecutor utiliza salida escalar positiva y MSE en la escala de entrenamiento, con RMSE reportada en unidades originales. No se atribuye esa adaptación a los clasificadores originales.

(e2-lit-fcn)=
## FCN 1D

### Fuente y evidencia publicada

Wang, Z., Yan, W. y Oates, T. (2017), *Time series classification from scratch with deep neural networks: A strong baseline*, IJCNN, pp. 1578–1585. Preprint v4 de 2016: [PDF](https://arxiv.org/pdf/1611.06455). Fig. 1 y §II.B, pp. 2–3: tres convoluciones, filtros 128/256/128 y kernels 8/5/3, BN, ReLU y GAP. §III.A, p. 3: 44 datasets UCR, particiones originales, normalización con estadísticas de entrenamiento; Adam 0.001, entropía cruzada; selección por mínimo loss de entrenamiento, sin validación cruzada. Tabla I, p. 4: FCN MPCE 0.0219, 18 victorias, ranking aritmético 3.977; Adiac error 0.143. No son métricas de regresión.

### Diagrama y fórmula de la adaptación

Diagrama propio, inspirado en Fig. 1(b):

    X [B,L,C] → Conv1D(128,8)+BN+ReLU
              → Conv1D(256,5)+BN+ReLU
              → Conv1D(128,3)+BN+ReLU
              → GAP temporal [B,128] → Dense lineal(H) → ŷ [B,H]

h^(j) = ReLU(BN(Conv1D_j(h^(j−1)))), z_c = L^−1 Σ_s h^(3)_(s,c).

Cambio explícito: salida lineal/MSE y objetivo futuro en lugar de softmax/entropía cruzada. Propuesta: stride 1, padding same y ventanas cerradas en t. Una convolución same dentro de una ventana íntegramente histórica no usa automáticamente futuro del objetivo; sí sería fuga aplicar convolución centrada a la serie completa antes de cortar en t. GAP puede diluir cuándo ocurrió una señal dentro de la ventana: es una limitación razonada que requiere ablación, no un fracaso publicado en BTC.

### Código, datos, licencia y coste

[Repositorio de autores](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline), [FCN.py](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py): usa actualmente Conv2D y reshape con dimensiones singleton, no una implementación Conv1D moderna literal. El archivo fija 2000 épocas. No trasladar mecánicamente esos tensores a multivariado. Licencia del código: **NR**, no se muestra LICENSE en raíz inspeccionada; disponibilidad pública no implica permiso amplio de reutilización. Licencia de cada dataset UCR: **NR** en esta inspección, debe verificarse individualmente. Datos BTC: **no evaluados** por la fuente.

Coste derivado, no medición publicada: O(BL Σ_j k_j C_(j−1)C_j) por pasada de convoluciones, activaciones O(BL Σ_j C_j); entrenamiento añade backward y estados del optimizador. Para Conv1D propuesta, pesos convolucionales sin bias/BN/cabeza = 1024C + 262144. Campo receptivo local antes de GAP = 1+(8−1)+(5−1)+(3−1)=14 puntos; GAP agrega toda la ventana, pero no convierte esos filtros en interacciones locales de longitud L. Latencia, RAM/VRAM, FLOPs publicados comparables con E2: **NR**. Ventaja a probar: baseline convolucional simple y paralelizable; límites: dependencia del escalado, deriva y pérdida de posición por GAP.

(e2-lit-resnet)=
## ResNet 1D

### Fuente y evidencia publicada

Misma referencia Wang et al.; §II.C, Fig. 1(c), pp. 2–3: tres bloques residuales y GAP. Tabla I, p. 4: ResNet MPCE 0.0231, 8 victorias, ranking aritmético 4.386; Adiac error 0.174. Comparación interna: FCN tiene menor MPCE, pero no se deduce superioridad universal. Su protocolo es el mismo descrito arriba. Coste de tiempo/hardware por modelo y métricas de forecasting BTC: **NR**.

### Diagrama y fórmula de la adaptación

    X → ResidualBlock(64) → ResidualBlock(128)
      → ResidualBlock(128) → GAP → Dense lineal(H) → ŷ

Cada bloque propuesto: Conv1D(k=8)+BN+ReLU → Conv1D(k=5)+BN+ReLU → Conv1D(k=3)+BN; sumar shortcut y aplicar ReLU. Forma: h_out = ReLU(F(h_in)+P(h_in)); P preserva dimensiones o las proyecta mediante conv 1×1 cuando sea necesario. [Código oficial ResNet.py, función build_resnet](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/ResNet.py) permite resolver el ancho por bloque y la proyección. El texto del paper es más abreviado que la figura/código; documentar exactamente dónde se coloca la última ReLU y si shortcut lleva BN.

Cambios E2 propuestos: Conv1D explícita; cabeza lineal H; pérdida supervisada de regresión; selección por validación temporal y no por test. La conexión residual facilita conservar representaciones, pero no garantiza que más profundidad supere FCN en este objetivo. Control requerido para la propuesta: presupuesto comparable de tuning y dimensionamiento de repeticiones conforme al nuevo plan temporal. No se introduce aquí un mínimo de tres semillas ni se reemplaza la política de potencia registrada; registrar todas las ejecuciones, no sólo la mejor.

### Coste, disponibilidad y límites

Coste derivado: O(BL Σ_9conv k_j C_(j−1)C_j), más las proyecciones 1×1. La suma residual cuesta O(BLC); activaciones y backward de nueve convoluciones incrementan memoria frente a tres capas de ancho comparable, sin asegurar un multiplicador fijo. Campo receptivo máximo de la ruta de nueve conv de stride 1 = 1+3[(8−1)+(5−1)+(3−1)] = 40; rutas shortcut tienen campos menores. Son cálculos de la adaptación Conv1D, no mediciones del script Conv2D. Parámetros exactos dependen de BN, bias y proyecciones: contar sobre el modelo final. Fuente de datos y licencia son las del apartado FCN. Licencia propia separada: **NR**. Riesgos a contrastar: sobreajuste, gasto mayor y sensibilidad a BN con batches pequeños o regímenes distintos.

(e2-lit-inception)=
## InceptionTime adaptado a regresión

### Fuente y evidencia publicada

Ismail Fawaz, H. et al. (2020), *InceptionTime: Finding AlexNet for Time Series Classification*, Data Mining and Knowledge Discovery 34, 1936–1962, DOI [10.1007/s10618-020-00710-y](https://doi.org/10.1007/s10618-020-00710-y); [PDF autor v3](https://arxiv.org/pdf/1909.04939). §3, Figs. 1–2, pp. 5–7: cinco redes, seis módulos por red; kernels 10/20/40, 32 filtros por rama, bottleneck 32, rama max-pool, residuales cada tres módulos, GAP/softmax. Fig. 6, p. 10: en 85 datasets UCR, victoria/empate/derrota contra HIVE-COTE 40/6/39, p>0.5; no superioridad significativa. §4–5, pp. 9–12: particiones originales, z-normalización, Adam; tiempos con una Quadro P6000 y redes secuenciales. Abstract: 8 millones de series en 13 horas. Esto no estima tiempo E2. Accuracy BTC/regresión: **NR**.

### Diagrama y fórmula de la adaptación

    X → [módulo ×3 + residual] → [módulo ×3 + residual]
      → GAP → Dense lineal(H) → ŷ_m
    Cinco inicializaciones independientes → promedio de ŷ_1,…,ŷ_5

Diagrama del módulo, redibujado conceptualmente:

    h → Conv1×1 → {Conv10 | Conv20 | Conv40} ┐
    h → MaxPool3 → Conv1×1                   ├→ Concatenate → BN → ReLU

I(h) = ReLU(BN(concat(C_10(B(h)), C_20(B(h)), C_40(B(h)), C_1(MaxPool(h))))).

ŷ_ensemble = (1/5) Σ_m ŷ_m. Esta última fórmula expresa promedio de pronósticos, **adaptación propia** del promedio de probabilidades original. Si sólo se entrena una red, rotular “Inception network adaptada” o “InceptionTime de un miembro”, no afirmar que reproduce el ensemble publicado. Una cabeza multihorizonte directa no es el modelo clasificador publicado.

[Implementación oficial, inception.py](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py): batch 64, depth 6, 1500 épocas por defecto, kernel_size inicial 41 reducido a 40 y subdividido; bottleneck se omite si entrada tiene un solo canal. Esto importa al reproducir univariado vs multivariado. Verificar padding, sesgos y la rama max-pool contra el archivo, no sólo la ilustración. Es una dependencia antigua, no una garantía de compatibilidad actual.

### Coste, datos, licencia y límites

[Repositorio oficial](https://github.com/hfawaz/InceptionTime) publica GPL-3.0 y resultados CSV para 85/128 datasets. La licencia del código no debe confundirse con la de cada dataset UCR: esta última **NR** en esta inspección. No copiar código GPL sin revisar compatibilidad del proyecto. El artículo, sus figuras y datasets tienen derechos independientes.

Coste derivado por módulo con M canales, bottleneck b y f filtros por rama: convoluciones aproximadamente BL[Mb + bf(10+20+40) + Mf], más pooling, BN y residuales. Cuando no se aplica bottleneck, sustituir b por M en las ramas y eliminar Mb. Con f=32 salen 128 canales. Cinco miembros multiplican aproximadamente por cinco el cómputo secuencial de una arquitectura idéntica; no es una predicción exacta de segundos, y entrenarlos a la vez exige más memoria. Medir aparte latencia de un miembro y ensemble. Riesgos del ensemble original: presupuesto aproximadamente cinco veces mayor, kernels largos respecto a L y promedio que oculte sesgo común. E2 evalúa una red individual; su coste medido no se multiplica por cinco ni se atribuye al ensemble publicado. La {ref}`comparación de artefactos de autor <e2-lit-inception-single>` separa ambas construcciones y sus datasets.

(e2-lit-sae)=
## LSTM-autoencoder con predictor

### Fuente identificada y distinción necesaria

Sagheer, A. y Kotb, M. (2019), *Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems*, Scientific Reports 9, 19038, DOI [10.1038/s41598-019-55320-6](https://doi.org/10.1038/s41598-019-55320-6); [PDF](https://www.nature.com/articles/s41598-019-55320-6.pdf). Es **preentrenamiento autoencoder seguido de ajuste supervisado**, no evidencia directa de entrenamiento multitarea conjunto. Tabla 1, p. 10: alquiler horario de bicicletas, LSTM-SAE de dos capas (5/47 unidades), dropout 0.1, lag 30, batch 73: RMSE 46.927, MAE 30.041, SMAPE 11.646; DLSTM de dos capas: 49.811/31.524/12.183. Son resultados de ese caso, no BTC. El artículo tiene licencia CC BY 4.0; esto no determina licencia del código.

### Diagrama y fórmula de la variante E2 propuesta

    X [B,L,C] → Encoder LSTM → z [B,d] ─→ cabeza lineal/MLP → ŷ [B,H]
                                    └→ Decoder LSTM → X_recon [B,L,C]

Para **entrenamiento conjunto propuesto**:

L_total = L_pred + λ L_recon,
L_recon = (NLC)^−1 Σ_t ||X_t−D(E(X_t))||_F²,
ŷ_t = g(E(X_t)).

λ es hiperparámetro a elegir sólo en validación. La escala relativa de ambos términos debe normalizarse, y λ=0 es una ablación esencial. Alternativa fiel al enfoque de la fuente: fase 1 minimizar reconstrucción, fase 2 inicializar la red predictiva con pesos aprendidos y optimizar supervisión. No llamar equivalentes ambos regímenes. No atribuir la pérdida conjunta a Sagheer/Kotb.

Ecuación recurrente resumida: c_s = f_s⊙c_(s−1)+i_s⊙ĉ_s; h_s=o_s⊙tanh(c_s), con puertas dependientes de x_s,h_(s−1). Describe la mecánica LSTM; encoder, decoder y pooling final deben fijarse explícitamente en E2. Reconstruir ventanas del test durante pretraining sigue siendo acceso al test: limitar toda fase de entrenamiento, incluso sin etiquetas, al fold permitido.

### Código, datos, coste y límites

[Repositorio del autor](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder): scripts para 1/2/3 capas, búsqueda y evaluación; pollution.csv; Keras 2.1.5, tensorflow-gpu 1.10.0, Hyperopt 0.1.1. El [script fuente fijado](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder/blob/8668132ef6f8855dbab289c5a2b2478f871c4c4b/2layer_evaluate.py) y la lectura primaria posterior sustentan la receta actual. Licencia del código: **NR** en la raíz inspeccionada; licencia del CSV: **NV**, sin asumir que hereda CC BY del artículo. El [registro actual de disponibilidad](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv) identifica UCI Beijing PM2.5 y distingue 43.824 observaciones brutas de las 43.800 preparadas que informa la fuente. No se reprodujeron los datos, splits ni experimentos originales del artículo; la reproducción de E2 publicada en §55 tiene otro objeto. Las incógnitas de la inspección inicial no sustituyen la extracción posterior de Tabla 2 y del protocolo.

Coste derivado de una capa LSTM de ancho d y entrada C: 4d(C+d+1) parámetros con una convención de bias único; implementación con dos sesgos cambia el conteo. Operaciones por secuencia O(Ld(C+d)); entrenamiento conjunto suma decoder y cabeza, además de backward. Activaciones O(BLd) por capa como orden orientativo, no pico real de VRAM. La recurrencia impone dependencia secuencial en L, así que igual número de parámetros no implica igual latencia que CNN. Inferencia sólo con encoder+cabeza puede descartar decoder si la implementación lo permite. Riesgos razonados: reconstrucción puede dedicar capacidad a variaciones que no ayudan al objetivo; λ introduce otra decisión; pretraining añade coste y potencial fuga.

## Registro de extracción y límites

- Wang PDF pp. 2–4: arquitectura/protocolo/Tabla I leídos en la inspección inicial. Los scripts oficiales FCN.py y ResNet.py se fijaron después al commit `3ecc8971fa768bd01ada7f3a613688255e6256c2` para verificar los cuerpos adaptados. Esa verificación no reproduce la clasificación UCR original. La inspección de licencia mantiene su alcance propio.
- InceptionTime PDF pp. 5–12: arquitectura, Fig. 6 y hardware inspeccionados. El cuerpo fuente se fijó posteriormente al commit `952d115e83fb3a66d75c5858c702f8dd5eeb18b7`; los enlaces master de la ficha inicial conservan su condición mutable. Los {ref}`resultados de autor de redes individuales y ensemble <e2-lit-inception-single>` son evidencia adicional con otro denominador.
- Sagheer/Kotb: la inspección inicial usó metadatos y Tabla 1 del PDF primario indexado. La lectura posterior del [texto primario PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911101/) examinó además métodos y tablas, incluida Tabla 2 de PM2.5, con localizadores en la extracción actual. El alcance inicial se conserva sin borrar la evidencia posterior; NR/NV no significan ausencia global en el artículo.
- En esta revisión bibliográfica no se entrenó ningún modelo, no se midió consumo real, no se evaluó BTC, no se revisó la licencia específica de fuentes de mercado, no se hizo backtest. Cualquier shortlist es metodológica, no recomendación financiera ni demostración de rentabilidad.

## Recomendación de integración

Los cuatro aportan mecanismos diferenciados, pero FCN, ResNet e Inception son convolucionales y LSTM-AE es recurrente: no acreditan por sí solos cuatro familias. El Top 10 completo añade MLP y atención, con cuatro familias documentadas en [§3](03_literatura.md). La integración ya está ejecutada: las fuentes se vinculan a clases/configuraciones en §44/§48; las seis semillas y los resultados de la cohorte común están en §6; parámetros y tiempos en §51; latencia batch 1 y memoria en §54; FLOPs core en §58. Las tres ablaciones del autoencoder y su inferencia posterior están en [§55](55_reproduccion_ablaciones.md). Estos productos son evidencia propia de E2 y no rellenan costes o resultados no reportados por los artículos. La [lista vigente del punto 3](../_static/e2_literature_current_v1/guide_checklist.csv) separa esos alcances.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/16_fichas_nuevas.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.
