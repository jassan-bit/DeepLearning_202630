# Proyecto Final de Deep Learning: Entregable 2

**Corte del 11/11/2026, 06:55:42 de Bogotá (11:55:42 UTC):** la campaña CUDA conserva **1.748 archivos de resultados**, 187 más que el corte histórico de 1.561. El último ajuste confirmado corresponde a **TimesNet, NAS-1, origen `2024-01-01`**: terminó a las **06:55:41**, con estado `completed`, **736 pasos de optimización** y 46 épocas ejecutadas (mejor época: 26). El siguiente trabajo quedó iniciado a las **06:55:42**. Son registros fechados de entrenamiento real; no miden actividad del kernel después del corte ni completan el benchmark final. [Avance y evidencias de §36](sections/36_avance_campana_cuda.md).

**Reanudación y presupuesto:** el diario registra una ampliación operativa de **36 a 48 horas acumuladas**, el 11/11 a las **06:40:17 de Bogotá**, y una sesión nueva a las **06:42:53**, con límite de **8 horas**. El `status.json` conserva la pausa anterior por presupuesto, con 1.744 resultados; los cuatro resultados posteriores y el nuevo inicio acreditan continuación. Documentar la ampliación existente no cambia el presupuesto ni reinicia Colab. Las horas son límites contables, no consumo GPU medido o facturado.

**Alcance científico:** continúa la selección HPO/NAS de desarrollo. FCN-1D, ResNet-1D, Inception-1D y LSTM-autoencoder tienen selección registrada para el primer origen; TimesNet sigue en NAS-1. No hay comparación integral terminada, ganador final ni R definitivo. La recuperación R3 y los cortes anteriores se conservan en [§35](sections/35_busqueda_cuda_reanudable.md) y [§36](sections/36_avance_campana_cuda.md).

**Perfiles históricos de recursos GPU:** [cohorte grande y TimesNet máximo](sections/34_perfil_gpu_ampliado.md). El segundo informe del usuario registra 15/15 perfiles completos, cero fallos y 680 pasos de optimización en Tesla T4. El TimesNet de 225 millones de parámetros completó dos épocas con un máximo de 5,080 GiB reservados por PyTorch. Estos perfiles miden recursos de una receta fijada; no son resultados de selección HPO/NAS. El cómputo necesario para completar la búsqueda y el R final siguen abiertos.

**Primer piloto GPU histórico:** [catorce referencias y sus recursos](sections/33_piloto_gpu.md). El primer informe registra 14/14 ajustes completos, cero fallos y 192 pasos sobre la cohorte pequeña. Ambos informes proceden de ejecuciones del usuario en Colab; Codex revisó sus importaciones. Los dos cortes históricos reúnen **29 perfiles y 872 pasos conocidos**, sin resultados comparativos de calidad predictiva; no se suman a los intentos de la campaña de §36.

**Infraestructura CUDA reproducible:** [backend GPU y notebook](sections/32_ejecucion_gpu.md). Fuente fijada por commit, Python 3.12.12 aislado del kernel de Colab, datos verificados y preparación de fuentes reutilizable. El corte de pruebas CPU de esa sección se conserva separado del piloto GPU posterior.

**Recursos y dimensionamiento en Codex Cloud:** [optimización de CPU y presupuesto condicionado](sections/31_optimizacion_nube.md). Completaron 85 de 86 perfiles ampliados y los treinta ajustes del nuevo piloto. Se eligieron el recorte de TimesNet y dos hilos. La nula del generador pasó su validación, pero ningún R estudiado respalda el objetivo de 0,80. La proyección primaria y el R final siguen abiertos; las horas con microperfil son una sensibilidad separada.

**Piloto real en Codex Cloud:** [recursos, variación entre semillas y presupuesto condicionado](sections/30_piloto_nube.md). Ajustes sobre desarrollo 2023–2024 en la CPU del servidor remoto; sin búsqueda HPO/NAS completa ni evaluación de las 608 fechas externas.

**Ciclo de entrenamiento integrado:** [18 sistemas, selección temporal, reajustes y perfilado CPU](sections/29_entrenamiento_temporal.md). Validación artificial con actualizaciones AdamW; sin campaña de mercado ni resultados nuevos de Bitcoin.

**Fábricas NAS verificadas:** [seis familias, 6.000 casos numéricos y fases auxiliares SAE](sections/28_fabricas_nas.md). Enlace de configuraciones HPO y pruebas sintéticas; sin optimización ni mercado.

**Diseño NAS/HPO para Codex:** [decisiones por receta, rangos y presupuesto contable](sections/27_nas_hpo_presupuesto.md). Seis variantes NAS y cinco cuerpos publicados fijos; planes generados sin entrenamiento. Perfiles de referencia publicados en §30; la medición ampliada y su proyección condicionada se incorporan en [§31](sections/31_optimizacion_nube.md). La duración total de la campaña sigue abierta.

**Registro por modelo:** [fuentes, adaptaciones, NAS, HPO, parámetros e integración](sections/26_registro_recetas.md). Once recetas identificadas y siete referencias separadas; decisiones experimentales aún abiertas.

**Preparación del benchmark:** [matriz de requisitos, decisiones y evidencia de cierre](sections/25_matriz_cierre.md). Orden de trabajo conforme a la guía; distingue avances parciales y pendientes antes de entrenar.

**Ejecutor común probado:** [18 sistemas, registro previo a cada observación y auditoría de disponibilidad](sections/24_ejecutor_comun.md). Pruebas sintéticas únicamente; las 608 fechas finales siguen sin verificar.

**Implementación de las fichas:** [cuerpos completos, dos fases SAE y controles temporales](sections/23_implementacion_recetas.md). Once modelos comprobados con datos artificiales; sin entrenamiento de mercado ni decisión NAS.

**Once propuestas concretas:** [fichas de cuerpos, adaptaciones y decisiones pendientes](sections/22_fichas_recetas.md). Ese corte conserva las propuestas originales; su implementación posterior se documenta en §23.


**Siguiente especificación:** [comparadores temporales, matriz NAS y costes por escenarios](sections/21_comparadores_nas.md). Propuesta sin entrenamiento, búsqueda ni cierre automático del protocolo.


**Guía original cotejada:** [correcciones y pendientes por página](sections/20_contraste_guia.md). La cuadrícula reducida no cubre el espacio mínimo; controles F2a, NAS condicional y batería estadística requieren completar el plan antes de entrenar.


**Actualización de planificación:** [§19: auditoría real de desarrollo y plan propuesto](./sections/19_plan_auditoria.md). Sin entrenamiento de mercado, HPO ni preregistro nuevo.

**Deep Learning · 2026-30 · Jassan Arteta**

## Catálogo vigente

El catálogo distingue los siguientes diez modelos base, una propuesta adicional y tres controles neuronales de E1:

1. **FCN-1D**.
2. **ResNet-1D**.
3. **Inception-1D**.
4. **LSTM-autoencoder con predictor**.
5. **TimesNet**.
6. **TSMixer**.
7. **TiDE**.
8. **TimeMixer**.
9. **PatchTST**.
10. **iTransformer**.

**Candidato 11: CNN-1D + LSTM + atención temporal.** Su superioridad no está demostrada. Es una hipótesis que debe compararse con controles y puede arrojar resultados negativos.

El catálogo sustituido se retira de la presentación. Estos nombres provienen de la literatura y de adaptaciones a las familias del curso; no son una transcripción literal de sus modelos. **MLP16, LSTM32 y GRU32 de E1** son controles separados. La campaña CUDA incluye esos tres controles junto con los diez modelos base y la propuesta: **catorce sistemas neuronales**. Los cuatro comparadores restantes del benchmark de dieciocho sistemas conservan su tratamiento separado. [Especificación y límites del catálogo](sections/15_revision_modelos.md).

## Qué está implementado y qué está pendiente

- **Implementación provisional disponible:** los once candidatos comparten fábrica y flujo sintético; 11/11 ejecutados sin omisiones, dos pasos supervisados cada uno y un paso de reconstrucción para el autoencoder. El híbrido ya tiene código y pruebas propias. El piloto real posterior se documenta en [§30](sections/30_piloto_nube.md), la medición de CPU en [§31](sections/31_optimizacion_nube.md) y los perfiles GPU en [§33](sections/33_piloto_gpu.md) y [§34](sections/34_perfil_gpu_ampliado.md); el benchmark completo sigue pendiente. [Recetas y evidencia](sections/18_integracion_sintetica.md).
- **Benchmark vigente:** la selección científica CUDA está en curso, con [cobertura parcial verificable en §36](sections/36_avance_campana_cuda.md). Todavía no hay benchmark integral ni ganador de la comparación final. Los trabajos de desarrollo de §§30–31 y los pilotos de recursos de §§33–34 conservan registros propios anteriores a sus ajustes. La [preparación de §35](sections/35_busqueda_cuda_reanudable.md) y los resultados de la campaña actual son evidencias distintas. Quedan completar la selección registrada, dimensionar las recetas seleccionadas, fijar R y realizar los reajustes, controles e inferencia finales. [Plan experimental](sections/06_benchmark_final.md).
- **Evidencia histórica real:** el piloto previo completó 220 trayectorias y 440 ajustes. El recibo del 06/11/2026 acredita HPO previo iniciado e incompleto, no ejecución actual. La réplica TimeMixer/ETTh1 es externa a Bitcoin. Estos recibos no se atribuyen al catálogo vigente. [Trazabilidad](sections/09_protocolo_registrado.md).

El contrato vigente se sincroniza con E1: D7×5 derivado de OHLCV mediante `relative_window`, σ₇ poblacional sin anualización, horizonte de un día y referencia histórica principal de 608 fechas. [Contrato completo y diferencias con los recibos anteriores](sections/17_contrato_sincronizado.md). El piloto de §30 reinicializa sus modelos; los resultados históricos no se trasladan al híbrido.

## Evidencia bibliográfica

**Ampliación bibliográfica del 08/11/2026:** la [sección 3](sections/03_literatura.md) incorpora una ronda nueva de WoS, Scopus e IEEE (**1.634 ocurrencias → 1.078 grupos**) y cuatro consultas arXiv (**1.199 apariciones → 1.004 IDs**). Los universos se mantienen separados hasta conciliarlos. Hay triaje provisional, tres lecturas focales y fuentes para las nuevas propuestas; no se declara elegibilidad global cerrada ni se modifican resultados experimentales.

**Avance de revisión bibliográfica:** los 59 prioritarios ya tienen lectura individual asistida por IA: 40 candidatos a texto completo, 17 inciertos y dos secundarios. La lectura de exclusiones propuestas alcanza 337/337: las 295 posteriores aportan 184 fuera del núcleo, 98 secundarios, 12 inciertos y un candidato directo. De los 40 candidatos iniciales, se examinaron 22: ocho textos completos (seis editoriales y dos preprints con equivalencia editorial pendiente) y catorce accesos parciales/no recuperados; 18 siguen sin búsqueda individual. No son inclusiones finales ni doble revisión humana. [Conteos y límites](sections/03_literatura.md).

## Ruta de lectura

- [Problema, datos e hipótesis](sections/01_punto_partida.md), [EDA](sections/02_eda.md) y [literatura](sections/03_literatura.md).
- [Implementación](sections/04_implementacion.md), [réplica externa](sections/05_paper_guia.md) y [benchmark pendiente](sections/06_benchmark_final.md).
- [Ruta temporal](sections/07_ruta_temporal.md), [reproducibilidad](sections/08_reproducibilidad.md) y [registro histórico](sections/09_protocolo_registrado.md).
- [Auditoría de datos](sections/10_auditoria_datos.md), [HPO](sections/11_hpo_equidad.md), [incertidumbre](sections/12_incertidumbre_errores.md), [inferencia](sections/13_agregacion_estadistica.md) y [arquitecturas](sections/14_arquitecturas.md).
- [Diez modelos y candidato 11](sections/15_revision_modelos.md), [fichas de las cuatro nuevas arquitecturas](sections/16_fichas_nuevas.md).
- [Preparación histórica del ejecutor CUDA](sections/35_busqueda_cuda_reanudable.md) y [avance parcial de la campaña del 10/11/2026](sections/36_avance_campana_cuda.md).

Los protocolos, muestras, scripts y recibos anteriores se preservan como evidencia inmutable. No se reetiquetan como entrenamientos del diseño actual ni se declara cerrada la revisión bibliográfica.
