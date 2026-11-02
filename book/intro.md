# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Secciones 1–4 · 2 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, utilizando precios de cierre obtenidos directamente de **Binance Spot (BTCUSDT)**. La estructura, el EDA, la revisión de literatura y el protocolo experimental se rigen por la guía del Entregable 1.

```{important}
Este es un borrador metodológico, no un protocolo congelado ni un experimento terminado. La fuente seleccionada es Binance Spot BTCUSDT. Se descargaron y verificaron los archivos oficiales del 01/01/2023 al 30/10/2026 inclusive UTC. El EDA de calidad, precios y retornos de desarrollo está ejecutado y documentado en la sección 2; se analizaron cuatro ventanas de volatilidad diaria como candidatas; la configuración final se fijará antes del benchmark. La adquisición verificada no equivale a un experimento terminado. No se han entrenado modelos de Deep Learning. Las decisiones no confirmadas están identificadas como **pendientes** o **propuestas**.
```

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación matemática, hipótesis propuesta, métricas y ficha técnica.
- [Análisis exploratorio de datos](sections/02_eda.md): adquisición directa desde Binance Spot BTCUSDT y análisis de calidad, precios y retornos de desarrollo.
- [Estado del arte](sections/03_estado_arte.md): estrategia de búsqueda y procedimiento para seleccionar tres modelos base; revisión y selección pendientes de ejecución.
- [Diseño experimental y piloto](sections/04_implementacion_evaluacion.md): pipeline, repeticiones, potencia y preregistro; entrenamiento y piloto pendientes.
- [Decisiones pendientes](sections/protocolo_pendiente.md): requisitos para cerrar la sección sin inventar información.
- [Referencias y materiales](sections/referencias.md): guía del curso y antecedente científico.

## Alcance del proyecto

El Entregable 1 completo exige EDA reproducible, revisión de literatura, selección justificada de **tres modelos simples de Deep Learning**, piloto y evaluación estadística. El análisis de calidad, precios y retornos de desarrollo está ejecutado y documentado en la sección 2. La definición final del objetivo, la revisión y selección de modelos, el piloto, el entrenamiento y la evaluación estadística permanecen pendientes. MLP, CNN1D y RNN simple son candidatos provisionales, no modelos seleccionados. Las familias observadas en clase son MLP, CNN, RNN, autoencoders y Transformer.

Solo se considera Bitcoin. Otras criptomonedas son una posible ampliación futura. Los documentos completos de los entregables 2 y 3 aún no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones están en `BUILD.md` en la raíz del repositorio. El libro usa Jupyter Book 1 con dependencias fijadas; se genera HTML localmente sin ejecutar notebooks. El repositorio fuente es público. El sitio publica únicamente el HTML generado en `book/_build/html`, desde la rama `gh-pages`. Los documentos originales del curso no se incorporan al HTML, aunque están disponibles en el repositorio fuente.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
