# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Sección 1 · 2 de noviembre de 2026

Este libro documenta la definición inicial de un problema de pronóstico de volatilidad de **Bitcoin**, calculada a partir de precios de cierre. El alcance actual es desarrollar la **Sección 1: Definición del Problema y Dataset**.

```{important}
Este es un borrador metodológico, no un protocolo congelado ni un experimento terminado. La fuente seleccionada es Binance Spot BTCUSDT. La adquisición directa verificada y la ejecución del EDA están pendientes. No se han entrenado modelos de Deep Learning. Las decisiones no confirmadas están identificadas como **pendientes** o **propuestas**.
```

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación matemática, hipótesis propuesta, métricas y ficha técnica.
- [Análisis exploratorio de datos](sections/02_eda.md): protocolo de EDA y adquisición directa desde Binance Spot BTCUSDT.
- [Decisiones pendientes](sections/protocolo_pendiente.md): requisitos para cerrar la sección sin inventar información.
- [Referencias y materiales](sections/referencias.md): guía del curso y antecedente científico.

## Alcance del proyecto

El Entregable 1 completo exigirá EDA reproducible, revisión de literatura, selección justificada de **tres modelos simples de Deep Learning**, piloto y evaluación estadística. Esas etapas **no forman parte del trabajo ejecutado en este avance**. MLP, CNN1D y RNN simple son candidatos provisionales, no modelos seleccionados. Las familias observadas en clase son MLP, CNN, RNN, autoencoders y Transformer.

Solo se considera Bitcoin. Otras criptomonedas son una posible ampliación futura. Los documentos completos de los entregables 2 y 3 aún no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones están en `BUILD.md` en la raíz del repositorio. El libro usa Jupyter Book 1 con dependencias fijadas; se genera HTML localmente sin ejecutar notebooks. El repositorio fuente es público. El sitio publica únicamente el HTML generado en `book/_build/html`, desde la rama `gh-pages`. Los documentos originales del curso no se incorporan al HTML, aunque están disponibles en el repositorio fuente.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
