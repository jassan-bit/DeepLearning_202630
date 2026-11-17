# Declaración operacional del estándar de E3

**Fecha real: 17 de noviembre de 2026. Estado: estándar y pendientes documentados para una futura campaña; tag de registro y revisión humana pendientes.**

Este documento se elabora después de los pilotos, los diez ajustes de la receta E3, la ampliación de E2 y el análisis estadístico posterior del calendario conocido. No es un preregistro de esos resultados, no se retrofecha y no representa una firma, aval humano o autorización de presupuesto.

| Campo | Compromiso y estado |
|---|---|
| Estándar único | Profundidad teórica: propiedad formal pertinente de la contribución final, con supuestos explícitos y verificación numérica; evaluación en al menos tres conjuntos adicionales de naturaleza distinta, elegidos antes de sus resultados; familia completa con Holm; tres pruebas distintas apropiadas por contraste; potencia por contraste ≥0,90 y potencia conjunta reportada; al menos diez semillas o más si lo exige el dimensionamiento. |
| Justificación de viabilidad | Las ejecuciones terminadas aportan tiempos reales: ampliación E2 5130,970719594005 s y consumo acumulado cerrado de E3 948,439625042 s, que incluye etapas previas. Esos tiempos no garantizan el coste de una campaña nueva ni asignan presupuesto. Deben dimensionarse recursos, fallos, tres conjuntos y repeticiones suficientes antes de confirmar viabilidad. |
| Justificación de semillas | Diez semillas observadas no son un cálculo de potencia. Se requiere piloto admisible, alternativa frente al margen, incertidumbre temporal y de inicialización, control familiar, MDE y potencia de la regla completa antes de fijar R. |
| Rivales conservados | FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST, iTransformer, MLP16, LSTM32, GRU32, Persistence, SeasonalNaive y ARIMA. No se retiran por su resultado. Los clásicos deterministas conservan una trayectoria. |
| Datos y propiedad | Tres conjuntos adicionales, sus fuentes/licencias/versiones/hashes y criterios de selección: pendientes. Propiedad pertinente de la receta/contribución final: por justificar; las pruebas de mecanismos anteriores no se heredan automáticamente. |
| Equidad y protocolo | Deben fijarse particiones y exposición previa, preprocesamiento, semillas, presupuesto, capacidad, pruebas y supuestos, familias, margen por escala, reglas de fallo y decisión. La comparación histórica E2=655 ejemplos de ajuste frente a E3=532 no se renombra como tratamiento idéntico. |
| Fecha, commit y tag | La fecha y el commit corresponden a la publicación real de este archivo. El tag de registro para una futura campaña sigue pendiente y deberá existir antes de congelar su protocolo; no se identifica un tag inexistente o retroactivo. |
| Responsabilidad | Elaborado con asistencia de IA y conforme al estándar aportado por el usuario. La persona autora deberá revisar, comprender, ratificar y defender el contenido; no se simula su firma ni una revisión independiente. |

El estándar no se reduce después de observar resultados. Si la potencia exige más semillas, se aumenta R antes del congelamiento; no se reduce la familia, las pruebas o los conjuntos para favorecer al candidato. Los resultados adversos y las desviaciones se documentan.

Fuentes: sección 2 y plantilla A.1 de la guía de E3; capturas aportadas por el usuario. Evidencia histórica: plan posterior 4a21b0329b99e864ec64fc2df49260aa874141bf, ejecución numérica 0213266404a0aa3af985d7306dce6b13892c71ec y capítulo 8 del informe.
