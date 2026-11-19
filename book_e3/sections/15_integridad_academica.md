# 15 Integridad académica

Las reglas de esta sección proceden de la guía, páginas 47–48, y se aplican a afirmaciones, código, datos, diseño experimental y documentación. Describirlas no constituye un veredicto sobre una persona, una sanción ni una calificación del avance. Las [condiciones de evaluación](14_evaluacion_calificacion.md) se conservan como reglas condicionales; R1 y R3 siguen pendientes, al igual que la comprensión y defensa académica exigidas para R2.

## Novedad, transferencia y atribución

Toda afirmación de novedad debe apoyarse en la revisión documentada de la sección 6.3 de la guía: búsquedas reproducibles, trabajos cercanos, clasificación de diferencias, procedencia y comprobación externa. La guía establece que una afirmación falsa o sin evidencia se trata como una falta de integridad y que el modelo se evalúa como si no cumpliera R1. **Mantener R1 pendiente mientras falta evidencia no equivale a declarar una novedad ya demostrada.**

Si el mecanismo procede de otra aplicación, deben citarse con precisión su fuente y contexto, justificar la transferencia y describir una adaptación algorítmica sustantiva. Cambiar únicamente los datos o hiperparámetros no acredita esa adaptación. La guía considera que presentar la transferencia sin atribución o modificación sustantiva no satisface R1 y remite su tratamiento al reglamento estudiantil.

Se permite consultar documentación, artículos y repositorios como inspiración, citándolos. Copiar código sin atribución, descargar/ejecutar una implementación ajena como si fuera el núcleo propio o presentar como propia una implementación de otra persona no satisface R2 según la guía. Usar primitivas del framework es distinto de atribuirse una arquitectura o mecanismo importado; el aporte debe localizarse y explicarse desde su formulación.

La [revisión y procedencia del candidato](06_modelo_original.md) mantienen abiertas las equivalencias con trabajos cercanos. La atención por contenido y los bloques CNN/LSTM son conocidos; la saturación radial tampoco se presenta como una función inventada aquí. La revisión focal, una lista BibTeX o una consulta bloqueada no prueban ausencia de antecedentes. Los registros distinguen lecturas verificadas, evidencia heredada y búsquedas todavía pendientes.

## Autoría, herramientas de IA y comprensión

La guía exige declarar el uso de IA **en el último capítulo del libro**, indicando para qué se utilizó. El [anexo final de reproducibilidad y autoría](cierre_reproducibilidad.md) conserva esa declaración y las contribuciones/confirmaciones por completar. La asistencia incluye formulación y documentación, código candidato, herramientas de auditoría/exportación y pruebas de software, con el alcance que consta en sus registros.

La persona autora es responsable de comprender y defender cada línea entregada, incluida la demostración formal, y de poder modificar el algoritmo durante la sustentación. La existencia de archivos, una prueba aprobada o una revisión de otro agente no certifica esa comprensión ni reemplaza la declaración personal. Las contribuciones y firmas humanas deben ser revisadas y confirmadas por sus titulares; no se generan firmas o avales en su nombre.

El código del candidato declara asistencia de IA y fue desarrollado desde las ecuaciones con primitivas de PyTorch. Esa evidencia de software y su prueba latente no se presentan como autoría académica defendida por el estudiante ni como cumplimiento automático de R2. La revisión y preparación de la [sustentación](13_sustentacion.md) deben conservar esa distinción.

## Disciplina experimental y registro de decisiones

La guía admite reportar de forma honesta un resultado desfavorable dentro de sus reglas de evaluación. Manipular particiones, semillas o pruebas para obtener significancia constituye una falta grave. Los efectos nulos, adversos y no concluyentes deben permanecer con su evidencia y límites; no se transforman en equivalencia por ausencia de rechazo.

Las prohibiciones concretas y su aplicación al registro del proyecto son:

| Regla de integridad de la guía | Registro o condición que debe conservar el proyecto |
|---|---|
| No elegir después de ver resultados los datasets, semillas, métricas, análisis o rivales que favorecen al modelo propio. | Fijar selección, estimandos y familias antes de la evaluación correspondiente; preservar todos los desenlaces y desviaciones como posteriores cuando lo sean. |
| No relajar el estándar de profundidad después de conocer resultados. | Conservar propiedad formal, evaluación adicional, potencia y número mínimo de pruebas/semillas exigidos; un resultado difícil no reduce las condiciones de R3. |
| No excluir rivales tras observar sus resultados. | Mantener la familia candidata de 16 rivales, sin retirar modelos desfavorables o incómodos; la memoria del dashboard no reduce la familia científica. |
| No dar a los benchmarks un presupuesto de ajuste o cómputo menor para favorecer al candidato. | Definir esfuerzo comparable, tiempos, fases, fallos y capacidad; declarar por separado diseño e HPO. El presupuesto E3 sigue pendiente de cierre. |
| No reajustar benchmarks después del congelamiento para alterar la comparación. | Mantener recetas y artefactos identificados. Una corrección real de errores debe conservar fecha, motivo, linaje y recálculo de los productos afectados, sin ocultarse como una receta original inalterada. |
| No reportar únicamente la mejor semilla de varias ejecuciones. | Mostrar la cohorte y dispersión pertinentes. Los cinco ONNX actuales usan la semilla representante 271801, fijada por el menor ID para ingeniería. Sus curvas y pronósticos individuales conservan ese alcance; las tablas actuales de E2 y E3 resumen diez semillas por red. Las tres exportaciones TorchScript y las seis semillas del corte histórico de E2 se conservan con su propia fecha, sin sustituir las medias vigentes por la mejor réplica. |
| No usar el conjunto de prueba para decidir hiperparámetros, umbrales o parada. | Separar desarrollo y confirmación y declarar exposición previa. Los periodos ya evaluados en E1/E2 no recuperan cegamiento al volver a puntuarse. |
| No usar los conjuntos adicionales para diseñar y presentarlos luego como evidencia independiente. | Elegirlos antes de conocer el desempeño; si influyen en decisiones de diseño, registrar ese uso y revisar su papel confirmatorio. D01 está admitido como archivo para la tarea RAW entre sesiones, sin evaluación de modelos; D02/D03 y la campaña común siguen pendientes. |
| No presentar una hipótesis decidida tras observar resultados como confirmatoria. | Conservar el carácter propuesto de D1–D5 y del protocolo en borrador; análisis nuevos sobre resultados conocidos mantienen su fecha y alcance exploratorio. |
| No omitir variantes evaluadas de la bitácora. | Registrar propuestas, parametrizaciones, pruebas, intentos, fallos y descartes. V1/v2 conservan cinco recetas entrenadas y el piloto por momentos completó otras tres. Las diez semillas aditivas repiten una receta; el Nd global sigue pendiente de conciliación y no se reduce al subtotal de ocho recetas. |
| No presentar como válida una demostración que no se sostiene. | Verificar cada supuesto, derivación y comprobación. La cota demostrada cubre la cabeza latente con parámetros fijos; no se atribuye sin prueba al pipeline completo OHLCV ni a mejor desempeño de mercado. |
| No omitir o alterar la fecha del protocolo pre-registrado. | Registrar fecha y tag Git reales anteriores a los experimentos definitivos; conservar versiones y desviaciones. El borrador actual no se retrofecha ni se presenta como preregistro congelado. |

La [sección 7](07_diseno_riguroso.md) define la disciplina de diseño y la [sección 8](08_verificacion_estadistica.md) publica los 48 contrastes posteriores con Holm global y decisión conjunta, conservando el cierre confirmatorio pendiente. No se elegirá la prueba más favorable ni se contará una hipótesis distinta como otra confirmación del mismo margen de RMSE sin justificar su estimando y alcance. Los análisis retrospectivos se conservan con su fecha real, estimando, motivación y exposición previa; una sensibilidad posterior no se presenta como decisión preregistrada.

Cuando se constata selección a posteriori de semillas, conjuntos, métricas, análisis o rivales para favorecer la conclusión, la guía la considera una falta grave y anula la condición R3. Esa regla es condicional: su inclusión aquí no atribuye una infracción a la persona autora. Los resultados adversos declarados con honestidad conservan el tope previsto por la evaluación; no justifican manipular el diseño para obtener rechazo.

## Aplicación al avance documentado

El estado real se apoya en fuentes y registros verificables, sin completar casillas por inferencia:

- Las [correcciones de E2](04_correcciones_e2.md) son de revisión interna y auditorías documentadas. El registro actual no incorpora observaciones particulares del profesor ni atribuye al docente una observación o aprobación sin evidencia.
- La [procedencia inicial del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/procedencia.md) y su [bitácora](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) separaban programación e intervenciones artificiales de la evaluación de mercado entonces pendiente. Los pilotos completados, las diez réplicas y el análisis posterior se actualizan en los capítulos 6–8, sin alterar ese corte inicial.
- El [protocolo canónico en borrador](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/protocolo.md) identifica decisiones sin cerrar y conserva el [borrador anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_d/protocolo_borrador.md) como antecedente; ninguno se registra como una campaña confirmatoria ejecutada ni se retrofecha.
- Las [tres TorchScript históricas y cinco ONNX actuales](05_configuraciones_exportacion.md), incluido TSMixer, conservan cohortes, tolerancias y límites distintos: 53 fechas para el recibo TorchScript y 608 para la paridad ONNX de una semilla por modelo. El perfil posterior E3 adopta el q existente de 271801, con cobertura individual de 94,08 % global y 85,56 % en 2026Q1, sólo para objetivos posteriores al 30 de diciembre de 2024. No atribuye al intervalo individual las medias de diez semillas ni promete cobertura futura. Los doce sistemas precalculados son nueve redes pendientes de exportación verificada y tres clásicos justificados por necesitar más historia o estado que siete barras. El perfil histórico de cuatro ONNX conserva sus archivos y fecha.
- La [declaración del último anexo](cierre_reproducibilidad.md) mantiene el uso de IA, la responsabilidad personal y las firmas/defensa pendientes, sin certificaciones simuladas.

Ante una nueva evidencia, corrección o desviación se deberá actualizar el registro correspondiente y la interpretación que dependa de ella. La consecuencia académica de una eventual infracción corresponde al procedimiento institucional; este capítulo no imputa una infracción ni asigna nota o tope al avance actual.

La revisión posterior conserva también la evidencia adversa: el control sólo piso mejora la MAE de E3, la proyección común reduce su ventaja RMSE y el intervalo posterior tiene subcobertura en 2026Q1. La búsqueda focal y el reanálisis en contenedor recibieron asistencia de IA. Sus recibos no son una certificación de novedad ni un acta humana; el [anexo final](cierre_reproducibilidad.md) mantiene la responsabilidad y confirmación personal pendientes.

## Declaración de IA para la revisión externa de R1

Se utilizó un asistente de IA de OpenAI, en este entorno de ChatGPT/Codex, para búsqueda y verificación bibliográfica, cribado de títulos, lectura focal de fuentes, comparación de fórmulas, redacción del dictamen y repetición de dos cadenas de búsqueda. La revisión técnica adicional se encargó con contexto separado de los dictámenes previos; ese procedimiento no equivale a una persona independiente ni garantiza ausencia de sesgos compartidos.

El autor comunicó el 18 de noviembre de 2026 que el profesor permite sustituir al revisor humano por IA si se menciona su uso. Esa modalidad se aplica al componente de revisión externa de R1 en [6.13](06_modelo_original.md), con dictamen, dos cadenas repetidas y {download}`declaración específica de uso <../_static/e3_r1_ai_substitution_20261118/ai_use_declaration.md>`. El componente se documenta como completado por esa vía; no se inventa una revisión humana o una firma.

La persona autora mantiene la responsabilidad de comprobar las fuentes, comprender y defender el trabajo y comunicar honestamente sus límites. El dictamen conserva su conclusión: la diferencia sustantiva todavía no está establecida y R1 global no queda acreditada por la modalidad de IA.
