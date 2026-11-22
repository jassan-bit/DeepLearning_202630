# Expediente evaluable: input-only-overlap-mse-1

Esta entrega conserva la candidata y su decisión técnica previa **CONTINUAR / SUSTENTADO_ACOTADO**. Reúne la argumentación ya examinada, la vincula a sus fuentes exactas y añade una demostración ejecutada de modificación numérica equivalente. No convierte esa revisión en aceptación docente ni en defensa del estudiante. No modifica calificaciones.

## 1. Problema y contribución defendida

La tarea es pronosticar al siguiente origen la desviación poblacional de siete retornos logarítmicos diarios, sin anualizar. Seis retornos de la etiqueta ya son observables en el origen. Un regresor directo de esa etiqueta puede ignorar su parte conocida y generar valores incompatibles con ella. El mecanismo aborda esta limitación algebraica verificable: separa la parte observada y parametriza una magnitud futura no negativa, antes de calcular el pronóstico.

La contribución defendida es **el algoritmo completo de ley atómica de la magnitud faltante, transformada por la geometría observada antes de la esperanza, con condicionamiento por forma normalizada causalmente al origen**. El núcleo procede de V3 propio y se atribuye. Frente al control interno `global_mse`, el incremento es sólo preprocesamiento: ese incremento aislado no acredita R1. Normalizar OHLCV, retirar CRPS, usar tres átomos, cambiar semillas o cambiar de activo tampoco constituyen por sí solos la contribución.

La motivación experimental se conserva con su alcance: V3 tuvo 30 ajustes y `atom_crps` fue 2,7043 % peor en RMSE que `atom_mse_only`; V4 y el V5 completo conservan sus abandonos. Los 796 trabajos pertenecen a la aditiva. La señal de `input_only_mse` fue seleccionada después de examinar 55 fechas expuestas, con cinco semillas; no explica causalmente esos resultados adversos ni confirma superioridad. Las mediciones de gradientes no aislaron una causa única. Se defiende un sesgo estructural plausible, no una solución ya demostrada al cambio de régimen.

La cuantificación descriptiva preservada es RMSE medio0,002051560824 para input-only y0,002109780697 para `global_mse`: diferencia −0,0000582198723 (2,7595 %), cuatro victorias de cinco y DE entre semillas de la candidata0,000072426935. Es una comparación interna postselección. No se elige ahora otro umbral a partir de esa ganancia. El soporte exacto elimina violaciones del suelo para las entradas admitidas; no demuestra que todos los rivales lo violasen ni que respetarlo cause mejor error. El recibo `analysis/AUDIT_SUMMARY.json` y los intervalos retrospectivos previos permiten comprobar esta separación entre limitación algebraica y señal de desarrollo.

## 2. Antecedentes reutilizados y diferencia exacta

Se reutiliza la revisión focal específica del 21 de noviembre de 2026 y el corpus acumulado. No se ejecutaron búsquedas nuevas en esta fase. Los nueve antecedentes externos y el antecedente interno se comparan en [la matriz previa](../../e3_input_only_candidate_20261121/docs/R1_THEORY_LITERATURE.md); sus recibos y límites quedan fijados en el manifiesto de esta entrega.

| Fuente examinada | Lo conservado o conocido | Diferencia y límite de la candidata |
|---|---|---|
| DAIN, Passalis et al. 2020, DOI 10.1109/TNNLS.2019.2944933, §II | Normalización financiera mediante desplazamiento, escala y compuerta | RMS analítico por ventana, sin capa DAIN. La diferencia sustantiva se argumenta por la lectura de U/A, no por normalizar finanzas. |
| RevIN, Kim et al. 2022, ICLR, §3.1 | Normalización por instancia y reversión afín | No hay reversión de escala local a la salida. No se presenta como RevIN nuevo ni se hereda su equivariancia. |
| Non-stationary Transformers, Liu et al. 2022, arXiv:2205.14415v4, §§3.1–3.2 | Información de escala que puede perderse al normalizar | No implementa su atención. Constituye una objeción: forzar iguales átomos/masas para igual forma puede perder información de régimen. |
| CryptoL, Taheri et al. 2026, arXiv:2609.11206v1, §§3.1–3.4 | OHLC cripto normalizado, MSE normalizada y restricciones de velas | Predice sigma7 en unidades físicas con soporte de etiqueta y ley de U. Las secciones examinadas no describen esta misma lectura; no se afirma ausencia mundial. |
| Schittenkopf et al. 1998, *Identifying Stochastic Processes with Mixture Density Networks*, WP11, §2 ecs.1–9 | Mezclas financieras condicionales | Ley positiva de U y esperanza después de T_A, supervisada por MSE puntual; no NLL gaussiana. Se mantiene la corrección bibliográfica: WP11 no es el distinto WP15 sobre predicción de volatilidad. |
| De Stefani et al. 2017, MIDAS, §§2–3, ecs.17–22 | Regresión directa de proxies de desviación móvil | Separación de parte observada y magnitud faltante. Cambiar la ventana o el ticker no basta. |
| FQF, Yang et al. 2019, NeurIPS32 | Posiciones/masas aprendidas | No usa Bellman ni actualizaciones de cuantiles; no hereda garantías de aprendizaje por refuerzo. |
| Gneiting y Raftery 2007, DOI 10.1198/016214506000001437 | CRPS conocido y reglas propias | La candidata es MSE-only. No reclama pérdida novedosa ni identificación/calibración de su ley latente. |
| HARNet, Reisenhofer et al. 2022, arXiv:2205.07719v1, §§2,4.1–4.3 | CNN causal con sesgo HAR para volatilidad | Se discute como guía/trabajo cercano; no incorpora filtros HAR ni hereda rendimiento. La lectura por identidad de etiqueta es la diferencia defendida. |
| V3 interno | Átomos positivos, masas y transformación antes de promediar | Es procedencia propia explícita, no una invención nueva fechada después del piloto ni una aceptación transferida automáticamente. |

La revisión acumulada documenta tres bases, seis cadenas bilingües, cinco trabajos semilla para seguimiento por citación y repositorios. La actualización focal específica tuvo siete consultas web; no se renombra como tres consultas nativas nuevas. El concepto de revisión IA separado repitió dos cadenas, en la modalidad de IA autorizada que conserva el expediente; no representa firma humana, aprobación institucional ni defensa. La conclusión es acotada al material realmente examinado: no se identificó allí un equivalente completo para el mismo fenómeno y tarea. No se prueba prioridad universal.

## 3. Formulación que corresponde al código

Sean r1,…,r6 los seis retornos disponibles y r7 el retorno aún no observado de la próxima etiqueta. Con m6=(Σr_i)/6:

    A = Σ(i=1..6)(r_i−m6)^2 / 7,
    U = |r7−m6|,  c = 6/49,
    Y = T_A(U) = sqrt(A+cU²).

Esta identidad poblacional de actualización de varianza es conocida. La propuesta no la presenta como un teorema inventado. Para una ventana general de n retornos, A=S_(n−1)/n y c=(n−1)/n²; esta generalización algebraica muestra un mecanismo transferible, sin afirmar que aquí se implementó otra longitud o tarea.

Para cada una de las siete filas OHLCV ya cerradas se calculan las coordenadas de logprecio relativas al último cierre conocido. Se dividen por s=max(q,10^−6 r_fit), donde q es la RMS de los seis retornos conocidos y r_fit la RMS congelada del ajuste. El volumen usa log(1+V). La media y desviación poblacional por canal proceden exclusivamente del ajuste y permanecen congeladas. La entrada X tiene forma B×7×5.

El encoder propio usa convolución causal 5→8, kernel3, tanh y LSTM unidireccional 8→12. En sus estados h_j:

    q_att = q_raw/sqrt(1+||q_raw||²), tau=.25+softplus(tau_raw),
    alpha_j = softmax_j(h_j·q_att/tau), z=Σ_j alpha_j h_j.

No se reutilizan la penalización ni la saturación de valores de la antigua cabeza aditiva; del objeto previo sólo se toman el encoder y las inicializaciones de query/temperatura. La ley latente es:

    d_k = 10^−4 + softplus(w_k·z+b_k),
    u_k = mean_fit(U) Σ(j<=k) d_j, k=1,2,3,
    p = softmax(a_1·z+g_1, a_2·z+g_2, 0),
    mu = Σ_k p_k T_A(u_k),
    loss = mean_i[((mu_i−Y_i)/mean_fit(Y))²].

El tercer logit fijo elimina un grado de traslación redundante; no cambia las probabilidades representables en el interior del simplex. Los átomos y la lectura operan en FP64 y el encoder en FP32. Hay 1262 parámetros entrenables. La receta no multiplica los átomos por la escala local; el argumento `scale` se valida por compatibilidad pero no participa en esta salida. La pérdida CRPS tiene peso cero. El entrenamiento congelado conserva AdamW, lr=.003, batch32, máximo40 épocas, paciencia8 y selección por RMSE de selección con primer empate; esta fase no ejecuta ese entrenamiento.

La ejecución usa `hypot(sqrt(A),sqrt(c)*u_k)` y una guardia de suelo para redondeo, en vez de elevar u_k al cuadrado en FP64. Las dos expresiones son iguales en aritmética real. Las diferencias de estabilidad no constituyen una nueva receta ni un nuevo diseño.

## 4. Propiedades demostradas y condiciones

**Identidad de la etiqueta.** Si S6=Σ(r_i−m6)² y m7=(6m6+r7)/7, la suma centrada de siete retornos es S7=S6+(6/7)(r7−m6)². Dividir por7 produce Y²=A+(6/49)U². Supone retornos alineados, sin anualización y desviación poblacional; no se traslada sin cambios a otras etiquetas.

**Soporte y media de la ley implementada.** d_k>0 implica 0<u1<u2<u3 y softmax produce p_k>0, Σp=1 en aritmética real. Cada T_A(u_k)≥sqrt(A); por convexidad de la combinación, mu≥sqrt(A). Para G=Σp_k δ_(u_k), mu=E_G[T_A(U)] exactamente. No se afirma que G sea la ley verdadera de mercado ni que mu sea su esperanza condicional verdadera. La positividad estricta puede redondearse; el código protege el suelo.

**Invariancia de representación, con hipótesis explícitas.** Bajo log P'=b+k log P, k>0, volumen fijo, estadísticas y parámetros congelados, y suelo de escala inactivo antes y después: q'=kq y las diferencias logarítmicas se multiplican por k; por tanto X'=X y los p/u son iguales. A'=k²A. Si el suelo se activa, esta invariancia no está garantizada. Tampoco es una invariancia conjunta a cambiar el volumen o volver a ajustar el preprocesador.

**Pérdida de equivariancia de salida.** Como u no se reescala, mu'=Σp sqrt(k²A+c u²). Para k>1, los átomos positivos satisfacen sqrt(k²A+c u²)<k sqrt(A+c u²), luego mu'<k mu. Para 0<k<1 se invierte la desigualdad. Con A=0, mu'=mu. Se preserva esta limitación explícita de input-only: no hereda la equivariancia del V5 completo.

**Causalidad en el origen.** Con siete barras cerradas, alineación correcta y ajuste congelado, X/A no requieren r7 ni etiquetas futuras. El encoder es causal respecto de X. Sin embargo, la normalización de una fila temprana de la ventana usa el último cierre y la RMS de toda la ventana conocida: no se declara causalidad por prefijo del mapa completo ni validez para predecir antes del cierre de la barra final. Una prueba de software no certifica la alineación de un proveedor real.

**No identificación distributiva bajo MSE.** Con A=0 y masas 1/3, los átomos (.5,1,1.5) y (.25,.75,2) tienen igual media1 pero diferentes varianzas. Ambos producen mu=sqrt(c) y la misma MSE para cualquier Y. MSE sólo supervisa la media transformada, no identifica p/u ni calibración o cobertura. No se reconstruyen intervalos probabilísticos desde estos átomos como si estuvieran validados.

**Gradiente de la cabeza, en el interior.** Si g_k=T_A(u_k), entonces ∂mu/∂u_j=p_j c u_j/g_j y ∂mu/∂ell_j=p_j(g_j−mu) para los logits libres j=1,2. Por la suma acumulada, ∂mu/∂d_j=mean_fit(U) Σ(k>=j)p_k c u_k/g_k. Se encadena la sigmoide de softplus y ∂loss/∂mu=2(mu−Y)/mean_fit(Y)². Son derivadas de lectura diferenciable, no una prueba de convergencia o superioridad. La guardia `maximum` coincide con la lectura en aritmética real y sólo introduce la selección de rama por redondeo en su frontera.

## 5. Hipótesis falsable y estado de evidencia

La hipótesis futura pertinente es que el algoritmo completo consiga una **media de RMSE individuales por semilla** menor que cada rival de la familia completa, con mejora práctica y sin degradación secundaria material conforme al protocolo admitido. Theta=mean_s sqrt(mean_t(error_(s,t)²)). No es sqrt(mean_(s,t)(error²)) ni RMSE del ensamble. La predicción estructural contrastable es el soporte de salida; no se equipara al efecto de mercado.

Los umbrales, las tres pruebas, la regla conjunta y el dimensionamiento corresponden a la batería separada de la fase2. No se fija retrospectivamente aquí una regla nueva para el piloto ni se declara una hipótesis preregistrada antes de resultados ya vistos. Los diagnósticos y los intervalos retrospectivos existentes son de desarrollo expuesto; sus intervalos frente a `global_mse` incluyen cero. R3 no se acredita en este expediente.

## 6. Correspondencia, procedencia y verificación

[code_correspondence.csv](code_correspondence.csv) identifica ecuación, función, líneas y SHA256 del archivo real. El [manifiesto](SOURCE_MANIFEST.json) fija la guía, identidad, revisión previa, recibos, runtime y este código didáctico. El wrapper `runtime/candidate.py` conserva la receta exacta `input_only_mse`; cuatro archivos propios preservan bytes del commit c4ca86f1431afeaa958ff0b544b453ab17a4d7b8, según `runtime/vendor/ORIGIN.json`. El commit de evidencia piloto es 05c1a637f13356dd1df6b8c5aa8f4676dda61c12. No se importan implementaciones ajenas de DAIN, RevIN, CryptoL, HARNet o FQF.

Se reutiliza el recibo auténtico de ocho pruebas previas específicas de la candidata: estado/salida/pérdida iguales a la receta original, formas, soporte, invariancia acotada, contraejemplo de equivariancia, independencia entre ventanas, degeneración y gradientes finitos sin pasos de optimización. Esta fase no vuelve a contarlas como nuevas.

El [ejercicio ejecutado](defense_exercises.py) añadió seis verificaciones con doce ventanas sintéticas. Incluye una modificación razonada y probada: reemplazar la evaluación ingenua sqrt(A+c u²) por `hypot`, ya presente en el runtime congelado. En el caso sintético extremo u~10^200 la expresión ingenua desborda y `hypot` permanece finita; en el rango moderado ambas coinciden. Esto demuestra la corrección de una decisión numérica concreta, no una causa observada de fallos del proyecto. Además contrasta causalidad del encoder, efecto nulo del argumento de restitución, soporte, invariancia y no identificación. El estado del modelo y los siete hashes runtime se verifican antes y después. No hay pasos de optimizador, datos de mercado, pesos guardados ni llamadas externas.

## 7. Derechos, IA y acto auténtico restante

Se declara asistencia de IA para revisión, pruebas, ecuaciones y redacción de esta entrega. Los conceptos IA separados tienen independencia funcional; no representan una persona externa ni certifican comprensión del estudiante. El uso de primitivas PyTorch está permitido por la guía; conservar propio código previo no equivale a importar una arquitectura ajena.

Este expediente no concede una licencia global al repositorio, datos o pesos. Las MIT/CC-BY del suplemento de Zenodo mantienen sólo su ámbito previo. Los derechos de las dependencias y artículos permanecen con sus titulares. Se citan fuentes sin redistribuir cuerpos; una URL pública no concede permiso de copia de datos o código. Se conservan las limitaciones documentadas de las consultas de licencias DAIN/RevIN. No se usan ni publican credenciales, identidad privada, llaves de respaldo o checkpoints.

Ya está preparado el objeto técnico que puede evaluarse: adaptación completa acotada frente al corpus, pruebas/procedencia y modificación ejecutable. El acto aún no observado es **la explicación y defensa efectivas del estudiante ante el evaluador, con una modificación razonada y respuestas comprobables**, y la valoración académica de R1/R2. [DEFENSE_GUIDE.md](DEFENSE_GUIDE.md) da el guion y la operación concreta. No se impone una firma adicional ni un nuevo permiso para continuar la corrección técnica. La revisión IA autorizada no sustituye ese dominio humano.

La guía §6 pp7–8 define R1/R2; §6.1–6.3 pp8–12 delimita adaptación y revisión; §14.5.1 pp43–44 exige sustento y demostración; §15 p48 mantiene responsabilidad del estudiante. La rúbrica p60 asigna revisión/diseño8 %, implementación/demostración10 % y sustentación7 %. Esos pesos no son incrementos de nota atribuibles al paquete. La valoración orientativa y sus topes no se recalculan aquí.
