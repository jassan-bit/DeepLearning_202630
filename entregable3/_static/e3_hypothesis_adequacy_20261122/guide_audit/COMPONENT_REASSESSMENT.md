# Revaloración orientativa del componente de hipótesis y adecuación

La valoración independiente solicitada cambia **2,8 → 3,0/5** para el componente
del **15 %**, exclusivamente por nueva evidencia numérica ejecutada sobre la
candidata actual. Es un juicio documental por IA; no es nota oficial ni una
equivalencia automática de los niveles de la guía. El componente permanece en
desarrollo.

| Magnitud | Anterior | Revisión actual |
|---|---:|---:|
| Componente hipótesis/adecuación | 2,8 | **3,0** |
| Aporte ponderado de ese componente | 0,420 | **0,450** |
| Puntaje ponderado completo, sin redondear | 4,092 | **4,122** |
| Puntaje mostrado antes del tope | 4,09 | **4,12** |
| Sustentación indicada por el autor | 4,5 | **4,5** |
| Nota bajo R1/R2 aceptadas y R3 no acreditada | 4,00 | **4,00** |

El cambio global es exactamente **0,030 puntos**. Todos los pesos y las otras
diez notas permanecen idénticos. La sustentación aporta 0,315, se trata como valor
del autor y no como defensa observada o R2 acreditada. La nota oficial sigue
vacía. Si se determina incumplimiento de R1 o R2, el tope pertinente sería 3,0;
una evaluación pendiente no se convierte automáticamente en ese incumplimiento.

## Evidencia que sí justifica el incremento limitado

Sobre 1760 predicciones ya guardadas, 32 trayectorias, 55 fechas expuestas y cinco
semillas se ejecutaron 128 posiciones de diagnósticos con Holm128, 64 intervalos
HAC puntuales de sesgo y dos remuestreos temporales pareados de 3999 sorteos
cada uno. Se conservaron los siete comparadores del piloto y la media de RMSE
individuales, con fechas comunes y semillas fijas. La lectura e integridad se
verificaron contra el ZIP original; una revisión separada confirmó la aritmética
y Holm mediante una biblioteca independiente. Estos diagnósticos/IC propios de
la candidata faltaban en el corte valorado en 2,8.

La mejora reconoce ejecución y calidad de interpretación, **no un resultado
favorable**. Hay 2/5 rechazos Mincer–Zarnowitz de input-only tras Holm128. Los
intervalos RMSE contra global incluyen cero con ambos bloques: bloque 7,
[-0,0001372754;0,0000293737], y bloque 14,
[-0,0001195573;0,0000191064]. Se reportan puntuales, exploratorios y condicionados
a las trayectorias guardadas, sin cobertura finita validada. Los no rechazos de
Ljung–Box/ARCH no demuestran adecuación; los IC que incluyen cero no prueban
equivalencia ni insuficiencia de la candidata en toda aplicación futura.

El complemento MCS estándar retiene a input-only entre siete recetas del piloto,
para pérdida MSE secundaria. Se incorpora como reporte ejecutado de alcance
limitado, **sin otro aumento por inclusión**, sin ganador único y sin sustituir
la primaria media-RMSE o la familia de 21 sistemas. La guía/matriz y las preparaciones de datos,
notebook y software no reciben crédito como experimentos de mercado.

## Límites que mantienen la valoración en desarrollo

El preflight identifica que la inferencia sobre MSE/raíz agregada no es una
prueba de la primaria heredada media-RMSE y que las cotas marginales por panel no
certifican por sí solas la regla compuesta adicional2/3 a error global 0,05. La
cota conservadora 2q=0,0957217463 no establece ese objetivo, sin demostrar por sí
misma inflación real. No se cambian automáticamente alpha, endpoint o reglas.

Faltan predicciones completas de planificación de 21 sistemas × 4 paneles y un dimensionamiento real,
predicciones fit/selection del mismo checkpoint para Tablas A/B, calibración y
cobertura de la candidata, producto MCS pertinente de familia completa y una
confirmación reservada con sus efectos y robustez. La nueva revisión no acredita
R3 ni transforma datos de desarrollo en prueba independiente.

Los hashes, fuentes y cálculos exactos están en COMPONENT_REASSESSMENT.json.
Asistencia de IA declarada; no se atribuye esta revisión a un tercero humano.
