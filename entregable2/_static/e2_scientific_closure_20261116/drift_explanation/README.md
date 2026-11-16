# Separación temporal de dominios: explicación exploratoria

Se conserva la alerta adversarial original: AUC OOF 0,9525386701486541,
p por permutación con reajuste 0,001. Este análisis posterior utiliza las mismas
655 fechas de entrenamiento y 608 de evaluación, sin la calibración. La respuesta
que aprenden los clasificadores es pertenencia de periodo, no volatilidad.

El plan precede a los nuevos cálculos, después de conocer esa alerta. Se utiliza
el mismo clasificador logístico balanceado y los cinco folds purgados originales.
La AUC completa se reproduce exactamente; no se altera su definición ni su p.

| Grupo de atributos | Dimensiones | AUC OOF descriptiva |
|---|---:|---:|
| OHLCV relativo completo | 35 | 0,9525386701486541 |
| OHLC de precios relativos | 28 | 0,49734331056649256 |
| log1p de volumen | 7 | 0,9026541783848935 |

Los siete rezagos de volumen mantienen información considerable para distinguir
los periodos con este procedimiento. Sus medias de log1p(volumen) son inferiores
en prueba en aproximadamente 0,96–0,98 desviaciones estándar del entrenamiento.
Por ejemplo, en t−1 la media cambia de 10,61498 a 9,75188 y la desviación de
0,89620 a 0,56062. Son estadísticas en escala log1p, no cambios porcentuales de
volumen bruto. Se publican las 35 medias, desviaciones, cuantiles y AUC univariadas.

Estos resultados no identifican una causa económica ni una descomposición aditiva
de la AUC. Que precios relativos por sí solos obtengan una AUC cercana a 0,5 no
prueba estacionariedad ni descarta cambios que este clasificador no detecte.
Los precios se expresan respecto al último Close disponible: no se examinó aquí
estabilidad de sus niveles absolutos. Los coeficientes no se publican porque la
rutina original no los expone; no se inventa una importancia de variables.

## Tratamiento de la alerta en el estudio

1. Mantener el cambio de distribución como una amenaza medida de transportabilidad.
   La auditoría causal de entradas y escaladores se evalúa separadamente: distinguir
   periodos no demuestra que haya entrado información futura.
2. Interpretar el benchmark como evaluación retrospectiva sobre esa transición
   temporal, con pesos y transformaciones congelados. No reponderar, retirar volumen,
   seleccionar otra receta o recalibrar con el test para hacer desaparecer la alerta.
3. Leer el desempeño temporal ya publicado por trimestre y ventanas móviles junto
   a los errores por regímenes de RQ2 y la cobertura condicional. Las fuentes y su
   alcance se ligan en treatment_evidence.json. Los subperiodos y ventanas solapadas
   no son datasets independientes; esos análisis no garantizan transportabilidad.
4. Una adaptación futura requerirá un diseño de desarrollo temporal y evaluación
   posterior propios. Este suplemento no ejecuta esa adaptación ni afirma que el
   drift se haya corregido.

Los folds del diagnóstico de distribución incluyen filas de fechas posteriores:
no constituyen un backtest causal de pronóstico. No se añaden p-valores,
confirmaciones, intervalos comparativos o afirmaciones de superioridad entre las
tres AUC. Las ventanas dependientes no se interpretan como observaciones iid.

Se ejecutaron 15 ajustes logísticos de dominio, cero entrenamientos de los
predictores y cero cambios a A2/RQ1, recetas, partición, escaladores o métricas.
