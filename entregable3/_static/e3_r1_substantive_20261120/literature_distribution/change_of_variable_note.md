# Cambio de variable: qué aporta y qué no aporta el score de V3

Análisis algebraico preparado con asistencia de IA; no se ejecutaron modelos ni
se añadieron observaciones, ajustes o contrastes. No se presenta como una regla
propia de puntuación inédita.

Sea `A>=0` observable al origen, `c=6/49`, `U>=0` y
`Y=g_A(U)=sqrt(A+c U²)`. Para una ley finita `G` de U, denote
`H=(g_A)#G`, su ley transformada. En orientación de pérdida,

```text
CRPS(G,u) = integral_0^infinity [F_G(v)-1{u<=v}]² dv.
```

La identidad es válida también para átomos: no se supone que G tenga densidad.
El cambio de variable `y=g_A(v)`, con inversa
`h_A(y)=sqrt((y²-A)/c)`, da, para `y0=g_A(u)`,

```text
CRPS(G,u)
 = integral_sqrt(A)^infinity [F_H(y)-1{y0<=y}]² w_A(y) dy,
w_A(y) = y/[sqrt(c)*sqrt(y²-A)], y>sqrt(A).
```

Se interpreta como integral impropia en el borde. Para A>0 el peso es singular
en el piso pero localmente integrable; no se afirma que sea acotado. Si A=0,
`g_0(v)=sqrt(c)*v` y el peso es la constante `1/sqrt(c)`.
Con la escala positiva de ajuste `sU`, el score usado por el código es el miembro
anterior dividido una sola vez por sU. Se requieren momentos/integrabilidad
apropiados para que el score y el término MSE poblacionales sean finitos.

Gneiting y Raftery (2007), §4.2, ecuaciones (20)–(21), proporcionan la definición
CDF y la representación por esperanzas. Su §6.4, ecuación (49), admite integrales
de scores binarios contra una medida Borel: el caso ponderado ya es una familia
conocida. Esta aplicación de cambio de variable es una observación algebraica
de esta revisión; no una ecuación atribuida al cuerpo no leído de Gneiting y
Ranjan (2011). El CRPS_U de V3 no es una nueva regla propia por sí mismo.

La equivalencia **no** hace a V3 idéntica a la cabeza aditiva de 1210 parámetros:
ésta emite un punto y se entrena con MSE. V3 parametriza una ley finita ordenada,
aplica `g_A` a sus átomos antes de promediar y entrena la ley con CRPS_U además
del error de esa media. Es una diferencia de representación y objetivo; no una
garantía de una clase puntual más rica o de menor RMSE.

Además `U=h_A(Y)` cuando se conoce A: la etiqueta auxiliar no es información
futura independiente adicional. A es observable y U sólo interviene como
etiqueta durante el ajuste. Es incorrecto afirmar que dos etiquetas deterministas
proporcionan por identidad más información que Y y A, aunque produzcan otro
sesgo inductivo y otros gradientes.

## Referencias verificadas

- Gneiting y Raftery (2007), cuerpo primario leído focalmente: §4.2, p.367
  impresa/PDFp9; §6.4, p.371 impresa/PDFp13, ecuación (49).
  https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Gneiting y Ranjan (2011), DOI10.1198/jbes.2010.08110: registro editorial e
  institucional observado, cuerpo no leído en esta revisión. No es necesario
  atribuirle una ecuación para justificar el cambio de variable anterior.
