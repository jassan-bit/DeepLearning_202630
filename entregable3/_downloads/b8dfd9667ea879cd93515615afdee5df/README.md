# Guía primaria y adecuación del dimensionamiento

Componente preparado con asistencia de IA. No entrena, predice, simula ni habilita confirmación. La fuente de entrada es `d6cf7f1af6800ec1ac85a26347bdde52380e362f`. Sus resultados de auditoría son posteriores a la evidencia que examinan y no son preregistros.

## Qué se ejecutó

- Lectura directa de las páginas 23–28 y 44–45 del PDF original, con SHA-256 y hechos separados en [primary/PRIMARY_GUIDE_FACTS.json](primary/PRIMARY_GUIDE_FACTS.json).
- [Matriz de doce requisitos](REQUIREMENT_EVIDENCE_GAP_MATRIX.csv), con sección/página, evidencia, alcance, trabajo ejecutable y requisito prospectivo. Sus estados son el corte de entrada: las ejecuciones nuevas de calibración y diagnóstico se integran por separado.
- Auditoría real de **71 fuentes distintas** y los **486 escenarios guardados**, con recuento de filas, claves únicas y banderas que impiden presentarlos como diseño suficiente. Las doce entradas inmediatas y sus hashes están en [INPUT_BINDINGS.json](INPUT_BINDINGS.json); los 71 hashes transitivos se comprueban desde sus manifiestos originales. Se conserva el primer recibo de lectura de metadatos en `DEVELOPMENT_METADATA_AUDIT_FIRST.json`; el recibo final incorpora además el cotejo directo de los dos CSV de escenarios.
- Conciliación de las **3/19 varianzas de desarrollo V3**, basadas en tres pares, y las **16 ausentes**. Las **76 celdas futuras V3** y las **84 de V4** permanecen vacías, sin imputarlas a cero.
- Comprobación de las **cinco referencias pareadas y 25 covarianzas V4**. Son del calendario D00 expuesto, con cinco semillas neuronales y un único trayecto de cada control analítico. No son entradas de los cuatro paneles futuros. La receta V4 está abandonada y sus 340 ajustes ilustrativos no se autorizan.
- [21 guardas de software](software_tests.log) y ejecución del [contrato de entradas](EXECUTED_POWER_INPUT_AUDIT.json). La persistencia requerida se conserva, pero faltan candidata viable, datos admitidos, métodos adecuados, márgenes y varianzas prospectivas. `R_final` y `T_final` siguen nulos.

Los ceros de ajustes, predicciones y remuestreos pertenecen exclusivamente a esta operación de lectura de metadatos. No borran los veinte ajustes CPU anteriores ni las ejecuciones estadísticas nuevas de otros componentes.

## La guía y el estimando

El Cuadro 4, página 26, admite para F2: MCS, DM-HLN por horizonte, Giacomini–White o Clark–West según anidamiento, Friedman sobre series/orígenes y bootstrap estacionario. Exige tres procedimientos distintos y al menos uno global o con control múltiple. Las secciones 8.4–8.7 exigen rechazos de igualdad, dirección y relevancia práctica, efectos, intervalos, adecuación y robustez.

La guía no exige literalmente tres nulos `theta <= delta`. Esa fue una decisión más fuerte de la extensión anterior. MCS tiene un papel global: no basta repetir su p en cada rival para afirmar tres contrastes. Se explicita la convención usual `alpha_MCS=0.10`, equivalente a confianza 0.90, evitando confundir los términos del texto.

GW requiere una hipótesis condicional e instrumentos/supuesto de estimación pertinentes. Clark–West requiere un anidamiento real y su ajuste, no dos redes con diferente número de parámetros. Friedman necesita unidades de series/orígenes defendibles; no convierte fechas correlacionadas o semillas en nuevos conjuntos independientes.

El objetivo primario ha sido **media de RMSE individuales** con incertidumbre de inicialización y calendario. Una comparación de pérdida cuadrada temporal, condicionada a una cohorte ya entrenada, es otro estimando y otra población de muestreo. Incluso el ranking puede invertirse: el [contraejemplo determinista](primary/PRIMARY_GUIDE_FACTS.json) tiene MSE por semilla `(0.01, 8.41)` frente a `(3, 3)`. Las medias de RMSE son 1.5 y 1.732, pero las medias de MSE son 4.21 y 3. No se transfiere una validación de un objeto al otro.

## Contrato ejecutable y límites de la referencia normal

`power_inputs_audit.py` verifica versión, estimando, unidades y población de semilla/calendario; bindings de evidencia; cuatro paneles; familia completa; pares de planificación; tres métodos; presupuesto; persistencia; calibración; covarianza y regla completa de potencia. Comprueba límites Monte Carlo por contraste y exige reportar la conjunta con precisión, sin exigirle universalmente 0.90.

La [plantilla](POWER_INPUT_CONTRACT.template.json) no selecciona otra candidata ni hereda los 19/57 o 21/63 de diseños abandonados. Incluso un contrato sintético con todas las casillas estructurales compatibles devuelve `power_certified_by_this_program=false` y `confirmation_enabled=false`: comparar JSON y hashes no acredita los supuestos científicos.

La función `linearized_budget_diagnostic` sólo ofrece una referencia normal explícitamente ilustrativa bajo

```
Var(theta_hat) = V_seed/R + LRV_calendar/T + LRV_interaction/(R*T).
```

Separa el margen práctico de la alternativa de planificación y usa una referencia bilateral Bonferroni declarada. El término `LRV_calendar/T` no disminuye al aumentar semillas: si supera la varianza tolerable de esa aproximación, la función devuelve bloqueo temporal y no un R ficticio. Conocer R por una fórmula normal no demuestra potencia de la intersección de tres pruebas con Holm, efectos, intervalos y márgenes. Esta referencia no se aplicó a los datos expuestos para seleccionar un diseño futuro.

La guía exige piloto de al menos cinco semillas y finales de diez o más según dimensionamiento; **0.90 por contraste y potencia conjunta reportada**. Las cantidades mínimas por sí solas no garantizan potencia. No se calcula potencia observada ni se usa el producto de potencias marginales como estimación real de la conjunta.

## Reproducir sin modelos

Desde este directorio:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test_power_inputs_audit.py' -v
PYTHONDONTWRITEBYTECODE=1 python power_inputs_audit.py \
  --repo-root /ruta/al/checkout \
  --contract POWER_INPUT_CONTRACT.template.json \
  --output /ruta/nueva/POWER_INPUT_AUDIT.json
```

El checkout debe contener los archivos del corte d6 identificados por los bindings. El programa rechaza rutas relativas que escapan, symlinks, hashes alterados y sobrescritura del resultado. No descarga insumos, no lanza campañas y no genera Colab confirmatorio.

Sigue faltando una propuesta científicamente viable, paneles admitidos, métodos calibrados para el estimando pertinente y un dimensionamiento prospectivo compatible con toda la regla. La sustentación 4.5 se conserva como valor proporcionado por el autor, sin afirmar que se observó la defensa. R3 no se acredita por este componente.
