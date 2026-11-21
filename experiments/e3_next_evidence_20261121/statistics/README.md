# Inferencia implementada y validación controlada ejecutada

Preparado con asistencia de IA. Se implementaron **dos procedimientos
direccionales por rival y un complemento MCS global**. Se ejecutaron los once
casos prefijados, **1024 réplicas Monte Carlo y 2.096.128 réplicas bootstrap**,
sin entrenar modelos ni generar predicciones o remuestreos de mercado.

**La configuración implementada no queda aprobada para confirmación.** En el
caso controlado de igualdad poblacional de los veinte riesgos, el MCS propuesto
rechazó 27/128 veces: **21,09 % frente a 10 % nominal**, con IC binomial exacto
individual de 95 % **[14,38 %, 29,19 %]**. Retuvo todos los modelos realmente
empatados en 101/128 réplicas, 78,91 %, frente al 90 % de confianza propuesto.
Esto concierne a esta extensión/configuración, no refuta la teoría del MCS
clásico. No se retocó el método ni se repitieron casos para buscar aceptación.

P1/P2 tampoco reciben aprobación universal: se observaron cinco señales
individuales de inflación marginal y quince de déficit de cobertura bajo los
criterios prefijados. Son intervalos individuales de exploración, **no una
certificación simultánea** sobre todos los rivales y escenarios. Holm puede
reducir falsos rechazos familiares, pero no vuelve válidos p-valores marginales
o métodos mal calibrados.

## Qué se conserva y qué se entrega

- `temporal_battery.py`: P1 bootstrap estacionario cruzado del funcional RMSE;
  P2 linealización semilla/calendario con HAC y factor HLN de h=1; P3 MCS global
  del funcional. Sus supuestos y extensiones se delimitan en
  `METHODS_IMPLEMENTED.md`.
- `CONTROLLED_VALIDATION_PLAN.json`: plan publicado **antes de los resultados**,
  SHA256 `20d2f73adff064d47d19b2394e708e5a84515eed410ee81b258c2a8226c710f5`,
  commit `550b6d26cbd1b82f40a347c24b180320776e5ca3`.
- `IMPLEMENTATION_FREEZE.json`: ocho archivos fijados por hash antes de ejecutar;
  incluye código, plan, 19 guardas y cambios de software anteriores al MC.
- `validate_controlled.py`: generador con verdad poblacional analítica, ejecución
  acotada, registro por réplica y tratamiento explícito de resultados parciales.
- `controlled_validation_run/`: recibo, once resúmenes, resumen global y las
  1024 réplicas detalladas comprimidas, con hashes de las entradas regenerables.
- `validation_compact.csv`: los once casos completos; `per_rival_validation.csv`:
  418 filas de caso/rival/procedimiento, sin seleccionar sólo resultados favorables.
- `VALIDATION_DECISION.json`: conclusión posterior, no preregistro ni modificación
  de la receta o del piloto V3 anterior.

La ejecución duró **40,863 s** dentro del máximo prefijado de 600 s. Runtime:
Python 3.12.14, NumPy 2.3.5 y SciPy 1.17.0; dos hilos BLAS/OMP. Se completaron
todos los casos sin errores. La verdad fue `base_rival − base_candidate` por la
construcción de los factores, nunca la diferencia observada de una muestra.
La distribución t5 tiene segundo momento estacionario exacto; su distribución
estacionaria completa sólo se aproxima con el burn-in declarado de 128 puntos.

## Familias y potencia: límites que permanecen

Los veinte sistemas son un DGP controlado, no diecinueve rivales reales nuevos.
Holm38 se refiere a los dos métodos direccionales por diecinueve rivales; la
reserva Holm57 añade diecinueve p=1 declarados **sin implementar el tercer test**.
La intersección de dos métodos no se renombra como potencia de la batería R3.
En la alternativa sintética prefijada, las tasas de intersección Holm38 por rival
fueron de 9,375 % a 20,3125 %; no acreditan potencia de mercado ni dimensionamiento.

La resolución mínima bootstrap es 1/2048. Al inicio de Holm57, el umbral es
0,05/57: sólo el menor p posible de P1 queda por debajo. Se conserva esta
resolución limitada en vez de añadir réplicas después de ver los resultados.

El diagnóstico de escala separa las dos fuentes: al pasar de R12/T128 a R24/T128,
la varianza entre semillas media estimada baja de 0,002339 a 0,001188, mientras la
temporal permanece cerca de 0,002. Al duplicar T con R12, la temporal baja a
0,001111 mientras permanece la variación entre semillas. Son ilustraciones del
DGP controlado, no estimaciones de los modelos reales o recomendaciones R/T.

La futura propuesta V4 cambia la familia: si incorpora V3 y `atom_mse_only`
como rivales, serían 21 rivales/63 componentes y 84 celdas para cuatro paneles.
**Este plan M20/19 rivales no se modificó y no valida aquella familia M22.**
Tampoco llena las varianzas ausentes del mercado ni selecciona R, T, MDE o δ
finales. Siguen pendientes un tercer test pertinente por contraste, revisión
metodológica, admisión de datos no expuestos y planificación suficiente.

## Reproducir el software y la ejecución controlada

Desde este directorio, en un entorno separado con Python 3.12.14 y las versiones
de `requirements-statistics.txt`:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m unittest discover -s . -p 'test_*.py' -v
```

La reproducción del MC requiere una salida nueva y el recibo del anclaje del
plan, suministrado por el paquete padre:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python validate_controlled.py \
  --plan CONTROLLED_VALIDATION_PLAN.json \
  --anchor ../CONTROLLED_PLAN_PUBLICATION.json \
  --freeze IMPLEMENTATION_FREEZE.json \
  --out /tmp/e3-controlled-independent-01
```

El comando rechaza cambiar el plan/código fijados o sobrescribir una salida.
Los resultados observados están ya guardados; repetirlos aquí no es necesario
para continuar el proyecto. Las entradas cuadradas y las réplicas se regeneran
por semilla; la igualdad bit a bit entre BLAS/plataformas diferentes no se
garantiza por publicar versiones de paquetes.

Para integrar el motor con futuros errores ya alineados, usar la función
`run_battery(q, model_names, candidate_index, margins, rng, Settings(...))`.
El llamador debe garantizar identidades de semillas, fechas comunes, fuentes,
unidades y separación de planificación/confirmación. La función no adquiere
datos ni convierte una entrada en un panel admitido. Sus resultados mantienen
`confirmation_enabled=false` y `three_tests_per_contrast_gate=false`.

El NO-GO de la receta V3, sus treinta ajustes y el 2,70 % adverso permanecen
intactos. No se creó un protocolo confirmatorio o Colab de mercado. Las pruebas
de software y esta simulación no son reproducción humana independiente ni R3.
