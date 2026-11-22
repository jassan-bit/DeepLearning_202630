# Defensa concreta del mecanismo y del código

Material preparado por IA; no es acta ni evidencia de una sustentación realizada. El estudiante debe poder exponerlo con sus palabras y operar sobre la copia didáctica ante quien evalúa. Duración sugerida: 12–15 minutos, ajustable al formato docente.

1. **Problema (2 minutos).** Dibujar siete retornos: seis conocidos y uno futuro. Explicar por qué sigma7 tiene un suelo ya observable y derivar A, U y c=6/49. Indicar población/ddof0, diario/sin anualizar y barras cerradas. No atribuir novedad a la identidad de varianza.
2. **Recorrido línea por línea (3 minutos).** Seguir `known_geometry`→`fit_inputs`→`transform`→`forward_components`→`loss_components`, con [la tabla de correspondencia](code_correspondence.csv). Explicar qué se ajusta sólo en fit y por qué `U` no aparece en la inferencia. Mostrar la referencia `LOCAL_OUTPUT`: input-only no restituye escala local.
3. **Mecanismo y propiedad (3 minutos).** Escribir u_k, p_k y sum p_k T_A(u_k). Explicar por qué transformar antes de promediar difiere de T_A(E[U]); demostrar soporte y luego el contraejemplo de equivariancia. Identificar qué parte es conocida/heredada y qué adaptación completa se defiende. La novedad no es la normalización aislada.
4. **Modificación razonada ejecutable (3 minutos).** Abrir `didactic_naive_readout` y `didactic_stable_readout` en la copia didáctica. Derivar sqrt(A+c u²)=hypot(sqrt(A),sqrt(c)u). Mostrar primero equivalencia moderada y después overflow de la forma ingenua para átomos~10^200. Explicar por qué no se cambian átomos, masas, funcional, parámetros o receta. Ejecutar el ejercicio; revisar D03/D04 y hashes antes/después. El runtime ya contenía la corrección y permanece intacto.
5. **Límites y preguntas (2–4 minutos).** Explicar el contraejemplo de no identificación por MSE, la diferencia entre causalidad del encoder y de normalización por prefijo, el alcance de R1 técnico y lo que R3 requeriría. Reconocer selección post hoc y resultados adversos V3/V4/V5.

Ejecución CPU, con el entorno del proyecto ya instalado:

```bash
python experiments/e3_primary_joint_repair_20261122/r1r2/defense_exercises.py \
  --candidate-root experiments/e3_input_only_candidate_20261121 \
  --output /tmp/e3-r1r2-defense-human-run.json
```

En el workspace de esta entrega se usó el Python CPU fijado del motor. No se instala otro entorno ni se monta Drive. El archivo `/tmp/...` registra la nueva ejecución, sin sobrescribir el recibo técnico de IA. El resultado PASS verifica operaciones: no certifica por sí solo quién las comprende. No se necesita clave ni token.

Preguntas que el estudiante debe poder responder:

| Pregunta | Respuesta técnica que permite comprobar comprensión |
|---|---|
| ¿Por qué dividir S6 por7 y no por6? | La etiqueta es la varianza poblacional final de siete retornos; S7/7 y la actualización de media producen6/49. |
| ¿Es T_A(E[U]) igual a E[T_A(U)]? | En general no; T_A es convexa para A>0, con segunda derivada cA/(A+c u²)^(3/2). Coinciden casos degenerados o A=0 en U≥0. La lectura implementa la segunda expresión de la ley finita. |
| ¿La ley atómica aprendida es una distribución calibrada? | No. El objetivo sólo identifica el promedio transformado; mostrar D06 y las dos leyes con diferente varianza. |
| ¿Por qué dos logits para tres átomos? | Softmax es invariante a trasladar todos los logits; fijar uno a cero conserva dos grados libres del simplex interior. |
| ¿El suelo impide sobreajuste o garantiza menor RMSE? | No; restringe soporte conforme a una identidad, y la pérdida puede favorecer una restricción sesgada o perder señal de régimen. |
| ¿Se aplica la cota de Lipschitz de la vieja cabeza aditiva? | No. Input-only usa otra lectura; sólo comparte el encoder y las inicializaciones query/temperatura. |
| ¿Normalizar por sí solo acredita R1? | No. Se compara el algoritmo completo con el corpus; frente a `global_mse` sólo cambia X. |
| ¿Qué supuesto pierde la invariancia? | El suelo puede activarse; también cambian las condiciones si se altera volumen, se reajustan estadísticas o no se transforman coherentemente los OHLC. |
| ¿Qué cambia al sustituir `sqrt` por `hypot`? | Estabilidad de evaluación en extremos, no la función matemática. No prueba que un overflow haya causado un fallo real del proyecto. |
| ¿Qué significa PASS? | Verificaciones sintéticas del programa. No superioridad de mercado, defensa humana, aprobación docente ni R3. |
| ¿Dónde está la primaria correcta? | theta=media_s RMSE_s; no raíz del MSE agregado ni RMSE de un ensamble. La batería que lo infiere está en fase2, no en estas pruebas. |

El acto auténtico faltante consiste en realizar esa explicación/demostración y que el evaluador valore R1/R2 según la guía. No se ha escrito una firma o aceptación ficticia. El autor puede conservar la salida y los comentarios reales que efectivamente se produzcan; no hay aquí un requisito inventado de certificación externa adicional.
