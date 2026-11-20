# Resolución científica y preparación prospectiva, 20 de noviembre de 2026

Fuente base: `8d03a73d529eb3f9b2e89992b4488cfeb88a99e6`.
Asistencia de IA declarada; no se atribuye firma o reproducción humana.

## Lo ejecutado

- Auditoría de 162 escenarios de semillas y 324 de calendario: 486 registros
  existentes con identidades, flags y hashes comprobados, sin repetirlos.
- Tabla de 19 rivales principales: sólo 3 tienen referencia de varianza en el
  piloto V3; 16 faltan. Las otras 3 varianzas de los 6 controles son ablaciones.
- Plantilla explícita de 76 celdas futuras (4 paneles×19 rivales), con valores
  ausentes como ausentes y no como cero.
- Auditoría de exposición y procedencia de D00–D03: ninguno está certificado
  como confirmación no expuesta. Clases de futuros dominios quedan registradas
  como no seleccionadas/no admitidas, sin inventar fuentes ni hashes.
- Código de preparación y agregación de potencia condicional, con pruebas de
  contrato de software. Conserva Holm 57 y la intersección de 3 métodos/rival.

La decisión científica permanece **NO_GO_CONFIRMATION_CURRENT_RECIPE**.
Se preservan 796 trabajos de la aditiva, 30 ajustes V3 y 51 trayectorias/20.400
predicciones retrospectivas. atom_crps fue 2,7043119888 % peor que atom_mse_only.
No se cambia la heurística ni se promueve una ablación. No hay GPU, ajustes,
inferencias, recalibración, nuevos p-valores o remuestreos científicos.

## Comandos CPU

Desde esta carpeta, con Python estándar:

```bash
python build_science_audit.py --root /ruta/al/repositorio --output /ruta/nueva/salida
python power_contract.py preflight --config prospective_design.DRAFT.json --output preflight_receipt.json --prepare-only
python -m unittest discover -s . -p test_power_contract.py -v
```

El primer comando verifica los insumos fijados y reconstruye sólo estos
productos documentales. `preflight` sin `--prepare-only` sale con código 2
mientras falten puertas. Ninguna opción puede entrenar o habilitar confirmación.

Para p-valores de una simulación futura de planificación, un panel y presupuesto
por archivo, nunca para reinterpretar resultados observados:

```bash
python power_contract.py summarize-simulation --config prospective_design.DRAFT.json --pvalues /ruta/planificacion.csv --output /ruta/resumen_condicional.json
```

Ese lector **no implementa ni valida** los tres procedimientos ni el DGP y no
produce una certificación de potencia real. Los tests usan fixtures de software
declarados, no experimentos científicos. `METHODS_REVIEW.md` y
`PLANNING_PILOT.md` contienen las decisiones y validaciones aún necesarias.

## Límites que no se cierran retrospectivamente

El margen BTC heredado no está justificado para V3, los nuevos márgenes y efectos
alternativos no están fijados, y la varianza de tres seeds sobre 55 fechas no
identifica la ley conjunta futura. Los 486 escenarios no determinan un R final.
Mínimo cinco seeds de planificación y diez de confirmación son pisos, no una
garantía de potencia. La guía exige reportar potencia conjunta, sin imponerle
un umbral 0,90. La batería temporal requiere demostrar alineamiento con RMSE,
tratamiento de semillas/fechas y calibración del error antes de dimensionarla.

El material es una preparación posterior a resultados, no un preregistro,
protocolo congelado o permiso de ejecución. No se genera Colab confirmatorio.
