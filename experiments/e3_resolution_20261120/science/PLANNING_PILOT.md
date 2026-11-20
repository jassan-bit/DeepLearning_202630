# Piloto de planificación preparado, no ejecutado

No se autoriza una campaña para buscar un resultado favorable. El NO-GO actual
se conserva: atom_crps no superó la heurística del piloto y no superó al mejor
control en los tres dominios de transferencia expuestos. La disponibilidad de
días y GPU ya comunicada por el autor no es un permiso faltante que haya que
volver a pedir. Faltan justificación científica, datos y dimensionamiento.

## Material que falta antes de producir una nueva trayectoria

1. Documento fechado que justifique por qué tiene sentido seguir la receta
   actual pese al resultado adverso. Una nueva pérdida/lectura sería otra versión
   y tendría su propio desarrollo, no una alteración retrospectiva del piloto.
2. Admisión y exposición auditadas de cuatro paneles futuros: uno principal y
   tres adicionales con al menos dos ejes de diferencia documentados. El
   registro futuro sólo identifica clases de dominio por evaluar, no fuentes
   seleccionadas ni datos admitidos. Fijar fuente, licencia, versión, hash,
   reloj, unidades y políticas de splits/roll, con custodio y log de acceso.
3. Particiones absolutas de ajuste, selección, calibración, planificación y
   confirmación. Los datos y resultados de planificación quedan excluidos de
   confirmación. Declarar si se permite historia observada compartida; para
   soportes disjuntos de ocho cierres, purgar siete targets entre particiones.
4. Márgenes prácticos y efectos alternativos por panel fundamentados en el uso,
   antes de resultados nuevos. El 2% del piloto es una regla de continuación,
   no una justificación automática de delta ni de Delta_alt.

## Diseño operativo de planificación

- Candidata y 19 rivales de `main_family_variance_inputs.csv`, sin eliminar los
  16 rivales ausentes del piloto por falta de resultados o por coste.
- Al menos cinco semillas emparejadas por receta estocástica para estimar
  dispersión de planificación; no contar repeticiones de controles deterministas
  como semillas nuevas. El mínimo confirmatorio posterior es diez o más.
- Con 15 redes incluyendo V3 y cuatro paneles, cinco seeds implican 300 ajustes
  de planificación antes de cualquier HPO nuevo. A120 s por ajuste son 10 h
  nominales; no son consumo medido ni presupuesto final, y excluyen descargas,
  HPO, clásicos, fallos y análisis. No se han lanzado esos ajustes.
- Usar el mismo preprocesamiento causal, particiones, selección, calibración y
  cohorte de hardware/software; contar oportunidades HPO y consumo por separado.
  Fijar parámetros/FLOPs y sensibilidad de capacidad comparable si procede.
- Guardar trayectoria completa por modelo/semilla/fecha, recursos y estados.
  Producir los 76 registros de `future_variance_template.csv`: varianza entre
  semillas, dependencia temporal, interacción y covarianza entre rivales.
- Validar la batería y el generador conjunto sólo sobre planificación. La
  revisión de `METHODS_REVIEW.md` especifica los puntos que impiden darla por
  aprobada. Registros con nulos, alternativas, semillas de simulación y versiones
  deben fijarse antes de calcular su potencia, sin usar la ganancia observada.

## Salidas para dimensionar

Para cada presupuesto candidato (R,T), un archivo CSV por panel con columnas:

`stage,dataset_id,replicate,R,T,rival,procedure,p_raw`

`stage` debe ser `planning_simulation`. Cada réplica contiene los 57 componentes
de la misma ley conjunta y el mismo presupuesto. El lector aplica Holm dentro
de cada réplica, calcula la intersección por rival y la conjunta sin asumir
independencia, y reporta precisión Monte Carlo. Un R sólo puede elegirse después
de validar métodos/DGP y demostrar la suficiencia de todos los contrastes bajo
alternativas operativas. Sensibilizar T, colas, dependencia y parámetros.

No se crean hoy un tag final, una celda Colab de confirmación, una fecha final
ni un R definitivo. La preparación ejecutable no se presenta como ejecución
de este piloto ni como potencia prospectiva ya acreditada.
