# Cierre comparativo de R1 y decisión de adaptación

Revisión del 19 de noviembre de 2026 en Colombia / 20 de noviembre en UTC,
preparada con asistencia de IA declarada. El libro base es `f666a215` y la
referencia científica evaluada permanece en `5a07d615`.

## Resultado concreto

La referencia aditiva de 1210 parámetros conserva los resultados de su auditoría
de 796 trabajos. La comparación delimitada de diez fuentes no establece una
diferencia sustantiva que acredite R1: atención con consulta aprendida, piso por
muestra más enlace positivo, MSE y las propiedades genéricas tienen antecedentes.
Eso no prueba ausencia global de novedad ni copia histórica.

Se recomienda investigar únicamente V3, ya implementada, como adaptación de
estructura y entrenamiento: ley de tres átomos del componente futuro U,
transformación exacta del objetivo solapado y CRPS_U auxiliar. Esta diferencia
es concreta y revisable, pero no acredita la prioridad de la composición ni su
utilidad predictiva. No se crea otra arquitectura ni se ejecutan ajustes aquí.

## Evidencia y acceso

- `primary_review`: diez trabajos cercanos; ocho cuerpos accesibles con lectura
  focal de secciones y dos fuentes limitadas a resumen/extracto. Seis relaciones
  de citación verificadas desde tres semillas.
- `search_review`: 50 registros con disposición explícita; 41 resúmenes completos,
  tres extractos parciales, un listado editorial y cinco sin resumen verificable.
  No se afirma haber leído los 50. La colección nativa original conserva sus
  180 ocurrencias y tres bases. Las 18 solicitudes nativas nuevas fallaron antes
  de recibir bibliografía: sus conteos son desconocidos.
- `abstract_followup`: fuentes primarias alternativas y sus límites, fusionadas
  por identificador con la etapa anterior. No son otras 18 publicaciones únicas.
- `code_review`: inspección acotada de README oficiales y contexto de tareas,
  con commits; no auditoría completa de repositorios ni copia de implementación.
- `adaptation_design`: modificación de V3, procedencia, controles y 100
  comprobaciones analíticas de leyes finitas. La ventaja de Jensen compara con
  sustitución por segundo momento bajo la ley correcta, no con la red aditiva.
- `model_identity.json`: fuentes verificadas y objetos separados.

`primary_review/priority_dispositions.json` conserva un subcorte anterior que
todavía no había obtenido el resumen SSRN 7452120; el seguimiento posterior y el
CSV final de 50 registros incorporan su recuperación. Los cortes se preservan
sin reescribir sus tiempos o hashes.

Sólo se publican paráfrasis, fuentes, localizadores, decisiones y verificaciones.
Los textos completos de artículos, los README completos y los cachés íntegros de
resúmenes usados durante la preparación quedan fuera de este paquete. Los
paquetes bibliográficos históricos mantienen sus propios recibos inmutables.

## Límites de diseño

Una cabeza escalar flexible puede representar la misma media puntual que V3.
Una pérdida auxiliar propia puede empeorar el riesgo puntual cuando la ley está
mal especificada o la capacidad es finita; el contraejemplo se conserva. La
representabilidad por tres átomos y el mínimo poblacional global son supuestos
restrictivos, no garantías de entrenamiento, calibración o generalización.

Las tres ablaciones deben ejecutarse con política común, declarando capacidad.
El control de agregación por segundo momento aún requiere integración para su
versión entrenada. Añadir V3 a los 19 sistemas actuales da 20; el planificador
histórico de 18 no dimensiona automáticamente esa familia. Presupuesto, margen,
hipótesis, reglas, semillas y potencia deben fijarse antes de la nueva ejecución.

Los cuatro conjuntos actuales ya tienen resultados expuestos. Si orientan la
adaptación, se usan como desarrollo/exploratorio para V3, no como confirmación
ciega recuperada. Los 796 trabajos no se renombran como resultados V3.

## Verificación sin mercado

```bash
python verify_packet.py
```

El verificador comprueba hashes, tamaños, identidades y conteos. No importa
PyTorch, carga pesos, lee precios, entrena o predice. El código de comprobación
analítica también es Python estándar y usa únicamente números artificiales.
R1 y R3 permanecen sin acreditar; la modalidad externa de IA autorizada en 6.13
no se reabre como requisito de firma humana ni sustituye la defensa académica.
