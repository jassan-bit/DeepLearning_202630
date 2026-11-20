# Criterios independientes de aceptación del cierre

Revisión por IA declarada, basada en la guía original de 60 páginas y en el
encargo del propietario. Esta revisión no es una firma humana, una aceptación
docente ni una ejecución de confirmación. Los requisitos estructurados y los
hashes de los insumos constan en INITIAL_REQUIREMENTS_QA.json.

## Ciencia y potencia

- El piloto para estimar la desviación de las diferencias por rival requiere
  al menos cinco semillas (guía, p. 27). La confirmación requiere al menos diez o
  más si lo exige la potencia. No se intercambian las varianzas de la aditiva y V3.
- Se exige potencia de al menos 0,90 por contraste y potencia conjunta reportada.
  La guía no exige conjuntamente 0,90. El producto de potencias no acredita la
  potencia conjunta bajo dependencia: todas las diferencias comparten candidata.
- Tres pruebas temporales distintas deben ser pertinentes a sus supuestos y
  todas deben rechazar. El máximo de sus p ajustados gobierna la decisión.
  Se distingue el número de rivales del número de componentes de p; Holm y el
  dimensionamiento usan una familia explícita y coherente, fijada antes de los
  resultados. Un MCS/diagrama global exigido no se reemplaza por contar métodos.
- Las fechas y las semillas tienen unidades de incertidumbre distintas; los
  errores de siete retornos solapados no se convierten en observaciones iid.
  Las repeticiones de una predicción determinista no añaden semillas independientes.
- Los tres conjuntos adicionales mantienen la misma tarea y cada uno difiere
  en al menos dos ejes. Deben tener procedencia, licencia, versión, hashes y
  disponibilidad previos, y no haber servido para diseño o selección. Un
  calendario expuesto permanece expuesto incluso al reentrenar.
- Las hipótesis,δ,tratamientos,pruebas,R y fechas finales no se inventan para
  rellenar el protocolo. Los faltantes bloquean su ejecución. Una especificación
  prospectiva preparada no se denomina protocolo final preregistrado.
- Se conservan NO-GO, 796 trabajos de la aditiva, 30 ajustes V3, 51 trayectorias y
  20.400 filas retrospectivas, y el 2,70 % adverso. No se promueve automáticamente
  una ablación. R3 no se acredita al completar papeles o poder ejecutar código.

## Archivo y derechos

La guía pide un archivo versionado del repositorio con identificador persistente
(p. 40), así como datos o instrucciones exactas de adquisición y licencias (p. 46).
No fija un tamaño obligatorio de 4,27 GB. El encargo actual sí exige conciliar
el maestro y distinguir cada archivo realmente incluido de los localizadores.

Un archivo completo de fuentes permitidas puede cerrar ese alcance concreto,
con exclusiones jurídicas justificadas e instrucciones exactas para obtener
insumos. No se lo llama archivo científico integral del maestro cuando faltan
bytes de pesos, predicciones o datos que se cuentan como componentes del maestro.
Cada descriptor debe declarar qué se incluye, qué se excluye y qué sólo se localiza.

El ZIP ya depositado conserva exactamente sus 916.861 bytes y su SHA256.
Los campos reservados internos reflejan el empaquetado previo a publicar; un
recibo posterior acredita su publicación. La licencia propia seleccionada del
suplemento no se extiende automáticamente al repositorio completo o a terceros.

## Reproducción y cierre externo

Los niveles A (tablas), B (predicciones aceptadas) y C (entrenamiento) se reportan
por separado. Se reutilizan recibos ejecutados y no se los convierte en una
reproducción humana desde cero. La excepción docente de IA para R1 no sustituye
al tercero que la guía pide en reproducibilidad.

El material para el tercero debe ser ejecutable y trazable: insumos fijados,
comandos, contenedor, semillas, productos esperados, tolerancias previas y comparación
por campo/unidad, con ausencias, no finitos y discrepancias explícitas. La plantilla
queda sin firma hasta que una persona ajena al código ejecute su propio alcance.

Los estados admisibles son ejecutado y verificado, preparado, bloqueado o pendiente
externo. Ninguna validación de sintaxis, fixture o archivo pasa por entrenamiento,
confirmación, publicación o firma humana.
