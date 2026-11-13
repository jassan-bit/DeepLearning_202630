# Continuación propuesta de E2: 72 → 80 horas acumuladas

Esta carpeta publica una **propuesta**, no una autorización ya concedida. El
presupuesto activo anterior conserva su límite de 72 horas. La propuesta sólo
puede aplicarse cuando el usuario introduce exactamente `80` en la pregunta de
confirmación de la nueva celda de Colab. Vacío, `0` y cualquier otra respuesta
cancelan la continuación antes de abrir el presupuesto o entrenar.

## Efecto de la confirmación

- El nuevo límite es **80 horas totales**, ocho más que el límite anterior.
- Se conservan todas las horas históricas y todos los cargos del diario actual.
  La ampliación no reinicia el presupuesto ni elimina intentos fallidos.
- Se conservan los modelos, datos, semillas, configuraciones seleccionadas y
  resultados existentes. No se vuelve a ejecutar HPO ni una oportunidad ya
  intentada para sortear el bloqueo.
- El límite por fase sigue siendo 1.800 segundos. Una tarea ordinaria reserva
  1.860 segundos; el LSTM-autoencoder, con tres fases, reserva 5.460 segundos.
- GitHub continúa siendo el único destino persistente. No se monta Google Drive.

En el cierre observado el 13 de noviembre de 2026 a las 07:27 UTC quedaban
aproximadamente 1.876,60 segundos del límite de 72 horas. Confirmar 80 añade
28.800 segundos al saldo existente; no implica que el trabajo vaya a durar ocho
horas ni garantiza que el saldo alcance para todos los análisis.

## Evidencia y conservación

El código registra la respuesta explícita en
`shared_budget/budget_continuation_80_approval.json`. Antes de usar el nuevo
saldo sella el prefijo completo del diario de 72 horas en un nuevo archivo
`shared_budget/budget_continuation_80.json`. El controlador conserva y publica
ambos documentos en el estado de GitHub antes de iniciar nuevas corridas.

Los archivos anteriores `reduced_migration.json`, `funding_extension.json` y
`resource_extension.json` mantienen sus bytes y sus pruebas E/H/P originales.
El adaptador reutiliza sus verificaciones y aplica el nuevo límite sólo a los
registros posteriores al prefijo sellado. Esos registros retienen la autoridad
H anterior y añaden una autoridad separada para esta confirmación de 80 horas.

Una reanudación valida el mismo prefijo y reutiliza la primera confirmación;
no concede otras ocho horas. Una reserva abierta, otro escritor, un prefijo
alterado, un consentimiento modificado o una tarea científica ya intentada
bloquean la continuación. Una interrupción con prueba local incompleta se
mantiene bloqueada para conciliarla; no se borra ni se adopta silenciosamente.

## Comprobaciones

Las pruebas usan diarios y objetos Git temporales, sin leer datos de mercado ni
ejecutar modelos. Comprueban la autorización explícita, las reservas originales,
la conservación de bytes, la reanudación tras superar el antiguo límite de
72 horas, la ausencia de doble crédito, las reservas desconocidas, el bloqueo
de otro escritor y la detección de alteraciones de autoridad y consentimiento.
