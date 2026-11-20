# Adecuación retrospectiva a partir de predicciones guardadas

Este paquete reúne evidencia existente y calcula diagnósticos auxiliares, sin
entrenar modelos de pronóstico, ejecutar inferencia o repetir remuestreos. Se
declara asistencia de IA en su código, análisis y redacción.

## Alcances que se mantienen separados

1. **Auditoría ampliada de 796 trabajos.** Se verificaron los hashes de las doce
   tablas originales. De ellas se extraen razones ajuste/validación/calibración/
   prueba, comparación con controles triviales, cobertura por trimestre y
   disponibilidad de recursos. Los cuatro paneles del candidato se localizaron
   mediante metadatos Git y se descargaron por sus rutas inmutables: 21.408.138
   bytes de predicciones guardadas, sin pesos. Sus cuatro SHA-256 coinciden con
   `panel_manifest.json`. Sólo se archivan los puntos de **prueba limpia**:
   18.080 filas, cuarenta trayectorias, diez semillas por conjunto. Sus cuarenta
   RMSE y MAE coinciden con las métricas originales dentro de 10⁻¹⁵.
2. **Antecedente histórico de 608 fechas.** Se conserva el archivo ya público
   del commit `510c3bddfdc9926322e3210ac61ee1f5eeffad2e`, con SHA-256
   `b82ba0a8a0371ea5643d9ff666a3ac5a1a4e91018b95a640bd88b1be50bb5133`.
   Sus diez semillas de otra ejecución no se atribuyen a los pesos nuevos de
   la auditoría ampliada.
3. **V3 queda fuera de este paquete.** Sus treinta ajustes tienen recibo,
   calendario y diagnósticos separados. Ningún resultado se transfiere entre
   esas ejecuciones por compartir arquitectura o subyacente.

El archivo `archive/input_manifest.json` fija las rutas, versiones, hashes de
origen, filtros y compresión. Los extractos gzip son copias verificables de
predicciones ya publicadas; no añaden una licencia ni ratifican derechos de
redistribución de las fuentes originales.

## Métodos y límites

Los errores se definen como objetivo menos pronóstico. Ljung–Box se calcula a
rezagos 7 y 14; ARCH-LM, a 7; Mincer–Zarnowitz contrasta conjuntamente intercepto
cero y pendiente uno con HAC Bartlett de rezago 14 y corrección HC1. Los rezagos
son **observaciones**: sesiones en D01 y días en las series cripto. Los objetivos
móviles comparten retornos; no se suponen observaciones iid ni diez mercados
independientes por disponer de diez semillas.

La familia exploratoria histórica contiene cuarenta valores p diagnósticos y
la ampliada ciento sesenta; cada una tiene su propio Holm, fijado antes de
calcular estos diagnósticos retrospectivos. No se afirma control global sobre
doscientos contrastes, pruebas de superioridad R3 ni preregistro prospectivo.
La validez de los valores p continúa condicionada a los supuestos asintóticos,
de momentos y dependencia pertinentes. Holm no repara un p marginal inválido.

Jarque–Bera se usa sólo como estadístico de forma, acompañado de asimetría y
curtosis. No se interpreta su p iid bajo dependencia serial ni se exige
normalidad residual para considerar útil un pronóstico de regresión. Una
falta de rechazo no certifica calibración, ausencia de dependencia o adecuación.

Las razones fuera/dentro de ajuste son descriptivas y comparan calendarios y
regímenes de volatilidad distintos. Una razón menor que uno no demuestra
ausencia de sobreajuste: no existe aquí un margen de brecha elegido
prospectivamente ni un análisis de potencia que permita aprobarlo.

## Resultados que modifican la interpretación

- Las razones RMSE de prueba/ajuste son 0,9364; 0,2945; 0,7802 y 0,6716 en
  D00–D03. El cambio de escala del objetivo entre periodos limita su lectura.
- Frente a `floor_only`, la MAE del candidato es **5,63 % mayor en D00 y
  26,59 % mayor en D01**, aunque su RMSE es menor. El objetivo de optimización
  y la métrica de evaluación importan.
- La familia ampliada muestra evidencia asintótica de heterocedasticidad en
  D00 y de desviación de la calibración lineal Mincer–Zarnowitz en D01. No son
  resultados universales de incapacidad del modelo ni pruebas de R3.
- Las coberturas medias globales no garantizan cobertura por trimestre. El
  mínimo trimestral descriptivo de D01 es 82,10 %, frente a nominal 90 %.
- Los 900 registros de recursos representan **796 trabajos**, con fases
  adicionales del autoencoder. Tres segmentos conservan cómputo activo
  desconocido tras una interrupción. El máximo exceso de plazo, 35.283,63 s,
  incluye tiempo de desconexión: **no son 9,8 horas de ejecución GPU**.
  FLOPs, energía y paridad de cómputo real total no se inventan.

Todos los valores, favorables y adversos, se conservan. `closure_matrix.csv`
indica qué se puede cerrar documentalmente y qué requiere evidencia nueva o
decisiones previas. R1 y R3 siguen sin acreditarse por este reanálisis.

## Reproducir sin red ni modelos

Entorno observado: Python 3.12.14, NumPy 2.3.5 y SciPy 1.17.0. La verificación
independiente utiliza además statsmodels 0.14.6. No se importa PyTorch.

Desde la raíz del repositorio:

```bash
python experiments/e3_evidence_close_20261120/adequacy/audit_existing_predictions.py \
  --verify-only \
  --source-dir book_e3/_static/e3_audit_completed_20261119T234320461006Z
python experiments/e3_evidence_close_20261120/adequacy/verify_diagnostic_reference.py
```

El primer comando comprueba los bytes de entrada/salida y recalcula en un
directorio temporal. Compara 256 registros y 3.292 celdas numéricas con tolerancia
relativa 10⁻⁹ y absoluta 10⁻¹² —10⁻¹⁵ para p—, manteniendo identidades y archivos
de origen exactos. El segundo verifica 250 comprobaciones numéricas contra
implementaciones independientes de statsmodels, además de 200 valores ajustados
por Holm. No certifica los supuestos estadísticos por concordancia de software.

Para reconstruir los extractos desde las fuentes originales, los dos colectores
verifican los hashes antes de copiar o filtrar. `fetch_candidate_panels.py` es
opcional: sólo descarga los cuatro CSV inmutables, no se ejecuta en CI y nunca
obtiene checkpoints. La compresión conserva el hash descomprimido del CSV.
