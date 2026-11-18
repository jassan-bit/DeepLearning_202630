# D01 admitido: archivo y tarea RAW de sesiones

Estado posterior a la revisión: **1 de 3 archivos admitidos, 0 evaluados por
modelos y 2 pendientes**. La decisión es `admission_decision.json` y la
política es `policy.md`.

El informe del usuario, ejecución `20261118T181551067867Z`, permanece sin
cambios en el commit `648f2b91b477b5f4226cad3af5928088accd74ed`, con código
`568aee0445a6eaba0f13c11deb2d6d64aa4f3fbd`. Ese informe registró cero
admitidos antes del cierre de calendario y definición RAW. El estado uno
es una decisión derivada posterior; no se reescribe el reporte original.

Se revisaron los tres miembros del ZIP, el ZIP y su manifiesto: cinco
archivos guardados. ZIP: 3964 bytes, SHA-256
`d1471459e242aa1ff74bd904a2ef5d3c8d5e7a0e4936e615d50fbf3dabe698cc`.
Su manifiesto tiene SHA-256
`82f023763cbe819dd2c2dce743ba39073c55f60f8ee7f7d251762a4a1119e974`.
El informe técnico está embebido en dataset_audit.json; el nombre legado
technical_review.json de su procedencia corresponde al archivo versionado
como technical_snapshot.json en el código ejecutado, no a otro miembro
que debiera existir en el ZIP.

D01 conserva el archivo inmutable WIKI/AAPL distribuido por Bokeh y su
permiso declarado de copia y redistribución en dominio público. Tiene
3270 fechas únicas/ordenadas y cero defectos OHLCV básicos. Las 3270 fechas
coinciden en orden con exchange-calendars 4.11.3, NASDAQ alias XNYS,
sin sesiones faltantes ni adicionales. Se comprobaron cierres de 9/11,
Reagan, Ford y Sandy; el alias no constituye un calendario independiente
ni una certificación de horarios intradía.

La política RAW_KEEP_ALL conserva precios y volumen sin ajustar, sin
eliminar ni rellenar filas y sin usar Adj Close. El objetivo es sigma7 de
precio cotizado bruto por sesión, no volatilidad económica ajustada por
acciones corporativas. Incluye sus efectos mecánicos como limitación.
No se etiquetan ni eliminan saltos automáticamente y no se observaron
resultados de modelos para elegir esta política.

## Comprobar sin instalar paquetes ni ejecutar modelos

```bash
python experiments/e3_d01_admission_20261118/verify.py
```

El verificador compara los datos y la instantánea de sesiones fijados
por SHA-256 y comprueba el contrato de admisión. La referencia original
se generó usando exchange-calendars==4.11.3 en un target aislado de /tmp,
con pandas 2.2.3, NumPy 2.3.5, pyluach 2.3.0, toolz 1.0.0,
korean-lunar-calendar 0.3.1 y tzdata 2026.5. calendar_check.json registra
versiones, hash de wheel comprobado contra PyPI y cierres extraordinarios;
source_manifest.json registra las fuentes del calendario contrastadas
byte a byte con ese wheel. No se alteró el entorno GPU del proyecto.

Admitir este archivo no habilita la campaña de modelos. Faltan D02/D03,
particiones, tratamiento, presupuesto, potencia y protocolo final. E2/E3
y sus checkpoints históricos existen y se conservan. No hubo nuevas
inferencias, ajustes, contrastes o valores de objetivos, y no se solicita
reentrenar automáticamente. R1/R3, defensa humana, reproducción externa,
DOI y aceptación pública de Render no quedan acreditados por este paso.
