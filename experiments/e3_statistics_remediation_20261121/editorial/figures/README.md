# Visualización de la validación controlada nueva

Asistencia de IA declarada. Esta figura muestra los cuatro escenarios previstos,
en su orden original, y las tres variantes globales sobre los mismos sorteos.
Lee exclusivamente el plan y el resumen ya guardados: no recalcula valores p,
intervalos, réplicas DGP o bootstrap.

`rendered/mcs_fresh_global_rates.png` y el SVG contienen las tasas observadas y
los IC95 % binomiales **individuales** reportados. Sólo los tres casos EPA nulos
muestran el nominal0,10 y el cribado previo del límite superior≤0,13. La alternativa
muestra rechazo global bajo desigualdad; no es tamaño, potencia de tres pruebas
ni una decisión pareada frente a cada rival. Las variantes no son tres métodos
primarios distintos.

Para volver a representar los resultados, preservando la salida original:

```bash
XDG_CACHE_HOME=/tmp/e3-stat-figure-cache MPLCONFIGDIR=/tmp/e3-stat-figure-mpl \
  python editorial/figures/plot_validation_rates.py \
  --plan calibration/VALIDATION_PLAN.json \
  --summary calibration/validation_fresh/validation_summary.json \
  --out /tmp/e3-stat-figure-display-new
```

La representación original usó Matplotlib3.10.8 con Agg y terminó con exit0.
Fontconfig informó que su caché predeterminada no era escribible; el PNG fue
inspeccionado y las etiquetas, cifras, barras y leyenda se renderizaron correctamente.
El recibo identifica el plan, resumen y programa por SHA256. La configuración
SVG elimina fecha dinámica y fija la sal de identificadores; no se promete
igualdad binaria entre versiones o plataformas distintas de Matplotlib/fuentes.
