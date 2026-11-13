# Reanudación E2 después de perder la máquina de Colab

El lanzador `notebooks/resume_e2_posthoc_fresh_colab.py` recupera una campaña
existente desde la rama `results/e2-reduced-delivery-v1`. Requiere A100 y el
secreto ya configurado `GITHUB_TOKEN`. No crea una campaña ni concede horas.

Primero restaura y verifica el diario original de 80 horas y el diario aditivo;
una reserva sin cierre bloquea la continuación. Después reserva 1.800 segundos
para restauración, instala los mismos entornos y recupera los predictores
congelados y los resultados post hoc terminados. La preparación completa corre
en un grupo de procesos aislado con límite global; al detenerla se cierra todo
ese grupo antes de liquidar su coste. Las publicaciones posteriores se cobran
como sobrecoste medido contra el mismo saldo. No se usa Google Drive.

La recuperación mantiene los ocho archivos de la configuración científica
`ae1a2d4f96807a20ea92c313297a5c09512a4817` y los siete del adaptador ambiental
`70316a2aa146a7f647ee36c88080022ec86f8005`. Reconstruye las rutas históricas y
exige los SHA originales del caché, recibo, perfil de paquetes, runtime y prueba
sintética. No modifica los worker_spec ni los recibos de corridas terminadas.
Las admisiones de esta sesión se guardan por separado.

Sólo después de cerrar y respaldar la preparación se ejecuta el controlador
original con su verificación de resultados completados. Los diagnósticos ya
terminados se reutilizan; no se vuelven a entrenar las corridas verificadas.
Una interrupción de entrenamiento conserva su reserva para conciliación.

Validación local: 53 pruebas de software, incluyendo selección de artefactos,
integridad del caché, CLI, conservación del runtime, límites de plazo, señales
y cierre real de un subproceso y su descendiente. No se ejecutó una restauración
real ni entrenamiento de mercado en la A100 privada del usuario desde este
workspace. La primera ejecución real debe realizarse en Colab.

Corrección del ejecutable base: se admite el enlace normal de `venv/bin/python`
con padres regulares y destino ejecutable, preservando la ruta del entorno y
verificando su Python/Torch/A100. El Python limpio conserva `--copies` y los
artefactos mantienen su prohibición de enlaces. Una instalación de paquetes
incompleta puede continuarse únicamente si todavía no tiene recibo ni admisión
ambiental; los resultados y cargos anteriores permanecen intactos.
