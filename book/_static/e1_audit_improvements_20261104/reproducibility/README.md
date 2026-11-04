# Reproducción aislada validada — cierre A05

Se actualizó `BUILD.md` al ciclo real de cuatro entradas, ajuste final y test ya completados. `scripts/restore_e1_private_artifacts.py` prepara un paquete privado e instala una réplica aislada; valida los seis TAR y cada miembro antes de restaurarlo, conserva todos los marcadores y rechaza sobrescrituras.

La ruta se probó desde el paquete privado, sin depender del checkout original: **113 insumos**, **6 archivos de retención**, **16.890 miembros restaurados**. El auditor independiente dio **PASS** en la copia, con 13.871 hashes y 7.680 BDS verificados. No se entrenaron modelos, ejecutaron nuevos pronósticos ni abrieron ZIP brutos de mercado.

Los recibos y el resumen contienen únicamente cantidades, firmas y el resultado de verificación. Los paquetes completos se conservan fuera del repositorio y del sitio público. La entrega a un evaluador usa un canal privado autorizado y la estructura descrita en `BUILD.md`; esta comprobación no convierte el backtest ya observado en una evaluación nueva.

Diez pruebas sintéticas verifican aislamiento, preservación de resultados, hashes, rutas, destinos públicos/otros checkouts, symlinks, duplicados y miembros faltantes.
