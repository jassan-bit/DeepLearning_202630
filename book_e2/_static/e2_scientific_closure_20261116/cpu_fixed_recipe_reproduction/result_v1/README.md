# Réplica científica CPU con recetas fijas: evidencia final

Ejecución posterior nueva: 78 ajustes supervisados reales (13 cuerpos ×6 semillas), tres reconstrucciones clásicas, calibración53, congelamiento81, admisión20ZIP y evaluación608. Las dos etapas compartidas completan83recibos. Se conserva la exposición histórica al test; no se recupera ceguera prospectiva.

`analysis/` copia la nueva A2 y el análisis primario de20000remuestreos ya ejecutado. Este exportador no ajusta modelos, no hace forwards y no calcula métricas ni contrastes. `predictions/` conserva los81JSON originales. `predictions_81.npz` deriva sus valores exactos; los targets provienen del NPZ auditado, unidos por608fechas y fuente diaria idéntica. El descriptor documenta los hashes y el origen separado de los targets.

El coste cobrado es la suma de83workers incluyendo importaciones;90tiempos de fase están contenidos en esos workers. No sumar fases o walltime del padre al cargo. Los fallos previos se conservan:dos de dependencia(5.254804967s) y,cuando aplica, ARIMA por el alias del recibo(3.327804276s). V4 reutiliza80workers verificados deV3 y sólo ejecuta ARIMA+score+primaria:no son83workers nuevos en el retry. Estos fallos se añaden al coste del estudio, sin reiniciar el presupuesto84h. No se redistribuyen etapas compartidas por modelo.

Los79checkpoints binarios se comprobaron localmente por SHA256 antes de excluirlos; el inventario declara hashes y tamaños. El profesor no puede verificar bytes omitidos ni reconstruir esos pesos usando sólo este export. Los dos predictores ingenuos JSON sí se incluyen. La reproducción de ajustes requiere código, fuentes, datos y cómputo del paquete correspondiente.

Alcance: réplica CPU con recetas históricas fijadas, sin repetirHPO; NAS no aplica a cuerpos definidos. No implica paridad A100/bitwise, reproducción del HPO original, IUT de coste/latencia para estos modelos nuevos ni cierre de toda la campaña original.
