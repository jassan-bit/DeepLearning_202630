# Verificación del paquete de nueva evidencia

Herramienta estándar Python 3.12, sin dependencias externas. Revisa el inventario exacto, tamaños y SHA256 de `PACKET_MANIFEST.json` y de ocho componentes. Calcula hashes por bloques; no deserializa checkpoints, ejecuta modelos, consulta servicios ni vuelve a hacer simulaciones.

Desde la raíz **del paquete público preparado**, ejecutar:

```bash
PYTHONDONTWRITEBYTECODE=1 python bundle_tools/verify_packet.py --packet . --output /tmp/e3-next-packet-verification.json
```

La salida `PASS` certifica estas comprobaciones de software, no la adecuación científica, la publicación remota ni la reproducción humana. El informe se escribe fuera del paquete inmutable. La carpeta de trabajo privada contiene archivos deliberadamente excluidos: no debe entregarse ni usarse como si fuera el paquete público.

El manifiesto raíz debe declarar `schema_version: 1`, una lista `files` de objetos `{path, bytes, sha256}` y una lista `components` de objetos `{manifest, sha256}`. Se excluye a sí mismo del inventario. Las rutas son relativas y canónicas; se rechazan miembros inesperados, enlaces simbólicos, duplicados, escapes de directorio y payloads de datos brutos, checkpoints, wheels o cachés. El manifiesto humano v1 y su suplemento v1.1 se verifican como componentes separados; el hijo no modifica el inventario congelado del padre.

Los SHA de los cinco componentes científicos/humanos ya congelados están fijados en el verificador. Planning, container_profile y bundle_tools se comprueban contra sus manifiestos y el inventario raíz. La procedencia pública se acredita por los recibos y commits previos; esta herramienta no sustituye la comprobación independiente de esos anclajes ni constituye una firma criptográfica de autoría.

Además se comprueba el alcance de las declaraciones actuales: **20 ajustes neuronales nuevos de desarrollo**, 1.210 predicciones guardadas, 22 trayectorias, seis recetas, decisión de abandonar V4 y ninguna promoción; conservación declarada de 796 trabajos y 30 ajustes anteriores; 1.024 réplicas controladas, 27/128 rechazos EPA en la extensión MCS y calibración no aprobada; cero conjuntos futuros admitidos; caso Docker de la fuente fijada `f5c960e…`, tablas principales 76+7 y persona externa pendiente. No se impone falsamente cero ajustes nuevos al trabajo global.

Las 17 guardas de corrupción usan exclusivamente archivos artificiales temporales:

```bash
PYTHONDONTWRITEBYTECODE=1 python bundle_tools/test_verify_packet.py
```

Estos programas y su revisión se prepararon con asistencia de IA declarada. La revisión automática no acredita R1, R2, R3 ni una ejecución humana independiente.
