# Verificación de la entrega documental

Verificador de sólo lectura preparado con asistencia de IA. No importa ni
ejecuta programas del paquete, grafos, cuadernos, entrenamiento o remuestreos.
No accede a la red ni crea ficheros en el paquete inspeccionado.

El padre integra `verify_packet.py` en la raíz del paquete y prepara
`PACKET_MANIFEST.json` con este esquema:

```json
{
  "schema_version": 1,
  "kind": "e3_resolution_packet_manifest",
  "base_commit": "8d03a73d529eb3f9b2e89992b4488cfeb88a99e6",
  "files": [{"path": "ruta/relativa", "bytes": 123, "sha256": "64 caracteres hexadecimales"}]
}
```

El manifiesto enumera todos los ficheros regulares salvo a sí mismo, incluidos
este verificador y sus recibos. No se ignoran ficheros ocultos ni `__pycache__`.
La generación debe preceder a la publicación y no incluir salidas de su propia
ejecución que modifiquen ese conjunto sin reconstruir el manifiesto.

```bash
python3 verify_packet.py --root /ruta/al/paquete
```

Salida 0: bytes y alcance declarado verificados. Salida 2: rechazo con razón.
También verifica los pins independientes de los manifiestos científicos y C,
los miembros internos, la conservación del NO-GO y del resultado adverso, el
alcance temporal de la primera cápsula de seis ONNX y la plantilla humana sin
firma. Si existen `observations/consolidation_receipt.json` y
`observations/review_release_receipt.json`, valida sus estados, alcances,
quince grafos y correspondencia de hash/tamaño del archivo publicado.

El verificador puede certificar identidad y coherencia de los recibos guardados;
no establece por sí mismo derechos de redistribución, confirmación científica o
reproducción humana. La verificación HTTP de assets corresponde al recibo
ejecutado del runner, no a una nueva descarga de este programa.

Las pruebas utilizan ficheros de software declarados, sin modelos ni datos de
mercado:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p test_verify_packet.py -v
```
