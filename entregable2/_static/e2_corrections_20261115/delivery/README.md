# Lectura actual de reproducibilidad y contrastes

Este directorio reúne valores existentes sin ejecutar estadísticas. Los 21 literales de B y los veinte p primarios quedan conservados.

A3 añade d_z e IC de semillas publicados posteriormente: la incertidumbre es condicional a seis pares y al calendario. No cambia Holm20, la ganancia o la decisión primaria. RQ2 conserva otra familia exploratoria.

Los resultados de brecha/RQ3 sólo se incorporan si existen resumen final, recibos/hash coherentes y las 78 paridades a tolerancia 1e−7. Un fallo o CSV parcial queda como procedencia, nunca como resultado completo. RQ3 no se convierte en una condición universal de B11.

El recibo de los 12 ZIP se conserva como snapshot de su versión pública; no es replay integral de entrenamiento. No se calcula porcentaje de cumplimiento o nota.

Repetición documental en una carpeta NUEVA:

```sh
python scripts/export_e2_corrections_readouts.py --output /tmp/e2-corrections-readouts-new
```
