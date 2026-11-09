# Evidencia sintética de las nuevas fábricas NAS

Ver [capítulo 28](../../sections/28_fabricas_nas.md) y summary.json. Los JSONL gzip contienen todos los casos; se verifican con SHA256 comprimido y descomprimido. No se ejecutó ningún optimizador ni se leyó mercado. El recorrido numérico usa un trial estructural registrado por familia (TimesNet: dos kernels); la geometría de seis kernels se prueba por meta y el extremo también numéricamente. No se afirma el producto18×1000.

Los checkers reproducibles están en scripts/check_e2_nas_factory.py y scripts/check_e2_nas_auxiliary.py. El entorno CPU está fijado en requirements-nas-cpu.lock.txt. CI ejecuta35 pruebas de mecanismos e integridad; no repite toda la enumeración numérica en cada publicación.

