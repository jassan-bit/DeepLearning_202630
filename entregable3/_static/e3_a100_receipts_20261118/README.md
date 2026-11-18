# Primera ejecución A100 persistida y revisada

El 18 de noviembre de 2026 se revisaron los recibos guardados por la persona
autora en la ejecución `20261118T172624131741Z`. Código ejecutado:
`1ff136f71c0ffcd1dbd918efbc078a8d9653ed21`. Resultados inmutables:
`47d2f316a50d9268cb6484a547e8a2f0cc0413eb`, bajo
`runs/colab_preflight/20261118T172624131741Z/`.

Se comprobaron por bytes los 40 archivos internos del ZIP, los 41 artefactos
del manifiesto y el propio manifiesto: 42 archivos guardados. El ZIP tiene
43.860 bytes y SHA-256
`44b7fb93a3f9f667bbe3cd7b6a8d4b1a190c0742ac62e78c0dbd127dfa62fd31`.
Su manifiesto tiene SHA-256
`61dc57e55c3385c398bf663754320d8dc53f46d15345c17be9762f172a2203c7`.
La rama conserva todos los 10.808 blobs de su fuente y sólo agrega los 42
archivos esperados. Su commit tiene como único padre la versión ejecutada.

Los 17 recibos individuales están aprobados y coinciden con el agregado.
Catorce redes declaran ejecución en cuda:0, backward y recarga idéntica,
con 456 tensores de gradiente finitos reportados. Tres controles usan CPU;
ARIMA hizo un ajuste exclusivamente sintético, con orden [2,0,1]. E3 es
InnovationModel con additive_positive_head y 1210 parámetros. Hardware:
NVIDIA A100-SXM4-40GB; Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6.

Las siete fuentes revisadas, el helper CUDA, el lock de 43 paquetes y el
script ejecutado coinciden con sus hashes remotos. `source_audit.json`
conserva esa comprobación asistida por IA. `model_peer_audit.json` documenta
la consistencia de los recibos y las 13 recetas neuronales heredadas.
`per_model.csv` resume cada perfil sin presentar desempeño de mercado.

## Repetir la comprobación de integridad y recibos

Desde un checkout con estos archivos, sin Torch, CUDA ni red:

```bash
python experiments/e3_a100_receipts_20261118/verify.py \
  --inputs experiments/e3_a100_receipts_20261118 \
  --output /tmp/e3-a100-verification-new
```

Para obtener los dos insumos desde el commit inmutable en una carpeta nueva,
añadir `--download` y elegir otro `--inputs`. El ejecutor comprueba los SHA
fijados, los miembros del ZIP y la coincidencia de los 17 recibos; regenera
`audit.json` y `per_model.csv`. No reconstruye modelos. El directorio de salida
debe ser nuevo para conservar recibos anteriores. La marca temporal cambia
al repetir; los campos técnicos y el CSV conservan sus valores.

## Alcance

La revisión comprueba archivos y consistencia de recibos. No se volvió a
usar GPU ni se reejecutó backward o recarga desde los pesos: los tensores
y checkpoints .pt no están en este paquete. Esas operaciones se documentan
por los recibos y el código cuyo hash se comprobó, no por una nueva ejecución
independiente. No acredita reproducción humana ajena al código, entrenamiento
completo, novedad R1 o superioridad R3.

La ejecución declara cero ajustes con mercado, cero predicciones de mercado
y cero contrastes nuevos. El guard mantiene la campaña real bloqueada:
cero datasets adicionales admitidos, manifiesto/tag previo y tratamiento,
presupuesto y potencia finales pendientes. La sesión L4 anterior conserva
su consola reportada; su ZIP perdido no fue recuperado por esta ejecución.
