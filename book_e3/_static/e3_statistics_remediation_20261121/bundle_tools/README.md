# Verificación offline del suplemento estadístico

Este verificador preparado con asistencia de IA comprueba archivos, tamaños,
SHA-256, pertenencia exacta y alcance de las afirmaciones. No importa los modelos,
no vuelve a calcular los diagnósticos y no ejecuta Monte Carlo.

Desde la raíz del repositorio publicado:

```bash
python experiments/e3_statistics_remediation_20261121/bundle_tools/verify_package.py \
  --packet experiments/e3_statistics_remediation_20261121
```

`PASS` significa integridad y coherencia del alcance. Una ejecución controlada
detenida por presupuesto puede conservarse como evidencia parcial y recibir
`PASS` de integridad; el informe mantiene `resource_stop` y no la presenta como
validación estadística completa. El verificador no acredita calibración a nivel
nominal, potencia, tres procedimientos primarios, R3 o una ejecución humana.

Puede escribirse un informe opcional con `--output` únicamente fuera del paquete.
Las guardas usan metadatos sintéticos mínimos para rechazar corrupción, archivos
extraños, cambios de fuente, muestras parciales promovidas, pérdida de hipótesis,
predicciones/calibradores inventados y falsas atribuciones de defensa observada.

```bash
python -m unittest discover \
  -s experiments/e3_statistics_remediation_20261121/bundle_tools \
  -p 'test_*.py' -v
```
