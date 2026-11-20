# Seguimiento científico V3, 20 de noviembre de 2026

Decisión ejecutada: **NO-GO para confirmación de la receta actual; R3 no acreditada.** Paquete posterior de auditoría asistida por IA, independiente de los archivos congelados anteriores. No es un prerregistro ni una evaluación humana.

- `SCIENTIFIC_DECISION.md`: razones, resultados adversos y límites de potencia.
- `decision.json`: estado operativo y condiciones de reapertura, sin ruta de entrenamiento.
- `requirements_matrix.csv`: requisito, evidencia, versión, comprobación, estado y faltante concreto.
- `guide_source.json`: identidad de la guía suministrada y páginas examinadas.
- `input_manifest.json`: 65 entradas anteriores fijadas por SHA-256.
- `audit_saved_science.py`: comprobación de hashes y aritmética de resultados guardados con la biblioteca estándar.
- `execution_receipt.json`: comprobación realmente ejecutada, duración y coste.
- `REOPENING_CHECKLIST.md`: material concreto para una eventual nueva decisión, sin simular autorización.

Desde la raíz del repositorio:

```bash
python experiments/e3_science_followup_20261120/audit_saved_science.py --root .
```

No se cargan modelos ni se vuelven a producir predicciones. Se comprueban quince trayectorias neuronales de desarrollo y 51 trayectorias de transferencia previamente guardadas. La referencia aditiva conserva sus 796 trabajos y el piloto V3 sus 30 ajustes; no se cuenta una semilla como un diseño nuevo.
