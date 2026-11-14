# Adjudicación focal de literatura E2, 14 de noviembre de 2026

Se adjudicaron individualmente los **30 objetivos identificables de recuperación dirigida**, a partir de registros y fichas primarias ya documentados. El corte contiene **13 incluidos aplicados**, **2 retenidos sólo como contexto de mecanismo con elegibilidad aplicada pendiente**, **7 exclusiones** y **8 sin informe primario completo acreditado en los recibos examinados**. No se completó la revisión global de 9.559 grupos, la cohorte de 1.078 ni los 40 candidatos originales.

Los 13 incluidos se separan en **11 estudios de volatilidad derivada de retornos** y **2 de volatilidad implícita/condicional**. No son 13 estimaciones combinables del mismo efecto. Ninguna fuente acredita eficacia transferible al contrato E2 σ7, D7×5, h1. Las dos fuentes con constructo no conciliado **no cuentan como inclusiones aplicadas**.

El trabajo sustantivo es la decisión por fuente y el enlace de evidencia disponible entre entregables: varias fuentes que un recibo E2 marcaba parciales o no recuperadas tenían secciones primarias o un cuerpo original ya documentados en E1. Esos enlaces recuperan evidencia para la adjudicación; no son nuevas descargas ni nuevas lecturas de artículos. Cinco intentos nuevos de acceso fallaron con proxy403 (`access_attempts.json`).

Archivos principales:

- `applied_target_decisions.csv` / `.json`: 30 decisiones individuales, motivo, rol, versión, target, arquitectura, métricas disponibles, límites y localizadores.
- `focused8_decisions.csv`: los ocho informes con fichas E2 públicas; 6 incluidos aplicados, 1 contexto con constructo pendiente y 1 excluido del núcleo DL.
- `mechanism_source_decisions.csv` / `.json`: 9 estudios primarios de los 10 sistemas vigentes, más 4 originales históricos como comparadores. Fuentes de mecanismo, sin sumarlos a eficacia aplicada.
- `historical43_source_roles.csv` / `.json`: rol asignado a las 43 fichas primarias E1 enlazables por DOI a grupos E2; incluye otros targets, comparadores y dudas de constructo. Se solapa con el corte de 30 y no debe sumarse a éste.
- `protocol.json`, `flow.json`, `validation.json`: criterios retrospectivos, identidades contables y verificaciones.
- `scope_audit.md`, `focused_synthesis.md`, `proposal_literature_insert.md`: alcance, síntesis crítica y propuesta de texto.
- `evidence_manifest.json`: hashes de las entradas públicas efectivamente consultadas. Un hash de ficha no acredita relectura del original.

Regeneración portable, con la biblioteca estándar de Python:

```bash
python build_eligibility.py --root /ruta/al/repositorio
```

Si el script está dentro del repositorio, puede encontrar un ancestro con `book_e2/` y `references/`; `--output` permite generar en otro directorio. Los criterios y decisiones están explícitos en el script: no ejecuta cribado automático de 9.559 títulos. No exige cinco semillas, código o receta completamente reproducible para la admisibilidad narrativa. Esos elementos permanecen como calidad NR/NV o riesgo de sesgo.

Una única adjudicación asistida por IA, sin doble revisión humana independiente y sin κ. No se modificaron capítulos ni artefactos publicados por esta tarea.
