El cierre preparó y verificó el índice de la evidencia existente. Conserva el maestro `20261119T031104515520Z` y el piloto V3 `20261120T031424176339Z`; no repite ajustes ni recalibra pesos o intervalos. Se declara asistencia de IA en esta auditoría y sus scripts.

Lo ejecutado y sus límites:

| Comprobación | Evidencia ejecutada | Alcance |
|---|---|---|
| Cadena histórica | Las 193 partes,386 documentos de recibo/almacenamiento y193 enlaces de commits; SHA256 encadenados, Git blob SHA-1 y fuente/protocolo/master verificados | Metadatos íntegros de la cadena. No se descargaron todos los payloads ni se repitió el entrenamiento |
| Inventario histórico |8.582 archivos lógicos;4.268.147.881 bytes declarados;780 marcadores de ajuste | El recibo final verificado declara796 resultados,780 ajustes (776 neuronales y4 ARIMA),4 conjuntos y3 adicionales. No contar resultados de baselines como nuevos ajustes |
| V3 | Snapshot 93 acumulativo, commit 8eaa581bc34b615a8507c9e384a3e4be29084720,259 archivos y2.359.027 bytes declarados; manifiesto Gitblob verificado |30 ajustes de desarrollo. El índice no transfiere resultados del modelo aditivo a V3 |
| Render público | GitHub Actions run 44754550086, job 136350741123, head 6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe; salud200 real tras 11 intentos | `/healthz` respondió 17 sistemas pero omitió `source_commit` y `source_commit_source`. No se inventó identidad; humo V2 no iniciado y aceptación pública pendiente |
| DOI público | En la misma ejecución, ZenodoAPI y registro público404; API: «The persistent identifier is not registered.»; resolver DOI404 «DOI Not Found» | No hay publicación verificable observada a 2026-11-20T14:34UTC. La reserva/borrador privado no equivale a publicación |
| Cliente congelado | Cliente V2 y legado verificados por SHA256;17 archivos de referencia de 5 predictores, pesos ONNX y calibrador E3 validados | Comprobación de referencias/readiness. No sustituye predicciones públicas ejecutadas |

`index/metadata_validation_receipt.json` contiene los resultados verificables. `index/audit193_parts.csv` identifica cada parte y sus hashes. El archivo terminal 193 contiene 35 archivos modificados; **no es un snapshot integral del maestro**. El snapshot 93 de V3 sí es acumulativo. Los 43 chunks JSONL del índice permiten resolver cada ruta al commit exacto que conservó su versión final.

Para validar el índice y planear una hidratación sin descargar datos:

```bash
python hydrate_evidence.py --group all --output /tmp/e3-evidence
python fetch_index_metadata.py
```

El primer comando revisa hashes/tamaños de los chunks y produce el presupuesto. El segundo sólo informa el plan para recuperar los metadatos usados en la validación. Para una selección explícita, por ejemplo los metadatos de admisión históricos:

```bash
python hydrate_evidence.py --group audit796 --select-prefix admission/ \
  --output /tmp/e3-evidence --hydrate --max-bytes 2000000
```

Cada byte recuperado se valida por tamaño/SHA256 y, cuando está declarado, Git blob SHA-1 antes de guardarse. El hidratador no deserializa checkpoints. No publica ni concede derechos sobre las fuentes. Para reconstruir los metadatos originales de la cadena, usar `fetch_index_metadata.py --fetch --max-bytes 6000000`, seguido de `python build_evidence_index.py`. La API pública sin autenticación puede limitar sus 193 lecturas de commit; opcionalmente se utiliza `GITHUB_TOKEN` del entorno sólo para GitHubAPI, sin imprimirlo ni persistirlo. Una recuperación parcial conserva la caché y puede reanudarse. Los comandos con `--hydrate` o `--fetch` están preparados; no se presentan como una hidratación integral ejecutada aquí.

Render requiere intervención concreta del autor porque `autoDeploy: false` y este entorno no dispone de acceso al panel privado. En el servicio `deeplearning-e3-dashboard`:

1. En el repositorio, verificar que el commit final integrado conserva `app.py` con los campos `source_commit`/`source_commit_source`, los cinco predictores servidos y el calibrador E3 fijado. Copiar ese SHA de 40 caracteres; no elegir una variante V3 experimental por su desempeño.
2. Abrir Render → servicio → **Manual Deploy → Deploy a specific commit** y pegar el SHA de 40 caracteres final. Si sólo aparece **Deploy latest commit**, confirmar primero que `main` apunta a ese mismo corte. No cambiar runtime, pesos, escalas, semillas, cuantiles ni variables de entrenamiento.
3. Esperar a **Live** y guardar el SHA de 40 caracteres efectivo mostrado por el despliegue. Si el despliegue conserva otro corte, corregirlo en Render; no alterar el recibo para hacerlo coincidir. Un `source_commit` observado por HTTP no constituye una atestación independiente de la plataforma.
4. Ejecutar **Read-only E3 public Render and DOI probe** desde Actions (`workflow_dispatch`), o la celda de CPU `render_public_acceptance_colab_cell.py` tras copiar el SHA de 40 caracteres efectivo del log de Render. Esta celda usa el helper V2 ya revisado y solicita la credencial sólo en el entorno local de Colab, preferentemente secreto `GITHUB_TOKEN` con `Contents: Read and write`, para guardar el ZIP y el recibo en una rama nueva `e3-public-acceptance-<id>`. No modifica `main`. La rutina de preservación está preparada y comprobada por sintaxis; no se ejecutó aquí contra GitHub con credenciales del autor.
5. Conservar el recibo y las capturas incluso si falla. Sólo `render_public_smoke_passed=true` de una prueba completa de 3 rutas,5 predicciones, callbacks, intervalo E3, explicación, advertencia fuera de dominio y huellas justificará la aceptación automatizada de ese perfil. Un workflow verde sólo indica que la observación terminó; revisar los flags del reporte.

La evidencia Actions se conservó en `github_actions_public_report.json` como recuperación **semántica** del JSON impreso en el log; su serialización local no se confunde con los bytes originales del artifact. `public_probe_execution_receipt.json` fija URLs, IDs, hash original reportado y digest/fecha de expiración del artifact. El log completo incluye la respuesta pública y sus hashes; no contiene tokens. Las observaciones locales de proxy403 anteriores se mantienen por separado y no se atribuyen a Render o Zenodo.

El cierre del DOI está preparado en `zenodo_fields.DRAFT.json`, `CITATION.cff.DRAFT`, `rights_by_source.csv` y `AUTHOR_ACTIONS.md`. Se debe confirmar licencia, autoría real y derechos por fuente, elegir suplemento documental o depósito integral autorizado y publicar los archivos correspondientes. El borrador anterior `e3-source-audit-1bae73b-v0.1.0-DRAFT.zip` sólo contenía 34 fuentes seleccionadas; no archiva este maestro de 4 GB. Tampoco el suplemento de índices actual contiene todos sus pesos/predicciones. No se ha publicado, firmado o relicenciado nada en Zenodo en nombre del autor.
