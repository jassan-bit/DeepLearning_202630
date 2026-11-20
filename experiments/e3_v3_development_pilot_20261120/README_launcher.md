El lanzador prepara un piloto de desarrollo nuevo y separado del master de 796
trabajos. No realiza llamadas de red al importarse. La fuente siempre se fija
mediante un SHA de Git completo que proporciona la celda de arranque publicada.

API de la celda de Colab:

```python
main(SOURCE_COMMIT)  # primera ejecución; continúa el mismo run local si existe
```

Tras reiniciar Colab, se usa exactamente el identificador que se imprimió al
iniciar y el mismo SHA de fuente:

```python
main(SOURCE_COMMIT, resume_run_id="IDENTIFICADOR_IMPRESO")
```

Si detecta ramas V3 previas y no dispone de una sesión local ni un identificador
explícito, se detiene para evitar iniciar accidentalmente otro piloto. La opción
`new_run=True` sólo se utiliza cuando se decide expresamente iniciar otro run y
no existe una sesión local activa. Una GPU/software diferente detiene la
continuación: el perfil inicial se conserva y no se reinicia el presupuesto.

El secreto `GITHUB_TOKEN` se consulta en los secretos de Colab. Si falta, se pide
con entrada oculta. Requiere acceso al repositorio y Contents: Read and write.
El token no se escribe en archivos, URLs, argumentos ni configuración de Git.
Sólo un hijo Git de red lo recibe por entorno mediante askpass temporal. El
proceso científico y la instalación de CUDA reciben entornos sin credenciales.

Cada run usa exclusivamente la rama `e3-v3-development-<id>` y el directorio
`runs/e3_v3_development_pilot/<id>`. Los snapshots consecutivos conservan todos
los archivos previos, manifiestan tamaños y SHA256 y verifican el ref remoto.
La publicación usa Git PACK y una comparación del HEAD esperado; conflictos y
resultados de escritura inciertos detienen la ejecución conservando evidencia.
El código fuente, protocolo anterior y otras ramas no se modifican.

El plan, la procedencia, el perfil GPU/software y los hashes de código se guardan
y verifican en GitHub antes del primer ajuste. Hay barreras de guardado antes de
cada ajuste, después del primer checkpoint validado y al terminar cada trabajo.
El motor conserva checkpoints locales por época. El plazo absoluto de 120 s por fase de
optimización incluye el tiempo de guardado; no se amplía al reanudar. La restauración
recupera sólo bytes efectivamente publicados. Si Colab pierde el disco antes del
primer checkpoint remoto y el plazo expira, el trabajo se declara incompleto;
no se inventa un estado ni se repite con un presupuesto nuevo.

Sólo se abren tres miembros `training_*` del NPZ fijo. Las particiones de test y
calibración no se cargan. El piloto usa desarrollo BTC previamente expuesto:
532 filas de ajuste, 54 de selección, 7 de purga y 55 de evaluación del piloto. La
evaluación espera a que terminen las 15 recetas HPO y los 15 ajustes finales.
Sus resultados no certifican R1, R3 ni superioridad confirmatoria.

Las pruebas de persistencia usan repositorios bare locales reales, filtrados sin
blobs. Cubren publicación incremental, restauración en caché nueva, conflictos,
archivos perdidos, rechazo de credenciales crudas/comprimidas y ACK perdido.
No se ejecutó entrenamiento con datos reales ni una GPU de Colab en Codex.
