E3: cierre de figuras de la campaña ya calculada

El registro y los recibos publicados muestran 796 trabajos completos, 76 paneles,
1.874.560 filas y el análisis estadístico terminado. Falló solamente la importación
de Matplotlib: el Python aislado heredó de Colab
`MPLBACKEND=module://matplotlib_inline.backend_inline`.

Este adaptador selecciona `MPLBACKEND=Agg` antes de iniciar el proceso de figuras.
Requiere la campaña original A100-SXM4-80GB y el master
`20261119T031104515520Z`. No inicia otro master. La ayuda de IA se declara.

El verificador exige los hashes exactos de los recibos y manifiestos guardados,
comprueba los 76 paneles y omite completamente el trabajador de entrenamiento.
El analizador científico original valida sus entradas y reutiliza su recibo final;
no repite los 9.999 remuestreos. El generador original exporta 16 figuras en PNG y
SVG. Sus 32 archivos se comprueban antes de que el controlador original marque
la etapa como completada y la guarde con el publicador Git PACK ya probado.

Durante este cierre, cada vista de persistencia se valida y se crea con los bytes
completos del productor. Así una interrupción no sustituye su manifiesto por la
anotación de una vista de entrenamiento en curso. No se omiten los controles del
publicador original ni se modifica el protocolo o el código científico fijado.

Ejecutar `resume_cell.py` en la sesión de Colab existente. Conserva el secreto
`GITHUB_TOKEN` en la interfaz de secretos de Colab; no incluirlo en el cuaderno.
El script reutiliza el publicador inmutable de GitHub, con su verificación SHA256.

Validación: 29 pruebas de finalización y alteración, más una regresión aislada
que reproduce el error exacto y genera 32 imágenes con tablas sintéticas marcadas.
Los resultados de esa regresión son pruebas de software; no son resultados de
mercado ni acreditan R1/R3. El preparador no ha ejecutado esta celda en la GPU
del usuario. R1/R3 siguen pendientes de sus criterios científicos.

Para repetir las pruebas, apuntar `E3_SCIENTIFIC_SOURCE` a un checkout del commit
`5a07d6157a9796ca54416dc5dea1e1527e731597` y ejecutar:

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

La regresión `regression_backend.py` requiere un Python aislado con
Matplotlib 3.10.8 y NumPy 2.3.5, sin `matplotlib_inline`. Seleccionarlo mediante
`E3_PLOT_TEST_PYTHON`; todas sus salidas están marcadas como sintéticas.
