# Ampliación gratuita: siete fuentes TCN y seis MLP

La búsqueda web complementaria del 07/11/2026 reúne trece obras distintas:
tres ya estaban identificadas en E1 y diez se incorporan como referencias nuevas.
Son nueve artículos de revista, tres publicaciones de congreso y un preprint
(Ge et al.). La identidad IEEE CAI 2024 de Lensky–Hao se verificó al finalizar:
el DOI IEEE y la versión arXiv corresponden a una sola obra.

## Acceso y lectura

Doce de las trece obras seleccionadas cuentan con fuente primaria completa
consultada: tres lecturas anteriores y nueve nuevas. Vo (2025) es acceso abierto
editorial; las secciones y tablas indexadas se consultaron, pero las aperturas
automatizadas devolvieron HTTP 429. Se conserva como lectura primaria parcial,
sin inventar descarga, texto íntegro leído o licencia de reutilización de archivos.
Se publican únicamente metadatos y enlaces: ningún PDF, resumen exportado,
contacto, texto íntegro o contenido privado se redistribuye.

## Papel de cada arquitectura

Las cuotas siete/seis organizan esta selección dirigida; no representan frecuencia
de uso en toda la literatura. Las fuentes de VIX, bolsa, divisas y materias primas
son transferencia financiera, separada de volatilidad realizada/histórica cripto.
Nguyen estudia CVI implícita y conserva otros componentes del modelo al sustituir
LSTM por TCN. Agal utiliza TCN dentro de FinAnalyzeNet: mencionar una TCN base
sin recuperar su receta ni fila numérica no añade un voto autónomo básico.

Las tres lecturas cripto nuevas verifican MLP y también otras bases: García-Medina
y Aguayo-Moreno, MLP/LSTM; Korkusuz y Sahiner, MLP/LSTM/RNN simple y RBFN/GFN/MFN
como tipos distintos; Lensky y Hao, MLP/LSTM/CNN. Las variantes y cabezas internas
no añaden estudios. Lensky conserva reporte incompleto de la receta MLP y queda
fuera de la sensibilidad estricta, aplicando la misma regla a fuentes antiguas y
nuevas. Korkusuz describe otro protocolo que Brauneis–Sahiner: diferente fuente,
frecuencia, activos, horizonte y ventana de evaluación. No se documentó reutilización
de ajustes o pronósticos; esto no acredita independencia estadística de mercados.

## Productos y reproducción

- `inputs.json` conserva adjudicación, versiones, calidad, datos desconocidos y procedencia.
- `readings.json` contiene las mismas trece fichas, compatibles con el compilador acumulado.
- `references.csv` y `references.bib` contienen autores, DOI y rutas gratuitas.
- `search_log.json` conserva consultas web efectivamente comunicadas por los investigadores.
- `summary.json`, `validation.json` y `manifest.json` verifican unidades e integridad.

`python scripts/build_e1_free_model_expansion.py --check` recompone estos metadatos.
No ejecuta consultas nativas, entrenamiento, pronósticos o lectura de reservas.
Los conteos acumulados se calculan en la etapa homogénea posterior, sin cambiar
los cortes históricos ni atribuir retrospectivamente estas fuentes a la selección
experimental. Desconocido no significa cero o ausencia. La receta y los resultados
de los autores no se reproducen por compilar sus referencias.
