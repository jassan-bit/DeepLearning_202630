# 5. Reproducibilidad y entrega

Este libro dispone de **configuración e índice propios** en `book_e3/`. La publicación reúne los tres libros del curso bajo un repositorio común y sirve E3 en `/DeepLearning_202630/entregable3/intro.html`.

## Construcción del informe

Desde la raíz del repositorio, con el entorno Jupyter Book instalado mediante `requirements-book.lock.txt`:

```bash
.venv/bin/jupyter-book build book_e3 --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py --book-dir book_e3
```

La construcción genera HTML a partir de estas fuentes y no entrena modelos, ejecuta experimentos ni adquiere datos. Las referencias web a otros entregables no son entradas necesarias para compilar E3.

## Estado de la entrega

| Componente | Estado |
|---|---|
| Libro con portada, índice y configuración propios | Preparado como estructura inicial |
| Guía específica y requisitos de E3 | Pendientes de incorporación |
| Desarrollo y resultados de E3 | Diseño conceptual documentado; sin implementación ni resultados |
| Artefactos académicos y archivos de entrega | Por definir con la guía |

Los futuros productos se identificarán por nombre, versión y evidencia. La disponibilidad de esta estructura no se interpreta como ejecución del trabajo académico ni como una entrega final terminada.

El [repositorio del curso](https://github.com/jassan-bit/DeepLearning_202630) conserva las fuentes; el [índice de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) permite acceder a cada libro por su dirección correspondiente.
