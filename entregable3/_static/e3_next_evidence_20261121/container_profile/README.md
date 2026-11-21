# Perfil de contenedor comprobado con dependencias fuera de línea

Preparado con asistencia de IA. La imagen de esta comprobación usa la misma
base Python y las mismas 89 versiones del kit humano. Este perfil añade los
hashes de los wheels efectivamente instalados. No contiene datos de mercado,
pesos, una firma ni una ejecución humana.

El primer intento no pudo escribir la configuración del cliente Docker en su
directorio predeterminado. Los dos siguientes fallaron al resolver DNS dentro
del constructor. Se usó un `DOCKER_CONFIG` propio y se descargaron los wheels
en el host; el cuarto build verificó sus hashes e instaló sin red. Son fallos
de infraestructura conservados, no falta de paquetes ni fallo del producto.

## Reconstruir el perfil en Linux x86_64 con Python 3.12

Los wheels binarios de 49.700.631 bytes no se incluyen en el paquete de código.
La persona que reproduzca los obtiene y verifica bajo sus términos originales:

```bash
python3.12 -m pip download --only-binary=:all: --no-deps --require-hashes \
  -r requirements-book-artifacts.lock.txt -d wheelhouse
docker build --network none -f Dockerfile.offline -t e3-book-reproduction-offline:f5c .
```

El lock de artefactos corresponde a CPython 3.12/Linux amd64, con wheels
compatibles `cp312`, `abi3` o universales. No es un lock de descarga para un
intérprete o plataforma distintos. El Dockerfile comprueba nombres, tamaños,
SHA256 de los 89 wheels, hash del lock original y `pip check`.

El kit humano conserva también su Dockerfile en línea con la base/versiones
fijadas. Este perfil comprobado mantiene su propio Dockerfile y recibo; no se
modificó aquel archivo congelado. La imagen local comprobada tiene ID
`sha256:69874b35ba5cab6c0f786fa06ade7d8560d8d9a805be6da7665e9fad6c87f80e`.
No se publicó en un registro OCI: quien reproduzca debe reconstruir la imagen
y registrar su propio ID. No se promete igualdad binaria del ID entre builds.

`BUILD_RECEIPT.json` y `pip-install-report.json` documentan la instalación real.
La ejecución del libro y la reagregación de tablas se registran por separado.
Ni construir la imagen ni su validación automática sustituyen a la persona
externa exigida por la guía.
