# Auditoría de anterioridad por campo y etapa

Se ligan registros Git originales a170ajustes HPO nuevos mediante sus journals físicos, huellas de resultado y cadenas de eventos. La fecha de los commits se compara con UTC de los journals, sin inventar una autoridad de tiempo externa. Los61resultados reutilizados conservan alcance histórico.

El primer protocolo reducido es c056c1cf (12/11/2026); la implementación es1e874549. La tabla informa el resultado de cada comparación; la existencia de contenido previo para esos ajustes no acredita preregistro integral antes de toda la historia ni recupera cegamiento.

Se entregan solicitudes exactas para174textos primarios y los campos que faltan en las exportaciones Scopus/WoS. No se borran informes inaccesibles ni se declara exhaustividad reparada.

Repetición: python scripts/audit_e2_registration_chronology.py --output NUEVA_CARPETA. Requiere el repositorio y los commits Git originales enumerados; el modo Git puede recuperar objetos originales faltantes. No entrena modelos.

Para comprobar el paquete sin repositorio, Git ni red: python run.py. Verifica206journals y170ajustes, y genera output/verification.json. Conserve o renombre output/ antes de repetir. Los tiempos Git/worker son los registrados; no se fabrican sellos externos ni cegamiento.

Los206journals se guardan con compresión gzip sin pérdida y mtime0. El verificador descomprime cada uno y comprueba la huella SHA256 del contenido Git original y toda la cadena de eventos; no se truncan registros.
