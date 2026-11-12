# Verificación de entorno E2: no entrena ni amplía presupuesto.
import hashlib as _e2_hashlib
import urllib.request as _e2_request
_e2_registration = 'c056c1cfb8ac771b32b13bf1382b9afc5e813912'
_e2_helper = _e2_request.urlopen("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + _e2_registration + "/notebooks/comprobar_e2_reducido_colab.py", timeout=30).read()
assert _e2_hashlib.sha256(_e2_helper).hexdigest() == '560b8a0b687c2ca4edd0841d8c41445bba41cf3c58a5ab44911a78954d975383', "El verificador no coincide con la fuente registrada"
_e2_namespace = {"__name__": "e2_reduced_colab_diagnostic"}
exec(compile(_e2_helper, "comprobar_e2_reducido_colab.py", "exec"), _e2_namespace)
_e2_diagnostic = _e2_namespace["run_colab_check"](_e2_registration, '13b9675e675e30ab958da0b58a448f8d63b187dd4183ae5efdb5b294ebeba914', namespace=globals(), budget_relative_path="experiments/e2_reduced_protocol_20261112/budget_reconciliation.json", budget_sha256='dfcccad150ce9cdb90859c745ae4d64f07bfcec6a9d0e2cfc21050a5cab972b5')
print("Entrenamiento iniciado:", _e2_diagnostic["training_started"])
