"""Manifest regression: nested vendor manifests are source, not self-reference."""
from pathlib import Path
import json
import tempfile
import unittest

import assemble_runtime


class PackagingTests(unittest.TestCase):
    def test_only_root_manifest_and_local_anchor_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'vendor').mkdir()
            (root/'vendor/SOURCE_MANIFEST.json').write_text('{"generated_fixture":true}\n')
            (root/'SOURCE_MANIFEST.json').write_text('old root manifest')
            (root/'PUBLIC_SOURCE_ANCHOR.json').write_text('LOCAL QA ANCHOR NOT PUBLIC SOURCE')
            (root/'bootstrap_colab.py').write_text('# synthetic packaging fixture\n')
            digest=assemble_runtime.manifest(root)
            value=json.loads((root/'SOURCE_MANIFEST.json').read_text())
            self.assertEqual(set(value['files']),{'vendor/SOURCE_MANIFEST.json','bootstrap_colab.py'})
            self.assertEqual(len(digest),64)


if __name__=='__main__':
    unittest.main(verbosity=2)
