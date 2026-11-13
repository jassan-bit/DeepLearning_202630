import io
import json
from pathlib import Path
import tempfile
import unittest
from e2_native_transverse_colab import parse_native_response, retrieve


def response(total=12, identifier='http://arxiv.org/abs/1234.5678v1'):
    return f'''<feed xmlns="http://www.w3.org/2005/Atom"
     xmlns:o="http://a9.com/-/spec/opensearch/1.1/">
     <o:totalResults>{total}</o:totalResults><entry><id>{identifier}</id>
     <title>Example  title</title><summary>Private abstract</summary>
     <published>2020-01-01T00:00:00Z</published><author><name>A</name></author>
     </entry></feed>'''.encode()


class NativeTransverseTests(unittest.TestCase):
    def test_native_total_distinct_from_retrieved_and_abstract_excluded(self):
        d = parse_native_response(response())
        self.assertEqual((d['native_total'], d['retrieved_records']), (12, 1))
        self.assertNotIn('abstract', json.dumps(d).lower())

    def test_inconsistent_count_rejected(self):
        with self.assertRaises(ValueError):
            parse_native_response(response(total=0))

    def test_api_error_record_rejected(self):
        with self.assertRaises(ValueError):
            parse_native_response(response(identifier='https://arxiv.org/api/errors'))

    def test_network_failure_remains_null(self):
        def unavailable(*args, **kwargs):
            raise OSError('offline')
        with tempfile.TemporaryDirectory() as directory:
            d = retrieve(Path(directory)/'receipt.json', opener=unavailable, clock=lambda: 'fixed', sleeper=lambda _: None)
            self.assertEqual(len(d['queries']), 3)
            self.assertTrue(all(q['native_total'] is None for q in d['queries']))

    def test_success_receipt_reused_without_requery(self):
        calls = []
        def opened(*args, **kwargs):
            calls.append(1)
            return io.BytesIO(response())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            d = retrieve(path, opener=opened, clock=lambda: 'fixed', sleeper=lambda _: None)
            self.assertEqual(d, retrieve(path, opener=opened))
            self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
