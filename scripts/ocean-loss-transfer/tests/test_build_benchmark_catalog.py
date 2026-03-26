from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_benchmark_catalog import build_benchmark_catalog  # noqa: E402


class BuildBenchmarkCatalogTests(unittest.TestCase):
    def test_build_catalog_marks_ready_and_ambiguous_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Benchmark'
            ready_dir = root / '通用Loss' / '1-Ready Paper'
            ready_dir.mkdir(parents=True, exist_ok=True)
            (ready_dir / 'paper.pdf').write_text('pdf', encoding='utf-8')
            repo_dir = ready_dir / 'Method-main'
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / 'train.py').write_text('print("ok")\n', encoding='utf-8')

            ambiguous_dir = root / '海洋Loss' / '2-Ambiguous Paper'
            ambiguous_dir.mkdir(parents=True, exist_ok=True)
            (ambiguous_dir / 'a.pdf').write_text('pdf-a', encoding='utf-8')
            (ambiguous_dir / 'b.pdf').write_text('pdf-b', encoding='utf-8')
            (ambiguous_dir / 'code-main.zip').write_text('zip', encoding='utf-8')
            repo_dir_2 = ambiguous_dir / 'Alt-main'
            repo_dir_2.mkdir(parents=True, exist_ok=True)
            (repo_dir_2 / 'core.py').write_text('print("alt")\n', encoding='utf-8')

            catalog = build_benchmark_catalog(str(root), max_depth=2)

        self.assertEqual(catalog['entry_count'], 2)
        self.assertEqual(catalog['status_counts']['ready'], 1)
        self.assertEqual(catalog['status_counts']['ambiguous'], 1)

        ready_entry = next(item for item in catalog['entries'] if item['status'] == 'ready')
        self.assertEqual(ready_entry['category'], '通用Loss')
        self.assertEqual(len(ready_entry['paper_pdf_candidates']), 1)
        self.assertEqual(len(ready_entry['code_candidates']), 1)
        self.assertTrue(ready_entry['suggested_paper_pdf_path'].endswith('paper.pdf'))
        self.assertTrue(ready_entry['suggested_code_path'].endswith('Method-main'))

        ambiguous_entry = next(item for item in catalog['entries'] if item['status'] == 'ambiguous')
        self.assertIn('multiple_pdfs', ambiguous_entry['notes'])
        self.assertIn('multiple_code_candidates', ambiguous_entry['notes'])

    def test_build_catalog_marks_incomplete_entry_without_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Benchmark'
            incomplete_dir = root / 'OnlyPdf'
            incomplete_dir.mkdir(parents=True, exist_ok=True)
            (incomplete_dir / 'paper.pdf').write_text('pdf', encoding='utf-8')

            catalog = build_benchmark_catalog(str(root), max_depth=1)

        self.assertEqual(catalog['entry_count'], 1)
        self.assertEqual(catalog['status_counts']['incomplete'], 1)
        entry = catalog['entries'][0]
        self.assertEqual(entry['status'], 'incomplete')
        self.assertIn('missing_code', entry['notes'])


if __name__ == '__main__':
    unittest.main()
