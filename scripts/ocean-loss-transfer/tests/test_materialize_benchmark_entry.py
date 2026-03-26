from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from materialize_benchmark_entry import materialize_benchmark_entry  # noqa: E402


class MaterializeBenchmarkEntryTests(unittest.TestCase):
    def test_materialize_archive_entry_extracts_repo_and_returns_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_pdf = root / 'paper.pdf'
            paper_pdf.write_text('pdf', encoding='utf-8')

            archive_path = root / 'repo-main.zip'
            with zipfile.ZipFile(archive_path, 'w') as zip_file:
                zip_file.writestr('repo-main/train.py', 'print("ok")\n')
                zip_file.writestr('repo-main/requirements.txt', 'torch\n')

            catalog_path = root / 'catalog.json'
            catalog_path.write_text(
                json.dumps(
                    {
                        'entries': [
                            {
                                'entry_id': 'demo-entry',
                                'paper_slug': 'demo-paper',
                                'status': 'ready',
                                'suggested_paper_pdf_path': str(paper_pdf),
                                'suggested_code_path': str(archive_path),
                                'code_candidates': [
                                    {
                                        'path': str(archive_path),
                                        'type': 'archive',
                                        'extract_required': True,
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            result = materialize_benchmark_entry(
                catalog_path=str(catalog_path),
                entry_id='demo-entry',
                cache_root=str(root / 'cache'),
            )

            self.assertEqual(result['status'], 'ready')
            self.assertTrue(result['materialized'])
            self.assertTrue(Path(result['code_repo_path']).is_dir())
            self.assertTrue((Path(result['code_repo_path']) / 'train.py').exists())
            self.assertEqual(result['paper_slug'], 'demo-paper')

    def test_materialize_repo_entry_returns_existing_repo_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_dir = root / 'repo-main'
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / 'train.py').write_text('print("ok")\n', encoding='utf-8')
            paper_pdf = root / 'paper.pdf'
            paper_pdf.write_text('pdf', encoding='utf-8')

            catalog_path = root / 'catalog.json'
            catalog_path.write_text(
                json.dumps(
                    {
                        'entries': [
                            {
                                'entry_id': 'repo-entry',
                                'paper_slug': 'repo-paper',
                                'status': 'ready',
                                'suggested_paper_pdf_path': str(paper_pdf),
                                'suggested_code_path': str(repo_dir),
                                'code_candidates': [
                                    {
                                        'path': str(repo_dir),
                                        'type': 'repo',
                                        'extract_required': False,
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            result = materialize_benchmark_entry(
                catalog_path=str(catalog_path),
                entry_id='repo-entry',
            )

            self.assertEqual(result['status'], 'ready')
            self.assertFalse(result['materialized'])
            self.assertEqual(result['code_repo_path'], str(repo_dir.resolve()))


if __name__ == '__main__':
    unittest.main()
