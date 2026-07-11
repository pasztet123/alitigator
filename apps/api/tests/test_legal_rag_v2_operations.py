from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from app.legal_rag_v2.reindex_embeddings import run as run_reindex
from scripts.run_legal_rag_v2_ab import assert_dev_cases, load_cases


class DevelopmentRunnerSafetyTests(unittest.TestCase):
    def test_holdout_path_is_rejected_before_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "Holdout inputs are forbidden"):
            assert_dev_cases(Path("data/benchmarks/private-holdout.json"))

    def test_development_cases_require_ids_and_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev-cases.json"
            path.write_text('[{"id":"case-1","question":"Pytanie?"}]', encoding="utf-8")
            self.assertEqual(load_cases(path)[0]["id"], "case-1")


class ReindexSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_hash_embeddings_require_double_opt_in(self) -> None:
        args = argparse.Namespace(
            dimensions=16,
            offline_hash=True,
            allow_offline_hash=False,
            model="text-embedding-3-large",
            index_path=Path("unused.sqlite3"),
            schema_version="test",
            chunker_version="test",
            limit=0,
            batch_size=1,
        )
        with self.assertRaisesRegex(RuntimeError, "explicit opt-in"):
            await run_reindex(args)


if __name__ == "__main__":
    unittest.main()
