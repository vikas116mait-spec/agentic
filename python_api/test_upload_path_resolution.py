from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python_api.store import (
    DATASETS_DIR,
    JOBS_DIR,
    resolve_dataset_storage_path,
    resolve_job_storage_path,
    serialize_dataset_storage_path,
    serialize_job_storage_path,
)


class UploadPathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def test_resolves_legacy_dataset_path_into_current_uploads_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(DATASETS_DIR)) as dataset_dir:
            dataset_path = Path(dataset_dir) / "sample.jsonl"
            dataset_path.write_text('{"messages":[{"role":"user","content":"Hi"}]}\n', encoding="utf-8")

            legacy_path = f"/app/uploads_python/datasets/{Path(dataset_dir).name}/sample.jsonl"
            resolved = resolve_dataset_storage_path(legacy_path)

        self.assertIsNotNone(resolved)
        self.assertEqual(str(resolved), str(dataset_path))

    def test_resolves_legacy_job_path_into_current_uploads_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(JOBS_DIR)) as job_dir:
            status_path = Path(job_dir) / "local_train_status.json"
            status_path.write_text("{}", encoding="utf-8")

            legacy_path = (
                "/mnt/nvme_disk2/User_data/vs95259v/Vikas/project/agentic/"
                f"uploads_python/jobs/{Path(job_dir).name}/local_train_status.json"
            )
            resolved = resolve_job_storage_path(legacy_path)

        self.assertIsNotNone(resolved)
        self.assertEqual(str(resolved), str(status_path))

    def test_serializes_current_dataset_and_job_paths_as_relative_uploads_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(DATASETS_DIR)) as dataset_dir, tempfile.TemporaryDirectory(
            dir=str(JOBS_DIR)
        ) as job_dir:
            dataset_path = Path(dataset_dir) / "sample.jsonl"
            dataset_path.write_text('{"messages":[{"role":"user","content":"Hi"}]}\n', encoding="utf-8")
            status_path = Path(job_dir) / "local_train_status.json"
            status_path.write_text("{}", encoding="utf-8")

            serialized_dataset = serialize_dataset_storage_path(dataset_path)
            serialized_job = serialize_job_storage_path(status_path)

        self.assertEqual(serialized_dataset, f"datasets/{Path(dataset_dir).name}/sample.jsonl")
        self.assertEqual(serialized_job, f"jobs/{Path(job_dir).name}/local_train_status.json")
