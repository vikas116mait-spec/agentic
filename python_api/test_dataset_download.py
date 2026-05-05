from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_api.errors import ApiError
from python_api.services import build_dataset_download_package
from python_api.store import UPLOADS_DIR


class DatasetDownloadPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        (UPLOADS_DIR / "datasets").mkdir(parents=True, exist_ok=True)

    def test_builds_download_package_for_dataset_in_uploads_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(UPLOADS_DIR / "datasets")) as dataset_dir:
            dataset_path = Path(dataset_dir) / "customer-support.jsonl"
            dataset_path.write_text('{"messages":[{"role":"user","content":"Hi"}]}\n', encoding="utf-8")

            with patch(
                "python_api.services.retrieve_dataset_detail",
                return_value={
                    "id": "dataset-1",
                    "originalFilename": "customer-support.jsonl",
                    "storagePath": str(dataset_path),
                },
            ):
                package = build_dataset_download_package("dataset-1")

        self.assertEqual(package["path"], str(dataset_path))
        self.assertEqual(package["filename"], "customer-support.jsonl")
        self.assertEqual(package["mediaType"], "application/x-ndjson")

    def test_rebases_legacy_container_dataset_path_to_current_uploads_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(UPLOADS_DIR / "datasets")) as dataset_dir:
            dataset_path = Path(dataset_dir) / "customer-support.jsonl"
            dataset_path.write_text('{"messages":[{"role":"user","content":"Hi"}]}\n', encoding="utf-8")
            legacy_path = f"/app/uploads_python/datasets/{Path(dataset_dir).name}/customer-support.jsonl"

            with patch(
                "python_api.services.retrieve_dataset_detail",
                return_value={
                    "id": "dataset-legacy",
                    "originalFilename": "customer-support.jsonl",
                    "storagePath": legacy_path,
                },
            ):
                package = build_dataset_download_package("dataset-legacy")

        self.assertEqual(package["path"], str(dataset_path))
        self.assertEqual(package["filename"], "customer-support.jsonl")
        self.assertEqual(package["mediaType"], "application/x-ndjson")

    def test_rejects_dataset_path_outside_uploads_directory(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            with patch(
                "python_api.services.retrieve_dataset_detail",
                return_value={
                    "id": "dataset-2",
                    "originalFilename": "outside.jsonl",
                    "storagePath": temp_file.name,
                },
            ):
                with self.assertRaises(ApiError) as context:
                    build_dataset_download_package("dataset-2")

        self.assertEqual(context.exception.code, "FORBIDDEN")

    def test_rejects_missing_dataset_file(self) -> None:
        missing_path = UPLOADS_DIR / "datasets" / "missing-dataset" / "demo.jsonl"
        with patch(
            "python_api.services.retrieve_dataset_detail",
            return_value={
                "id": "dataset-3",
                "originalFilename": "demo.jsonl",
                "storagePath": str(missing_path),
            },
        ):
            with self.assertRaises(ApiError) as context:
                build_dataset_download_package("dataset-3")

        self.assertEqual(context.exception.code, "NOT_FOUND")
