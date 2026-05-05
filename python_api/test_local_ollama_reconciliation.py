from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from python_api.local_qlora.state import read_status_file, write_status_file
from python_api.services import ensure_ollama_model_available, retrieve_job_detail
from python_api.store import JOBS_DIR, empty_state


class LocalOllamaReconciliationTests(unittest.TestCase):
    def _build_state(self, *, job_id: str, status_path: Path, events_path: Path, adapter_path: Path) -> dict:
        state = empty_state()
        state["datasets"].append(
            {
                "id": "dataset-1",
                "name": "demo-dataset",
                "validationStatus": "VALID",
                "createdAt": "2026-05-05T00:00:00+00:00",
                "updatedAt": "2026-05-05T00:00:00+00:00",
            }
        )
        state["jobs"].append(
            {
                "id": job_id,
                "datasetId": "dataset-1",
                "modelProvider": "local",
                "modelProviderLabel": "Local GPU QLoRA",
                "providerJobId": "1234",
                "providerJobUrl": None,
                "providerNamespace": None,
                "baseModel": "Qwen/Qwen2.5-1.5B-Instruct",
                "status": "succeeded",
                "statusMessage": "Local GPU QLoRA training finished with warnings.",
                "fineTunedModel": f"jobs/{job_id}/adapter",
                "modelRepoId": f"jobs/{job_id}/adapter",
                "modelRepoUrl": None,
                "datasetRepoId": None,
                "datasetRepoPath": None,
                "datasetRepoUrl": None,
                "trackioUrl": None,
                "trainingBackend": "local_qlora",
                "trainingPreset": "balanced",
                "trainedTokens": None,
                "ollamaModelName": "demo-model",
                "lastSyncedAt": "2026-05-05T00:00:00+00:00",
                "progressJson": {
                    "stage": "succeeded",
                    "lossHistory": [],
                    "warnings": ["GGUF export failed (adapter is still saved): demo"],
                    "ollamaRegistered": False,
                },
                "resultFilesJson": [
                    {"type": "local_adapter", "path": f"jobs/{job_id}/adapter"},
                ],
                "localStatusPath": f"jobs/{job_id}/local_train_status.json",
                "localEventsPath": f"jobs/{job_id}/local_train_events.jsonl",
                "localConfigPath": f"jobs/{job_id}/local_train_config.json",
                "localLogPath": f"jobs/{job_id}/local_train.log",
                "localMetricsPath": f"jobs/{job_id}/local_train_metrics.json",
                "localArtifactsPath": f"jobs/{job_id}/adapter",
                "createdAt": "2026-05-05T00:00:00+00:00",
                "updatedAt": "2026-05-05T00:00:00+00:00",
                "finishedAt": "2026-05-05T00:00:00+00:00",
            }
        )
        return state

    def _write_job_files(self, *, job_dir: Path) -> tuple[Path, Path, Path, Path]:
        adapter_path = job_dir / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (job_dir / "local_train_config.json").write_text("{}", encoding="utf-8")
        (job_dir / "local_train.log").write_text("", encoding="utf-8")
        (job_dir / "local_train_metrics.json").write_text("{}", encoding="utf-8")
        status_path = job_dir / "local_train_status.json"
        events_path = job_dir / "local_train_events.jsonl"
        return adapter_path, status_path, events_path, job_dir / "gguf"

    def test_retrieve_job_detail_reconciles_registered_local_model(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(JOBS_DIR)) as job_dir_raw:
            job_dir = Path(job_dir_raw)
            job_id = job_dir.name
            adapter_path, status_path, events_path, gguf_dir = self._write_job_files(job_dir=job_dir)
            gguf_dir.mkdir(parents=True, exist_ok=True)
            gguf_path = gguf_dir / "model.q8_0.gguf"
            gguf_path.write_text("gguf", encoding="utf-8")

            write_status_file(
                status_path,
                {
                    "status": "succeeded",
                    "stage": "succeeded",
                    "statusMessage": "Local GPU QLoRA training finished with warnings. GGUF export failed.",
                    "warnings": ["GGUF export failed (adapter is still saved): demo"],
                    "ollamaRegistered": False,
                    "lossHistory": [{"step": 10, "loss": 1.2}, {"step": 20, "loss": 0.9}],
                    "resultFilesJson": [{"type": "local_adapter", "path": f"jobs/{job_id}/adapter"}],
                },
            )

            state = self._build_state(job_id=job_id, status_path=status_path, events_path=events_path, adapter_path=adapter_path)

            def fake_update_state(mutator):
                return mutator(state)

            with patch("python_api.services.load_state", side_effect=lambda: state), patch(
                "python_api.services.update_state",
                side_effect=fake_update_state,
            ), patch(
                "python_api.services._run_ollama_cli",
                return_value=CompletedProcess(args=["ollama", "show", "demo-model"], returncode=0, stdout="", stderr=""),
            ):
                detail = retrieve_job_detail(job_id)

            self.assertEqual(detail["status"], "succeeded")
            self.assertTrue(detail["progressJson"]["ollamaRegistered"])
            self.assertIn("Model available in Ollama", detail["statusMessage"])
            self.assertIn("local_gguf", {item["type"] for item in detail["resultFilesJson"]})
            self.assertIn("local_modelfile", {item["type"] for item in detail["resultFilesJson"]})

            status_payload = read_status_file(status_path)
            self.assertTrue(status_payload["ollamaRegistered"])
            self.assertEqual(status_payload["warnings"], [])
            self.assertTrue((gguf_dir / "Modelfile").is_file())
            self.assertIn(str(gguf_path.resolve()), (gguf_dir / "Modelfile").read_text(encoding="utf-8"))

    def test_ensure_ollama_model_available_auto_registers_from_discovered_gguf(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(JOBS_DIR)) as job_dir_raw:
            job_dir = Path(job_dir_raw)
            job_id = job_dir.name
            adapter_path, status_path, events_path, gguf_dir = self._write_job_files(job_dir=job_dir)
            gguf_dir.mkdir(parents=True, exist_ok=True)
            gguf_path = gguf_dir / "recovered.q8_0.gguf"
            gguf_path.write_text("gguf", encoding="utf-8")

            write_status_file(
                status_path,
                {
                    "status": "succeeded",
                    "stage": "succeeded",
                    "statusMessage": "Local GPU QLoRA training finished with warnings. GGUF export failed.",
                    "warnings": ["GGUF export failed (adapter is still saved): demo"],
                    "ollamaRegistered": False,
                    "resultFilesJson": [{"type": "local_adapter", "path": f"jobs/{job_id}/adapter"}],
                },
            )

            state = self._build_state(job_id=job_id, status_path=status_path, events_path=events_path, adapter_path=adapter_path)
            recorded_calls: list[tuple[str, ...]] = []

            def fake_update_state(mutator):
                return mutator(state)

            def fake_run_ollama_cli(*args: str):
                recorded_calls.append(tuple(args))
                if args[0] == "show":
                    return CompletedProcess(args=args, returncode=1, stdout="", stderr="not found")
                if args[0] == "create":
                    return CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")
                return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with patch("python_api.services.load_state", side_effect=lambda: state), patch(
                "python_api.services.update_state",
                side_effect=fake_update_state,
            ), patch(
                "python_api.services._run_ollama_cli",
                side_effect=fake_run_ollama_cli,
            ):
                ensure_ollama_model_available("demo-model")

            self.assertEqual(recorded_calls[0][0], "show")
            self.assertEqual(recorded_calls[1][0], "create")
            self.assertTrue((gguf_dir / "Modelfile").is_file())
            self.assertIn(str(gguf_path.resolve()), (gguf_dir / "Modelfile").read_text(encoding="utf-8"))

            status_payload = read_status_file(status_path)
            self.assertTrue(status_payload["ollamaRegistered"])
            self.assertEqual(status_payload["warnings"], [])
            self.assertIn("local_gguf", {item["type"] for item in status_payload["resultFilesJson"]})
            self.assertIn("local_modelfile", {item["type"] for item in status_payload["resultFilesJson"]})
            self.assertTrue(state["jobs"][0]["progressJson"]["ollamaRegistered"])


if __name__ == "__main__":
    unittest.main()
