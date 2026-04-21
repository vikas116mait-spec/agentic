from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from python_api.local_qlora import ensure_local_training_ready, spawn_local_training_job
from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.local_qlora.train import LocalProgressCallback, _recommended_dataset_num_proc, run_local_qlora_training
from python_api.services import _default_local_ollama_model_name, _default_model_profiles, _ensure_model_profiles_initialized


def runtime_summary(*, unsloth_available: bool = True) -> dict[str, object]:
    return {
        "python": sys.executable,
        "dependencies": {
            "torch": True,
            "transformers": True,
            "datasets": True,
            "peft": True,
            "trl": True,
            "accelerate": True,
            "bitsandbytes": True,
        },
        "gpuCount": 1,
        "unslothAvailable": unsloth_available,
    }


class FakeTorch:
    class cuda:  # noqa: N801 - mirror torch.cuda attribute style
        @staticmethod
        def is_bf16_supported() -> bool:
            return False


class FakeModel:
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")


class FakeTokenizer:
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeTrainer:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.callback = None

    def add_callback(self, callback: object) -> None:
        self.callback = callback

    def train(self) -> SimpleNamespace:
        return SimpleNamespace(
            metrics={
                "global_step": 4,
                "train_runtime": 1.5,
                "train_loss": 0.42,
                "train_steps_per_second": 2.0,
                "train_samples_per_second": 4.0,
            }
        )

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.25}


class LocalTrainingPreflightTests(unittest.TestCase):
    def test_requires_explicit_training_python(self) -> None:
        with patch("python_api.local_qlora.local_training_enabled", return_value=True), patch(
            "python_api.local_qlora.configured_local_training_python", return_value=""
        ):
            with self.assertRaisesRegex(RuntimeError, "Set LOCAL_TRAINING_PYTHON"):
                ensure_local_training_ready()

    def test_warns_when_unsloth_is_missing_for_gguf_export(self) -> None:
        with patch("python_api.local_qlora.local_training_enabled", return_value=True), patch(
            "python_api.local_qlora.configured_local_training_python", return_value=sys.executable
        ), patch("python_api.local_qlora._probe_training_runtime", return_value=runtime_summary(unsloth_available=False)):
            summary = ensure_local_training_ready(export_gguf=True)

        warnings = summary.get("warnings") or []
        self.assertTrue(any("Unsloth" in str(warning) for warning in warnings))

    def test_requires_ollama_when_push_is_enabled(self) -> None:
        with patch("python_api.local_qlora.local_training_enabled", return_value=True), patch(
            "python_api.local_qlora.configured_local_training_python", return_value=sys.executable
        ), patch("python_api.local_qlora._probe_training_runtime", return_value=runtime_summary()), patch(
            "python_api.local_qlora.ensure_ollama_runtime_ready",
            side_effect=RuntimeError("Ollama is not reachable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Ollama is not reachable"):
                ensure_local_training_ready(export_gguf=True, push_to_ollama=True)


class LocalTrainingDefaultsTests(unittest.TestCase):
    def test_auto_generated_ollama_name_uses_dataset_model_and_job_id(self) -> None:
        generated = _default_local_ollama_model_name(
            "Customer Support Dataset",
            "Qwen/Qwen2.5-3B-Instruct",
            "1234567890abcdef",
        )

        self.assertEqual(generated, "customer-support-dataset-qwen2-5-3b-12345678")

    def test_legacy_job_default_migrates_to_local_profile(self) -> None:
        state = {
            "model_profiles_initialized": True,
            "model_profiles": deepcopy(_default_model_profiles()),
            "model_profile_defaults": {
                "playgroundBaseProfileId": "profile-medium",
                "playgroundCompareProfileId": "profile-large",
                "agentBaseProfileId": "profile-medium",
                "agentModelProfileId": "profile-thinking",
                "jobBaseProfileId": "profile-large",
            },
        }

        _ensure_model_profiles_initialized(state)

        self.assertEqual(state["model_profile_defaults"]["jobBaseProfileId"], "profile-local-qlora")

    def test_small_dataset_uses_single_tokenization_worker(self) -> None:
        self.assertEqual(_recommended_dataset_num_proc(81, 9), 1)

    def test_medium_dataset_caps_tokenization_workers_conservatively(self) -> None:
        self.assertLessEqual(_recommended_dataset_num_proc(400, 40), 2)

    def test_progress_callback_ignores_unhandled_trainer_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "job"
            config = LocalQLoraJobConfig(
                job_id="job1234567890abcd",
                dataset_id="dataset-1",
                dataset_name="demo-dataset",
                dataset_path=str(Path(temp_dir) / "dataset.jsonl"),
                base_model="Qwen/Qwen2.5-1.5B-Instruct",
                working_dir=str(working_dir),
                output_dir=str(working_dir / "artifacts"),
                model_output_path=str(working_dir / "artifacts" / "adapter"),
                config_path=str(working_dir / "local_train_config.json"),
                status_path=str(working_dir / "local_train_status.json"),
                events_path=str(working_dir / "local_train_events.jsonl"),
                log_path=str(working_dir / "local_train.log"),
                metrics_path=str(working_dir / "local_train_metrics.json"),
            )
            callback = LocalProgressCallback(config=config, status_path=Path(config.status_path), gpu_count=1)

            callback.on_epoch_begin(None, None, None)


class LocalTrainingWarningsTests(unittest.TestCase):
    def make_config(self, temp_dir: str, *, push_to_ollama: bool = True) -> LocalQLoraJobConfig:
        working_dir = Path(temp_dir) / "job"
        return LocalQLoraJobConfig(
            job_id="job1234567890abcd",
            dataset_id="dataset-1",
            dataset_name="demo-dataset",
            dataset_path=str(Path(temp_dir) / "dataset.jsonl"),
            base_model="Qwen/Qwen2.5-3B-Instruct",
            working_dir=str(working_dir),
            output_dir=str(working_dir / "artifacts"),
            model_output_path=str(working_dir / "artifacts" / "adapter"),
            config_path=str(working_dir / "local_train_config.json"),
            status_path=str(working_dir / "local_train_status.json"),
            events_path=str(working_dir / "local_train_events.jsonl"),
            log_path=str(working_dir / "local_train.log"),
            metrics_path=str(working_dir / "local_train_metrics.json"),
            export_gguf=True,
            push_to_ollama=push_to_ollama,
            ollama_model_name="demo-model",
        )

    def common_patches(self, *, peft_config: object | None) -> list[patch]:
        fake_model = FakeModel()
        fake_tokenizer = FakeTokenizer()
        fake_trainer_factory = lambda **kwargs: FakeTrainer(kwargs["model"])
        return [
            patch(
                "python_api.local_qlora.train.build_sft_datasets",
                return_value=(object(), object(), {"totalRecords": 12, "trainRecords": 10, "evalRecords": 2}),
            ),
            patch(
                "python_api.local_qlora.train.load_quantized_model",
                return_value=(fake_model, fake_tokenizer, peft_config, FakeTorch()),
            ),
            patch("python_api.local_qlora.train._instantiate_sft_config", return_value=object()),
            patch("python_api.local_qlora.train._instantiate_trainer", side_effect=fake_trainer_factory),
            patch(
                "python_api.local_qlora.train.summarize_eval_metrics",
                return_value={"evalLoss": 0.25, "perplexity": 1.28},
            ),
            patch("python_api.local_qlora.train.local_gpu_count", return_value=1),
        ]

    def test_training_succeeds_with_warning_when_unsloth_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)
            patches = self.common_patches(peft_config=object())
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                run_local_qlora_training(config)

            status = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            warning_messages = status.get("warnings") or []
            result_types = {item["type"] for item in status.get("resultFilesJson") or []}

            self.assertEqual(status["status"], "succeeded")
            self.assertFalse(status["ollamaRegistered"])
            self.assertTrue(any("Unsloth" in warning for warning in warning_messages))
            self.assertIn("local_adapter", result_types)
            self.assertNotIn("local_gguf", result_types)

    def test_training_keeps_gguf_when_ollama_registration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)
            gguf_path = Path(config.model_output_path) / "gguf" / "model.gguf"
            modelfile_path = gguf_path.parent / "Modelfile"

            def fake_export(*args: object, **kwargs: object) -> str:
                gguf_path.parent.mkdir(parents=True, exist_ok=True)
                gguf_path.write_text("gguf", encoding="utf-8")
                return str(gguf_path)

            def fake_modelfile(*args: object, **kwargs: object) -> str:
                modelfile_path.write_text("FROM demo", encoding="utf-8")
                return str(modelfile_path)

            patches = self.common_patches(peft_config=None)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patch(
                "python_api.local_qlora.export.export_gguf",
                side_effect=fake_export,
            ), patch(
                "python_api.local_qlora.export.write_ollama_modelfile",
                side_effect=fake_modelfile,
            ), patch(
                "python_api.local_qlora.export.push_to_ollama",
                side_effect=RuntimeError("Ollama offline"),
            ):
                run_local_qlora_training(config)

            status = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            warning_messages = status.get("warnings") or []
            result_types = {item["type"] for item in status.get("resultFilesJson") or []}

            self.assertEqual(status["status"], "succeeded")
            self.assertFalse(status["ollamaRegistered"])
            self.assertTrue(any("Ollama registration failed" in warning for warning in warning_messages))
            self.assertIn("local_gguf", result_types)
            self.assertIn("local_modelfile", result_types)


class LocalTrainingSpawnTests(unittest.TestCase):
    def make_config(self, temp_dir: str) -> LocalQLoraJobConfig:
        working_dir = Path(temp_dir) / "job"
        return LocalQLoraJobConfig(
            job_id="job1234567890abcd",
            dataset_id="dataset-1",
            dataset_name="demo-dataset",
            dataset_path=str(Path(temp_dir) / "dataset.jsonl"),
            base_model="Qwen/Qwen2.5-3B-Instruct",
            working_dir=str(working_dir),
            output_dir=str(working_dir / "artifacts"),
            model_output_path=str(working_dir / "artifacts" / "adapter"),
            config_path=str(working_dir / "local_train_config.json"),
            status_path=str(working_dir / "local_train_status.json"),
            events_path=str(working_dir / "local_train_events.jsonl"),
            log_path=str(working_dir / "local_train.log"),
            metrics_path=str(working_dir / "local_train_metrics.json"),
            export_gguf=True,
            push_to_ollama=False,
            ollama_model_name="",
        )

    def test_spawn_redirects_background_output_to_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)
            popen_kwargs: dict[str, object] = {}

            class FakePopen:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    popen_kwargs.update(kwargs)
                    self.pid = 4321

            class FakeThread:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

                def start(self) -> None:
                    return None

            with patch("python_api.local_qlora.local_training_python", return_value=sys.executable), patch(
                "python_api.local_qlora._select_single_gpu",
                return_value={
                    "index": 2,
                    "name": "GPU 2",
                    "memoryTotalMb": 81920,
                    "memoryUsedMb": 4096,
                    "memoryFreeMb": 77824,
                },
            ), patch("python_api.local_qlora.subprocess.Popen", side_effect=FakePopen), patch(
                "python_api.local_qlora.threading.Thread",
                side_effect=lambda *args, **kwargs: FakeThread(),
            ):
                spawn_local_training_job(config, runtime_summary=runtime_summary())

            stdout_target = popen_kwargs["stdout"]
            self.assertNotEqual(stdout_target, sys.stdout)
            self.assertEqual(getattr(stdout_target, "name", None), config.log_path)
            self.assertEqual(popen_kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(popen_kwargs["env"]["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
