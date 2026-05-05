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

from python_api.local_qlora import ensure_local_training_ready, inspect_local_training_runtime, spawn_local_training_job
from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.local_qlora.data import build_evaluation_example, format_training_record
from python_api.local_qlora.export import write_ollama_modelfile
from python_api.local_qlora.model import _resolve_target_modules
from python_api.local_qlora.train import LocalProgressCallback, _recommended_dataset_num_proc, run_local_qlora_training
from python_api.services import _default_local_ollama_model_name, _default_model_profiles, _ensure_model_profiles_initialized
from python_api.store import DATASETS_DIR, JOBS_DIR


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
        "compilerTools": {
            "cc": "/usr/bin/gcc",
            "cxx": "/usr/bin/g++",
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


class FakeChatTemplateTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = " || ".join(f"{message['role']}={message['content']}" for message in messages)
        return f"TEMPLATE::{rendered}::tokenize={tokenize}::gen={add_generation_prompt}"


class FakeLeafModule:
    def children(self):
        return ()


class FakeDiscoveredTargetModuleModel:
    def __init__(self, leaf_names: list[str]) -> None:
        self.leaf_names = leaf_names

    def named_modules(self):
        yield ("", self)
        for index, leaf_name in enumerate(self.leaf_names):
            yield (f"model.layers.{index}.{leaf_name}", FakeLeafModule())

    def children(self):
        return ()


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

    def test_requires_compiler_toolchain_when_unsloth_is_available(self) -> None:
        summary = runtime_summary()
        summary["compilerTools"] = {"cc": None, "cxx": None}

        with patch("python_api.local_qlora.local_training_enabled", return_value=True), patch(
            "python_api.local_qlora.configured_local_training_python", return_value=sys.executable
        ), patch("python_api.local_qlora._probe_training_runtime", return_value=summary):
            with self.assertRaisesRegex(RuntimeError, "native compiler toolchain"):
                ensure_local_training_ready()

    def test_runtime_inspection_warns_when_compiler_is_missing(self) -> None:
        summary = runtime_summary()
        summary["compilerTools"] = {"cc": None, "cxx": None}

        with patch("python_api.local_qlora.local_training_enabled", return_value=True), patch(
            "python_api.local_qlora.configured_local_training_python", return_value=sys.executable
        ), patch("python_api.local_qlora._probe_training_runtime", return_value=summary):
            payload = inspect_local_training_runtime()

        warning_messages = payload.get("warnings") or []
        self.assertTrue(any("compiler toolchain" in str(message) for message in warning_messages))


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
                model_output_path=str(working_dir / "adapter"),
                config_path=str(working_dir / "local_train_config.json"),
                status_path=str(working_dir / "local_train_status.json"),
                events_path=str(working_dir / "local_train_events.jsonl"),
                log_path=str(working_dir / "local_train.log"),
                metrics_path=str(working_dir / "local_train_metrics.json"),
            )
            callback = LocalProgressCallback(config=config, status_path=Path(config.status_path), gpu_count=1)

            callback.on_epoch_begin(None, None, None)

    def test_fast_preset_uses_shorter_defaults(self) -> None:
        config = LocalQLoraJobConfig(
            job_id="job-fast",
            dataset_id="dataset-1",
            dataset_name="demo-dataset",
            dataset_path="/tmp/dataset.jsonl",
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            working_dir="/tmp/job-fast",
            output_dir="/tmp/job-fast/artifacts",
            model_output_path="/tmp/job-fast/adapter",
            config_path="/tmp/job-fast/local_train_config.json",
            status_path="/tmp/job-fast/local_train_status.json",
            events_path="/tmp/job-fast/local_train_events.jsonl",
            log_path="/tmp/job-fast/local_train.log",
            metrics_path="/tmp/job-fast/local_train_metrics.json",
            training_preset="fast",
        )

        resolved = config.resolved_hyperparameters()

        self.assertEqual(resolved["num_train_epochs"], 1)
        self.assertEqual(resolved["max_steps"], 60)
        self.assertEqual(resolved["gradient_accumulation_steps"], 4)
        self.assertEqual(resolved["max_seq_length"], 1024)
        self.assertEqual(resolved["lora_alpha"], 32)
        self.assertEqual(resolved["lora_dropout"], 0.0)
        self.assertEqual(resolved["target_module_strategy"], "attention_mlp")
        self.assertEqual(resolved["eval_max_samples"], 3)

    def test_job_config_persists_portable_upload_paths_and_resolves_them_on_load(self) -> None:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(DATASETS_DIR)) as dataset_dir, tempfile.TemporaryDirectory(
            dir=str(JOBS_DIR)
        ) as job_dir:
            dataset_path = Path(dataset_dir) / "dataset.jsonl"
            dataset_path.write_text('{"messages":[{"role":"user","content":"Hi"}]}\n', encoding="utf-8")
            working_dir = Path(job_dir)

            config = LocalQLoraJobConfig(
                job_id="job-portable",
                dataset_id="dataset-1",
                dataset_name="demo-dataset",
                dataset_path=str(dataset_path),
                base_model="Qwen/Qwen2.5-1.5B-Instruct",
                working_dir=str(working_dir),
                output_dir=str(working_dir / "artifacts"),
                model_output_path=str(working_dir / "adapter"),
                config_path=str(working_dir / "local_train_config.json"),
                status_path=str(working_dir / "local_train_status.json"),
                events_path=str(working_dir / "local_train_events.jsonl"),
                log_path=str(working_dir / "local_train.log"),
                metrics_path=str(working_dir / "local_train_metrics.json"),
            )

            payload = config.to_dict()
            self.assertEqual(payload["datasetPath"], f"datasets/{Path(dataset_dir).name}/dataset.jsonl")
            self.assertEqual(payload["configPath"], f"jobs/{working_dir.name}/local_train_config.json")
            self.assertEqual(payload["statusPath"], f"jobs/{working_dir.name}/local_train_status.json")

            config_path = working_dir / "local_train_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = LocalQLoraJobConfig.from_file(config_path)

        self.assertEqual(loaded.dataset_path, str(dataset_path))
        self.assertEqual(loaded.config_path, str(config_path))
        self.assertEqual(loaded.status_path, str(working_dir / "local_train_status.json"))

    def test_quality_preset_still_allows_manual_override(self) -> None:
        config = LocalQLoraJobConfig(
            job_id="job-quality",
            dataset_id="dataset-1",
            dataset_name="demo-dataset",
            dataset_path="/tmp/dataset.jsonl",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            working_dir="/tmp/job-quality",
            output_dir="/tmp/job-quality/artifacts",
            model_output_path="/tmp/job-quality/adapter",
            config_path="/tmp/job-quality/local_train_config.json",
            status_path="/tmp/job-quality/local_train_status.json",
            events_path="/tmp/job-quality/local_train_events.jsonl",
            log_path="/tmp/job-quality/local_train.log",
            metrics_path="/tmp/job-quality/local_train_metrics.json",
            training_preset="quality",
            hyperparameters={"learning_rate": 5e-5},
        )

        resolved = config.resolved_hyperparameters()

        self.assertEqual(resolved["per_device_train_batch_size"], 1)
        self.assertEqual(resolved["gradient_accumulation_steps"], 8)
        self.assertEqual(resolved["learning_rate"], 5e-5)
        self.assertEqual(resolved["lora_r"], 48)
        self.assertEqual(resolved["target_module_strategy"], "expanded")

    def test_llama32_instruction_records_use_chat_headers(self) -> None:
        formatted = format_training_record(
            {
                "instruction": "Summarize this case",
                "input": "A short contract dispute.",
                "output": "It is a contract dispute summary.",
            },
            "unsloth/Llama-3.2-1B-Instruct",
        )

        self.assertEqual(
            formatted["text"],
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Summarize this case\n\nA short contract dispute.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "It is a contract dispute summary.<|eot_id|>",
        )

    def test_tokenizer_chat_template_is_preferred_when_available(self) -> None:
        formatted = format_training_record(
            {
                "instruction": "Summarize this case",
                "input": "A short contract dispute.",
                "output": "It is a contract dispute summary.",
            },
            "Qwen/Qwen2.5-3B-Instruct",
            tokenizer=FakeChatTemplateTokenizer(),
        )

        self.assertEqual(
            formatted["text"],
            "TEMPLATE::user=Summarize this case\n\nA short contract dispute. || "
            "assistant=It is a contract dispute summary.::tokenize=False::gen=False",
        )

    def test_instruction_records_use_consistent_chat_fallback_without_template(self) -> None:
        formatted = format_training_record(
            {
                "instruction": "Say hello",
                "output": "Hello there!",
            },
            "Qwen/Qwen2.5-3B-Instruct",
        )

        self.assertEqual(
            formatted["text"],
            "### User:\nSay hello\n\n### Assistant:\nHello there!",
        )

    def test_build_evaluation_example_uses_generation_prompt(self) -> None:
        example = build_evaluation_example(
            {
                "instruction": "Summarize this case",
                "input": "A short contract dispute.",
                "output": "It is a contract dispute summary.",
            },
            model_id="Qwen/Qwen2.5-3B-Instruct",
            tokenizer=FakeChatTemplateTokenizer(),
        )

        self.assertIsNotNone(example)
        self.assertEqual(example["target"], "It is a contract dispute summary.")
        self.assertIn("gen=True", example["prompt"])

    def test_classic_falcon_models_use_falcon_target_modules_without_model_discovery(self) -> None:
        self.assertEqual(
            _resolve_target_modules("tiiuae/falcon-7b"),
            ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )

    def test_falcon3_models_prefer_loaded_model_leaf_names_over_repo_name(self) -> None:
        model = FakeDiscoveredTargetModuleModel(
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.assertEqual(
            _resolve_target_modules("tiiuae/Falcon3-3B-Instruct", model=model),
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    def test_target_module_strategy_can_limit_to_attention_layers(self) -> None:
        self.assertEqual(
            _resolve_target_modules(
                "Qwen/Qwen2.5-3B-Instruct",
                strategy="attention_only",
            ),
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    def test_ollama_modelfile_includes_inference_parameters_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "model.gguf"
            modelfile = Path(
                write_ollama_modelfile(
                    str(gguf_path),
                    inference_parameters={
                        "temperature": 0.18,
                        "top_p": 0.88,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        "num_ctx": 1536,
                    },
                )
            )

            contents = modelfile.read_text(encoding="utf-8")

        self.assertIn("PARAMETER temperature 0.18", contents)
        self.assertIn("PARAMETER top_p 0.88", contents)
        self.assertIn("PARAMETER top_k 40", contents)
        self.assertIn("PARAMETER repeat_penalty 1.1", contents)
        self.assertIn("PARAMETER num_ctx 1536", contents)


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
            model_output_path=str(working_dir / "adapter"),
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
                return_value={"evalLoss": 0.25, "perplexity": 1.28, "generationExactMatch": 0.5},
            ),
            patch(
                "python_api.local_qlora.train.run_holdout_generation_evaluation",
                return_value={"sampleCount": 2, "exactMatch": 0.5, "tokenF1": 0.6, "rougeL": 0.55},
            ),
            patch("python_api.local_qlora.train.local_gpu_count", return_value=1),
        ]

    def test_training_exports_gguf_via_fallback_when_unsloth_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir, push_to_ollama=False)
            gguf_path = Path(config.working_dir) / "gguf" / "model.q8_0.gguf"
            modelfile_path = gguf_path.parent / "Modelfile"
            patches = self.common_patches(peft_config=object())
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patch(
                "python_api.local_qlora.export.export_saved_adapter_to_gguf",
                side_effect=lambda **kwargs: str(gguf_path),
            ), patch(
                "python_api.local_qlora.export.write_ollama_modelfile",
                side_effect=lambda *args, **kwargs: str(modelfile_path),
            ):
                gguf_path.parent.mkdir(parents=True, exist_ok=True)
                gguf_path.write_text("gguf", encoding="utf-8")
                modelfile_path.write_text("FROM demo", encoding="utf-8")
                run_local_qlora_training(config)

            status = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            warning_messages = status.get("warnings") or []
            result_types = {item["type"] for item in status.get("resultFilesJson") or []}

            self.assertEqual(status["status"], "succeeded")
            self.assertFalse(status["ollamaRegistered"])
            self.assertEqual(warning_messages, [])
            self.assertIn("local_adapter", result_types)
            self.assertIn("local_gguf", result_types)
            self.assertIn("local_modelfile", result_types)

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

    def test_training_recovers_when_unsloth_gguf_export_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)
            config.gguf_quantization = "q8_0"
            gguf_path = Path(config.working_dir) / "gguf" / "model.q8_0.gguf"
            modelfile_path = gguf_path.parent / "Modelfile"

            def fake_fallback_export(**kwargs: object) -> str:
                gguf_path.parent.mkdir(parents=True, exist_ok=True)
                gguf_path.write_text("gguf", encoding="utf-8")
                return str(gguf_path)

            def fake_modelfile(*args: object, **kwargs: object) -> str:
                modelfile_path.write_text("FROM demo", encoding="utf-8")
                return str(modelfile_path)

            patches = self.common_patches(peft_config=None)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patch(
                "python_api.local_qlora.export.export_gguf",
                side_effect=RuntimeError("Unsloth: GGUF conversion failed: EOF when reading a line"),
            ), patch(
                "python_api.local_qlora.export.export_saved_adapter_to_gguf",
                side_effect=fake_fallback_export,
            ), patch(
                "python_api.local_qlora.export.write_ollama_modelfile",
                side_effect=fake_modelfile,
            ), patch(
                "python_api.local_qlora.export.push_to_ollama",
            ) as push_mock:
                run_local_qlora_training(config)

            status = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            warning_messages = status.get("warnings") or []
            result_types = {item["type"] for item in status.get("resultFilesJson") or []}

            self.assertEqual(status["status"], "succeeded")
            self.assertFalse(warning_messages)
            self.assertTrue(status["ollamaRegistered"])
            self.assertIn("local_gguf", result_types)
            self.assertIn("local_modelfile", result_types)
            push_mock.assert_called_once()


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
            model_output_path=str(working_dir / "adapter"),
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
