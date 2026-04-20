from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from python_api.env import ensure_env_loaded
from python_api.errors import ApiError
from python_api.agentic_workflow import AgentRunWorkflow
from python_api.services import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_BASE_MODEL,
    cancel_job_record,
    build_job_download_package,
    create_model_profile,
    create_agent_run_record,
    create_dataset_record,
    create_job_record,
    dashboard_summary,
    delete_model_profile,
    get_provider_display_name,
    get_agent_run_detail,
    list_model_profiles,
    list_agent_runs,
    list_datasets,
    list_job_events,
    list_jobs,
    provider_is_configured,
    mark_agent_run_cancelled,
    provider_status_payload,
    retrieve_dataset_detail,
    retrieve_job_detail,
    run_playground_prompt,
    save_agent_run_snapshot,
    sync_job_record,
    update_model_profile,
    update_model_profile_defaults,
    upload_dataset_record_to_openai,
)
from python_api.temporal_runtime import temporal_runtime

ensure_env_loaded()


def error_response(code: str, message: str, status: int = 400, details: list[Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
            }
        },
    )


def handle_api_error(error: Exception) -> JSONResponse:
    if isinstance(error, ApiError):
        return error_response(error.code, error.message, error.status, error.details)
    return error_response("INTERNAL_SERVER_ERROR", str(error), 500)


class CreateJobRequest(BaseModel):
    datasetId: str
    baseModel: str = DEFAULT_BASE_MODEL
    modelProvider: str | None = None
    hyperparameters: dict[str, Any] | None = None
    exportGguf: bool = True
    ggufQuantization: str = "q4_k_m"
    pushToOllama: bool = False
    ollamaModelName: str = ""
    numEpochs: int | None = None
    learningRate: float | None = None
    perDeviceBatchSize: int | None = None


class PlaygroundRunRequest(BaseModel):
    baseModel: str = DEFAULT_BASE_MODEL
    baseModelProvider: str | None = None
    fineTunedModel: str | None = None
    fineTunedModelProvider: str | None = None
    prompt: str


class CreateAgentRunRequest(BaseModel):
    goal: str
    datasetId: str | None = None
    baseModel: str = DEFAULT_BASE_MODEL
    baseModelProvider: str | None = None
    evaluationPrompt: str | None = None
    agentModel: str | None = DEFAULT_AGENT_MODEL
    agentModelProvider: str | None = None


class ModelProfileRequest(BaseModel):
    name: str
    provider: str
    model: str
    category: str = "custom"
    description: str | None = None


class ModelProfileDefaultsRequest(BaseModel):
    playgroundBaseProfileId: str | None = None
    playgroundCompareProfileId: str | None = None
    agentBaseProfileId: str | None = None
    agentModelProfileId: str | None = None
    jobBaseProfileId: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup_task = asyncio.create_task(temporal_runtime.start())
    try:
        yield
    finally:
        if not startup_task.done():
            startup_task.cancel()
        await temporal_runtime.stop()


app = FastAPI(title="Agentic Python API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "temporal": temporal_runtime.status_payload()}


@app.get("/dashboard/summary")
def get_dashboard_summary():
    try:
        return dashboard_summary()
    except Exception as error:  # pragma: no cover
        return handle_api_error(error)


@app.get("/settings/model-profiles")
def get_model_profiles():
    try:
        return list_model_profiles()
    except Exception as error:
        return handle_api_error(error)


@app.post("/settings/model-profiles")
def post_model_profile(request: ModelProfileRequest):
    try:
        return create_model_profile(
            request.name,
            request.provider,
            request.model,
            request.category,
            request.description,
        )
    except Exception as error:
        return handle_api_error(error)


@app.patch("/settings/model-profiles/defaults")
def patch_model_profile_defaults(request: ModelProfileDefaultsRequest):
    try:
        return update_model_profile_defaults(request.model_dump())
    except Exception as error:
        return handle_api_error(error)


@app.patch("/settings/model-profiles/{profile_id}")
def patch_model_profile(profile_id: str, request: ModelProfileRequest):
    try:
        return update_model_profile(
            profile_id,
            request.name,
            request.provider,
            request.model,
            request.category,
            request.description,
        )
    except Exception as error:
        return handle_api_error(error)


@app.delete("/settings/model-profiles/{profile_id}")
def remove_model_profile(profile_id: str):
    try:
        return delete_model_profile(profile_id)
    except Exception as error:
        return handle_api_error(error)


@app.get("/datasets")
def get_datasets(validationStatus: str | None = None):
    try:
        return list_datasets(validationStatus)
    except Exception as error:  # pragma: no cover
        return handle_api_error(error)


@app.post("/datasets")
async def create_dataset(file: UploadFile = File(...), name: str | None = Form(None)):
    try:
        original_filename = file.filename or "dataset.jsonl"
        payload = await file.read()
        return create_dataset_record(original_filename, payload, name)
    except Exception as error:  # pragma: no cover
        return handle_api_error(error)


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return retrieve_dataset_detail(dataset_id)
    except Exception as error:
        return handle_api_error(error)


@app.post("/datasets/{dataset_id}/upload-to-openai")
def upload_dataset_to_openai(dataset_id: str):
    try:
        return upload_dataset_record_to_openai(dataset_id)
    except Exception as error:
        return handle_api_error(error)


@app.get("/jobs")
def get_jobs():
    try:
        return list_jobs()
    except Exception as error:  # pragma: no cover
        return handle_api_error(error)


@app.post("/jobs")
def create_job(request: CreateJobRequest):
    try:
        return create_job_record(
            request.datasetId,
            request.baseModel,
            request.hyperparameters,
            request.modelProvider,
            export_gguf=request.exportGguf,
            gguf_quantization=request.ggufQuantization,
            push_to_ollama=request.pushToOllama,
            ollama_model_name=request.ollamaModelName,
            num_epochs=request.numEpochs,
            learning_rate=request.learningRate,
            per_device_batch_size=request.perDeviceBatchSize,
        )
    except Exception as error:
        return handle_api_error(error)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return retrieve_job_detail(job_id)
    except Exception as error:
        return handle_api_error(error)


@app.post("/jobs/{job_id}/sync")
def sync_job(job_id: str):
    try:
        return sync_job_record(job_id)
    except Exception as error:
        return handle_api_error(error)


@app.get("/jobs/{job_id}/events")
def get_job_events(job_id: str):
    try:
        return list_job_events(job_id)
    except Exception as error:
        return handle_api_error(error)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return cancel_job_record(job_id)
    except Exception as error:
        return handle_api_error(error)


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str, type: str = "adapter"):
    try:
        package = build_job_download_package(job_id, artifact_type=type)
        return FileResponse(package["path"], media_type=package["mediaType"], filename=package["filename"])
    except Exception as error:
        return handle_api_error(error)


@app.post("/playground/run")
def run_playground(request: PlaygroundRunRequest):
    try:
        return run_playground_prompt(
            request.prompt,
            request.baseModel,
            request.fineTunedModel,
            request.baseModelProvider,
            request.fineTunedModelProvider,
        )
    except Exception as error:
        return handle_api_error(error)


@app.get("/agent/runtime")
def get_agent_runtime():
    return temporal_runtime.status_payload()


@app.get("/agent/runs")
def get_agent_runs():
    try:
        return list_agent_runs()
    except Exception as error:
        return handle_api_error(error)


@app.post("/agent/runs")
async def create_agent_run(request: CreateAgentRunRequest):
    if not request.goal.strip():
        return error_response("AGENT_GOAL_REQUIRED", "A goal is required to start an agent run.", 400)

    required_providers = {
        request.baseModelProvider or None,
        request.agentModelProvider or request.baseModelProvider or None,
    }
    unresolved = [provider for provider in required_providers if provider and not provider_is_configured(provider)]
    if unresolved:
        return error_response(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            (
                "Set the required credentials for "
                f"{', '.join(get_provider_display_name(provider) for provider in sorted(unresolved))} "
                "before starting an agent run."
            ),
            503,
            [provider_status_payload()],
        )

    runtime_ready = await temporal_runtime.start()
    if not runtime_ready or not temporal_runtime.client:
        status = temporal_runtime.status_payload()
        return error_response(
            "AGENT_RUNTIME_UNAVAILABLE",
            status.get("error") or "Temporal is still starting. Try again in a few seconds.",
            503,
            [status],
        )

    run_id = uuid4().hex
    workflow_id = f"agent-run-{run_id}"
    snapshot = {
        "id": run_id,
        "workflowId": workflow_id,
        "status": "queued",
        "goal": request.goal.strip(),
        "datasetId": request.datasetId,
        "baseModel": request.baseModel or DEFAULT_BASE_MODEL,
        "baseModelProvider": request.baseModelProvider,
        "evaluationPrompt": request.evaluationPrompt,
        "agentModel": request.agentModel or DEFAULT_AGENT_MODEL,
        "agentModelProvider": request.agentModelProvider,
        "summary": None,
        "latestJobId": None,
        "fineTunedModel": None,
        "fineTunedModelProvider": None,
        "lastError": None,
        "latestResponseId": None,
        "sleepSeconds": None,
        "steps": [],
    }

    try:
        await temporal_runtime.client.start_workflow(
            AgentRunWorkflow.run,
            snapshot,
            id=workflow_id,
            task_queue=temporal_runtime.task_queue,
        )
        return create_agent_run_record(
            run_id=run_id,
            workflow_id=workflow_id,
            goal=request.goal.strip(),
            dataset_id=request.datasetId,
            base_model=request.baseModel or DEFAULT_BASE_MODEL,
            base_model_provider=request.baseModelProvider,
            evaluation_prompt=request.evaluationPrompt,
            agent_model=request.agentModel or DEFAULT_AGENT_MODEL,
            agent_model_provider=request.agentModelProvider,
        )
    except Exception as error:  # pragma: no cover
        return handle_api_error(ApiError("AGENT_RUN_FAILED", str(error), 500))


@app.get("/agent/runs/{run_id}")
async def get_agent_run(run_id: str):
    try:
        snapshot = get_agent_run_detail(run_id)
    except Exception as error:
        return handle_api_error(error)

    if snapshot.get("status") in {"queued", "running", "waiting"}:
        runtime_ready = temporal_runtime.status == "ready" or await temporal_runtime.start()
        if runtime_ready and temporal_runtime.client:
            try:
                handle = temporal_runtime.client.get_workflow_handle(snapshot["workflowId"])
                live_snapshot = await handle.query("snapshot")
                return save_agent_run_snapshot(live_snapshot)
            except Exception as error:  # pragma: no cover
                snapshot["runtimeError"] = str(error)
                return snapshot

    return snapshot


@app.post("/agent/runs/{run_id}/cancel")
async def cancel_agent_run(run_id: str):
    try:
        snapshot = get_agent_run_detail(run_id)
    except Exception as error:
        return handle_api_error(error)

    if snapshot.get("status") in {"succeeded", "failed", "cancelled"}:
        return snapshot

    runtime_ready = temporal_runtime.status == "ready" or await temporal_runtime.start()
    if not runtime_ready or not temporal_runtime.client:
        status = temporal_runtime.status_payload()
        return error_response(
            "AGENT_RUNTIME_UNAVAILABLE",
            status.get("error") or "Temporal is unavailable, so the run could not be cancelled cleanly.",
            503,
            [status],
        )

    try:
        handle = temporal_runtime.client.get_workflow_handle(snapshot["workflowId"])
        await handle.cancel()
    except Exception as error:  # pragma: no cover
        return handle_api_error(ApiError("AGENT_CANCEL_FAILED", str(error), 500))

    try:
        return mark_agent_run_cancelled(run_id)
    except Exception as error:
        return handle_api_error(error)
