from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

from temporalio import activity

from python_api.errors import ApiError
from python_api.services import (
    DEFAULT_AGENT_MODEL,
    create_job_record,
    get_model_client,
    get_model_provider,
    get_provider_display_name,
    list_datasets,
    list_jobs,
    provider_supports_fine_tuning,
    retrieve_dataset_detail,
    run_playground_prompt,
    save_agent_run_snapshot,
    sync_job_record,
    upload_dataset_record_to_openai,
)
from python_api.store import utc_now_iso


AGENT_INSTRUCTIONS = """
You are the orchestration agent for a fine-tuning workspace.

Your job is to complete the user's fine-tuning goal using tools and only report facts grounded in tool results.

Rules:
- Do not invent dataset IDs, job IDs, model names, or statuses.
- Prefer the pinned dataset_id from the run context when one is provided.
- Before creating a fine-tuning job, make sure the dataset is valid and uploaded to the training provider when that mode supports it.
- After creating or discovering a fine-tuning job, sync it before making claims about its state.
- If a job is still active, call wait_for_seconds instead of ending the run early.
- If an evaluation prompt is available and the fine-tuned model has succeeded, run the playground comparison before finishing.
- Keep final summaries concise and operational.
""".strip()


BASE_AGENT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_datasets",
        "description": "List datasets available in the workspace so you can choose one for fine-tuning.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "validation_status": {
                    "type": "string",
                    "enum": ["VALID", "INVALID", "PENDING"],
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_dataset",
        "description": "Retrieve the details and validation summary for a single dataset.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "upload_dataset_to_openai",
        "description": "Upload a validated dataset to the OpenAI Files API so it can be used for fine-tuning.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_jobs",
        "description": "List fine-tuning jobs already created in the workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_job",
        "description": "Create a supervised fine-tuning job from a dataset.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "base_model": {"type": "string"},
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "sync_job",
        "description": "Fetch the latest status and recent events for a fine-tuning job.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_playground",
        "description": "Run a prompt against the base model and the fine-tuned model to compare outputs.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "base_model": {"type": "string"},
                "fine_tuned_model": {"type": "string"},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "wait_for_seconds",
        "description": "Pause the workflow for a short period before continuing to monitor an active job.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "minimum": 10, "maximum": 300},
                "reason": {"type": "string"},
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
]

MAX_TOOL_LOOPS = 8
LOCAL_UNSUPPORTED_TOOL_NAMES = {"upload_dataset_to_openai", "create_job", "sync_job", "wait_for_seconds"}


def _truncate(value: str, limit: int = 220) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _step(title: str, detail: str, *, kind: str = "tool", status: str = "completed") -> dict[str, Any]:
    return {
        "id": f"step-{utc_now_iso()}",
        "title": title,
        "detail": _truncate(detail),
        "kind": kind,
        "status": status,
        "createdAt": utc_now_iso(),
    }


def _build_run_context(snapshot: dict[str, Any], *, continuing: bool) -> str:
    recent_steps = snapshot.get("steps", [])[-5:]
    step_lines = [f"- {step['title']}: {step['detail']}" for step in recent_steps] or ["- No steps recorded yet."]

    header = "Continue the active run." if continuing else "Start a new agentic fine-tuning run."
    base_provider = snapshot.get("baseModelProvider") or get_model_provider()
    agent_provider = snapshot.get("agentModelProvider") or base_provider
    provider_lines = [
        f"Base model provider: {get_provider_display_name(base_provider)} ({base_provider})",
        f"Agent model provider: {get_provider_display_name(agent_provider)} ({agent_provider})",
        (
            "Managed fine-tuning support: available for the selected base provider."
            if provider_supports_fine_tuning(base_provider)
            else "Managed fine-tuning support: unavailable for the selected base provider. Focus on dataset review and playground runs."
        ),
    ]

    return "\n".join(
        [
            header,
            *provider_lines,
            f"Goal: {snapshot['goal']}",
            f"Pinned dataset id: {snapshot.get('datasetId') or 'none'}",
            f"Preferred base model: {snapshot.get('baseModel') or 'none'}",
            f"Preferred base model provider: {snapshot.get('baseModelProvider') or 'none'}",
            f"Evaluation prompt: {snapshot.get('evaluationPrompt') or 'none'}",
            f"Agent model: {snapshot.get('agentModel') or 'none'}",
            f"Agent model provider: {snapshot.get('agentModelProvider') or 'none'}",
            f"Latest known job id: {snapshot.get('latestJobId') or 'none'}",
            f"Latest known fine-tuned model: {snapshot.get('fineTunedModel') or 'none'}",
            f"Latest known fine-tuned model provider: {snapshot.get('fineTunedModelProvider') or 'none'}",
            "Recent run history:",
            *step_lines,
        ]
    )


def _model_name(snapshot: dict[str, Any]) -> str:
    return snapshot.get("agentModel") or DEFAULT_AGENT_MODEL


def _agent_provider(snapshot: dict[str, Any]) -> str:
    return snapshot.get("agentModelProvider") or get_model_provider()


def _response_tool_definitions() -> list[dict[str, Any]]:
    if provider_supports_fine_tuning():
        return deepcopy(BASE_AGENT_TOOL_DEFINITIONS)
    return [tool for tool in deepcopy(BASE_AGENT_TOOL_DEFINITIONS) if tool["name"] not in LOCAL_UNSUPPORTED_TOOL_NAMES]


def _chat_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
                "strict": tool["strict"],
            },
        }
        for tool in _response_tool_definitions()
    ]


def _call_agent_model(snapshot: dict[str, Any], input_items: list[dict[str, Any]], previous_response_id: str | None):
    client = get_model_client(_agent_provider(snapshot), model=_model_name(snapshot))
    request: dict[str, Any] = {
        "model": _model_name(snapshot),
        "instructions": AGENT_INSTRUCTIONS,
        "input": input_items,
        "tools": _response_tool_definitions(),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": True,
        "max_output_tokens": 900,
    }
    if previous_response_id:
        request["previous_response_id"] = previous_response_id
    return client.responses.create(**request)


def _tool_result_payload(output: Any) -> str:
    return json.dumps(output, default=str)


def _assistant_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    serialized = []
    for tool_call in tool_calls or []:
        serialized.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
        )
    return serialized


def _execute_tool(name: str, args: dict[str, Any], snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if name == "list_datasets":
        datasets = [
            {
                "id": dataset["id"],
                "name": dataset["name"],
                "validationStatus": dataset["validationStatus"],
                "recordCount": dataset["recordCount"],
                "openaiFileId": dataset.get("openaiFileId"),
                "createdAt": dataset["createdAt"],
            }
            for dataset in list_datasets(args.get("validation_status"))
        ]
        return (
            {
                "output": {"datasets": datasets},
                "step": _step("Listed datasets", f"Found {len(datasets)} datasets in the workspace."),
            },
            {},
        )

    if name == "get_dataset":
        dataset = retrieve_dataset_detail(args["dataset_id"])
        return (
            {
                "output": dataset,
                "step": _step(
                    "Loaded dataset details",
                    f"Dataset {dataset['id']} is {dataset['validationStatus']} with {dataset['recordCount']} records.",
                ),
            },
            {"datasetId": dataset["id"]},
        )

    if name == "upload_dataset_to_openai":
        dataset = upload_dataset_record_to_openai(args["dataset_id"])
        return (
            {
                "output": dataset,
                "step": _step(
                    "Uploaded dataset to OpenAI",
                    f"Dataset {dataset['id']} is now linked to file {dataset['openaiFileId']}.",
                ),
            },
            {"datasetId": dataset["id"]},
        )

    if name == "list_jobs":
        jobs = [
            {
                "id": job["id"],
                "datasetId": job["datasetId"],
                "datasetName": job.get("datasetName"),
                "status": job["status"],
                "baseModel": job["baseModel"],
                "fineTunedModel": job.get("fineTunedModel"),
                "createdAt": job["createdAt"],
            }
            for job in list_jobs()
        ]
        return (
            {
                "output": {"jobs": jobs},
                "step": _step("Listed jobs", f"Found {len(jobs)} fine-tuning jobs."),
            },
            {},
        )

    if name == "create_job":
        dataset_id = args["dataset_id"]
        base_model = args.get("base_model") or snapshot.get("baseModel")
        base_provider = snapshot.get("baseModelProvider")
        job = create_job_record(dataset_id, base_model, provider=base_provider)
        return (
            {
                "output": job,
                "step": _step(
                    "Created fine-tuning job",
                    f"Created job {job['id']} for dataset {dataset_id} using {job['baseModel']}.",
                ),
            },
            {
                "datasetId": dataset_id,
                "baseModel": job["baseModel"],
                "baseModelProvider": job.get("modelProvider") or base_provider,
                "latestJobId": job["id"],
                "fineTunedModel": job.get("fineTunedModel"),
                "fineTunedModelProvider": job.get("modelProvider"),
            },
        )

    if name == "sync_job":
        job = sync_job_record(args["job_id"])
        detail = f"Job {job['id']} is {job['status']}."
        if job.get("fineTunedModel"):
            detail = f"{detail} Fine-tuned model: {job['fineTunedModel']}."
        return (
            {
                "output": job,
                "step": _step("Synced fine-tuning job", detail),
            },
            {
                "latestJobId": job["id"],
                "fineTunedModel": job.get("fineTunedModel"),
                "fineTunedModelProvider": job.get("modelProvider"),
            },
        )

    if name == "run_playground":
        prompt = args["prompt"]
        base_model = args.get("base_model") or snapshot.get("baseModel")
        fine_tuned_model = args.get("fine_tuned_model") or snapshot.get("fineTunedModel")
        result = run_playground_prompt(
            prompt,
            base_model,
            fine_tuned_model,
            snapshot.get("baseModelProvider"),
            snapshot.get("fineTunedModelProvider"),
        )
        return (
            {
                "output": result,
                "step": _step(
                    "Ran playground comparison",
                    "Generated side-by-side outputs for the base model and fine-tuned model.",
                ),
            },
            {},
        )

    if name == "wait_for_seconds":
        seconds = max(10, min(int(args["seconds"]), 300))
        reason = args.get("reason") or "Waiting before the next sync."
        return (
            {
                "output": {"seconds": seconds, "reason": reason},
                "step": _step("Paused run", f"Waiting {seconds} seconds. {reason}", kind="wait", status="waiting"),
            },
            {"sleepSeconds": seconds},
        )

    raise ApiError("UNKNOWN_TOOL", f"Unknown tool `{name}`.", 400)


def _complete_snapshot(working_snapshot: dict[str, Any], summary: str) -> dict[str, Any]:
    working_snapshot["summary"] = summary
    working_snapshot["status"] = "succeeded"
    working_snapshot["updatedAt"] = utc_now_iso()
    working_snapshot["finishedAt"] = utc_now_iso()
    working_snapshot["steps"].append(
        _step("Completed agent run", summary, kind="summary", status="completed")
    )
    return working_snapshot


def _fail_snapshot(working_snapshot: dict[str, Any], message: str) -> dict[str, Any]:
    working_snapshot["status"] = "failed"
    working_snapshot["lastError"] = message
    working_snapshot["updatedAt"] = utc_now_iso()
    working_snapshot["finishedAt"] = utc_now_iso()
    working_snapshot["steps"].append(
        _step("Agent run failed", message, kind="summary", status="failed")
    )
    return working_snapshot


def _run_openai_agent_turn(snapshot: dict[str, Any]) -> dict[str, Any]:
    working_snapshot = deepcopy(snapshot)
    working_snapshot["status"] = "running"
    working_snapshot["sleepSeconds"] = None
    working_snapshot["updatedAt"] = utc_now_iso()
    previous_response_id = working_snapshot.get("latestResponseId")

    input_items: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_run_context(working_snapshot, continuing=bool(previous_response_id)),
        }
    ]

    for _ in range(MAX_TOOL_LOOPS):
        response = _call_agent_model(working_snapshot, input_items, previous_response_id)
        previous_response_id = response.id
        working_snapshot["latestResponseId"] = response.id

        tool_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not tool_calls:
            summary = response.output_text.strip() or "The run completed without a text summary."
            return _complete_snapshot(working_snapshot, summary)

        next_input_items: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments or "{}")
                tool_result, snapshot_updates = _execute_tool(tool_call.name, arguments, working_snapshot)
                working_snapshot.update({key: value for key, value in snapshot_updates.items() if value is not None})
                working_snapshot["steps"].append(tool_result["step"])
                next_input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": _tool_result_payload(tool_result["output"]),
                    }
                )
                if snapshot_updates.get("sleepSeconds"):
                    working_snapshot["status"] = "waiting"
                    working_snapshot["updatedAt"] = utc_now_iso()
                    return working_snapshot
            except ApiError as error:
                failure_step = _step(tool_call.name, error.message, kind="tool", status="failed")
                working_snapshot["steps"].append(failure_step)
                next_input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": _tool_result_payload(
                            {
                                "ok": False,
                                "error": {
                                    "code": error.code,
                                    "message": error.message,
                                    "details": error.details,
                                },
                            }
                        ),
                    }
                )

        working_snapshot["updatedAt"] = utc_now_iso()
        input_items = next_input_items

    return _fail_snapshot(working_snapshot, "The agent exceeded the maximum tool-call loop limit.")


def _run_local_agent_turn(snapshot: dict[str, Any]) -> dict[str, Any]:
    working_snapshot = deepcopy(snapshot)
    working_snapshot["status"] = "running"
    working_snapshot["sleepSeconds"] = None
    working_snapshot["updatedAt"] = utc_now_iso()
    working_snapshot["latestResponseId"] = None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": _build_run_context(working_snapshot, continuing=False)},
    ]

    client = get_model_client(_agent_provider(working_snapshot), model=_model_name(working_snapshot))
    tools = _chat_tool_definitions()

    for _ in range(MAX_TOOL_LOOPS):
        completion = client.chat.completions.create(
            model=_model_name(working_snapshot),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = completion.choices[0].message
        tool_calls = list(message.tool_calls or [])

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = _assistant_tool_calls(tool_calls)
        messages.append(assistant_message)

        if not tool_calls:
            summary = (message.content or "").strip() or "The run completed without a text summary."
            return _complete_snapshot(working_snapshot, summary)

        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                tool_result, snapshot_updates = _execute_tool(tool_call.function.name, arguments, working_snapshot)
                working_snapshot.update({key: value for key, value in snapshot_updates.items() if value is not None})
                working_snapshot["steps"].append(tool_result["step"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _tool_result_payload(tool_result["output"]),
                    }
                )
                if snapshot_updates.get("sleepSeconds"):
                    working_snapshot["status"] = "waiting"
                    working_snapshot["updatedAt"] = utc_now_iso()
                    return working_snapshot
            except ApiError as error:
                failure_step = _step(tool_call.function.name, error.message, kind="tool", status="failed")
                working_snapshot["steps"].append(failure_step)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _tool_result_payload(
                            {
                                "ok": False,
                                "error": {
                                    "code": error.code,
                                    "message": error.message,
                                    "details": error.details,
                                },
                            }
                        ),
                    }
                )

        working_snapshot["updatedAt"] = utc_now_iso()

    return _fail_snapshot(working_snapshot, "The agent exceeded the maximum tool-call loop limit.")


def _run_agent_turn(snapshot: dict[str, Any]) -> dict[str, Any]:
    if _agent_provider(snapshot) == "openai":
        return _run_openai_agent_turn(snapshot)
    return _run_local_agent_turn(snapshot)


@activity.defn(name="agent_turn")
async def run_agent_turn_activity(snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        updated_snapshot = await asyncio.to_thread(_run_agent_turn, snapshot)
    except Exception as error:  # pragma: no cover
        failed_snapshot = deepcopy(snapshot)
        failed_snapshot["status"] = "failed"
        failed_snapshot["lastError"] = str(error)
        failed_snapshot["updatedAt"] = utc_now_iso()
        failed_snapshot["finishedAt"] = utc_now_iso()
        failed_snapshot["steps"] = failed_snapshot.get("steps", []) + [
            _step("Agent run failed", str(error), kind="summary", status="failed")
        ]
        save_agent_run_snapshot(failed_snapshot)
        raise

    save_agent_run_snapshot(updated_snapshot)
    return updated_snapshot
