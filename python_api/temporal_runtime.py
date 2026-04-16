from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from python_api.agentic_activities import run_agent_turn_activity
from python_api.agentic_workflow import AgentRunWorkflow
from python_api.services import provider_status_payload


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_host_port(target: str) -> tuple[str, int]:
    host, _, port = target.partition(":")
    if not host or not port:
        return "127.0.0.1", 7233
    return host, int(port)

class TemporalRuntime:
    def __init__(self) -> None:
        self.status = "stopped"
        self.mode = "uninitialized"
        self.error: str | None = None
        self.address = os.environ.get("TEMPORAL_ADDRESS", "127.0.0.1:7233")
        self.namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
        self.task_queue = os.environ.get("TEMPORAL_TASK_QUEUE", "agentic-agent-queue")
        self.auto_start_dev_server = _env_flag("TEMPORAL_AUTO_START_DEV_SERVER", True)
        self.dev_server_ui = _env_flag("TEMPORAL_DEV_SERVER_UI", False)
        self.client: Client | None = None
        self.environment: WorkflowEnvironment | None = None
        self.worker: Worker | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> bool:
        while self.status == "starting":
            await asyncio.sleep(0.2)

        async with self._lock:
            if self.status == "ready":
                return True

            self.status = "starting"
            self.error = None

            try:
                await self._connect_client()
                self.worker = Worker(
                    self.client,
                    task_queue=self.task_queue,
                    workflows=[AgentRunWorkflow],
                    activities=[run_agent_turn_activity],
                )
                self.worker_task = asyncio.create_task(self.worker.run())
                self.worker_task.add_done_callback(self._handle_worker_completion)
                self.status = "ready"
                return True
            except Exception as error:  # pragma: no cover
                self.status = "error"
                self.error = str(error)
                return False

    async def _connect_client(self) -> None:
        if self.client:
            return

        if self.auto_start_dev_server:
            try:
                self.client = await Client.connect(self.address, namespace=self.namespace)
                self.mode = "connected_existing_server"
                return
            except Exception:
                host, port = _parse_host_port(self.address)
                self.environment = await WorkflowEnvironment.start_local(
                    namespace=self.namespace,
                    ip=host,
                    port=port,
                    ui=self.dev_server_ui,
                    dev_server_log_level="warn",
                )
                self.client = self.environment.client
                self.mode = "embedded_dev_server"
                return

        self.client = await Client.connect(self.address, namespace=self.namespace)
        self.mode = "external_server"

    def _handle_worker_completion(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as error:  # pragma: no cover
            self.status = "error"
            self.error = f"Temporal worker stopped unexpectedly: {error}"

    async def stop(self) -> None:
        async with self._lock:
            if self.worker:
                await self.worker.shutdown()
            if self.worker_task:
                try:
                    await self.worker_task
                except Exception:
                    pass
            if self.environment:
                await self.environment.shutdown()
            self.worker = None
            self.worker_task = None
            self.environment = None
            self.client = None
            self.status = "stopped"
            self.mode = "uninitialized"

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "address": self.address,
            "namespace": self.namespace,
            "taskQueue": self.task_queue,
            "error": self.error,
            "autoStartDevServer": self.auto_start_dev_server,
            **provider_status_payload(),
        }


temporal_runtime = TemporalRuntime()
