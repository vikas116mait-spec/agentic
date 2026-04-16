from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow


@workflow.defn
class AgentRunWorkflow:
    def __init__(self) -> None:
        self.snapshot: dict[str, Any] = {}

    @workflow.run
    async def run(self, initial_snapshot: dict[str, Any]) -> dict[str, Any]:
        self.snapshot = dict(initial_snapshot)
        self.snapshot["status"] = "running"
        while True:
            updated_snapshot = await workflow.execute_activity(
                "agent_turn",
                self.snapshot,
                start_to_close_timeout=timedelta(minutes=5),
                summary=f"Agent turn for {self.snapshot.get('id', 'unknown-run')}",
            )
            self.snapshot = dict(updated_snapshot)
            if self.snapshot.get("status") in {"succeeded", "failed", "cancelled"}:
                return self.snapshot

            wait_seconds = int(self.snapshot.get("sleepSeconds") or 0)
            if wait_seconds > 0:
                await workflow.sleep(wait_seconds, summary=f"Waiting {wait_seconds}s before next agent turn")
                self.snapshot["status"] = "running"
                self.snapshot["sleepSeconds"] = None

    @workflow.query(name="snapshot")
    def snapshot_query(self) -> dict[str, Any]:
        return self.snapshot
