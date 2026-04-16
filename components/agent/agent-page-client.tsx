"use client";

import { useEffect, useMemo, useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { Textarea } from "@/components/ui/textarea";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate } from "@/lib/utils";

type RuntimeStatus = {
  status: string;
  mode: string;
  address: string;
  namespace: string;
  taskQueue: string;
  error: string | null;
  autoStartDevServer: boolean;
  modelProvider: string;
  modelProviderLabel: string;
  providerBaseUrl: string | null;
  providerConfigured: boolean;
  supportsFineTuning: boolean;
  defaultBaseModel: string;
  defaultAgentModel: string;
};

type DatasetOption = {
  id: string;
  name: string;
  validationStatus: string;
  recordCount: number;
  createdAt: string;
};

type AgentStep = {
  id: string;
  title: string;
  detail: string;
  kind: string;
  status: string;
  createdAt: string;
};

type AgentRun = {
  id: string;
  workflowId: string;
  status: string;
  goal: string;
  baseModel: string;
  datasetId: string | null;
  evaluationPrompt: string | null;
  agentModel: string | null;
  summary: string | null;
  latestJobId: string | null;
  fineTunedModel: string | null;
  lastError: string | null;
  createdAt: string;
  updatedAt: string;
  finishedAt: string | null;
  steps: AgentStep[];
};

const ACTIVE_STATUSES = new Set(["queued", "running", "waiting"]);
const OPENAI_RUN_GOAL =
  "Take my latest valid dataset, upload it to OpenAI, create a supervised fine-tune, keep monitoring the job until it finishes, and run the evaluation prompt once the model is ready.";
const LOCAL_RUN_GOAL =
  "Review my latest valid dataset, tell me whether it looks ready for fine-tuning later, and run the evaluation prompt against my local model so I can iterate cheaply for now.";

function mergeRun(runs: AgentRun[], updatedRun: AgentRun) {
  const withoutUpdated = runs.filter((run) => run.id !== updatedRun.id);
  return [updatedRun, ...withoutUpdated].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
}

export function AgentPageClient() {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<AgentRun | null>(null);
  const [goal, setGoal] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [evaluationPrompt, setEvaluationPrompt] = useState("Summarize the user request in one sentence.");
  const [agentModel, setAgentModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === datasetId) ?? null,
    [datasetId, datasets]
  );

  useEffect(() => {
    async function loadInitialState() {
      try {
        const [runtimePayload, datasetPayload, runPayload] = await Promise.all([
          pythonApiFetch<RuntimeStatus>("/agent/runtime"),
          pythonApiFetch<DatasetOption[]>("/datasets"),
          pythonApiFetch<AgentRun[]>("/agent/runs")
        ]);
        setRuntime(runtimePayload);
        setDatasets(datasetPayload);
        setRuns(runPayload);
        setSelectedRunId((current) => current ?? runPayload[0]?.id ?? null);
        setGoal((current) => current || (runtimePayload.supportsFineTuning ? OPENAI_RUN_GOAL : LOCAL_RUN_GOAL));
        setBaseModel((current) => current || runtimePayload.defaultBaseModel);
        setAgentModel((current) => current || runtimePayload.defaultAgentModel);
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Could not load the agent workspace.");
      } finally {
        setLoading(false);
      }
    }

    void loadInitialState();
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setSelectedRun(null);
      return;
    }

    let cancelled = false;
    let intervalId: number | null = null;

    async function fetchRun() {
      try {
        const run = await pythonApiFetch<AgentRun>(`/agent/runs/${selectedRunId}`);
        if (cancelled) {
          return;
        }
        setSelectedRun(run);
        setRuns((current) => mergeRun(current, run));
        if (!ACTIVE_STATUSES.has(run.status) && intervalId !== null) {
          window.clearInterval(intervalId);
        }
      } catch (requestError) {
        if (!cancelled) {
          setActionMessage(requestError instanceof Error ? requestError.message : "Could not refresh the run.");
        }
      }
    }

    void fetchRun();
    intervalId = window.setInterval(() => {
      void fetchRun();
    }, 3000);

    return () => {
      cancelled = true;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [selectedRunId]);

  async function refreshRuntime() {
    const payload = await pythonApiFetch<RuntimeStatus>("/agent/runtime");
    setRuntime(payload);
  }

  async function handleCreateRun() {
    setSubmitting(true);
    setActionMessage(null);

    try {
      const createdRun = await pythonApiFetch<AgentRun>("/agent/runs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          goal,
          datasetId: datasetId || undefined,
          baseModel,
          evaluationPrompt: evaluationPrompt || undefined,
          agentModel: agentModel || undefined
        })
      });
      setRuns((current) => mergeRun(current, createdRun));
      setSelectedRunId(createdRun.id);
      setSelectedRun(createdRun);
      setActionMessage("Agent run started.");
      await refreshRuntime();
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Could not start the agent run.");
      await refreshRuntime().catch(() => undefined);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCancelRun() {
    if (!selectedRun) {
      return;
    }

    setCancelling(true);
    setActionMessage(null);

    try {
      const updatedRun = await pythonApiFetch<AgentRun>(`/agent/runs/${selectedRun.id}/cancel`, {
        method: "POST"
      });
      setSelectedRun(updatedRun);
      setRuns((current) => mergeRun(current, updatedRun));
      setActionMessage("Agent run cancelled.");
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Could not cancel the run.");
    } finally {
      setCancelling(false);
    }
  }

  if (error) {
    return <ErrorAlert title="Could not load agent workspace" description={error} />;
  }

  if (loading || !runtime) {
    return <LoadingState label="Loading agent workspace..." />;
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-black/45">Temporal + {runtime.modelProviderLabel}</p>
            <h1 className="mt-2 font-display text-4xl">Agentic control room</h1>
            <p className="mt-3 max-w-3xl text-sm text-black/60">
              Launch a durable run that can choose tools and keep pushing your workflow forward without babysitting each
              click.
            </p>
          </div>
          <StatusBadge value={runtime.status} />
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Runtime mode</p>
            <p className="mt-2 text-sm font-semibold">{runtime.mode}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Temporal address</p>
            <p className="mt-2 text-sm">{runtime.address}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Task queue</p>
            <p className="mt-2 text-sm">{runtime.taskQueue}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Model provider</p>
            <p className="mt-2 text-sm">{runtime.modelProviderLabel}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Provider status</p>
            <p className="mt-2 text-sm">{runtime.providerConfigured ? "Configured" : "Missing setup"}</p>
          </div>
        </div>

        {runtime.providerBaseUrl ? (
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Provider base URL</p>
            <p className="mt-2 break-all text-sm">{runtime.providerBaseUrl}</p>
          </div>
        ) : null}

        {!runtime.supportsFineTuning ? (
          <div className="rounded-2xl border border-black/10 bg-white p-4 text-sm text-black/65">
            Local mode is active. Agent runs can inspect datasets and use the playground with open-source models, but
            managed fine-tuning jobs stay disabled until you switch back to `LLM_PROVIDER=openai`.
          </div>
        ) : null}

        {runtime.error ? <ErrorAlert title="Temporal runtime issue" description={runtime.error} /> : null}
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Card className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">New run</p>
            <h2 className="mt-2 font-display text-3xl">Tell the agent what to do</h2>
          </div>

          <Textarea value={goal} onChange={(event) => setGoal(event.target.value)} />

          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-2 text-sm">
              <span className="font-medium text-black/70">Pinned dataset</span>
              <select
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
                value={datasetId}
                onChange={(event) => setDatasetId(event.target.value)}
              >
                <option value="">Let the agent choose</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} | {dataset.validationStatus} | {dataset.recordCount} records
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-2 text-sm">
              <span className="font-medium text-black/70">Base model</span>
              <input
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
                value={baseModel}
                onChange={(event) => setBaseModel(event.target.value)}
                placeholder={runtime.defaultBaseModel}
              />
            </label>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-2 text-sm">
              <span className="font-medium text-black/70">Evaluation prompt</span>
              <input
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
                value={evaluationPrompt}
                onChange={(event) => setEvaluationPrompt(event.target.value)}
                placeholder="Optional prompt to run after training"
              />
            </label>

            <label className="space-y-2 text-sm">
              <span className="font-medium text-black/70">Agent model</span>
              <input
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
                value={agentModel}
                onChange={(event) => setAgentModel(event.target.value)}
                placeholder={runtime.defaultAgentModel}
              />
            </label>
          </div>

          {selectedDataset ? (
            <div className="rounded-2xl bg-white p-4 text-sm text-black/65">
              Pinned dataset: {selectedDataset.name} | {selectedDataset.validationStatus} | created{" "}
              {formatDate(selectedDataset.createdAt)}
            </div>
          ) : null}

          {actionMessage ? <p className="text-sm text-black/70">{actionMessage}</p> : null}

          <div className="flex flex-wrap gap-3">
            <Button onClick={handleCreateRun} disabled={submitting || !goal.trim() || !runtime.providerConfigured}>
              {submitting ? "Starting..." : "Start agent run"}
            </Button>
            <Button variant="ghost" onClick={() => void refreshRuntime()}>
              Refresh runtime
            </Button>
          </div>
        </Card>

        <Card className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Recent runs</p>
            <h2 className="mt-2 font-display text-3xl">Run history</h2>
          </div>

          {runs.length === 0 ? <p className="text-sm text-black/60">No agent runs yet. Start one from the left panel.</p> : null}

          <div className="space-y-3">
            {runs.map((run) => (
              <button
                key={run.id}
                type="button"
                onClick={() => setSelectedRunId(run.id)}
                className={`w-full rounded-2xl border px-4 py-4 text-left transition ${
                  selectedRunId === run.id ? "border-brand bg-white shadow-panel" : "border-black/10 bg-white/70 hover:bg-white"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="line-clamp-2 font-semibold">{run.goal}</p>
                  <StatusBadge value={run.status} />
                </div>
                <p className="mt-2 text-sm text-black/60">
                  {formatDate(run.updatedAt)} | {run.baseModel}
                </p>
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Selected run</p>
            <h2 className="mt-2 font-display text-3xl">{selectedRun ? "Live execution trace" : "Choose a run"}</h2>
            <p className="mt-2 text-sm text-black/60">
              {selectedRun
                ? `Updated ${formatDate(selectedRun.updatedAt)}`
                : "Pick a run from the history panel to inspect its tool calls and final summary."}
            </p>
          </div>
          {selectedRun && ACTIVE_STATUSES.has(selectedRun.status) ? (
            <Button variant="danger" onClick={handleCancelRun} disabled={cancelling}>
              {cancelling ? "Cancelling..." : "Cancel run"}
            </Button>
          ) : null}
        </div>

        {!selectedRun ? <LoadingState label="No run selected yet." /> : null}

        {selectedRun ? (
          <>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl bg-white p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Status</p>
                <div className="mt-2">
                  <StatusBadge value={selectedRun.status} />
                </div>
              </div>
              <div className="rounded-2xl bg-white p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Dataset</p>
                <p className="mt-2 break-all text-sm">{selectedRun.datasetId ?? "Agent-selected"}</p>
              </div>
              <div className="rounded-2xl bg-white p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Latest job</p>
                <p className="mt-2 break-all text-sm">{selectedRun.latestJobId ?? "None yet"}</p>
              </div>
              <div className="rounded-2xl bg-white p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Output model</p>
                <p className="mt-2 break-all text-sm">{selectedRun.fineTunedModel ?? "Pending"}</p>
              </div>
            </div>

            <div className="rounded-2xl bg-white p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Goal</p>
              <p className="mt-3 text-sm text-black/80">{selectedRun.goal}</p>
              {selectedRun.summary ? (
                <>
                  <p className="mt-5 text-xs uppercase tracking-[0.2em] text-black/45">Summary</p>
                  <p className="mt-3 text-sm text-black/80">{selectedRun.summary}</p>
                </>
              ) : null}
              {selectedRun.lastError ? (
                <>
                  <p className="mt-5 text-xs uppercase tracking-[0.2em] text-black/45">Last error</p>
                  <p className="mt-3 text-sm text-danger">{selectedRun.lastError}</p>
                </>
              ) : null}
            </div>

            <div className="space-y-3">
              {selectedRun.steps.length === 0 ? <p className="text-sm text-black/60">The run has not emitted any steps yet.</p> : null}
              {selectedRun.steps.map((step) => (
                <div key={step.id} className="rounded-2xl border border-black/10 bg-white p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <p className="font-semibold">{step.title}</p>
                      <p className="mt-1 text-sm text-black/60">{step.detail}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <StatusBadge value={step.status} />
                      <p className="text-xs text-black/45">{formatDate(step.createdAt)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </Card>
    </div>
  );
}
