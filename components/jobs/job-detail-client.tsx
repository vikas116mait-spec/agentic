"use client";

import { useEffect, useState } from "react";
import { JobEventList } from "@/components/jobs/job-event-list";
import { LossChart } from "@/components/jobs/loss-chart";
import { TestModelPanel } from "@/components/jobs/test-model-panel";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { LoadingState } from "@/components/ui/loading-state";
import { getPythonApiBaseUrl, pythonApiFetch } from "@/lib/python-api";
import { formatDate, formatDuration, formatNumber } from "@/lib/utils";

type GpuMetric = {
  index: number;
  memUsedMb: number;
  memTotalMb: number;
  utilizationPct: number;
  temperatureC: number;
};

type JobProgressSnapshot = {
  stage: string | null;
  runtimeSummary: {
    gpuCount?: number | null;
    runtimePython?: string | null;
    multiGpu?: boolean | null;
    selectedGpu?: number | string | null;
    selectedGpuFreeMb?: number | null;
  } | null;
  datasetStats: {
    totalRecords?: number | null;
    trainRecords?: number | null;
    evalRecords?: number | null;
  } | null;
  progress: {
    percent?: number | null;
    currentStep?: number | null;
    maxSteps?: number | null;
    epoch?: number | null;
    totalEpochs?: number | null;
    elapsedSeconds?: number | null;
    etaSeconds?: number | null;
    trainLoss?: number | null;
    evalLoss?: number | null;
    learningRate?: number | null;
    stepsPerSecond?: number | null;
    samplesPerSecond?: number | null;
    gpuCount?: number | null;
    perDeviceTrainBatchSize?: number | null;
    gradientAccumulationSteps?: number | null;
  } | null;
  metrics: {
    summary?: {
      evalLoss?: number | null;
      perplexity?: number | null;
      eval_runtime?: number | null;
      eval_samples_per_second?: number | null;
      eval_steps_per_second?: number | null;
    } | null;
  } | null;
  gpuMetrics?: GpuMetric[] | null;
  lossHistory?: Array<{ step: number; loss: number }> | null;
  unslothActive?: boolean | null;
};

type ResultFile = {
  type: string;
  path: string;
};

type JobDetail = {
  id: string;
  modelProvider: string;
  modelProviderLabel: string;
  providerJobId: string | null;
  providerJobUrl: string | null;
  baseModel: string;
  status: string;
  statusMessage: string | null;
  fineTunedModel: string | null;
  modelRepoId: string | null;
  modelRepoUrl: string | null;
  datasetRepoId: string | null;
  datasetRepoUrl: string | null;
  trackioUrl: string | null;
  trainingBackend: string | null;
  trainedTokens: number | null;
  ollamaModelName: string | null;
  lastSyncedAt: string | null;
  progressJson: JobProgressSnapshot | null;
  resultFilesJson: ResultFile[] | null;
  dataset: {
    name: string;
  } | null;
  createdAt: string;
  events: Array<{
    id: string;
    level: string;
    message: string;
    createdAt: string;
    eventType: string | null;
  }>;
};

function formatDecimal(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function formatScientific(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toExponential(digits);
}

function stageLabel(stage: string | null | undefined) {
  if (!stage) {
    return "Waiting";
  }

  const labels: Record<string, string> = {
    queued: "Queued",
    loading_model: "Loading model",
    training: "Training",
    evaluating: "Evaluating",
    saving: "Saving adapter",
    exporting: "Exporting GGUF",
    failed: "Failed",
    succeeded: "Completed",
    cancelled: "Cancelled"
  };

  return labels[stage] ?? stage.replaceAll("_", " ");
}

function stageSummary(stage: string | null | undefined) {
  const summaries: Record<string, string> = {
    queued: "The run is queued and the local trainer process is starting up.",
    loading_model: "The base model is being downloaded or loaded into 4-bit memory.",
    training: "LoRA adapters are actively training on your dataset.",
    evaluating: "The model is being checked on the evaluation split.",
    saving: "The final adapter files and metrics are being written to disk.",
    exporting: "Converting the trained model to GGUF format. This may take a few minutes.",
    failed: "The run stopped with an error. Check the latest event and trainer log for the traceback.",
    succeeded: "Training finished and the adapter is ready to use.",
    cancelled: "The run was cancelled before completion."
  };

  return summaries[stage ?? ""] ?? "The trainer is preparing the run. ETA becomes more accurate after the first few logged steps.";
}

export function JobDetailClient({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let cancelled = false;

    pythonApiFetch<JobDetail>(`/jobs/${jobId}`)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setJob(payload);
        setError(null);
      })
      .catch((requestError: Error) => {
        if (cancelled) {
          return;
        }
        setError(requestError.message);
      });

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  async function syncJob({ silent = false }: { silent?: boolean } = {}) {
    if (!silent) {
      setSyncing(true);
      setActionMessage(null);
    }
    try {
      const updated = await pythonApiFetch<JobDetail>(`/jobs/${jobId}/sync`, { method: "POST" });
      setJob(updated);
      setError(null);
      if (!silent) {
        setActionMessage("Job synced.");
      }
    } catch (requestError) {
      if (!silent) {
        setActionMessage(requestError instanceof Error ? requestError.message : "Sync failed.");
      }
    } finally {
      if (!silent) {
        setSyncing(false);
      }
    }
  }

  async function handleSync() {
    await syncJob();
  }

  async function handleCancel() {
    setCancelling(true);
    setActionMessage(null);
    try {
      const updated = await pythonApiFetch<JobDetail>(`/jobs/${jobId}/cancel`, { method: "POST" });
      setJob(updated);
      setError(null);
      setActionMessage("Cancel request sent.");
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Cancel failed.");
    } finally {
      setCancelling(false);
    }
  }

  const shouldPoll = job ? ["queued", "running", "validating_files"].includes(job.status) : false;

  useEffect(() => {
    if (!shouldPoll) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void syncJob({ silent: true });
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [jobId, shouldPoll]);

  if (error) {
    return <ErrorAlert title="Could not load job" description={error} />;
  }

  if (!job) {
    return <LoadingState variant="detail" />;
  }

  const progress = job.progressJson?.progress;
  const datasetStats = job.progressJson?.datasetStats;
  const runtimeSummary = job.progressJson?.runtimeSummary;
  const metricsSummary = job.progressJson?.metrics?.summary;
  const gpuMetrics = job.progressJson?.gpuMetrics ?? null;
  const lossHistory = job.progressJson?.lossHistory ?? [];
  const unslothActive = job.progressJson?.unslothActive ?? null;
  const progressPercent = progress?.percent ?? null;
  const currentStage = stageLabel(job.progressJson?.stage);
  const stageDescription = job.statusMessage ?? stageSummary(job.progressJson?.stage);
  const progressWidth = progressPercent === null || progressPercent === undefined ? 6 : Math.max(6, Math.min(100, progressPercent));
  const canDownloadLocalModel = job.modelProvider === "local" && job.status === "succeeded" && Boolean(job.fineTunedModel);
  const downloadUrl = `${getPythonApiBaseUrl()}/jobs/${job.id}/download`;
  const ggufFile = job.resultFilesJson?.find((f) => f.type === "local_gguf") ?? null;

  return (
    <div className="space-y-6">
      <Breadcrumb crumbs={[{ label: "Training runs", href: "/jobs" }, { label: job.dataset?.name ?? job.id }]} />
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-black/45">Fine-tuning job</p>
            <h1 className="mt-2 font-display text-4xl">{job.dataset?.name ?? "Unknown dataset"}</h1>
            <p className="mt-3 text-sm text-black/60">
              Created {formatDate(job.createdAt)} | {job.modelProviderLabel} | Base model {job.baseModel}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {unslothActive === true && (
              <span className="inline-flex items-center gap-1 rounded-full bg-brand/10 px-3 py-1 text-xs font-semibold text-brand">
                ⚡ Unsloth
              </span>
            )}
            {unslothActive === false && (
              <span className="inline-flex items-center gap-1 rounded-full bg-black/8 px-3 py-1 text-xs font-semibold text-black/50">
                HF+PEFT
              </span>
            )}
            <StatusBadge value={job.status} />
          </div>
        </div>

        {/* Progress + setup row */}
        <div className="grid gap-3 lg:grid-cols-[1.4fr_0.6fr]">
          {/* Live progress */}
          <div className="rounded-[1.5rem] bg-white p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Live progress</p>
                <p className="mt-2 font-display text-2xl">{currentStage}</p>
                <p className="mt-2 text-sm text-black/60">
                  {shouldPoll ? "Auto-refreshing every 5 seconds while this run is active." : "This run is no longer actively refreshing."}
                </p>
              </div>
              <div className="text-right shrink-0">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Completion</p>
                <p className="mt-2 text-lg font-semibold">
                  {progressPercent !== null && progressPercent !== undefined ? `${formatDecimal(progressPercent, 1)}%` : "--"}
                </p>
              </div>
            </div>

            <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-black/8">
              <div
                className="h-full rounded-full bg-brand transition-all duration-500"
                style={{ width: `${progressWidth}%` }}
              />
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Current step", value: `${formatNumber(progress?.currentStep)} / ${formatNumber(progress?.maxSteps)}` },
                { label: "Epoch", value: `${formatDecimal(progress?.epoch)} / ${formatDecimal(progress?.totalEpochs)}` },
                { label: "Elapsed", value: formatDuration(progress?.elapsedSeconds) },
                { label: "ETA", value: formatDuration(progress?.etaSeconds) },
              ].map(({ label, value }) => (
                <div key={label} className="rounded-2xl bg-muted/60 px-3 py-3">
                  <p className="text-xs uppercase tracking-[0.15em] text-black/40">{label}</p>
                  <p className="mt-1.5 text-sm font-medium">{value}</p>
                </div>
              ))}
            </div>

            <p className="mt-5 text-sm text-black/65">{stageDescription}</p>

            {lossHistory.length > 0 && (
              <div className="mt-5">
                <p className="mb-2 text-xs uppercase tracking-[0.2em] text-black/45">Training loss</p>
                <LossChart data={lossHistory} height={72} />
              </div>
            )}
          </div>

          {/* Training setup + dataset split */}
          <div className="flex flex-col gap-3">
            <div className="rounded-[1.5rem] bg-white p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Training setup</p>
              <div className="mt-4 grid grid-cols-2 gap-2">
                {[
                  { label: "Backend", value: job.trainingBackend ?? job.modelProviderLabel },
                  { label: "GPUs", value: formatNumber(runtimeSummary?.gpuCount ?? progress?.gpuCount) },
                  { label: "Device", value: String(runtimeSummary?.selectedGpu ?? "--") },
                  { label: "Mode", value: runtimeSummary?.multiGpu ? "Multi-GPU" : "Single GPU" },
                  { label: "Batch", value: `${formatNumber(progress?.perDeviceTrainBatchSize)} × ${formatNumber(progress?.gradientAccumulationSteps)} acc` },
                  { label: "Free memory", value: `${formatNumber(runtimeSummary?.selectedGpuFreeMb)} MB` },
                ].map(({ label, value }) => (
                  <div key={label} className="rounded-xl bg-muted/60 px-2.5 py-2">
                    <p className="text-[10px] text-black/40 leading-tight">{label}</p>
                    <p className="mt-1 text-xs font-medium text-black/80 leading-tight">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[1.5rem] bg-white p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Dataset split</p>
              <div className="mt-4 grid grid-cols-3 gap-2">
                {[
                  { label: "Total", value: formatNumber(datasetStats?.totalRecords) },
                  { label: "Train", value: formatNumber(datasetStats?.trainRecords) },
                  { label: "Eval", value: formatNumber(datasetStats?.evalRecords) },
                ].map(({ label, value }) => (
                  <div key={label} className="rounded-xl bg-muted/60 px-2.5 py-2.5 text-center">
                    <p className="text-[10px] text-black/40">{label}</p>
                    <p className="mt-1 text-base font-semibold text-black/80">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* GPU metrics — full width so cards have room */}
        {gpuMetrics && gpuMetrics.length > 0 ? (
          <div className="rounded-[1.5rem] bg-white p-5">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">GPU metrics</p>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
              {gpuMetrics.map((gpu) => {
                const memPct = gpu.memTotalMb > 0 ? Math.round((gpu.memUsedMb / gpu.memTotalMb) * 100) : 0;
                return (
                  <div key={gpu.index} className="rounded-2xl bg-muted/60 px-3 py-3 space-y-2 min-w-0">
                    <div className="flex items-center justify-between gap-1 text-xs">
                      <span className="font-semibold text-black/70">GPU {gpu.index}</span>
                      <span className="text-black/45 shrink-0">{gpu.temperatureC}°C</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/10">
                      <div
                        className="h-full rounded-full bg-brand transition-all duration-500"
                        style={{ width: `${memPct}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-black/45">
                      <span>{memPct}%</span>
                      <span>{gpu.utilizationPct}% util</span>
                    </div>
                    <p className="text-[10px] text-black/40 leading-tight">
                      {(gpu.memUsedMb / 1024).toFixed(0)} / {(gpu.memTotalMb / 1024).toFixed(0)} GB
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Provider job id</p>
            <p className="mt-2 break-all text-sm">{job.providerJobId ?? "Pending"}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Fine-tuned model</p>
            <p className="mt-2 break-all text-sm">{job.fineTunedModel ?? "Pending"}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Trained tokens</p>
            <p className="mt-2 text-sm">{formatNumber(job.trainedTokens)}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Training speed</p>
            <p className="mt-2 text-sm">
              {progress?.stepsPerSecond ? `${formatDecimal(progress.stepsPerSecond)} steps/s` : "--"}
            </p>
            <p className="mt-1 text-xs text-black/55">
              {progress?.samplesPerSecond ? `${formatDecimal(progress.samplesPerSecond)} samples/s` : ""}
            </p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Last synced</p>
            <p className="mt-2 text-sm">{formatDate(job.lastSyncedAt)}</p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Train loss</p>
            <p className="mt-2 text-sm">{formatDecimal(progress?.trainLoss, 4)}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Eval loss</p>
            <p className="mt-2 text-sm">{formatDecimal(progress?.evalLoss ?? metricsSummary?.evalLoss, 4)}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Perplexity</p>
            <p className="mt-2 text-sm">{formatDecimal(metricsSummary?.perplexity, 2)}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Learning rate</p>
            <p className="mt-2 text-sm">{formatScientific(progress?.learningRate, 2)}</p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {job.providerJobUrl ? (
            <div className="rounded-2xl bg-white p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Provider job URL</p>
              <a className="mt-2 block break-all text-sm text-brand underline" href={job.providerJobUrl} target="_blank" rel="noreferrer">
                {job.providerJobUrl}
              </a>
            </div>
          ) : null}

          {job.modelRepoUrl || job.modelRepoId ? (
            <div className="rounded-2xl bg-white p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">
                {job.modelRepoUrl ? "Model repo" : "Model artifact"}
              </p>
              {job.modelRepoUrl ? (
                <a className="mt-2 block break-all text-sm text-brand underline" href={job.modelRepoUrl} target="_blank" rel="noreferrer">
                  {job.modelRepoId ?? job.modelRepoUrl}
                </a>
              ) : (
                <p className="mt-2 break-all text-sm">{job.modelRepoId}</p>
              )}
            </div>
          ) : null}

          {job.datasetRepoUrl || job.datasetRepoId ? (
            <div className="rounded-2xl bg-white p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">
                {job.datasetRepoUrl ? "Dataset repo" : "Training data"}
              </p>
              {job.datasetRepoUrl ? (
                <a className="mt-2 block break-all text-sm text-brand underline" href={job.datasetRepoUrl} target="_blank" rel="noreferrer">
                  {job.datasetRepoId ?? job.datasetRepoUrl}
                </a>
              ) : (
                <p className="mt-2 break-all text-sm">{job.datasetRepoId}</p>
              )}
            </div>
          ) : null}

          {job.trackioUrl ? (
            <div className="rounded-2xl bg-white p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Trackio</p>
              <a className="mt-2 block break-all text-sm text-brand underline" href={job.trackioUrl} target="_blank" rel="noreferrer">
                {job.trackioUrl}
              </a>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap gap-3">
          {canDownloadLocalModel ? (
            <Button variant="secondary" onClick={() => window.location.assign(downloadUrl)}>
              Download adapter
            </Button>
          ) : null}
          {ggufFile ? (
            <Button variant="secondary" onClick={() => window.location.assign(`${getPythonApiBaseUrl()}/jobs/${job.id}/download?type=gguf`)}>
              Download GGUF
            </Button>
          ) : null}
          <Button onClick={handleSync} disabled={syncing}>
            {syncing ? "Syncing..." : "Sync status"}
          </Button>
          <Button variant="danger" onClick={handleCancel} disabled={cancelling}>
            {cancelling ? "Cancelling..." : "Cancel job"}
          </Button>
        </div>

        {actionMessage ? <p className="text-sm text-black/70">{actionMessage}</p> : null}
        {canDownloadLocalModel ? (
          <p className="text-sm text-black/55">
            Download includes the trained LoRA adapter, tokenizer files, metrics, and metadata for this run.
            {ggufFile ? " GGUF export is also available for use with Ollama and llama.cpp." : ""}
          </p>
        ) : null}
      </Card>

      {job.status === "succeeded" && job.fineTunedModel ? (
        <Card className="space-y-5 bg-white/88">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Test model</p>
            <p className="mt-2 font-display text-2xl">Chat with your fine-tuned model</p>
            <p className="mt-1.5 text-sm text-black/55">
              Send prompts to <span className="font-mono text-xs text-black/70">{job.fineTunedModel}</span> and see responses in real time.
            </p>
          </div>
          <TestModelPanel
            fineTunedModel={job.fineTunedModel}
            modelProvider={job.modelProvider}
            baseModel={job.baseModel}
            ollamaModelName={job.ollamaModelName}
          />
        </Card>
      ) : null}

      <JobEventList
        events={job.events.map((event) => ({
          ...event,
          createdAt: new Date(event.createdAt)
        }))}
      />
    </div>
  );
}
