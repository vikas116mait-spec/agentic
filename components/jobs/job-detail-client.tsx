"use client";

import { useEffect, useState } from "react";
import { JobEventList } from "@/components/jobs/job-event-list";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { getPythonApiBaseUrl, pythonApiFetch } from "@/lib/python-api";
import { formatDate, formatDuration, formatNumber } from "@/lib/utils";

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
  lastSyncedAt: string | null;
  progressJson: JobProgressSnapshot | null;
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
    return <LoadingState label="Loading job..." />;
  }

  const progress = job.progressJson?.progress;
  const datasetStats = job.progressJson?.datasetStats;
  const runtimeSummary = job.progressJson?.runtimeSummary;
  const metricsSummary = job.progressJson?.metrics?.summary;
  const progressPercent = progress?.percent ?? null;
  const currentStage = stageLabel(job.progressJson?.stage);
  const stageDescription = job.statusMessage ?? stageSummary(job.progressJson?.stage);
  const progressWidth = progressPercent === null || progressPercent === undefined ? 6 : Math.max(6, Math.min(100, progressPercent));
  const canDownloadLocalModel = job.modelProvider === "local" && job.status === "succeeded" && Boolean(job.fineTunedModel);
  const downloadUrl = `${getPythonApiBaseUrl()}/jobs/${job.id}/download`;

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-black/45">Fine-tuning job</p>
            <h1 className="mt-2 font-display text-4xl">{job.dataset?.name ?? "Unknown dataset"}</h1>
            <p className="mt-3 text-sm text-black/60">
              Created {formatDate(job.createdAt)} | {job.modelProviderLabel} | Base model {job.baseModel}
            </p>
          </div>
          <StatusBadge value={job.status} />
        </div>

        <div className="grid gap-3 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="rounded-[1.5rem] bg-white p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Live progress</p>
                <p className="mt-2 font-display text-2xl">{currentStage}</p>
                <p className="mt-2 text-sm text-black/60">
                  {shouldPoll ? "Auto-refreshing every 5 seconds while this run is active." : "This run is no longer actively refreshing."}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Completion</p>
                <p className="mt-2 text-lg font-semibold">
                  {progressPercent !== null && progressPercent !== undefined ? `${formatDecimal(progressPercent, 1)}%` : "--"}
                </p>
              </div>
            </div>

            <div className="mt-4 h-3 overflow-hidden rounded-full bg-black/8">
              <div
                className="h-full rounded-full bg-brand transition-all duration-500"
                style={{ width: `${progressWidth}%` }}
              />
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl bg-muted/60 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Current step</p>
                <p className="mt-2 text-sm">
                  {formatNumber(progress?.currentStep)} / {formatNumber(progress?.maxSteps)}
                </p>
              </div>
              <div className="rounded-2xl bg-muted/60 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Epoch</p>
                <p className="mt-2 text-sm">
                  {formatDecimal(progress?.epoch)} / {formatDecimal(progress?.totalEpochs)}
                </p>
              </div>
              <div className="rounded-2xl bg-muted/60 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">Elapsed</p>
                <p className="mt-2 text-sm">{formatDuration(progress?.elapsedSeconds)}</p>
              </div>
              <div className="rounded-2xl bg-muted/60 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-black/45">ETA</p>
                <p className="mt-2 text-sm">{formatDuration(progress?.etaSeconds)}</p>
              </div>
            </div>

            <p className="mt-5 text-sm text-black/65">{stageDescription}</p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
            <div className="rounded-[1.5rem] bg-white p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Training setup</p>
              <div className="mt-4 space-y-3 text-sm text-black/70">
                <p>Backend: {job.trainingBackend ?? job.modelProviderLabel}</p>
                <p>GPUs: {formatNumber(runtimeSummary?.gpuCount ?? progress?.gpuCount)}</p>
                <p>Selected device: {runtimeSummary?.selectedGpu ?? "--"}</p>
                <p>Mode: {runtimeSummary?.multiGpu ? "Multi-GPU" : "Single GPU"}</p>
                <p>
                  Batch: {formatNumber(progress?.perDeviceTrainBatchSize)} x accumulation{" "}
                  {formatNumber(progress?.gradientAccumulationSteps)}
                </p>
                <p>Free memory at launch: {formatNumber(runtimeSummary?.selectedGpuFreeMb)} MB</p>
              </div>
            </div>

            <div className="rounded-[1.5rem] bg-white p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">Dataset split</p>
              <div className="mt-4 space-y-3 text-sm text-black/70">
                <p>Total records: {formatNumber(datasetStats?.totalRecords)}</p>
                <p>Train records: {formatNumber(datasetStats?.trainRecords)}</p>
                <p>Eval records: {formatNumber(datasetStats?.evalRecords)}</p>
              </div>
            </div>
          </div>
        </div>

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
              Download trained adapter
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
          </p>
        ) : null}
      </Card>

      <JobEventList
        events={job.events.map((event) => ({
          ...event,
          createdAt: new Date(event.createdAt)
        }))}
      />
    </div>
  );
}
