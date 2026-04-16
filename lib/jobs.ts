import type { FineTuneJobStatus } from "@/lib/types";

const knownStatuses = new Set([
  "validating_files",
  "queued",
  "running",
  "succeeded",
  "failed",
  "cancelled",
  "paused"
]);

export function normalizeJobStatus(status: string | null | undefined): FineTuneJobStatus {
  if (!status) {
    return "unknown";
  }

  return knownStatuses.has(status) ? (status as FineTuneJobStatus) : "unknown";
}

export function mapFineTuneJob(job: {
  status?: string | null;
  fine_tuned_model?: string | null;
  trained_tokens?: number | null;
  estimated_finish?: number | null;
  result_files?: string[] | null;
  finished_at?: number | null;
  method?: unknown;
}) {
  return {
    status: normalizeJobStatus(job.status),
    fineTunedModel: job.fine_tuned_model ?? null,
    trainedTokens: job.trained_tokens ?? null,
    estimatedFinishAt: job.estimated_finish ? new Date(job.estimated_finish * 1000) : null,
    resultFilesJson: job.result_files ?? null,
    hyperparametersJson: job.method ?? null,
    finishedAt: job.finished_at ? new Date(job.finished_at * 1000) : null
  };
}
