"use client";

import { useEffect, useState } from "react";
import { JobEventList } from "@/components/jobs/job-event-list";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate, formatNumber } from "@/lib/utils";

type JobDetail = {
  id: string;
  openaiJobId: string;
  baseModel: string;
  status: string;
  fineTunedModel: string | null;
  trainedTokens: number | null;
  lastSyncedAt: string | null;
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

export function JobDetailClient({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    pythonApiFetch<JobDetail>(`/jobs/${jobId}`)
      .then(setJob)
      .catch((requestError: Error) => setError(requestError.message));
  }, [jobId]);

  async function handleSync() {
    setSyncing(true);
    setActionMessage(null);
    try {
      const updated = await pythonApiFetch<JobDetail>(`/jobs/${jobId}/sync`, { method: "POST" });
      setJob(updated);
      setActionMessage("Job synced.");
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Sync failed.");
    } finally {
      setSyncing(false);
    }
  }

  async function handleCancel() {
    setCancelling(true);
    setActionMessage(null);
    try {
      const updated = await pythonApiFetch<JobDetail>(`/jobs/${jobId}/cancel`, { method: "POST" });
      setJob(updated);
      setActionMessage("Cancel request sent.");
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Cancel failed.");
    } finally {
      setCancelling(false);
    }
  }

  if (error) {
    return <ErrorAlert title="Could not load job" description={error} />;
  }

  if (!job) {
    return <LoadingState label="Loading job..." />;
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-black/45">Fine-tuning job</p>
            <h1 className="mt-2 font-display text-4xl">{job.dataset?.name ?? "Unknown dataset"}</h1>
            <p className="mt-3 text-sm text-black/60">
              Created {formatDate(job.createdAt)} | Base model {job.baseModel}
            </p>
          </div>
          <StatusBadge value={job.status} />
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Provider job id</p>
            <p className="mt-2 break-all text-sm">{job.openaiJobId}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Fine-tuned model</p>
            <p className="mt-2 text-sm">{job.fineTunedModel ?? "Pending"}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Trained tokens</p>
            <p className="mt-2 text-sm">{formatNumber(job.trainedTokens)}</p>
          </div>
          <div className="rounded-2xl bg-white p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-black/45">Last synced</p>
            <p className="mt-2 text-sm">{formatDate(job.lastSyncedAt)}</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <Button onClick={handleSync} disabled={syncing}>
            {syncing ? "Syncing..." : "Sync status"}
          </Button>
          <Button variant="danger" onClick={handleCancel} disabled={cancelling}>
            {cancelling ? "Cancelling..." : "Cancel job"}
          </Button>
        </div>

        {actionMessage ? <p className="text-sm text-black/70">{actionMessage}</p> : null}
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
