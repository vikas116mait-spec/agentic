"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Zap } from "lucide-react";
import { EmptyState } from "@/components/ui/empty-state";
import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate } from "@/lib/utils";

type JobListItem = {
  id: string;
  status: string;
  baseModel: string;
  fineTunedModel: string | null;
  modelProvider: string;
  modelProviderLabel: string;
  datasetName: string;
  trainingBackend: string | null;
  createdAt: string;
  progressJson?: {
    unslothActive?: boolean | null;
    lossHistory?: Array<{ step: number; loss: number }> | null;
  } | null;
  resultFilesJson?: Array<{ type: string; path: string }> | null;
};

function BackendBadge({ job }: { job: JobListItem }) {
  if (job.modelProvider !== "local") return null;

  const unsloth = job.progressJson?.unslothActive;
  if (unsloth === true) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-brand/10 px-2 py-0.5 text-xs font-semibold text-brand">
        <Zap className="h-3 w-3" />
        Unsloth
      </span>
    );
  }
  if (unsloth === false) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-black/8 px-2 py-0.5 text-xs text-black/50">
        HF+PEFT
      </span>
    );
  }
  return null;
}

function MiniSparkline({ data }: { data: Array<{ step: number; loss: number }> }) {
  if (!data || data.length < 2) return null;

  const w = 60;
  const h = 20;
  const minL = Math.min(...data.map((d) => d.loss));
  const maxL = Math.max(...data.map((d) => d.loss));
  const range = maxL - minL || 1;
  const minS = data[0].step;
  const maxS = data[data.length - 1].step;
  const sRange = maxS - minS || 1;

  const pts = data
    .map((d) => {
      const x = ((d.step - minS) / sRange) * w;
      const y = h - ((d.loss - minL) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline
        points={pts}
        fill="none"
        stroke="var(--color-brand, #16a34a)"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function JobsPageClient() {
  const [jobs, setJobs] = useState<JobListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<JobListItem[]>("/jobs")
      .then(setJobs)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) return <ErrorAlert title="Could not load jobs" description={error} />;
  if (!jobs) return <LoadingState label="Loading jobs..." />;

  if (jobs.length === 0) {
    return (
      <EmptyState
        title="No jobs yet"
        description="Upload one dataset, then start a fine-tuning run. OpenAI, Hugging Face Jobs, and Local GPU QLoRA training profiles all show up here."
        action={
          <div className="flex gap-3">
            <Link href="/datasets/new">
              <Button>Upload dataset</Button>
            </Link>
            <Link href="/jobs/new">
              <Button variant="ghost">Create job</Button>
            </Link>
          </div>
        }
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="font-display text-2xl">Training runs</p>
        <Link href="/jobs/new">
          <Button size="sm">Start run</Button>
        </Link>
      </div>

      {jobs.map((job) => {
        const hasGguf = job.resultFilesJson?.some((f) => f.type === "local_gguf") ?? false;
        const lossHistory = job.progressJson?.lossHistory ?? [];
        const lastLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : null;

        return (
          <Link key={job.id} href={`/jobs/${job.id}`} className="group block">
            <div className="flex items-center gap-4 rounded-[1.5rem] border border-black/8 bg-white/80 px-5 py-4 shadow-sm transition-all hover:border-black/14 hover:shadow-md">
              {/* left accent bar */}
              <div className={`hidden sm:block w-1 self-stretch rounded-full shrink-0 ${
                job.status === "succeeded" || job.status === "completed"
                  ? "bg-brand/40"
                  : job.status === "running" || job.status === "starting"
                    ? "bg-accent/60"
                    : job.status === "failed" || job.status === "error"
                      ? "bg-danger/40"
                      : "bg-black/10"
              }`} />

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-semibold truncate">{job.datasetName}</p>
                  <BackendBadge job={job} />
                  {hasGguf && (
                    <span className="inline-flex items-center rounded-full bg-accent/20 px-2 py-0.5 text-xs font-semibold text-black/65">
                      GGUF
                    </span>
                  )}
                </div>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5">
                  <span className="text-xs text-black/40">{job.modelProviderLabel}</span>
                  <span className="text-xs text-black/25">·</span>
                  <span className="text-xs font-mono text-black/50 truncate max-w-[220px]">{job.baseModel}</span>
                  <span className="text-xs text-black/25">·</span>
                  <span className="text-xs text-black/40">{formatDate(job.createdAt)}</span>
                </div>
                {job.fineTunedModel && (
                  <p className="mt-1 truncate text-xs text-brand/80">{job.fineTunedModel}</p>
                )}
              </div>

              <div className="flex shrink-0 items-center gap-3">
                {lossHistory.length > 1 && (
                  <div className="hidden md:flex flex-col items-end gap-0.5">
                    <MiniSparkline data={lossHistory} />
                    {lastLoss !== null && (
                      <span className="text-xs text-black/35">loss {lastLoss.toFixed(4)}</span>
                    )}
                  </div>
                )}
                <StatusBadge value={job.status} />
                <ArrowRight className="h-4 w-4 text-black/20 transition group-hover:translate-x-0.5 group-hover:text-black/40" />
              </div>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
