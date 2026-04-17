"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
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
  modelProviderLabel: string;
  datasetName: string;
  createdAt: string;
};

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

      {jobs.map((job) => (
        <Link key={job.id} href={`/jobs/${job.id}`} className="block">
          <div className="flex items-center justify-between rounded-[1.5rem] border border-black/8 bg-white/80 p-5 shadow-sm transition hover:shadow-md">
            <div className="min-w-0">
              <p className="truncate font-semibold">{job.datasetName}</p>
              <p className="mt-1 truncate text-sm text-black/50">
                {job.modelProviderLabel} · {job.baseModel} · {formatDate(job.createdAt)}
              </p>
              {job.fineTunedModel && (
                <p className="mt-1 truncate text-xs text-brand">{job.fineTunedModel}</p>
              )}
            </div>
            <div className="ml-4 flex shrink-0 items-center gap-3">
              <StatusBadge value={job.status} />
              <ArrowRight className="h-4 w-4 text-black/30" />
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}
