"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
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

  if (error) {
    return <ErrorAlert title="Could not load jobs" description={error} />;
  }

  if (!jobs) {
    return <LoadingState label="Loading jobs..." />;
  }

  if (jobs.length === 0) {
    return (
      <EmptyState
        title="No jobs yet"
        description="Once a validated dataset is ready and OpenAI mode is enabled, you can create a fine-tuning job here."
        action={
          <Link href="/jobs/new">
            <Button>Create job</Button>
          </Link>
        }
      />
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div>
          <p className="font-display text-3xl">Fine-tuning jobs</p>
          <p className="mt-2 text-sm text-black/60">Track status, inspect events, and jump into testing when the model is ready.</p>
        </div>
        <Link href="/jobs/new">
          <Button>New job</Button>
        </Link>
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="text-black/45">
            <tr>
              <th className="pb-3">Dataset</th>
              <th className="pb-3">Status</th>
              <th className="pb-3">Base model</th>
              <th className="pb-3">Fine-tuned model</th>
              <th className="pb-3">Created</th>
              <th className="pb-3">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-black/5">
            {jobs.map((job) => (
              <tr key={job.id}>
                <td className="py-4 font-semibold">{job.datasetName}</td>
                <td className="py-4">
                  <StatusBadge value={job.status} />
                </td>
                <td className="py-4">{job.baseModel}</td>
                <td className="py-4">{job.fineTunedModel ?? "--"}</td>
                <td className="py-4">{formatDate(job.createdAt)}</td>
                <td className="py-4">
                  <Link href={`/jobs/${job.id}`} className="font-semibold text-brand">
                    View
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
