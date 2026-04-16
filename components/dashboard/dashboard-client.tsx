"use client";

import { useEffect, useState } from "react";
import { Activity, Database, Rocket, Sparkles } from "lucide-react";
import { StatCard } from "@/components/stat-card";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate } from "@/lib/utils";

type DashboardSummary = {
  totalDatasets: number;
  totalJobs: number;
  runningJobs: number;
  completedModels: number;
  latestActivity: Array<{
    id: string;
    datasetName: string;
    baseModel: string;
    status: string;
    createdAt: string;
  }>;
};

export function DashboardClient() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<DashboardSummary>("/dashboard/summary")
      .then(setData)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) {
    return <ErrorAlert title="Python API unavailable" description={`Start the Python service on port 8001. ${error}`} />;
  }

  if (!data) {
    return <LoadingState label="Loading dashboard..." />;
  }

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Total datasets" value={data.totalDatasets} detail="Validated and pending uploads" icon={<Database />} />
        <StatCard label="Total jobs" value={data.totalJobs} detail="Jobs launched in your workspace" icon={<Rocket />} />
        <StatCard label="Running jobs" value={data.runningJobs} detail="Currently syncing against the training provider" icon={<Activity />} />
        <StatCard label="Completed models" value={data.completedModels} detail="Ready to test in playground" icon={<Sparkles />} />
      </section>

      <Card>
        <p className="font-display text-2xl">Latest activity</p>
        <div className="mt-4 space-y-3">
          {data.latestActivity.length === 0 ? <p className="text-sm text-black/60">No activity yet. Upload your first dataset to begin.</p> : null}
          {data.latestActivity.map((job) => (
            <div key={job.id} className="rounded-2xl bg-white p-4">
              <p className="font-semibold">{job.datasetName}</p>
              <p className="mt-1 text-sm text-black/60">
                {job.baseModel} | {job.status} | {formatDate(job.createdAt)}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
