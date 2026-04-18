"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Database } from "lucide-react";
import { EmptyState } from "@/components/ui/empty-state";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";
import { formatDate, formatNumber } from "@/lib/utils";

type DatasetListItem = {
  id: string;
  name: string;
  validationStatus: string;
  recordCount: number;
  createdAt: string;
};

export function DatasetsPageClient() {
  const [datasets, setDatasets] = useState<DatasetListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    pythonApiFetch<DatasetListItem[]>("/datasets")
      .then(setDatasets)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) return <ErrorAlert title="Could not load datasets" description={error} />;
  if (!datasets) return <LoadingState label="Loading datasets..." />;

  if (datasets.length === 0) {
    return (
        <EmptyState
          title="No datasets yet"
          description="Upload a .jsonl file to get started. Each line can use either a messages array or instruction/input/output fields."
          action={
          <Link href="/datasets/new">
            <Button>Upload dataset</Button>
          </Link>
        }
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="font-display text-2xl">Datasets</p>
        <Link href="/datasets/new">
          <Button size="sm">Upload new</Button>
        </Link>
      </div>

      {datasets.map((dataset) => (
        <Link key={dataset.id} href={`/datasets/${dataset.id}`} className="group block">
          <div className="flex items-center gap-4 rounded-[1.5rem] border border-black/8 bg-white/80 px-5 py-4 shadow-sm transition-all hover:border-black/14 hover:shadow-md">
            {/* record count bar */}
            <div className="hidden sm:flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-brand/8 text-brand">
              <Database className="h-4 w-4" />
            </div>

            <div className="min-w-0 flex-1">
              <p className="truncate font-semibold">{dataset.name}</p>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5">
                <span className="text-xs font-medium text-black/55">
                  {formatNumber(dataset.recordCount)} records
                </span>
                <span className="text-xs text-black/25">·</span>
                <span className="text-xs text-black/40">{formatDate(dataset.createdAt)}</span>
              </div>
            </div>

            <div className="ml-2 flex shrink-0 items-center gap-3">
              <StatusBadge value={dataset.validationStatus} />
              <ArrowRight className="h-4 w-4 text-black/20 transition group-hover:translate-x-0.5 group-hover:text-black/40" />
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}
