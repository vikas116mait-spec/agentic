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

  if (error) {
    return <ErrorAlert title="Python API unavailable" description={error} />;
  }

  if (!datasets) {
    return <LoadingState label="Loading datasets..." />;
  }

  if (datasets.length === 0) {
    return (
      <EmptyState
        title="No datasets yet"
        description="Upload your first JSONL dataset to start validating records."
        action={
          <Link href="/datasets/new">
            <Button>Upload dataset</Button>
          </Link>
        }
      />
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div>
          <p className="font-display text-3xl">Datasets</p>
          <p className="mt-2 text-sm text-black/60">Every dataset is validated first, then optionally prepared for managed training later.</p>
        </div>
        <Link href="/datasets/new">
          <Button>New dataset</Button>
        </Link>
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="text-black/45">
            <tr>
              <th className="pb-3">Name</th>
              <th className="pb-3">Status</th>
              <th className="pb-3">Records</th>
              <th className="pb-3">Created</th>
              <th className="pb-3">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-black/5">
            {datasets.map((dataset) => (
              <tr key={dataset.id}>
                <td className="py-4 font-semibold">{dataset.name}</td>
                <td className="py-4">
                  <StatusBadge value={dataset.validationStatus} />
                </td>
                <td className="py-4">{formatNumber(dataset.recordCount)}</td>
                <td className="py-4">{formatDate(dataset.createdAt)}</td>
                <td className="py-4">
                  <Link href={`/datasets/${dataset.id}`} className="font-semibold text-brand">
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
