"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ValidationSummaryCard } from "@/components/dataset/validation-summary-card";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";
import { formatBytes, formatDate } from "@/lib/utils";
import type { DatasetValidationSummary } from "@/lib/types";

type DatasetDetail = {
  id: string;
  name: string;
  originalFilename: string;
  fileSizeBytes: number;
  createdAt: string;
  validationStatus: string;
  validationSummary: DatasetValidationSummary;
  openaiFileId: string | null;
};

type RuntimeStatus = {
  modelProviderLabel: string;
  supportsFineTuning: boolean;
};

export function DatasetDetailClient({ datasetId }: { datasetId: string }) {
  const router = useRouter();
  const [dataset, setDataset] = useState<DatasetDetail | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    Promise.all([
      pythonApiFetch<DatasetDetail>(`/datasets/${datasetId}`),
      pythonApiFetch<RuntimeStatus>("/agent/runtime")
    ])
      .then(([datasetPayload, runtimePayload]) => {
        setDataset(datasetPayload);
        setRuntime(runtimePayload);
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [datasetId]);

  async function handleUploadToOpenAI() {
    setUploading(true);
    setActionMessage(null);
    try {
      const updated = await pythonApiFetch<DatasetDetail>(`/datasets/${datasetId}/upload-to-openai`, {
        method: "POST"
      });
      setDataset(updated);
      setActionMessage("Training file uploaded.");
      router.refresh();
    } catch (requestError) {
      setActionMessage(requestError instanceof Error ? requestError.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  if (error) {
    return <ErrorAlert title="Could not load dataset" description={error} />;
  }

  if (!dataset) {
    return <LoadingState label="Loading dataset..." />;
  }

  return (
    <div className="space-y-6">
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-black/45">Dataset detail</p>
            <h1 className="mt-2 font-display text-4xl">{dataset.name}</h1>
            <p className="mt-3 text-sm text-black/60">
              {dataset.originalFilename} | {formatBytes(dataset.fileSizeBytes)} | {formatDate(dataset.createdAt)}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <StatusBadge value={dataset.validationStatus} />
            <Link href={`/jobs/new?datasetId=${dataset.id}`}>
              <Button>Create job</Button>
            </Link>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <Button
            className="w-full"
            disabled={dataset.validationStatus !== "VALID" || uploading || runtime?.supportsFineTuning === false}
            onClick={handleUploadToOpenAI}
          >
            {uploading ? "Uploading..." : runtime?.supportsFineTuning === false ? "Training upload unavailable" : "Upload training file"}
          </Button>
          <Link href={`/jobs/new?datasetId=${dataset.id}`}>
            <Button className="w-full" variant="ghost">
              Train from this dataset
            </Button>
          </Link>
        </div>

        {actionMessage ? <p className="text-sm text-black/70">{actionMessage}</p> : null}
        {runtime?.supportsFineTuning === false ? (
          <p className="text-sm text-black/60">
            {runtime.modelProviderLabel} mode is active, so remote fine-tuning steps stay disabled until you switch back
            to OpenAI.
          </p>
        ) : null}
        {dataset.openaiFileId ? <p className="text-sm text-brand">Training file id: {dataset.openaiFileId}</p> : null}
      </Card>

      <ValidationSummaryCard summary={dataset.validationSummary} />
    </div>
  );
}
