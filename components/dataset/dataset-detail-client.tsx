"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ValidationSummaryCard } from "@/components/dataset/validation-summary-card";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { LoadingState } from "@/components/ui/loading-state";
import { getPythonApiBaseUrl, pythonApiFetch } from "@/lib/python-api";
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
  managedFineTuningAvailable: boolean;
};

export function DatasetDetailClient({ datasetId }: { datasetId: string }) {
  const [dataset, setDataset] = useState<DatasetDetail | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

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
    setUploadMessage(null);
    try {
      const updated = await pythonApiFetch<DatasetDetail>(`/datasets/${datasetId}/upload-to-openai`, {
        method: "POST"
      });
      setDataset(updated);
      setUploadMessage("Uploaded to OpenAI — ready to train.");
    } catch (requestError) {
      setUploadMessage(requestError instanceof Error ? requestError.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  if (error) return <ErrorAlert title="Could not load dataset" description={error} />;
  if (!dataset) return <LoadingState variant="detail" />;

  const isValid = dataset.validationStatus === "VALID";
  const canUploadToOpenAI = isValid && runtime?.managedFineTuningAvailable !== false && !dataset.openaiFileId;
  const datasetDownloadUrl = `${getPythonApiBaseUrl()}/datasets/${dataset.id}/download`;

  return (
    <div className="space-y-6">
      <Breadcrumb crumbs={[{ label: "Datasets", href: "/datasets" }, { label: dataset.name }]} />
      <Card className="space-y-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="font-display text-3xl">{dataset.name}</h1>
            <p className="mt-1.5 text-sm text-black/50">
              {formatBytes(dataset.fileSizeBytes)} · {formatDate(dataset.createdAt)}
            </p>
          </div>
          <StatusBadge value={dataset.validationStatus} />
        </div>

        <div className="flex flex-wrap gap-3">
          {isValid ? (
            <Link href={`/jobs/new?datasetId=${dataset.id}`}>
              <Button>Start fine-tuning job</Button>
            </Link>
          ) : null}
          <Button variant="secondary" onClick={() => window.location.assign(datasetDownloadUrl)}>
            Download dataset
          </Button>
          <Link href="/guides/downloads#dataset-file">
            <Button variant="ghost">How to use this file</Button>
          </Link>
          {canUploadToOpenAI && (
            <Button variant="ghost" onClick={handleUploadToOpenAI} disabled={uploading}>
              {uploading ? "Uploading..." : "Pre-upload to OpenAI"}
            </Button>
          )}
        </div>

        {!isValid ? (
          <p className="rounded-2xl bg-danger/10 p-4 text-sm text-danger">
            Fix the errors below before this dataset can be used for training.
          </p>
        ) : null}

        {dataset.openaiFileId && (
          <p className="text-xs text-black/45">OpenAI file: {dataset.openaiFileId}</p>
        )}
        {uploadMessage && (
          <p className="text-sm text-black/65">{uploadMessage}</p>
        )}
        {runtime?.managedFineTuningAvailable === false && (
          <p className="text-xs text-black/45">
            Add an <code>OPENAI_API_KEY</code> for managed fine-tuning, or set <code>HF_TOKEN</code> for the paid
            Hugging Face Jobs path. For the recommended free setup, keep <code>LOCAL_TRAINING_ENABLED=1</code> and
            point <code>LOCAL_TRAINING_PYTHON</code> at your training venv. Ollama profiles can still be used in
            Playground and Agent right now.
          </p>
        )}
      </Card>

      <ValidationSummaryCard summary={dataset.validationSummary} />
    </div>
  );
}
