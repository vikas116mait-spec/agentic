"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { Input } from "@/components/ui/input";
import { LoadingState } from "@/components/ui/loading-state";
import { pythonApiFetch } from "@/lib/python-api";

type DatasetOption = { id: string; name: string };
type RuntimeStatus = {
  supportsFineTuning: boolean;
  modelProviderLabel: string;
  defaultBaseModel: string;
};

export function CreateJobForm({ initialDatasetId = "" }: { initialDatasetId?: string }) {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetOption[] | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [datasetId, setDatasetId] = useState(initialDatasetId);
  const [baseModel, setBaseModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      pythonApiFetch<DatasetOption[]>("/datasets?validationStatus=VALID"),
      pythonApiFetch<RuntimeStatus>("/agent/runtime")
    ])
      .then(([items, runtimePayload]) => {
        setDatasets(items);
        setRuntime(runtimePayload);
        setBaseModel((current) => current || runtimePayload.defaultBaseModel);
        if (!datasetId && items[0]?.id) {
          setDatasetId(items[0].id);
        }
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [initialDatasetId]);

  async function handleCreate() {
    setLoading(true);
    setError(null);

    try {
      const payload = await pythonApiFetch<{ id: string }>("/jobs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          datasetId,
          baseModel
        })
      });
      router.push(`/jobs/${payload.id}`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not create job.");
      setLoading(false);
      return;
    }
  }

  if (error && !datasets) {
    return <ErrorAlert title="Could not load datasets" description={error} />;
  }

  if (!datasets) {
    return <LoadingState label="Loading valid datasets..." />;
  }

  if (datasets.length === 0) {
    return <ErrorAlert title="No valid datasets yet" description="Upload and validate a dataset first, then come back to create a job." />;
  }

  if (runtime && !runtime.supportsFineTuning) {
    return (
      <ErrorAlert
        title="Fine-tuning jobs are disabled in local mode"
        description={`${runtime.modelProviderLabel} is active for low-cost local inference. Switch to OpenAI later when you want managed fine-tuning jobs again.`}
      />
    );
  }

  return (
    <div className="space-y-4">
      <select
        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
        value={datasetId}
        onChange={(event) => setDatasetId(event.target.value)}
      >
        {datasets.map((dataset) => (
          <option key={dataset.id} value={dataset.id}>
            {dataset.name}
          </option>
        ))}
      </select>

      <Input
        value={baseModel}
        onChange={(event) => setBaseModel(event.target.value)}
        placeholder={runtime?.defaultBaseModel ?? "Base model"}
        required
      />

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      <Button disabled={loading || !datasetId} onClick={handleCreate}>
        {loading ? "Creating..." : "Create job"}
      </Button>
    </div>
  );
}
