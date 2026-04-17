"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import { findModelProfile, isFineTuningProfile, isRunnableProfile, modelProfileLabel } from "@/lib/model-profiles";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfilesResponse } from "@/lib/types";

type DatasetOption = { id: string; name: string };

const selectClassName = "w-full rounded-2xl border border-black/10 bg-white px-4 py-3";

export function CreateJobForm({ initialDatasetId = "" }: { initialDatasetId?: string }) {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetOption[] | null>(null);
  const [profilesData, setProfilesData] = useState<ModelProfilesResponse | null>(null);
  const [datasetId, setDatasetId] = useState(initialDatasetId);
  const [profileId, setProfileId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      pythonApiFetch<DatasetOption[]>("/datasets?validationStatus=VALID"),
      pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles")
    ])
      .then(([items, profilesPayload]) => {
        const fineTuningProfiles = profilesPayload.profiles.filter(isFineTuningProfile);
        const preferredProfile = findModelProfile(fineTuningProfiles, profilesPayload.defaults.jobBaseProfileId);
        setDatasets(items);
        setProfilesData(profilesPayload);
        setProfileId((current) => current || preferredProfile?.id || fineTuningProfiles[0]?.id || "");
        if (!datasetId && items[0]?.id) {
          setDatasetId(items[0].id);
        }
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [initialDatasetId]);

  const fineTuningProfiles = useMemo(
    () => (profilesData?.profiles ?? []).filter(isFineTuningProfile),
    [profilesData?.profiles]
  );
  const runnableProfiles = useMemo(
    () => (profilesData?.profiles ?? []).filter(isRunnableProfile),
    [profilesData?.profiles]
  );
  const selectedProfile = useMemo(
    () => findModelProfile(fineTuningProfiles, profileId),
    [fineTuningProfiles, profileId]
  );

  async function handleCreate() {
    if (!selectedProfile) {
      setError("Choose a fine-tuning profile first.");
      return;
    }

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
          baseModel: selectedProfile.model,
          modelProvider: selectedProfile.provider
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

  if (!datasets || !profilesData) {
    return <LoadingState label="Loading valid datasets..." />;
  }

  if (datasets.length === 0) {
    return <ErrorAlert title="No valid datasets yet" description="Upload and validate a dataset first, then come back to create a job." />;
  }

  if (fineTuningProfiles.length === 0) {
    return (
      <div className="space-y-4 rounded-[1.5rem] border border-black/8 bg-white/80 p-6 shadow-sm">
        <ErrorAlert
          title="No fine-tuning profiles available"
          description="Create or enable an OpenAI, Hugging Face Jobs, or Local GPU QLoRA profile in Models. Set OPENAI_API_KEY for OpenAI, HF_TOKEN for Hugging Face Jobs, or keep LOCAL_TRAINING_ENABLED on to use your own GPU."
        />
        {runnableProfiles.length > 0 ? (
          <div className="rounded-2xl bg-amber-50 p-4 text-sm text-black/70">
            Other runnable profiles are still available for Playground and Agent:{" "}
            {runnableProfiles.map(modelProfileLabel).join(" • ")}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-3">
          <Link href="/settings">
            <Button>Open models</Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm">
          <span className="font-medium text-black/70">1. Dataset</span>
          <select className={selectClassName} value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2 text-sm">
          <span className="font-medium text-black/70">2. Base model</span>
          <select className={selectClassName} value={profileId} onChange={(event) => setProfileId(event.target.value)}>
            {fineTuningProfiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {modelProfileLabel(profile)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {selectedProfile ? (
        <div className="rounded-2xl bg-white p-4 text-sm text-black/65">
          <p className="font-medium text-black/80">Selected run</p>
          <p className="mt-1">Dataset: {datasets.find((dataset) => dataset.id === datasetId)?.name ?? "Choose a dataset"}</p>
          <p className="mt-1">Model: {modelProfileLabel(selectedProfile)}</p>
        </div>
      ) : null}

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      <Button disabled={loading || !datasetId || !selectedProfile} onClick={handleCreate}>
        {loading ? "Starting..." : "Start fine-tuning"}
      </Button>

      <p className="text-sm text-black/45">
        Tip: use Local GPU QLoRA for free local experiments.
      </p>
    </div>
  );
}
