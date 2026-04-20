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
const checkboxLabelClassName = "flex items-center gap-2 text-sm text-black/70 cursor-pointer";

export function CreateJobForm({ initialDatasetId = "" }: { initialDatasetId?: string }) {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetOption[] | null>(null);
  const [profilesData, setProfilesData] = useState<ModelProfilesResponse | null>(null);
  const [datasetId, setDatasetId] = useState(initialDatasetId);
  const [profileId, setProfileId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Export options — only relevant for local QLoRA jobs
  const [exportGguf, setExportGguf] = useState(true);
  const [ggufQuantization, setGgufQuantization] = useState("q4_k_m");
  const [pushToOllama, setPushToOllama] = useState(false);
  const [ollamaModelName, setOllamaModelName] = useState("");

  // Hyperparameters — collapsible, local provider only
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [numEpochs, setNumEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [perDeviceBatchSize, setPerDeviceBatchSize] = useState(2);

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

  const isLocalProvider = selectedProfile?.provider === "local";

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
          modelProvider: selectedProfile.provider,
          ...(isLocalProvider && {
            exportGguf,
            ggufQuantization,
            pushToOllama,
            ollamaModelName,
            numEpochs,
            learningRate,
            perDeviceBatchSize,
          }),
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

      {isLocalProvider ? (
        <>
        <div className="rounded-[1.5rem] border border-black/8 bg-white/80 p-5 space-y-4">
          <p className="text-sm font-medium text-black/70">Export options</p>
          <p className="text-sm text-black/55">
            The base model is downloaded automatically for local fine-tuning. After training, the adapter is saved on
            disk and a GGUF export is enabled by default so you can download a ready-to-use artifact.
          </p>

          <label className={checkboxLabelClassName}>
            <input
              type="checkbox"
              checked={exportGguf}
              onChange={(e) => setExportGguf(e.target.checked)}
              className="h-4 w-4 rounded"
            />
            Export to GGUF after training (requires Unsloth)
          </label>

          {exportGguf ? (
            <div className="ml-6 space-y-3">
              <label className="space-y-1 text-sm">
                <span className="text-black/60">Quantization format</span>
                <select
                  className={selectClassName}
                  value={ggufQuantization}
                  onChange={(e) => setGgufQuantization(e.target.value)}
                >
                  <option value="q4_k_m">Q4_K_M — recommended, best speed/quality balance</option>
                  <option value="q8_0">Q8_0 — higher quality, larger file</option>
                  <option value="f16">F16 — full precision, largest file</option>
                  <option value="q2_k">Q2_K — smallest file, lower quality</option>
                  <option value="q5_k_m">Q5_K_M — high quality, moderate size</option>
                </select>
              </label>

              <label className={checkboxLabelClassName}>
                <input
                  type="checkbox"
                  checked={pushToOllama}
                  onChange={(e) => setPushToOllama(e.target.checked)}
                  className="h-4 w-4 rounded"
                />
                Push to local Ollama after export
              </label>

              {pushToOllama ? (
                <label className="space-y-1 text-sm">
                  <span className="text-black/60">Ollama model name</span>
                  <input
                    type="text"
                    placeholder="e.g. my-fine-tuned-model"
                    value={ollamaModelName}
                    onChange={(e) => setOllamaModelName(e.target.value)}
                    className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                  />
                  <p className="text-xs text-black/45">
                    After training, run <code className="font-mono">ollama run {ollamaModelName || "your-model-name"}</code> to use it.
                  </p>
                </label>
              ) : null}
            </div>
          ) : null}
        </div>

        {/* Hyperparameters */}
        <div className="rounded-[1.5rem] border border-black/8 bg-white/80 p-5 space-y-4">
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex w-full items-center justify-between text-sm font-medium text-black/70"
          >
            <span>Training hyperparameters</span>
            <span className="text-xs text-black/40">{showAdvanced ? "Hide" : "Customize"}</span>
          </button>

          {showAdvanced && (
            <div className="grid gap-4 md:grid-cols-3 pt-2">
              <label className="space-y-1.5 text-sm">
                <span className="font-medium text-black/70">Epochs</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={numEpochs}
                  onChange={(e) => setNumEpochs(Number(e.target.value))}
                  className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                />
                <p className="text-xs text-black/40">Default: 3. More epochs = more overfitting risk.</p>
              </label>
              <label className="space-y-1.5 text-sm">
                <span className="font-medium text-black/70">Learning rate</span>
                <input
                  type="number"
                  step={1e-5}
                  min={1e-6}
                  max={1e-2}
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                />
                <p className="text-xs text-black/40">Default: 0.0002. Lower = slower, more stable.</p>
              </label>
              <label className="space-y-1.5 text-sm">
                <span className="font-medium text-black/70">Batch size (per device)</span>
                <select
                  value={perDeviceBatchSize}
                  onChange={(e) => setPerDeviceBatchSize(Number(e.target.value))}
                  className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                >
                  <option value={1}>1 — lowest VRAM</option>
                  <option value={2}>2 — default</option>
                  <option value={4}>4 — faster, needs more VRAM</option>
                  <option value={8}>8 — large GPU only</option>
                </select>
                <p className="text-xs text-black/40">Default: 2. Reduce if you get OOM errors.</p>
              </label>
            </div>
          )}
        </div>
        </>
      ) : null}

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      <Button disabled={loading || !datasetId || !selectedProfile} onClick={handleCreate}>
        {loading ? "Starting..." : "Start fine-tuning"}
      </Button>

      <p className="text-sm text-black/45">
        Tip: use Local GPU QLoRA for free local experiments. Install Unsloth for 2x speed and 70% less VRAM.
      </p>
    </div>
  );
}
