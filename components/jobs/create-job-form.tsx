"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-state";
import {
  findModelProfile,
  isRunnableProfile,
  isSelectableFineTuningProfile,
  modelProfileLabel,
  sortFineTuningProfiles,
} from "@/lib/model-profiles";
import {
  LOCAL_TRAINING_PRESETS,
  TARGET_MODULE_STRATEGIES,
  describeModelTier,
  getLocalTrainingPreset,
  type LocalTrainingPresetId,
  type TargetModuleStrategy,
} from "@/lib/local-training";
import { pythonApiFetch } from "@/lib/python-api";
import type { ModelProfilesResponse } from "@/lib/types";

type DatasetOption = { id: string; name: string };
type LocalTrainingRuntime = {
  enabled: boolean;
  configuredPython: string | null;
  pythonExists: boolean;
  providerConfigured: boolean;
  runtimePython: string | null;
  gpuCount: number;
  unslothAvailable: boolean;
  missingDependencies: string[];
  warnings: string[];
  ollama: {
    host: string;
    cliAvailable: boolean;
    reachable: boolean;
    error: string | null;
  };
};

const selectClassName = "w-full rounded-2xl border border-black/10 bg-white px-4 py-3";
const checkboxLabelClassName = "flex items-center gap-2 text-sm text-black/70 cursor-pointer";

export function CreateJobForm({ initialDatasetId = "" }: { initialDatasetId?: string }) {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetOption[] | null>(null);
  const [profilesData, setProfilesData] = useState<ModelProfilesResponse | null>(null);
  const [localRuntime, setLocalRuntime] = useState<LocalTrainingRuntime | null>(null);
  const [datasetId, setDatasetId] = useState(initialDatasetId);
  const [profileId, setProfileId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Export options — only relevant for local QLoRA jobs
  const [exportGguf, setExportGguf] = useState(true);
  const [ggufQuantization, setGgufQuantization] = useState("q4_k_m");
  const [pushToOllama, setPushToOllama] = useState(true);
  const [ollamaModelName, setOllamaModelName] = useState("");

  // Hyperparameters — collapsible, local provider only
  const [trainingPreset, setTrainingPreset] = useState<LocalTrainingPresetId>("balanced");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [numEpochs, setNumEpochs] = useState(LOCAL_TRAINING_PRESETS.balanced.numEpochs);
  const [learningRate, setLearningRate] = useState(LOCAL_TRAINING_PRESETS.balanced.learningRate);
  const [perDeviceBatchSize, setPerDeviceBatchSize] = useState(LOCAL_TRAINING_PRESETS.balanced.perDeviceBatchSize);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(LOCAL_TRAINING_PRESETS.balanced.gradientAccumulationSteps);
  const [maxSeqLength, setMaxSeqLength] = useState(LOCAL_TRAINING_PRESETS.balanced.maxSeqLength);
  const [loraRank, setLoraRank] = useState(LOCAL_TRAINING_PRESETS.balanced.loraRank);
  const [loraAlpha, setLoraAlpha] = useState(LOCAL_TRAINING_PRESETS.balanced.loraAlpha);
  const [loraDropout, setLoraDropout] = useState(LOCAL_TRAINING_PRESETS.balanced.loraDropout);
  const [warmupRatio, setWarmupRatio] = useState(LOCAL_TRAINING_PRESETS.balanced.warmupRatio);
  const [weightDecay, setWeightDecay] = useState(LOCAL_TRAINING_PRESETS.balanced.weightDecay);
  const [evalRatio, setEvalRatio] = useState(LOCAL_TRAINING_PRESETS.balanced.evalRatio);
  const [evalMaxSamples, setEvalMaxSamples] = useState(LOCAL_TRAINING_PRESETS.balanced.evalMaxSamples);
  const [evalMaxNewTokens, setEvalMaxNewTokens] = useState(LOCAL_TRAINING_PRESETS.balanced.evalMaxNewTokens);
  const [targetModuleStrategy, setTargetModuleStrategy] = useState<TargetModuleStrategy>(LOCAL_TRAINING_PRESETS.balanced.targetModuleStrategy);
  const [inferenceTemperature, setInferenceTemperature] = useState(LOCAL_TRAINING_PRESETS.balanced.inferenceTemperature);
  const [inferenceTopP, setInferenceTopP] = useState(LOCAL_TRAINING_PRESETS.balanced.inferenceTopP);
  const [inferenceTopK, setInferenceTopK] = useState(LOCAL_TRAINING_PRESETS.balanced.inferenceTopK);
  const [inferenceRepeatPenalty, setInferenceRepeatPenalty] = useState(LOCAL_TRAINING_PRESETS.balanced.inferenceRepeatPenalty);

  function preferredFreeProfile(profiles: ModelProfilesResponse["profiles"]) {
    return (
      profiles.find((profile) => profile.provider === "local" && profile.model === "Qwen/Qwen2.5-3B-Instruct") ??
      profiles.find((profile) => profile.provider === "local") ??
      null
    );
  }

  useEffect(() => {
    Promise.all([
      pythonApiFetch<DatasetOption[]>("/datasets?validationStatus=VALID"),
      pythonApiFetch<ModelProfilesResponse>("/settings/model-profiles"),
      pythonApiFetch<LocalTrainingRuntime>("/settings/local-training/runtime").catch(() => null),
    ])
      .then(([items, profilesPayload, runtimePayload]) => {
        const fineTuningProfiles = sortFineTuningProfiles(profilesPayload.profiles.filter(isSelectableFineTuningProfile));
        const freeProfile = preferredFreeProfile(fineTuningProfiles);
        const preferredProfile = findModelProfile(fineTuningProfiles, profilesPayload.defaults.jobBaseProfileId);
        setDatasets(items);
        setProfilesData(profilesPayload);
        setLocalRuntime(runtimePayload);
        setProfileId((current) => current || freeProfile?.id || preferredProfile?.id || fineTuningProfiles[0]?.id || "");
        if (!datasetId && items[0]?.id) {
          setDatasetId(items[0].id);
        }
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [initialDatasetId]);

  useEffect(() => {
    if (!exportGguf && pushToOllama) {
      setPushToOllama(false);
    }
  }, [exportGguf, pushToOllama]);

  const fineTuningProfiles = useMemo(
    () => sortFineTuningProfiles((profilesData?.profiles ?? []).filter(isSelectableFineTuningProfile)),
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
  const activePreset = getLocalTrainingPreset(trainingPreset);
  const selectedModelTier = selectedProfile ? describeModelTier(selectedProfile.model) : null;

  const isLocalProvider = selectedProfile?.provider === "local";
  const selectedProfileNeedsSetup = Boolean(selectedProfile && !selectedProfile.providerConfigured);

  function applyTrainingPreset(presetId: LocalTrainingPresetId) {
    const preset = LOCAL_TRAINING_PRESETS[presetId];
    setTrainingPreset(presetId);
    setNumEpochs(preset.numEpochs);
    setLearningRate(preset.learningRate);
    setPerDeviceBatchSize(preset.perDeviceBatchSize);
    setGradientAccumulationSteps(preset.gradientAccumulationSteps);
    setMaxSeqLength(preset.maxSeqLength);
    setLoraRank(preset.loraRank);
    setLoraAlpha(preset.loraAlpha);
    setLoraDropout(preset.loraDropout);
    setWarmupRatio(preset.warmupRatio);
    setWeightDecay(preset.weightDecay);
    setEvalRatio(preset.evalRatio);
    setEvalMaxSamples(preset.evalMaxSamples);
    setEvalMaxNewTokens(preset.evalMaxNewTokens);
    setTargetModuleStrategy(preset.targetModuleStrategy);
    setInferenceTemperature(preset.inferenceTemperature);
    setInferenceTopP(preset.inferenceTopP);
    setInferenceTopK(preset.inferenceTopK);
    setInferenceRepeatPenalty(preset.inferenceRepeatPenalty);
  }

  async function handleCreate() {
    if (!selectedProfile) {
      setError("Choose a fine-tuning profile first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const localHyperparameters = isLocalProvider
        ? {
            num_train_epochs: numEpochs,
            learning_rate: learningRate,
            per_device_train_batch_size: perDeviceBatchSize,
            gradient_accumulation_steps: gradientAccumulationSteps,
            max_seq_length: maxSeqLength,
            lora_r: loraRank,
            lora_alpha: loraAlpha,
            lora_dropout: loraDropout,
            warmup_ratio: warmupRatio,
            weight_decay: weightDecay,
            eval_ratio: evalRatio,
            eval_max_samples: evalMaxSamples,
            eval_max_new_tokens: evalMaxNewTokens,
            target_module_strategy: targetModuleStrategy,
            inference_temperature: inferenceTemperature,
            inference_top_p: inferenceTopP,
            inference_top_k: inferenceTopK,
            inference_repeat_penalty: inferenceRepeatPenalty,
          }
        : undefined;
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
            hyperparameters: localHyperparameters,
            trainingPreset,
            exportGguf,
            ggufQuantization,
            pushToOllama: exportGguf ? pushToOllama : false,
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
          description="Create or enable a Local GPU QLoRA profile in Models for the free path. Set LOCAL_TRAINING_ENABLED=1 and LOCAL_TRAINING_PYTHON to your training venv, then come back here."
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
                {profile.providerConfigured ? "" : " | setup needed"}
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
          {selectedModelTier ? (
            <p className="mt-1">
              Fit: <span className="font-medium text-black/80">{selectedModelTier.tier}</span> - {selectedModelTier.useCase}
            </p>
          ) : null}
        </div>
      ) : null}

      {isLocalProvider ? (
        <div className="space-y-4">
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-black/70">
            Recommended free path: Local GPU QLoRA trains on your own NVIDIA GPU, saves the adapter locally, and can auto-export to GGUF for Ollama.
          </div>

          <div className="rounded-[1.5rem] border border-black/8 bg-white/80 p-5 space-y-4">
            <div>
              <p className="text-sm font-medium text-black/70">Training speed profile</p>
              <p className="mt-1 text-sm text-black/55">
                Choose a preset first. You can still fine-tune the numbers manually below if you need more control.
              </p>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              {Object.values(LOCAL_TRAINING_PRESETS).map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => applyTrainingPreset(preset.id)}
                  className={[
                    "rounded-[1.25rem] border px-4 py-4 text-left transition",
                    trainingPreset === preset.id
                      ? "border-brand bg-brand/5 shadow-sm"
                      : "border-black/10 bg-white hover:border-black/25",
                  ].join(" ")}
                >
                  <p className="text-sm font-semibold text-black/80">{preset.label}</p>
                  <p className="mt-2 text-xs text-black/55">{preset.description}</p>
                  <div className="mt-3 space-y-1 text-xs text-black/50">
                    <p>{preset.maxSteps ? `${preset.maxSteps} step cap` : `${preset.numEpochs} epochs`}</p>
                    <p>Batch {preset.perDeviceBatchSize} x Acc {preset.gradientAccumulationSteps}</p>
                    <p>Context {preset.maxSeqLength} tokens</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {localRuntime ? (
            <div className="rounded-[1.5rem] border border-black/8 bg-white/80 p-5 space-y-4">
              <div>
                <p className="text-sm font-medium text-black/70">Local runtime readiness</p>
                <p className="mt-1 text-sm text-black/55">
                  Live snapshot of the training runtime, accelerator path, and Ollama availability.
                </p>
              </div>
              <div className="grid gap-3 md:grid-cols-4">
                {[
                  {
                    label: "Trainer runtime",
                    value: localRuntime.providerConfigured ? "Ready" : "Needs setup",
                    hint: localRuntime.configuredPython ?? "Set LOCAL_TRAINING_PYTHON",
                  },
                  {
                    label: "GPU",
                    value: localRuntime.gpuCount > 0 ? `${localRuntime.gpuCount} detected` : "No CUDA GPU",
                    hint: localRuntime.gpuCount > 0 ? "Local QLoRA can use your GPU." : "CPU fallback is much slower.",
                  },
                  {
                    label: "Unsloth",
                    value: localRuntime.unslothAvailable ? "Accelerated" : "Standard path",
                    hint: localRuntime.unslothAvailable ? "Faster training and GGUF export." : "Falls back to HuggingFace + PEFT.",
                  },
                  {
                    label: "Ollama",
                    value: localRuntime.ollama?.reachable ? "Reachable" : "Offline",
                    hint: localRuntime.ollama?.host ?? "http://127.0.0.1:11434",
                  },
                ].map((item) => (
                  <div key={item.label} className="rounded-2xl bg-muted/60 px-4 py-3">
                    <p className="text-[11px] uppercase tracking-[0.15em] text-black/40">{item.label}</p>
                    <p className="mt-2 text-sm font-semibold text-black/80">{item.value}</p>
                    <p className="mt-1 text-xs text-black/50">{item.hint}</p>
                  </div>
                ))}
              </div>
              {localRuntime.warnings.length > 0 ? (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-black/70">
                  {localRuntime.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {selectedProfileNeedsSetup ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-black/70">
          Local GPU QLoRA is selected, but this workspace still needs setup. Point <code>LOCAL_TRAINING_PYTHON</code> at a real training venv Python, then restart the Python API so the free path becomes runnable.
        </div>
      ) : null}

      {isLocalProvider ? (
        <>
        <div className="rounded-[1.5rem] border border-black/8 bg-white/80 p-5 space-y-4">
          <p className="text-sm font-medium text-black/70">Export options</p>
          <p className="text-sm text-black/55">
            The base model is downloaded automatically for local fine-tuning. After training, the adapter is saved on
            disk, GGUF export is enabled by default, and Ollama registration is the recommended end state when your local Ollama runtime is available.
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
                    placeholder="Leave blank to auto-generate"
                    value={ollamaModelName}
                    onChange={(e) => setOllamaModelName(e.target.value)}
                    className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                  />
                  <p className="text-xs text-black/45">
                    Leave this blank to auto-generate a name from the dataset and base model. After training, run <code className="font-mono">ollama run {ollamaModelName || "generated-model-name"}</code> to use it.
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
            <div className="space-y-4 pt-2">
              <div className="rounded-2xl bg-muted/60 px-4 py-3 text-sm text-black/65">
                <p className="font-medium text-black/80">Preset summary</p>
                <p className="mt-1">{activePreset.description}</p>
                <p className="mt-2 text-xs text-black/50">
                  Current preset uses LoRA rank {activePreset.loraRank}, gradient accumulation {activePreset.gradientAccumulationSteps}, and context length {activePreset.maxSeqLength}.
                </p>
              </div>
              <div className="space-y-5">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-black/45">Schedule</p>
                  <div className="mt-3 grid gap-4 md:grid-cols-3">
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
                      <p className="text-xs text-black/40">Preset default: {activePreset.numEpochs}. More epochs increase adaptation and overfitting risk.</p>
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
                      <p className="text-xs text-black/40">Preset default: {activePreset.learningRate}. Lower is slower but usually steadier.</p>
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
                      <p className="text-xs text-black/40">Preset default: {activePreset.perDeviceBatchSize}. Reduce first if you hit OOM.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Gradient accumulation</span>
                      <input
                        type="number"
                        min={1}
                        max={64}
                        value={gradientAccumulationSteps}
                        onChange={(e) => setGradientAccumulationSteps(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Higher values simulate a larger batch without increasing immediate VRAM.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Max sequence length</span>
                      <input
                        type="number"
                        min={256}
                        max={8192}
                        step={128}
                        value={maxSeqLength}
                        onChange={(e) => setMaxSeqLength(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Longer context helps longer examples but raises VRAM and training time.</p>
                    </label>
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-black/45">Adapter capacity</p>
                  <div className="mt-3 grid gap-4 md:grid-cols-3">
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">LoRA rank</span>
                      <input
                        type="number"
                        min={4}
                        max={256}
                        step={4}
                        value={loraRank}
                        onChange={(e) => setLoraRank(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Higher rank increases adaptation capacity and memory use.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">LoRA alpha</span>
                      <input
                        type="number"
                        min={4}
                        max={512}
                        step={4}
                        value={loraAlpha}
                        onChange={(e) => setLoraAlpha(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Scaling factor paired with rank. Stronger updates usually use a higher alpha.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">LoRA dropout</span>
                      <input
                        type="number"
                        min={0}
                        max={0.5}
                        step={0.01}
                        value={loraDropout}
                        onChange={(e) => setLoraDropout(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Small dropout can regularize tiny datasets; 0 keeps the strongest fit.</p>
                    </label>
                    <label className="space-y-1.5 text-sm md:col-span-2">
                      <span className="font-medium text-black/70">Target layer strategy</span>
                      <select
                        value={targetModuleStrategy}
                        onChange={(e) => setTargetModuleStrategy(e.target.value as TargetModuleStrategy)}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      >
                        {TARGET_MODULE_STRATEGIES.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-black/40">
                        {TARGET_MODULE_STRATEGIES.find((option) => option.id === targetModuleStrategy)?.description}
                      </p>
                    </label>
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-black/45">Regularization and evaluation</p>
                  <div className="mt-3 grid gap-4 md:grid-cols-3">
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Warmup ratio</span>
                      <input
                        type="number"
                        min={0}
                        max={0.5}
                        step={0.01}
                        value={warmupRatio}
                        onChange={(e) => setWarmupRatio(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">A short warmup often helps stability on small, noisy datasets.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Weight decay</span>
                      <input
                        type="number"
                        min={0}
                        max={0.2}
                        step={0.005}
                        value={weightDecay}
                        onChange={(e) => setWeightDecay(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Regularizes updates and can help avoid memorizing tiny datasets.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Eval split ratio</span>
                      <input
                        type="number"
                        min={0}
                        max={0.4}
                        step={0.01}
                        value={evalRatio}
                        onChange={(e) => setEvalRatio(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">How much of the dataset to hold out for quality checks.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Generation eval samples</span>
                      <input
                        type="number"
                        min={0}
                        max={32}
                        value={evalMaxSamples}
                        onChange={(e) => setEvalMaxSamples(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Runs a small held-out answer quality check after training.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Generation eval max new tokens</span>
                      <input
                        type="number"
                        min={32}
                        max={1024}
                        step={16}
                        value={evalMaxNewTokens}
                        onChange={(e) => setEvalMaxNewTokens(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Cap for each held-out generation during evaluation.</p>
                    </label>
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-black/45">Inference defaults</p>
                  <div className="mt-3 grid gap-4 md:grid-cols-3">
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Temperature</span>
                      <input
                        type="number"
                        min={0}
                        max={2}
                        step={0.01}
                        value={inferenceTemperature}
                        onChange={(e) => setInferenceTemperature(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Lower values make responses steadier and more literal.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Top-p</span>
                      <input
                        type="number"
                        min={0.1}
                        max={1}
                        step={0.01}
                        value={inferenceTopP}
                        onChange={(e) => setInferenceTopP(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Nucleus sampling cap for the registered Ollama model.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Top-k</span>
                      <input
                        type="number"
                        min={1}
                        max={200}
                        value={inferenceTopK}
                        onChange={(e) => setInferenceTopK(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Token shortlist size used during Ollama inference.</p>
                    </label>
                    <label className="space-y-1.5 text-sm">
                      <span className="font-medium text-black/70">Repeat penalty</span>
                      <input
                        type="number"
                        min={1}
                        max={2}
                        step={0.01}
                        value={inferenceRepeatPenalty}
                        onChange={(e) => setInferenceRepeatPenalty(Number(e.target.value))}
                        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm"
                      />
                      <p className="text-xs text-black/40">Helps reduce looping and repeated phrases in the exported Ollama model.</p>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        </>
      ) : null}

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      <Button disabled={loading || !datasetId || !selectedProfile || selectedProfileNeedsSetup} onClick={handleCreate}>
        {loading ? "Starting..." : selectedProfileNeedsSetup ? "Complete local setup to continue" : "Start fine-tuning"}
      </Button>

      <p className="text-sm text-black/45">
        Tip: for free fine-tuning, point <code>LOCAL_TRAINING_PYTHON</code> at your <code>.venv-train</code> runtime and install Unsloth for faster GGUF export with less VRAM.
      </p>
    </div>
  );
}
