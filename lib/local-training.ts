export type LocalTrainingPresetId = "fast" | "balanced" | "quality";
export type TargetModuleStrategy = "auto" | "attention_only" | "attention_mlp" | "expanded";

export type LocalTrainingPreset = {
  id: LocalTrainingPresetId;
  label: string;
  shortLabel: string;
  description: string;
  maxSteps?: number;
  numEpochs: number;
  learningRate: number;
  perDeviceBatchSize: number;
  gradientAccumulationSteps: number;
  maxSeqLength: number;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  warmupRatio: number;
  weightDecay: number;
  evalRatio: number;
  evalMaxSamples: number;
  evalMaxNewTokens: number;
  targetModuleStrategy: TargetModuleStrategy;
  inferenceTemperature: number;
  inferenceTopP: number;
  inferenceTopK: number;
  inferenceRepeatPenalty: number;
};

export const LOCAL_TRAINING_PRESETS: Record<LocalTrainingPresetId, LocalTrainingPreset> = {
  fast: {
    id: "fast",
    label: "Fast smoke test",
    shortLabel: "Fastest",
    description: "Shortest run for pipeline validation and quick iteration on smaller datasets.",
    maxSteps: 60,
    numEpochs: 1,
    learningRate: 2e-4,
    perDeviceBatchSize: 2,
    gradientAccumulationSteps: 4,
    maxSeqLength: 1024,
    loraRank: 16,
    loraAlpha: 32,
    loraDropout: 0,
    warmupRatio: 0.05,
    weightDecay: 0.01,
    evalRatio: 0.1,
    evalMaxSamples: 3,
    evalMaxNewTokens: 128,
    targetModuleStrategy: "attention_mlp",
    inferenceTemperature: 0.2,
    inferenceTopP: 0.9,
    inferenceTopK: 40,
    inferenceRepeatPenalty: 1.08,
  },
  balanced: {
    id: "balanced",
    label: "Balanced",
    shortLabel: "Balanced",
    description: "Recommended default for most local GPUs: stable quality without overly slow training.",
    numEpochs: 3,
    learningRate: 1e-4,
    perDeviceBatchSize: 2,
    gradientAccumulationSteps: 4,
    maxSeqLength: 1536,
    loraRank: 32,
    loraAlpha: 64,
    loraDropout: 0.05,
    warmupRatio: 0.08,
    weightDecay: 0.02,
    evalRatio: 0.12,
    evalMaxSamples: 6,
    evalMaxNewTokens: 160,
    targetModuleStrategy: "auto",
    inferenceTemperature: 0.18,
    inferenceTopP: 0.88,
    inferenceTopK: 40,
    inferenceRepeatPenalty: 1.1,
  },
  quality: {
    id: "quality",
    label: "Best quality",
    shortLabel: "Quality",
    description: "Slower but more thorough settings for longer runs and better adaptation quality.",
    numEpochs: 4,
    learningRate: 7e-5,
    perDeviceBatchSize: 1,
    gradientAccumulationSteps: 8,
    maxSeqLength: 2048,
    loraRank: 48,
    loraAlpha: 96,
    loraDropout: 0.05,
    warmupRatio: 0.1,
    weightDecay: 0.02,
    evalRatio: 0.15,
    evalMaxSamples: 8,
    evalMaxNewTokens: 192,
    targetModuleStrategy: "expanded",
    inferenceTemperature: 0.15,
    inferenceTopP: 0.85,
    inferenceTopK: 50,
    inferenceRepeatPenalty: 1.12,
  },
};

export const TARGET_MODULE_STRATEGIES: Array<{
  id: TargetModuleStrategy;
  label: string;
  description: string;
}> = [
  {
    id: "auto",
    label: "Auto",
    description: "Inspect the model and adapt the best known projection layers automatically.",
  },
  {
    id: "attention_only",
    label: "Attention only",
    description: "Touch only attention projections for the smallest VRAM footprint.",
  },
  {
    id: "attention_mlp",
    label: "Attention + MLP",
    description: "Train both attention and feed-forward projections for stronger adaptation.",
  },
  {
    id: "expanded",
    label: "Expanded",
    description: "Try the widest supported projection set found on the model for maximum adaptation capacity.",
  },
];

export function getLocalTrainingPreset(presetId: string | null | undefined): LocalTrainingPreset {
  if (presetId === "fast" || presetId === "quality" || presetId === "balanced") {
    return LOCAL_TRAINING_PRESETS[presetId];
  }
  return LOCAL_TRAINING_PRESETS.balanced;
}

function extractParameterBillions(modelId: string): number | null {
  // Match "0.5b", "1.5b", "8b", "14b", "70b", "72b", etc. -- but not partial token like "1b" inside "14b".
  const match = modelId.toLowerCase().match(/(?<![\d.])(\d+(?:\.\d+)?)\s*b\b/);
  if (!match) {
    return null;
  }
  const value = Number.parseFloat(match[1]);
  return Number.isFinite(value) ? value : null;
}

export function describeModelTier(modelId: string) {
  const billions = extractParameterBillions(modelId);

  if (billions === null) {
    return {
      tier: "Custom",
      useCase: "Review VRAM and runtime needs before using this model heavily.",
    };
  }

  if (billions <= 1) {
    return {
      tier: "Low VRAM",
      useCase: "Best for smoke tests and the smallest local GPUs.",
    };
  }

  if (billions <= 4) {
    return {
      tier: "Balanced",
      useCase: "Good default for fast local iteration with usable quality.",
    };
  }

  if (billions <= 9) {
    return {
      tier: "Stronger quality",
      useCase: "Better outputs, but slower training or inference and more VRAM use.",
    };
  }

  if (billions <= 16) {
    return {
      tier: "High VRAM",
      useCase: "Higher accuracy 13-16B class. Recommended ~16 GB+ VRAM with QLoRA.",
    };
  }

  if (billions <= 40) {
    return {
      tier: "Heavy GPU",
      useCase: "Strong 30B-class models. Plan for ~24 GB+ VRAM, possibly with CPU offload.",
    };
  }

  return {
    tier: "Multi-GPU / cloud",
    useCase: "70B+ models usually need multiple GPUs or paid cloud fine-tuning.",
  };
}
