export type LocalTrainingPresetId = "fast" | "balanced" | "quality";

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
    maxSeqLength: 1024,
  },
  quality: {
    id: "quality",
    label: "Best quality",
    shortLabel: "Quality",
    description: "Slower but more thorough settings for longer runs and better adaptation quality.",
    numEpochs: 4,
    learningRate: 8e-5,
    perDeviceBatchSize: 1,
    gradientAccumulationSteps: 8,
    maxSeqLength: 1536,
  },
};

export function getLocalTrainingPreset(presetId: string | null | undefined): LocalTrainingPreset {
  if (presetId === "fast" || presetId === "quality" || presetId === "balanced") {
    return LOCAL_TRAINING_PRESETS[presetId];
  }
  return LOCAL_TRAINING_PRESETS.balanced;
}

export function describeModelTier(modelId: string) {
  const normalized = modelId.toLowerCase();

  if (normalized.includes("0.5b") || normalized.includes("1b")) {
    return {
      tier: "Low VRAM",
      useCase: "Best for smoke tests and the smallest local GPUs.",
    };
  }

  if (
    normalized.includes("1.5b") ||
    normalized.includes("1.7b") ||
    normalized.includes("2b") ||
    normalized.includes("3b") ||
    normalized.includes("4b")
  ) {
    return {
      tier: "Balanced",
      useCase: "Good default for fast local iteration with usable quality.",
    };
  }

  if (normalized.includes("7b") || normalized.includes("8b") || normalized.includes("9b")) {
    return {
      tier: "Stronger quality",
      useCase: "Better outputs, but slower training or inference and more VRAM use.",
    };
  }

  return {
    tier: "Custom",
    useCase: "Review VRAM and runtime needs before using this model heavily.",
  };
}
