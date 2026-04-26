import type { ModelProfile } from "@/lib/types";

const LOCAL_FINE_TUNING_MODEL_ORDER = new Map([
  ["Qwen/Qwen2.5-3B-Instruct", 0],
  ["Qwen/Qwen2.5-1.5B-Instruct", 1],
  ["Qwen/Qwen2.5-0.5B-Instruct", 2]
]);

export function findModelProfile(profiles: ModelProfile[], profileId: string | null | undefined) {
  if (!profileId) {
    return null;
  }

  return profiles.find((profile) => profile.id === profileId) ?? null;
}

export function modelProfileLabel(profile: ModelProfile) {
  return `${profile.name} | ${profile.providerLabel} | ${profile.model}`;
}

export function isRunnableProfile(profile: ModelProfile) {
  return profile.providerConfigured && profile.supportsInference;
}

export function isFineTuningProfile(profile: ModelProfile) {
  return profile.providerConfigured && profile.supportsFineTuning;
}

export function isSelectableFineTuningProfile(profile: ModelProfile) {
  return profile.supportsFineTuning && (profile.providerConfigured || profile.provider === "local");
}

export function sortFineTuningProfiles(profiles: ModelProfile[]) {
  return [...profiles].sort((left, right) => {
    const providerPriority = (profile: ModelProfile) => {
      if (profile.provider === "local") return 0;
      if (profile.provider === "openai") return 1;
      if (profile.provider === "huggingface") return 2;
      return 3;
    };

    const providerDelta = providerPriority(left) - providerPriority(right);
    if (providerDelta !== 0) {
      return providerDelta;
    }

    const modelPriority = (profile: ModelProfile) => LOCAL_FINE_TUNING_MODEL_ORDER.get(profile.model) ?? 50;
    const modelDelta = modelPriority(left) - modelPriority(right);
    if (modelDelta !== 0) {
      return modelDelta;
    }

    return modelProfileLabel(left).localeCompare(modelProfileLabel(right));
  });
}
