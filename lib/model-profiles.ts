import type { ModelProfile } from "@/lib/types";

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
