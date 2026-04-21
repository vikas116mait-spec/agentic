import { describe, expect, it } from "vitest";
import { isSelectableFineTuningProfile, sortFineTuningProfiles } from "@/lib/model-profiles";
import type { ModelProfile } from "@/lib/types";

function profile(overrides: Partial<ModelProfile>): ModelProfile {
  return {
    id: overrides.id ?? "profile",
    name: overrides.name ?? "Profile",
    provider: overrides.provider ?? "local",
    providerLabel: overrides.providerLabel ?? "Local GPU QLoRA",
    providerConfigured: overrides.providerConfigured ?? true,
    supportsInference: overrides.supportsInference ?? true,
    supportsFineTuning: overrides.supportsFineTuning ?? true,
    model: overrides.model ?? "Qwen/Qwen2.5-3B-Instruct",
    category: overrides.category ?? "large",
    description: overrides.description ?? "",
    createdAt: overrides.createdAt ?? "2026-01-01T00:00:00Z",
    updatedAt: overrides.updatedAt ?? "2026-01-01T00:00:00Z"
  };
}

describe("sortFineTuningProfiles", () => {
  it("puts the recommended local 3B profile first", () => {
    const sorted = sortFineTuningProfiles([
      profile({ id: "hf", provider: "huggingface", providerLabel: "Hugging Face Jobs (Paid Cloud)" }),
      profile({ id: "local-15b", model: "Qwen/Qwen2.5-1.5B-Instruct", name: "Qwen 1.5B Local" }),
      profile({ id: "local-3b", model: "Qwen/Qwen2.5-3B-Instruct", name: "Local GPU QLoRA" }),
      profile({ id: "openai", provider: "openai", providerLabel: "OpenAI", model: "gpt-4.1-mini-2025-04-14" })
    ]);

    expect(sorted.map((item) => item.id)).toEqual(["local-3b", "local-15b", "openai", "hf"]);
  });
});

describe("isSelectableFineTuningProfile", () => {
  it("keeps the local free path visible even before setup is complete", () => {
    expect(
      isSelectableFineTuningProfile(
        profile({
          id: "local-needs-setup",
          provider: "local",
          providerConfigured: false,
        })
      )
    ).toBe(true);
  });

  it("still hides unconfigured paid fine-tuning providers", () => {
    expect(
      isSelectableFineTuningProfile(
        profile({
          id: "hf-needs-setup",
          provider: "huggingface",
          providerConfigured: false,
        })
      )
    ).toBe(false);
  });
});
