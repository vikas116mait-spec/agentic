import { describe, expect, it } from "vitest";
import { describeModelTier, getLocalTrainingPreset } from "@/lib/local-training";

describe("getLocalTrainingPreset", () => {
  it("falls back to balanced for unknown preset ids", () => {
    expect(getLocalTrainingPreset("unknown").id).toBe("balanced");
  });

  it("returns the configured fast preset values", () => {
    const preset = getLocalTrainingPreset("fast");

    expect(preset.numEpochs).toBe(1);
    expect(preset.gradientAccumulationSteps).toBe(2);
  });
});

describe("describeModelTier", () => {
  it("marks small models as low VRAM", () => {
    expect(describeModelTier("Qwen/Qwen2.5-0.5B-Instruct").tier).toBe("Low VRAM");
  });

  it("marks 2B-class models as balanced", () => {
    expect(describeModelTier("ibm-granite/granite-3.1-2b-instruct").tier).toBe("Balanced");
  });

  it("marks 7B models as stronger quality", () => {
    expect(describeModelTier("qwen2.5:7b").tier).toBe("Stronger quality");
  });
});
