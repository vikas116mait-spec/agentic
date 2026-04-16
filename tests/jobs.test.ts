import { describe, expect, it } from "vitest";
import { mapFineTuneJob, normalizeJobStatus } from "@/lib/jobs";

describe("normalizeJobStatus", () => {
  it("keeps known statuses", () => {
    expect(normalizeJobStatus("running")).toBe("running");
  });

  it("falls back for unknown statuses", () => {
    expect(normalizeJobStatus("mystery_status")).toBe("unknown");
  });
});

describe("mapFineTuneJob", () => {
  it("maps API payload values into database shape", () => {
    const mapped = mapFineTuneJob({
      status: "succeeded",
      fine_tuned_model: "ft:model",
      trained_tokens: 123,
      estimated_finish: 1746484925,
      result_files: ["file-1"],
      finished_at: 1746485841,
      method: { type: "supervised" }
    });

    expect(mapped.status).toBe("succeeded");
    expect(mapped.fineTunedModel).toBe("ft:model");
    expect(mapped.trainedTokens).toBe(123);
    expect(mapped.resultFilesJson).toEqual(["file-1"]);
    expect(mapped.hyperparametersJson).toEqual({ type: "supervised" });
    expect(mapped.estimatedFinishAt).toBeInstanceOf(Date);
    expect(mapped.finishedAt).toBeInstanceOf(Date);
  });
});
