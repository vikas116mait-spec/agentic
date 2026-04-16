import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { validateJsonlFile } from "@/lib/jsonl";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

async function createTempJsonl(contents: string, fileName = "dataset.jsonl") {
  const directory = await mkdtemp(path.join(os.tmpdir(), "agentic-jsonl-"));
  tempDirs.push(directory);
  const filePath = path.join(directory, fileName);
  await writeFile(filePath, contents, "utf8");
  return filePath;
}

describe("validateJsonlFile", () => {
  it("accepts valid chat training records", async () => {
    const filePath = await createTempJsonl(
      JSON.stringify({
        messages: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there" }
        ]
      })
    );

    const result = await validateJsonlFile(filePath);

    expect(result.validRecords).toBe(1);
    expect(result.invalidRecords).toBe(0);
    expect(result.errors).toHaveLength(0);
  });

  it("flags malformed JSON and missing assistant content", async () => {
    const filePath = await createTempJsonl(
      [
        '{"messages":[{"role":"user","content":"Hello"}]}',
        '{"messages":[{"role":"assistant","content":""}]}',
        '{"messages":'
      ].join("\n")
    );

    const result = await validateJsonlFile(filePath);

    expect(result.totalRecords).toBe(3);
    expect(result.invalidRecords).toBe(3);
    expect(result.errors.map((error) => error.line)).toEqual([1, 2, 3]);
  });
});
