import { promises as fs } from "node:fs";
import path from "node:path";
import { z } from "zod";
import type {
  DatasetValidationError,
  DatasetValidationExample,
  DatasetValidationSummary,
  DatasetValidationWarning
} from "@/lib/types";

const messageSchema = z.object({
  role: z.string().min(1, "role is required"),
  content: z.any()
});

const trainingRecordSchema = z.object({
  messages: z.array(messageSchema).min(1, "messages must not be empty")
});

export async function validateJsonlFile(filePath: string): Promise<DatasetValidationSummary> {
  if (path.extname(filePath).toLowerCase() !== ".jsonl") {
    return {
      totalRecords: 0,
      validRecords: 0,
      invalidRecords: 1,
      errors: [{ line: 0, message: "File must use the .jsonl extension." }],
      warnings: [],
      examples: []
    };
  }

  const content = await fs.readFile(filePath, "utf8");
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);

  const errors: DatasetValidationError[] = [];
  const warnings: DatasetValidationWarning[] = [];
  const examples: DatasetValidationExample[] = [];
  let validRecords = 0;

  lines.forEach((line, index) => {
    const lineNumber = index + 1;

    try {
      const record = JSON.parse(line) as unknown;
      const parsed = trainingRecordSchema.safeParse(record);

      if (!parsed.success) {
        parsed.error.issues.forEach((issue) => {
          errors.push({
            line: lineNumber,
            message: issue.message
          });
        });
        return;
      }

      const assistantMessages = parsed.data.messages.filter((message) => message.role === "assistant");
      if (assistantMessages.length === 0) {
        errors.push({
          line: lineNumber,
          message: "At least one assistant message is required."
        });
        return;
      }

      const hasEmptyAssistant = assistantMessages.some((message) => {
        if (typeof message.content === "string") {
          return message.content.trim().length === 0;
        }

        if (Array.isArray(message.content)) {
          return message.content.length === 0;
        }

        return !message.content;
      });

      if (hasEmptyAssistant) {
        errors.push({
          line: lineNumber,
          message: "Assistant content should not be empty."
        });
        return;
      }

      const hasNonStandardRole = parsed.data.messages.some((message) => {
        return !["system", "user", "assistant", "developer", "tool"].includes(message.role);
      });

      if (hasNonStandardRole) {
        warnings.push({
          line: lineNumber,
          message: "Record contains a non-standard role value."
        });
      }

      validRecords += 1;

      if (examples.length < 3) {
        examples.push({
          line: lineNumber,
          preview: parsed.data
        });
      }
    } catch {
      errors.push({
        line: lineNumber,
        message: "Line is not valid JSON."
      });
    }
  });

  return {
    totalRecords: lines.length,
    validRecords,
    invalidRecords: lines.length - validRecords,
    errors,
    warnings,
    examples
  };
}
