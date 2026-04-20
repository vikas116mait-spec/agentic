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

const chatTrainingRecordSchema = z.object({
  messages: z.array(messageSchema).min(1, "messages must not be empty")
});

const instructionTrainingRecordSchema = z.object({
  instruction: z.string().trim().min(1, "instruction must not be empty"),
  input: z.string().optional().default(""),
  output: z.string().trim().min(1, "output must not be empty")
});

function validateRecord(record: unknown) {
  const chatParsed = chatTrainingRecordSchema.safeParse(record);
  if (chatParsed.success) {
    const assistantMessages = chatParsed.data.messages.filter((message) => message.role === "assistant");
    if (assistantMessages.length === 0) {
      return {
        errors: ["At least one assistant message is required."],
        warnings: [],
        preview: null
      };
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
      return {
        errors: ["Assistant content should not be empty."],
        warnings: [],
        preview: null
      };
    }

    const warnings = chatParsed.data.messages.some((message) => {
      return !["system", "user", "assistant", "developer", "tool"].includes(message.role);
    })
      ? ["Record contains a non-standard role value."]
      : [];

    return {
      errors: [],
      warnings,
      preview: chatParsed.data
    };
  }

  const instructionParsed = instructionTrainingRecordSchema.safeParse(record);
  if (instructionParsed.success) {
    return {
      errors: [],
      warnings: [],
      preview: instructionParsed.data
    };
  }

  return {
    errors: ["Each line must contain either a `messages` array or `instruction`/`output` fields."],
    warnings: [],
    preview: null
  };
}

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
      const validation = validateRecord(record);

      if (validation.errors.length > 0) {
        validation.errors.forEach((message) => {
          errors.push({
            line: lineNumber,
            message
          });
        });
        return;
      }

      validation.warnings.forEach((message) => {
        warnings.push({
          line: lineNumber,
          message
        });
      });

      validRecords += 1;

      if (examples.length < 3 && validation.preview) {
        examples.push({
          line: lineNumber,
          preview: validation.preview
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
