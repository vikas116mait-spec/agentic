import fs from "node:fs";
import OpenAI from "openai";

type ModelProvider = "ollama" | "openai" | "huggingface" | "local" | "groq" | "gemini" | "cerebras" | "together";

const VALID_PROVIDERS = new Set<string>(["ollama", "openai", "huggingface", "local", "groq", "gemini", "cerebras", "together"]);

function getModelProvider(): ModelProvider {
  const configured = process.env.LLM_PROVIDER?.trim().toLowerCase();
  if (configured && !VALID_PROVIDERS.has(configured)) {
    throw new Error(`Unsupported LLM_PROVIDER \`${configured}\`. Supported: ollama, openai, huggingface, local, groq, gemini, cerebras, together.`);
  }

  if (configured && VALID_PROVIDERS.has(configured)) {
    return configured as ModelProvider;
  }

  return process.env.OPENAI_API_KEY ? "openai" : "ollama";
}

function getOllamaBaseURL() {
  const normalized = (process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434").trim().replace(/\/+$/, "");
  return normalized.endsWith("/v1") ? `${normalized}/` : `${normalized}/v1/`;
}

function ensureFineTuningAvailable() {
  if (getModelProvider() === "openai") {
    return;
  }

  throw new Error(
    "This legacy Next.js fine-tuning path is OpenAI-only. Use the Python API jobs flow for Hugging Face Jobs or Local GPU QLoRA training."
  );
}

function getClient() {
  const provider = getModelProvider();
  if (provider === "huggingface" || provider === "local") {
    throw new Error("This provider is a training backend in this app, not a direct inference client.");
  }

  if (getModelProvider() === "openai") {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY is not configured.");
    }

    return new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      baseURL: process.env.OPENAI_BASE_URL
    });
  }

  return new OpenAI({
    apiKey: process.env.OLLAMA_API_KEY ?? "ollama",
    baseURL: getOllamaBaseURL()
  });
}

export async function uploadTrainingFileToOpenAI(filePath: string) {
  ensureFineTuningAvailable();
  return getClient().files.create({
    file: fs.createReadStream(filePath),
    purpose: "fine-tune"
  });
}

export async function createFineTuningJob(
  trainingFileId: string,
  baseModel: string,
  methodConfig?: Record<string, unknown>
) {
  ensureFineTuningAvailable();
  return getClient().fineTuning.jobs.create({
    training_file: trainingFileId,
    model: baseModel,
    method: {
      type: "supervised",
      supervised: methodConfig ? { hyperparameters: methodConfig } : undefined
    }
  });
}

export async function retrieveFineTuningJob(openaiJobId: string) {
  ensureFineTuningAvailable();
  return getClient().fineTuning.jobs.retrieve(openaiJobId);
}

export async function listFineTuningEvents(openaiJobId: string) {
  ensureFineTuningAvailable();
  return getClient().fineTuning.jobs.listEvents(openaiJobId);
}

export async function cancelFineTuningJob(openaiJobId: string) {
  ensureFineTuningAvailable();
  return getClient().fineTuning.jobs.cancel(openaiJobId);
}

export async function runModel(messages: Array<{ role: "system" | "user" | "assistant"; content: string }>, model: string) {
  if (getModelProvider() === "ollama") {
    const response = await getClient().chat.completions.create({
      model,
      messages
    });

    return response.choices[0]?.message?.content ?? "";
  }

  const response = await getClient().responses.create({
    model,
    input: messages.map((message) => ({
      role: message.role,
      content: [{ type: "input_text", text: message.content }]
    }))
  });

  return response.output_text ?? "";
}
