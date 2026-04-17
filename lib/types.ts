export type DatasetValidationError = {
  line: number;
  message: string;
};

export type DatasetValidationWarning = {
  line?: number;
  message: string;
};

export type DatasetValidationExample = {
  line: number;
  preview: unknown;
};

export type ModelProvider = "ollama" | "openai" | "huggingface" | "local";

export type ModelProfileCategory = "small" | "medium" | "large" | "thinking" | "custom";

export type ProviderStatus = {
  provider: ModelProvider;
  label: string;
  baseUrl: string | null;
  configured: boolean;
  supportsInference: boolean;
  supportsFineTuning: boolean;
};

export type ModelProfile = {
  id: string;
  name: string;
  provider: ModelProvider;
  providerLabel: string;
  providerConfigured: boolean;
  supportsInference: boolean;
  supportsFineTuning: boolean;
  model: string;
  category: ModelProfileCategory;
  description: string | null;
  createdAt: string;
  updatedAt: string;
};

export type ModelProfileDefaults = {
  playgroundBaseProfileId: string | null;
  playgroundCompareProfileId: string | null;
  agentBaseProfileId: string | null;
  agentModelProfileId: string | null;
  jobBaseProfileId: string | null;
};

export type ModelProfilesResponse = {
  profiles: ModelProfile[];
  defaults: ModelProfileDefaults;
  providers: ProviderStatus[];
};

export type DatasetValidationSummary = {
  totalRecords: number;
  validRecords: number;
  invalidRecords: number;
  errors: DatasetValidationError[];
  warnings: DatasetValidationWarning[];
  examples: DatasetValidationExample[];
};

export type FineTuneJobStatus =
  | "validating_files"
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "paused"
  | "unknown";

export type PlaygroundRunResult = {
  baseOutput: string | null;
  tunedOutput: string | null;
};

export type ApiErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "DATASET_NOT_FOUND"
  | "DATASET_INVALID"
  | "OPENAI_UPLOAD_FAILED"
  | "FINE_TUNING_UNSUPPORTED"
  | "JOB_CREATION_FAILED"
  | "JOB_SYNC_FAILED"
  | "MODEL_PROVIDER_INVALID"
  | "MODEL_PROVIDER_NOT_CONFIGURED"
  | "PLAYGROUND_RUN_FAILED"
  | "VALIDATION_ERROR"
  | "NOT_FOUND";
