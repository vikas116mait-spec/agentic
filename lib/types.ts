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
