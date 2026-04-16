import { z } from "zod";

export const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8, "Password must be at least 8 characters."),
  name: z.string().trim().min(1).max(80).optional()
});

export const createJobSchema = z.object({
  datasetId: z.string().min(1),
  baseModel: z.string().min(1),
  hyperparameters: z
    .object({
      n_epochs: z.union([z.literal("auto"), z.number().int().positive()]).optional(),
      batch_size: z.union([z.literal("auto"), z.number().int().positive()]).optional(),
      learning_rate_multiplier: z.union([z.literal("auto"), z.number().positive()]).optional()
    })
    .optional()
});

export const playgroundRunSchema = z.object({
  baseModel: z.string().min(1),
  fineTunedModel: z.string().min(1).optional().nullable(),
  prompt: z.string().min(1, "Prompt is required.")
});
