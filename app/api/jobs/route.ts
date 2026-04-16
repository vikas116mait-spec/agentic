import { Prisma } from "@prisma/client";
import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";
import { createJobSchema } from "@/lib/validators";
import { createFineTuningJob, uploadTrainingFileToOpenAI } from "@/lib/openai";
import { mapFineTuneJob } from "@/lib/jobs";

export async function POST(request: Request) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      throw new AppError("UNAUTHORIZED", "You must be signed in.", 401);
    }

    const input = createJobSchema.parse(await request.json());
    const dataset = await db.dataset.findFirst({
      where: {
        id: input.datasetId,
        userId: user.id
      }
    });

    if (!dataset) {
      throw new AppError("DATASET_NOT_FOUND", "Dataset not found.", 404);
    }

    if (dataset.validationStatus !== "VALID") {
      throw new AppError("DATASET_INVALID", "Only valid datasets can be used for fine-tuning.", 400);
    }

    let openaiFileId = dataset.openaiFileId;
    if (!openaiFileId) {
      const uploaded = await uploadTrainingFileToOpenAI(dataset.storagePath);
      openaiFileId = uploaded.id;
      await db.dataset.update({
        where: { id: dataset.id },
        data: {
          openaiFileId,
          uploadedToOpenAIAt: new Date()
        }
      });
    }

    const openaiJob = await createFineTuningJob(openaiFileId, input.baseModel, input.hyperparameters);
    const mapped = mapFineTuneJob(openaiJob);

    const job = await db.fineTuneJob.create({
      data: {
        userId: user.id,
        datasetId: dataset.id,
        openaiJobId: openaiJob.id,
        baseModel: input.baseModel,
        methodType: "supervised",
        status: mapped.status,
        fineTunedModel: mapped.fineTunedModel,
        trainedTokens: mapped.trainedTokens,
        estimatedFinishAt: mapped.estimatedFinishAt,
        resultFilesJson: mapped.resultFilesJson ? (mapped.resultFilesJson as Prisma.InputJsonValue) : Prisma.JsonNull,
        hyperparametersJson: input.hyperparameters ? (input.hyperparameters as Prisma.InputJsonValue) : Prisma.JsonNull,
        lastSyncedAt: new Date(),
        finishedAt: mapped.finishedAt
      }
    });

    return NextResponse.json(job);
  } catch (error) {
    return errorResponse(error);
  }
}
