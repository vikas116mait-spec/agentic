import { Prisma } from "@prisma/client";
import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";
import { cancelFineTuningJob } from "@/lib/openai";
import { mapFineTuneJob } from "@/lib/jobs";

export async function POST(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      throw new AppError("UNAUTHORIZED", "You must be signed in.", 401);
    }

    const { id } = await params;
    const job = await db.fineTuneJob.findFirst({
      where: {
        id,
        userId: user.id
      }
    });

    if (!job) {
      throw new AppError("NOT_FOUND", "Job not found.", 404);
    }

    const cancelled = await cancelFineTuningJob(job.openaiJobId);
    const mapped = mapFineTuneJob(cancelled);

    const updated = await db.fineTuneJob.update({
      where: { id: job.id },
      data: {
        status: mapped.status,
        fineTunedModel: mapped.fineTunedModel,
        trainedTokens: mapped.trainedTokens,
        estimatedFinishAt: mapped.estimatedFinishAt,
        resultFilesJson: mapped.resultFilesJson ? (mapped.resultFilesJson as Prisma.InputJsonValue) : Prisma.JsonNull,
        hyperparametersJson: mapped.hyperparametersJson ? (mapped.hyperparametersJson as Prisma.InputJsonValue) : Prisma.JsonNull,
        finishedAt: mapped.finishedAt,
        lastSyncedAt: new Date()
      }
    });

    return NextResponse.json(updated);
  } catch (error) {
    return errorResponse(error);
  }
}
