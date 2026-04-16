import { Prisma } from "@prisma/client";
import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";
import { listFineTuningEvents, retrieveFineTuningJob } from "@/lib/openai";
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

    const [remoteJob, eventPage] = await Promise.all([
      retrieveFineTuningJob(job.openaiJobId),
      listFineTuningEvents(job.openaiJobId)
    ]);

    const mapped = mapFineTuneJob(remoteJob);

    await db.$transaction([
      db.fineTuneJob.update({
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
      }),
      ...eventPage.data.slice(0, 50).map((event) =>
        db.jobEvent.upsert({
          where: {
            id: `${job.id}-${event.id}`
          },
          create: {
            id: `${job.id}-${event.id}`,
            fineTuneJobId: job.id,
            level: event.level ?? "info",
            message: event.message,
            eventType: event.type ?? null,
            createdAt: new Date((event.created_at ?? Math.floor(Date.now() / 1000)) * 1000)
          },
          update: {
            level: event.level ?? "info",
            message: event.message,
            eventType: event.type ?? null
          }
        })
      )
    ]);

    const updated = await db.fineTuneJob.findUnique({
      where: { id: job.id },
      include: {
        events: {
          orderBy: { createdAt: "desc" },
          take: 50
        }
      }
    });

    return NextResponse.json(updated);
  } catch (error) {
    return errorResponse(error);
  }
}
