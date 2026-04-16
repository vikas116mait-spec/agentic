import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";
import { uploadTrainingFileToOpenAI } from "@/lib/openai";

export async function POST(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      throw new AppError("UNAUTHORIZED", "You must be signed in.", 401);
    }

    const { id } = await params;
    const dataset = await db.dataset.findFirst({
      where: {
        id,
        userId: user.id
      }
    });

    if (!dataset) {
      throw new AppError("DATASET_NOT_FOUND", "Dataset not found.", 404);
    }

    if (dataset.validationStatus !== "VALID") {
      throw new AppError("DATASET_INVALID", "Dataset must validate successfully before upload.", 400);
    }

    const uploaded = await uploadTrainingFileToOpenAI(dataset.storagePath);
    const updated = await db.dataset.update({
      where: { id: dataset.id },
      data: {
        openaiFileId: uploaded.id,
        uploadedToOpenAIAt: new Date()
      }
    });

    return NextResponse.json(updated);
  } catch (error) {
    return errorResponse(error);
  }
}
