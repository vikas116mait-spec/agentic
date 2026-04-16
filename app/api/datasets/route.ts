import { Prisma } from "@prisma/client";
import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { validateJsonlFile } from "@/lib/jsonl";
import { saveUploadedFile } from "@/lib/storage";
import { getCurrentUser } from "@/lib/auth";

export async function POST(request: Request) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      throw new AppError("UNAUTHORIZED", "You must be signed in to upload datasets.", 401);
    }

    const formData = await request.formData();
    const file = formData.get("file");
    const name = formData.get("name");

    if (!(file instanceof File)) {
      throw new AppError("VALIDATION_ERROR", "A JSONL file is required.", 400);
    }

    if (!file.name.toLowerCase().endsWith(".jsonl")) {
      throw new AppError("DATASET_INVALID", "Only .jsonl files are supported.", 400);
    }

    const dataset = await db.dataset.create({
      data: {
        userId: user.id,
        name: typeof name === "string" && name.trim().length > 0 ? name : file.name,
        originalFilename: file.name,
        storagePath: "",
        fileSizeBytes: file.size,
        validationSummaryJson: {} as Prisma.InputJsonValue,
        recordCount: 0
      }
    });

    const stored = await saveUploadedFile(file, user.id, dataset.id);
    const summary = await validateJsonlFile(stored.storagePath);

    const updated = await db.dataset.update({
      where: { id: dataset.id },
      data: {
        storagePath: stored.storagePath,
        originalFilename: stored.originalFilename,
        fileSizeBytes: stored.size,
        recordCount: summary.totalRecords,
        validationStatus: summary.invalidRecords > 0 ? "INVALID" : "VALID",
        validationSummaryJson: summary as Prisma.InputJsonValue
      }
    });

    return NextResponse.json({
      id: updated.id,
      validationSummary: summary
    });
  } catch (error) {
    return errorResponse(error);
  }
}
