import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
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

    const events = await db.jobEvent.findMany({
      where: {
        fineTuneJobId: job.id
      },
      orderBy: { createdAt: "desc" }
    });

    return NextResponse.json(events);
  } catch (error) {
    return errorResponse(error);
  }
}
