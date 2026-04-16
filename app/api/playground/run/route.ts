import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { getCurrentUser } from "@/lib/auth";
import { runModel } from "@/lib/openai";
import { playgroundRunSchema } from "@/lib/validators";

export async function POST(request: Request) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      throw new AppError("UNAUTHORIZED", "You must be signed in.", 401);
    }

    const input = playgroundRunSchema.parse(await request.json());
    const messages = [{ role: "user" as const, content: input.prompt }];

    const [baseOutput, tunedOutput] = await Promise.all([
      runModel(messages, input.baseModel),
      input.fineTunedModel ? runModel(messages, input.fineTunedModel) : Promise.resolve(null)
    ]);

    const run = await db.playgroundRun.create({
      data: {
        userId: user.id,
        baseModel: input.baseModel,
        fineTunedModel: input.fineTunedModel ?? null,
        prompt: input.prompt,
        baseOutput,
        tunedOutput
      }
    });

    return NextResponse.json({
      id: run.id,
      baseOutput,
      tunedOutput
    });
  } catch (error) {
    return errorResponse(error);
  }
}
