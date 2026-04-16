import bcrypt from "bcryptjs";
import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { AppError, errorResponse } from "@/lib/errors";
import { signupSchema } from "@/lib/validators";

export async function POST(request: Request) {
  try {
    const json = await request.json();
    const input = signupSchema.parse(json);

    const existing = await db.user.findUnique({
      where: {
        email: input.email
      }
    });

    if (existing) {
      throw new AppError("VALIDATION_ERROR", "An account with this email already exists.", 409);
    }

    const passwordHash = await bcrypt.hash(input.password, 10);
    const user = await db.user.create({
      data: {
        email: input.email,
        name: input.name,
        passwordHash
      }
    });

    return NextResponse.json({
      id: user.id,
      email: user.email
    });
  } catch (error) {
    return errorResponse(error);
  }
}
