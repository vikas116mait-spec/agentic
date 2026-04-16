import { NextResponse } from "next/server";
import type { ApiErrorCode } from "@/lib/types";

export class AppError extends Error {
  code: ApiErrorCode;
  status: number;
  details: unknown[];

  constructor(code: ApiErrorCode, message: string, status = 400, details: unknown[] = []) {
    super(message);
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export function errorResponse(error: unknown) {
  if (error instanceof AppError) {
    return NextResponse.json(
      {
        error: {
          code: error.code,
          message: error.message,
          details: error.details
        }
      },
      { status: error.status }
    );
  }

  console.error(error);

  return NextResponse.json(
    {
      error: {
        code: "VALIDATION_ERROR",
        message: "Something went wrong while processing your request.",
        details: []
      }
    },
    { status: 500 }
  );
}
