import { promises as fs } from "node:fs";
import path from "node:path";
import { slugifyFileName } from "@/lib/utils";

const uploadsRoot = path.join(process.cwd(), "uploads");

export async function ensureUploadsRoot() {
  await fs.mkdir(uploadsRoot, { recursive: true });
  return uploadsRoot;
}

export async function saveUploadedFile(file: File, userId: string, datasetId: string) {
  const root = await ensureUploadsRoot();
  const userDir = path.join(root, userId, datasetId);
  await fs.mkdir(userDir, { recursive: true });

  const originalFilename = file.name || "dataset.jsonl";
  const fileName = slugifyFileName(originalFilename);
  const storagePath = path.join(userDir, fileName);
  const buffer = Buffer.from(await file.arrayBuffer());
  await fs.writeFile(storagePath, buffer);

  return {
    storagePath,
    originalFilename,
    size: buffer.byteLength
  };
}
