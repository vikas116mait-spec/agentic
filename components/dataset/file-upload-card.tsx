"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { pythonApiFetch } from "@/lib/python-api";

export function FileUploadCard() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function pickFile(picked: File | null | undefined) {
    if (picked && picked.name.endsWith(".jsonl")) {
      setFile(picked);
      setMessage(null);
    } else if (picked) {
      setMessage("Please choose a .jsonl file.");
    }
  }

  async function handleUpload() {
    if (!file) {
      setMessage("Choose a .jsonl file first.");
      return;
    }
    setLoading(true);
    setMessage(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name.replace(/\.jsonl$/i, ""));
    try {
      const payload = await pythonApiFetch<{ id: string }>("/datasets", {
        method: "POST",
        body: formData
      });
      router.push(`/datasets/${payload.id}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
      setLoading(false);
    }
  }

  return (
    <Card className="mx-auto max-w-lg space-y-5">
      <div>
        <p className="font-display text-2xl">Upload a dataset</p>
        <p className="mt-1.5 text-sm text-black/55">
          Bring one `.jsonl` file. It is validated line by line before you start training.
        </p>
      </div>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
        className={`flex w-full cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed px-6 py-14 text-center transition ${
          dragging ? "border-brand bg-brand/5" : file ? "border-brand/40 bg-brand/5" : "border-black/15 bg-white hover:border-brand/40"
        }`}
      >
        <Upload className={`h-10 w-10 ${file ? "text-brand" : "text-black/30"}`} />
        {file ? (
          <>
            <p className="mt-4 font-semibold text-brand">{file.name}</p>
            <p className="mt-1 text-sm text-black/45">Click to choose a different file</p>
          </>
        ) : (
          <>
            <p className="mt-4 font-semibold">Drop a .jsonl file here</p>
            <p className="mt-1 text-sm text-black/45">or click to browse</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".jsonl"
          className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
      </button>

      {message ? <p className="text-sm text-danger">{message}</p> : null}

      <Button className="w-full" onClick={handleUpload} disabled={loading || !file}>
        {loading ? "Uploading..." : "Upload and validate"}
      </Button>

      <p className="text-center text-xs text-black/40">
        Both chat-style `messages` and `instruction/input/output` formats are supported.
      </p>
    </Card>
  );
}
