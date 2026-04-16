"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { pythonApiFetch } from "@/lib/python-api";

export function FileUploadCard() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleUpload() {
    if (!file) {
      setMessage("Select a .jsonl file before uploading.");
      return;
    }

    setLoading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name || file.name.replace(/\.jsonl$/i, ""));

    try {
      const payload = await pythonApiFetch<{ id: string }>("/datasets", {
        method: "POST",
        body: formData
      });
      router.push(`/datasets/${payload.id}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
      setLoading(false);
      return;
    }
  }

  return (
    <Card className="space-y-4">
      <div>
        <p className="font-display text-2xl">Upload a training dataset</p>
        <p className="mt-2 text-sm text-black/60">
          We validate the file line by line before it can be sent to OpenAI.
        </p>
      </div>

      <input
        className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3"
        placeholder="Dataset name"
        value={name}
        onChange={(event) => setName(event.target.value)}
      />

      <label className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-black/15 bg-white px-6 py-12 text-center">
        <Upload className="h-10 w-10 text-brand" />
        <p className="mt-4 font-semibold">Choose a .jsonl file</p>
        <p className="mt-2 text-sm text-black/55">Uploaded files stay on your server and are only sent to OpenAI on request.</p>
        <input
          type="file"
          accept=".jsonl"
          className="hidden"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
      </label>

      {file ? <p className="text-sm text-black/65">Selected: {file.name}</p> : null}
      {message ? <p className="text-sm text-danger">{message}</p> : null}

      <Button onClick={handleUpload} disabled={loading}>
        {loading ? "Uploading..." : "Upload and validate"}
      </Button>
    </Card>
  );
}
