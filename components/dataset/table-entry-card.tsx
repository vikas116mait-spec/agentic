"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { pythonApiFetch } from "@/lib/python-api";

type Format = "instruction" | "chat";

type InstructionRow = { instruction: string; input: string; output: string };
type ChatRow = { system: string; user: string; assistant: string };

function emptyInstruction(): InstructionRow {
  return { instruction: "", input: "", output: "" };
}
function emptyChat(): ChatRow {
  return { system: "", user: "", assistant: "" };
}

function rowsToJsonl(format: Format, rows: InstructionRow[] | ChatRow[]): string {
  const records = (rows as (InstructionRow | ChatRow)[])
    .filter((row) => {
      if (format === "instruction") {
        const r = row as InstructionRow;
        return r.instruction.trim() && r.output.trim();
      }
      const r = row as ChatRow;
      return r.user.trim() && r.assistant.trim();
    })
    .map((row) => {
      if (format === "instruction") {
        const r = row as InstructionRow;
        const record: Record<string, string> = {
          instruction: r.instruction.trim(),
          output: r.output.trim(),
        };
        if (r.input.trim()) record.input = r.input.trim();
        return record;
      }
      const r = row as ChatRow;
      const messages: { role: string; content: string }[] = [];
      if (r.system.trim()) messages.push({ role: "system", content: r.system.trim() });
      messages.push({ role: "user", content: r.user.trim() });
      messages.push({ role: "assistant", content: r.assistant.trim() });
      return { messages };
    });
  return records.map((r) => JSON.stringify(r)).join("\n");
}

const cellClass =
  "w-full resize-none rounded-xl border border-black/10 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand/40 min-h-[60px]";

export function TableEntryCard() {
  const router = useRouter();
  const [format, setFormat] = useState<Format>("instruction");
  const [name, setName] = useState("");
  const [instructionRows, setInstructionRows] = useState<InstructionRow[]>([emptyInstruction()]);
  const [chatRows, setChatRows] = useState<ChatRow[]>([emptyChat()]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const rows = format === "instruction" ? instructionRows : chatRows;
  const validCount = rows.filter((row) => {
    if (format === "instruction") {
      const r = row as InstructionRow;
      return r.instruction.trim() && r.output.trim();
    }
    const r = row as ChatRow;
    return r.user.trim() && r.assistant.trim();
  }).length;

  function updateInstruction(index: number, field: keyof InstructionRow, value: string) {
    setInstructionRows((prev) => prev.map((r, i) => (i === index ? { ...r, [field]: value } : r)));
  }

  function updateChat(index: number, field: keyof ChatRow, value: string) {
    setChatRows((prev) => prev.map((r, i) => (i === index ? { ...r, [field]: value } : r)));
  }

  function addRow() {
    if (format === "instruction") setInstructionRows((prev) => [...prev, emptyInstruction()]);
    else setChatRows((prev) => [...prev, emptyChat()]);
  }

  function removeRow(index: number) {
    if (format === "instruction") {
      setInstructionRows((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== index) : prev));
    } else {
      setChatRows((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== index) : prev));
    }
  }

  async function handleUpload() {
    if (validCount === 0) {
      setMessage("Add at least one complete row before uploading.");
      return;
    }
    const datasetName = name.trim() || `manual-dataset-${Date.now()}`;
    const jsonl = rowsToJsonl(
      format,
      format === "instruction" ? instructionRows : chatRows
    );
    const blob = new Blob([jsonl], { type: "text/plain" });
    const file = new File([blob], `${datasetName}.jsonl`, { type: "text/plain" });
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", datasetName);
    setLoading(true);
    setMessage(null);
    try {
      const payload = await pythonApiFetch<{ id: string }>("/datasets", {
        method: "POST",
        body: formData,
      });
      router.push(`/datasets/${payload.id}`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Upload failed.");
      setLoading(false);
    }
  }

  return (
    <Card className="mx-auto max-w-5xl space-y-5">
      <div>
        <p className="font-display text-2xl">Enter data manually</p>
        <p className="mt-1.5 text-sm text-black/55">
          Fill in the table — each row becomes one JSONL training record.
        </p>
      </div>

      {/* Format & name */}
      <div className="flex flex-wrap items-end gap-4">
        <label className="space-y-1 text-sm">
          <span className="font-medium text-black/70">Format</span>
          <div className="flex rounded-2xl border border-black/10 overflow-hidden">
            {(["instruction", "chat"] as Format[]).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFormat(f)}
                className={`px-4 py-2 text-sm capitalize transition ${
                  format === f
                    ? "bg-brand text-white font-medium"
                    : "bg-white text-black/60 hover:bg-black/5"
                }`}
              >
                {f === "instruction" ? "Instruction / Output" : "Chat messages"}
              </button>
            ))}
          </div>
        </label>

        <label className="flex-1 min-w-[200px] space-y-1 text-sm">
          <span className="font-medium text-black/70">Dataset name</span>
          <input
            type="text"
            placeholder="e.g. my-training-data"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-2xl border border-black/10 bg-white px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </label>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-black/10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-black/10 bg-black/3">
              <th className="w-8 px-3 py-2 text-left text-xs font-medium text-black/40">#</th>
              {format === "instruction" ? (
                <>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">Instruction *</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">Input (optional)</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">Output *</th>
                </>
              ) : (
                <>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">System (optional)</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">User *</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-black/60">Assistant *</th>
                </>
              )}
              <th className="w-10 px-2 py-2" />
            </tr>
          </thead>
          <tbody className="divide-y divide-black/6">
            {format === "instruction"
              ? instructionRows.map((row, i) => (
                  <tr key={i} className="bg-white hover:bg-black/1 transition">
                    <td className="px-3 py-2 text-xs text-black/30 align-top pt-3">{i + 1}</td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="What is the capital of France?"
                        value={row.instruction}
                        onChange={(e) => updateInstruction(i, "instruction", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="(optional context)"
                        value={row.input}
                        onChange={(e) => updateInstruction(i, "input", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="Paris"
                        value={row.output}
                        onChange={(e) => updateInstruction(i, "output", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2 align-top pt-3">
                      <button
                        type="button"
                        onClick={() => removeRow(i)}
                        disabled={instructionRows.length === 1}
                        className="text-black/25 hover:text-danger transition disabled:opacity-30"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))
              : chatRows.map((row, i) => (
                  <tr key={i} className="bg-white hover:bg-black/1 transition">
                    <td className="px-3 py-2 text-xs text-black/30 align-top pt-3">{i + 1}</td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="You are a helpful assistant."
                        value={row.system}
                        onChange={(e) => updateChat(i, "system", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="What is the capital of France?"
                        value={row.user}
                        onChange={(e) => updateChat(i, "user", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2">
                      <textarea
                        className={cellClass}
                        placeholder="Paris"
                        value={row.assistant}
                        onChange={(e) => updateChat(i, "assistant", e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-2 align-top pt-3">
                      <button
                        type="button"
                        onClick={() => removeRow(i)}
                        disabled={chatRows.length === 1}
                        className="text-black/25 hover:text-danger transition disabled:opacity-30"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
          </tbody>
        </table>
      </div>

      <button
        type="button"
        onClick={addRow}
        className="flex items-center gap-2 text-sm text-brand hover:text-brand/80 transition"
      >
        <Plus className="h-4 w-4" />
        Add row
      </button>

      {validCount > 0 && (
        <p className="text-xs text-black/45">
          {validCount} complete row{validCount !== 1 ? "s" : ""} will be saved as JSONL.
        </p>
      )}

      {message ? <p className="text-sm text-danger">{message}</p> : null}

      <Button className="w-full" onClick={handleUpload} disabled={loading || validCount === 0}>
        {loading ? "Saving..." : "Save as dataset"}
      </Button>
    </Card>
  );
}
