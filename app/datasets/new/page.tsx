"use client";

import { useState } from "react";
import { FileUploadCard } from "@/components/dataset/file-upload-card";
import { TableEntryCard } from "@/components/dataset/table-entry-card";

type Tab = "upload" | "manual";

export default function NewDatasetPage() {
  const [tab, setTab] = useState<Tab>("upload");

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      {/* Tab switcher */}
      <div className="flex gap-1 rounded-2xl border border-black/10 bg-black/3 p-1 w-fit">
        {([
          { key: "upload", label: "Upload file" },
          { key: "manual", label: "Enter data manually" },
        ] as { key: Tab; label: string }[]).map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={`rounded-xl px-5 py-2 text-sm font-medium transition ${
              tab === key
                ? "bg-white text-black shadow-sm"
                : "text-black/50 hover:text-black/80"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "upload" ? <FileUploadCard /> : <TableEntryCard />}
    </div>
  );
}
