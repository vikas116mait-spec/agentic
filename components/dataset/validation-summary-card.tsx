import type { DatasetValidationSummary } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { JsonPreview } from "@/components/ui/json-preview";

const taskColors: Record<string, string> = {
  coding:         "bg-brand/12 text-brand",
  qa:             "bg-sky-100 text-sky-700",
  translation:    "bg-purple-100 text-purple-700",
  summarization:  "bg-orange-100 text-orange-700",
  math:           "bg-yellow-100 text-yellow-800",
  explanation:    "bg-teal-100 text-teal-700",
  writing:        "bg-pink-100 text-pink-700",
  reasoning:      "bg-indigo-100 text-indigo-700",
  classification: "bg-red-100 text-red-700",
  extraction:     "bg-lime-100 text-lime-700",
  default:        "bg-black/8 text-black/60",
};

function TaskTag({ task }: { task: string }) {
  const key = task.toLowerCase().replace(/\s+/g, "_");
  const cls = taskColors[key] ?? taskColors.default;
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize ${cls}`}>
      {task}
    </span>
  );
}

export function ValidationSummaryCard({ summary }: { summary: DatasetValidationSummary }) {
  return (
    <Card className="space-y-5">
      <div>
        <p className="font-display text-2xl">Validation summary</p>
        <p className="mt-2 text-sm text-black/60">
          Focused diagnostics help you fix dataset problems before training starts.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl bg-white p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Total records</p>
          <p className="mt-2 text-2xl font-semibold">{summary.totalRecords}</p>
        </div>
        <div className="rounded-2xl bg-white p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Valid</p>
          <p className="mt-2 text-2xl font-semibold text-brand">{summary.validRecords}</p>
        </div>
        <div className="rounded-2xl bg-white p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-black/45">Invalid</p>
          <p className="mt-2 text-2xl font-semibold text-danger">{summary.invalidRecords}</p>
        </div>
      </div>

      <div>
        <p className="font-semibold">Examples</p>
        <div className="mt-3 space-y-3">
          {summary.examples.length === 0 ? <p className="text-sm text-black/55">No valid examples parsed yet.</p> : null}
          {summary.examples.map((example) => {
            const task = typeof example.preview === "object" && example.preview !== null
              ? (example.preview as Record<string, unknown>).task as string | undefined
              : undefined;
            return (
              <div key={example.line}>
                <div className="mb-2 flex items-center gap-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-black/45">Line {example.line}</p>
                  {task && <TaskTag task={task} />}
                </div>
                <JsonPreview value={example.preview} />
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <p className="font-semibold">Errors</p>
        <div className="mt-3 space-y-2">
          {summary.errors.length === 0 ? <p className="text-sm text-brand">No errors found.</p> : null}
          {summary.errors.slice(0, 12).map((error) => (
            <div key={`${error.line}-${error.message}`} className="rounded-2xl bg-danger/10 p-3 text-sm text-danger">
              Line {error.line}: {error.message}
            </div>
          ))}
        </div>
      </div>

      <div>
        <p className="font-semibold">Warnings</p>
        <div className="mt-3 space-y-2">
          {summary.warnings.length === 0 ? <p className="text-sm text-black/55">No warnings.</p> : null}
          {summary.warnings.slice(0, 8).map((warning, index) => (
            <div key={`${warning.line ?? index}-${warning.message}`} className="rounded-2xl bg-accent/20 p-3 text-sm">
              {warning.line ? `Line ${warning.line}: ` : ""}
              {warning.message}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
