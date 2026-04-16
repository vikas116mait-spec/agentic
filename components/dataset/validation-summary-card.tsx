import type { DatasetValidationSummary } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { JsonPreview } from "@/components/ui/json-preview";

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
          {summary.examples.map((example) => (
            <div key={example.line}>
              <p className="mb-2 text-xs uppercase tracking-[0.2em] text-black/45">Line {example.line}</p>
              <JsonPreview value={example.preview} />
            </div>
          ))}
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
