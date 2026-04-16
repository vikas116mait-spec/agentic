import { Card } from "@/components/ui/card";

export function PlaygroundComparison({
  baseOutput,
  tunedOutput
}: {
  baseOutput: string | null;
  tunedOutput: string | null;
}) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="bg-white">
        <p className="text-xs uppercase tracking-[0.2em] text-black/45">Base model</p>
        <p className="mt-3 whitespace-pre-wrap text-sm text-black/75">
          {baseOutput ?? "Run a prompt to see the base model response."}
        </p>
      </Card>
      <Card className="bg-white">
        <p className="text-xs uppercase tracking-[0.2em] text-black/45">Second model</p>
        <p className="mt-3 whitespace-pre-wrap text-sm text-black/75">
          {tunedOutput ?? "Select another model to compare its output here."}
        </p>
      </Card>
    </div>
  );
}
