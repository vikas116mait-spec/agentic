import { AlertTriangle } from "lucide-react";

export function ErrorAlert({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-3xl border border-danger/20 bg-danger/10 p-4 text-danger">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <p className="font-semibold">{title}</p>
          <p className="mt-1 text-sm text-danger/85">{description}</p>
        </div>
      </div>
    </div>
  );
}
