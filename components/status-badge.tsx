import { cn } from "@/lib/utils";

const palette: Record<string, string> = {
  VALID: "bg-brand/15 text-brand",
  INVALID: "bg-danger/15 text-danger",
  PENDING: "bg-accent/20 text-black",
  succeeded: "bg-brand/15 text-brand",
  running: "bg-accent/30 text-black",
  queued: "bg-black/10 text-black",
  waiting: "bg-[#d8efe9] text-brand",
  failed: "bg-danger/15 text-danger",
  cancelled: "bg-black/10 text-black",
  paused: "bg-black/10 text-black",
  validating_files: "bg-accent/30 text-black",
  completed: "bg-brand/15 text-brand",
  ready: "bg-brand/15 text-brand",
  starting: "bg-accent/30 text-black",
  stopped: "bg-black/10 text-black",
  error: "bg-danger/15 text-danger",
  unknown: "bg-black/10 text-black"
};

export function StatusBadge({ value }: { value: string }) {
  return (
    <span className={cn("inline-flex rounded-full px-3 py-1 text-xs font-semibold", palette[value] ?? palette.unknown)}>
      {value}
    </span>
  );
}
