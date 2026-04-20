import { cn } from "@/lib/utils";

type StatusConfig = {
  className: string;
  dot?: "pulse" | "solid" | "none";
};

const palette: Record<string, StatusConfig> = {
  VALID:            { className: "bg-brand/12 text-brand",            dot: "solid" },
  INVALID:          { className: "bg-danger/12 text-danger",           dot: "none" },
  PENDING:          { className: "bg-accent/20 text-black/70",         dot: "pulse" },
  succeeded:        { className: "bg-brand/12 text-brand",             dot: "solid" },
  completed:        { className: "bg-brand/12 text-brand",             dot: "solid" },
  ready:            { className: "bg-brand/12 text-brand",             dot: "solid" },
  running:          { className: "bg-accent/25 text-black/80",         dot: "pulse" },
  starting:         { className: "bg-accent/25 text-black/80",         dot: "pulse" },
  validating_files: { className: "bg-accent/25 text-black/80",         dot: "pulse" },
  waiting:          { className: "bg-[#d8efe9] text-brand",            dot: "pulse" },
  queued:           { className: "bg-black/8 text-black/60",           dot: "none" },
  paused:           { className: "bg-black/8 text-black/60",           dot: "none" },
  cancelled:        { className: "bg-black/8 text-black/60",           dot: "none" },
  stopped:          { className: "bg-black/8 text-black/60",           dot: "none" },
  failed:           { className: "bg-danger/12 text-danger",           dot: "none" },
  error:            { className: "bg-danger/12 text-danger",           dot: "none" },
  unknown:          { className: "bg-black/8 text-black/50",           dot: "none" },
};

function formatLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function StatusBadge({ value }: { value: string }) {
  const config = palette[value] ?? palette.unknown;

  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold", config.className)}>
      {config.dot === "pulse" && (
        <span className="relative flex h-1.5 w-1.5 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {config.dot === "solid" && (
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />
      )}
      {formatLabel(value)}
    </span>
  );
}
